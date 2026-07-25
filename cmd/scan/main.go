// Command scan estimates Model FLOPs Utilization (MFU) for a running
// inference server from its Prometheus /metrics endpoint and the GPU
// model. No eBPF, no root. Every number it prints is a modeled
// estimate, not a kernel measurement.
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ingero-io/scan/internal/device"
	"github.com/ingero-io/scan/internal/mfu"
	"github.com/ingero-io/scan/internal/scrape"
	"github.com/spf13/cobra"
)

// version is set at release build time via -ldflags "-X main.version=vX.Y.Z".
var version = "dev"

var (
	flagEndpoint   string
	flagEngine     string
	flagModel      string
	flagGPU        string
	flagGPUCount   int
	flagParams     float64
	flagRate       float64
	flagInterval   time.Duration
	flagPrometheus string
)

var rootCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan your GPU for wasted spend - the MFU gap (no root, no eBPF)",
	Long: `scan reads a serving engine's Prometheus /metrics endpoint twice over a
short window, derives achieved output tokens/sec, and estimates Model FLOPs
Utilization (MFU) against the GPU's peak: the gap between the utilization your
dashboard shows (read live from nvidia-smi or amd-smi) and the real work the GPU
is doing.

Runs on NVIDIA and on AMD/ROCm hosts, and needs a GPU: it reads live utilization
and exits if none is present. Everything it prints is an ESTIMATE (MFU is
modeled, not measured). No eBPF, no root - it only reads metrics the engine
already exposes.

  scan --endpoint http://localhost:8000/metrics --model llama-3-70b
  scan --model mixtral-8x7b --rate 2.49`,
	RunE:          run,
	SilenceUsage:  true,
	SilenceErrors: true,
	Version:       version, // enables `scan --version`
}

func init() {
	f := rootCmd.Flags()
	f.StringVar(&flagEndpoint, "endpoint", "http://localhost:8000/metrics", "serving engine Prometheus /metrics URL")
	f.StringVar(&flagEngine, "engine", "vllm", "serving engine: vllm|sglang|tgi")
	f.StringVar(&flagModel, "model", "", "model name (e.g. llama-3-70b); or use --params")
	f.Float64Var(&flagParams, "params", 0, "active params/token for an unknown model (e.g. 8e9)")
	f.StringVar(&flagGPU, "gpu", "", "override the detected GPU model used for the rate/peak tables")
	f.IntVar(&flagGPUCount, "gpu-count", 0, "override the detected GPU count")
	f.Float64Var(&flagRate, "rate", 0, "your real USD/hr per GPU (overrides the bundled list-price estimate)")
	f.DurationVar(&flagInterval, "interval", 15*time.Second, "sampling window")
	f.StringVar(&flagPrometheus, "prometheus", "", "serve Prometheus exposition on this addr (e.g. :9100); re-samples every --interval instead of running once")
}

func main() {
	// A signal-cancelled context so the long-running --prometheus loop (and any
	// in-flight device read or scrape) stops cleanly on Ctrl-C / SIGTERM. The
	// one-shot path ignores it.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := rootCmd.ExecuteContext(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(cmd *cobra.Command, args []string) error {
	if flagModel == "" && flagParams <= 0 {
		return fmt.Errorf("--model or --params is required")
	}
	if flagInterval <= 0 {
		return fmt.Errorf("--interval must be > 0, got %s", flagInterval)
	}
	parser, err := mfu.ParserFor(flagEngine)
	if err != nil {
		return err
	}

	// scan is a GPU tool. A real GPU must be present: the vendor tool's own
	// reading is the evidence the numbers describe a real device, so we require
	// it even when --gpu/--gpu-count name the model explicitly (those only
	// override the name and count used for the rate/peak tables; they do not let
	// scan run GPU-less).
	dev, derr := device.Detect(cmd.Context())
	if derr != nil {
		return fmt.Errorf("no GPU detected (%w)\n  scan measures the live MFU gap on a serving GPU and must run on the GPU host", derr)
	}
	gpu, count := flagGPU, flagGPUCount
	if gpu == "" {
		gpu = dev.Name
	}
	if count == 0 {
		count = dev.Count
	}

	if flagPrometheus != "" {
		return runPrometheus(cmd, parser, dev.Vendor, gpu, count)
	}

	est, rateSource, err := sampleEstimate(cmd.Context(), cmd.ErrOrStderr(), parser, dev.Vendor, gpu, count)
	if err != nil {
		return err
	}
	fmt.Fprint(cmd.OutOrStdout(), mfu.Render(est, rateSource))
	return nil
}

// sampleEstimate runs one scrape -> sample-window -> scrape cycle and computes
// the MFU Estimate. It is the single estimate path shared by the one-shot CLI
// and the Prometheus loop, so the MFU math is never duplicated. rateSource is
// empty when no rate was available.
func sampleEstimate(ctx context.Context, logw io.Writer, parser scrape.Parser, vendor device.Vendor, gpu string, count int) (mfu.Estimate, string, error) {
	t0, err := scrapeOutputTokens(ctx, parser, flagEndpoint)
	if err != nil {
		return mfu.Estimate{}, "", fmt.Errorf("scrape %s: %w", flagEndpoint, err)
	}
	fmt.Fprintf(logw, "sampling %s for %s...\n", flagEndpoint, flagInterval)
	// Wait out the throughput window and, over the same window, poll the live
	// GPU utilization so the report contrasts MFU against a real reading.
	util, err := device.SampleBusy(ctx, vendor, flagInterval)
	if err != nil {
		return mfu.Estimate{}, "", err
	}
	var utilPtr *float64
	if util >= 0 {
		utilPtr = &util
	}
	t1, err := scrapeOutputTokens(ctx, parser, flagEndpoint)
	if err != nil {
		return mfu.Estimate{}, "", fmt.Errorf("scrape %s: %w", flagEndpoint, err)
	}

	tps := mfu.TokensPerSec(t0, t1, flagInterval)
	if tps <= 0 {
		return mfu.Estimate{}, "", fmt.Errorf("no output-token throughput observed for engine %q over %s (idle, or %q does not export generation tokens)", flagEngine, flagInterval, flagEngine)
	}

	rate, rateSource := flagRate, "--rate"
	if rate <= 0 {
		if r, prov, ok := mfu.LookupRate(gpu); ok {
			rate, rateSource = r, prov+" list price"
		} else {
			rateSource = ""
		}
	}

	est, err := mfu.Compute(mfu.Input{
		Model: flagModel, ParamsOverride: flagParams, GPU: gpu, GPUCount: count,
		TokensPerSec: tps, HourlyUSDPerGPU: rate, GPUUtilPct: utilPtr,
		UtilTool: vendor.Tool(),
	})
	if err != nil {
		return mfu.Estimate{}, "", err
	}
	return est, rateSource, nil
}

// runPrometheus serves the public MFU-gap board on flagPrometheus, re-sampling
// every --interval into an atomic snapshot the /metrics handler reads. It loops
// until the command context is cancelled (SIGINT). A transient sample error
// (engine briefly down, idle) is logged and retried on the next interval rather
// than killing the exporter.
func runPrometheus(cmd *cobra.Command, parser scrape.Parser, vendor device.Vendor, gpu string, count int) error {
	ctx := cmd.Context()
	snap := &snapshot{}
	srv := serveMetrics(flagPrometheus, snap)

	fmt.Fprintf(cmd.ErrOrStderr(), "scan: serving Prometheus exposition on %s/metrics (re-sampling every %s)\n", flagPrometheus, flagInterval)
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			fmt.Fprintln(cmd.ErrOrStderr(), "scan: metrics server error:", err)
		}
	}()
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	for {
		// sampleEstimate blocks for --interval (the sampling window), so the loop
		// paces itself; no extra sleep is needed.
		est, rateSource, err := sampleEstimate(ctx, cmd.ErrOrStderr(), parser, vendor, gpu, count)
		if err != nil {
			if ctx.Err() != nil {
				return nil // cancelled mid-sample: clean exit
			}
			fmt.Fprintln(cmd.ErrOrStderr(), "scan: sample error (keeping last snapshot):", err)
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(flagInterval):
				continue
			}
		}
		snap.set(renderPrometheus(est, rateSource != ""))
	}
}

// httpClient bounds a hung /metrics endpoint (likely, since the engine
// is the thing under stress) instead of blocking the scrape forever.
var httpClient = &http.Client{Timeout: 30 * time.Second}

// scrapeOutputTokens fetches the engine /metrics body once and returns
// the running output-token counter total.
func scrapeOutputTokens(ctx context.Context, parser scrape.Parser, url string) (float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return 0, err
	}
	samples, err := parser.Parse(body)
	if err != nil {
		return 0, err
	}
	return mfu.SumOutputTokens(samples), nil
}

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
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/ingero-io/scan/internal/mfu"
	"github.com/ingero-io/scan/internal/scrape"
	"github.com/spf13/cobra"
)

// version is set at release build time via -ldflags "-X main.version=vX.Y.Z".
var version = "dev"

var (
	flagEndpoint string
	flagEngine   string
	flagModel    string
	flagGPU      string
	flagGPUCount int
	flagParams   float64
	flagRate     float64
	flagInterval time.Duration
)

var rootCmd = &cobra.Command{
	Use:   "scan",
	Short: "Scan your GPU for wasted spend - the MFU gap (no root, no eBPF)",
	Long: `scan reads a serving engine's Prometheus /metrics endpoint twice over a
short window, derives achieved output tokens/sec, and estimates Model FLOPs
Utilization (MFU) against the GPU's peak: the gap between the utilization your
dashboard shows (read live from nvidia-smi) and the real work the GPU is doing.

Requires an NVIDIA GPU - it reads live utilization and exits if none is present.
Everything it prints is an ESTIMATE (MFU is modeled, not measured). No eBPF,
no root - it only reads metrics the engine already exposes.

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
}

func main() {
	if err := rootCmd.Execute(); err != nil {
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

	// scan is a GPU tool. A real NVIDIA GPU must be present: nvidia-smi is the
	// evidence the numbers describe a real device, so we require it even when
	// --gpu/--gpu-count name the model explicitly (those only override the name
	// and count used for the rate/peak tables; they do not let scan run GPU-less).
	dn, dc, derr := detectGPU(cmd.Context())
	if derr != nil {
		return fmt.Errorf("no NVIDIA GPU detected (%w)\n  scan measures the live MFU gap on a serving GPU and must run on the GPU host", derr)
	}
	gpu, count := flagGPU, flagGPUCount
	if gpu == "" {
		gpu = dn
	}
	if count == 0 {
		count = dc
	}

	t0, err := scrapeOutputTokens(cmd.Context(), parser, flagEndpoint)
	if err != nil {
		return fmt.Errorf("scrape %s: %w", flagEndpoint, err)
	}
	fmt.Fprintf(cmd.ErrOrStderr(), "sampling %s for %s...\n", flagEndpoint, flagInterval)
	// Wait out the throughput window and, over the same window, poll the live
	// GPU utilization so the report contrasts MFU against a real reading.
	util, err := sampleGPUUtil(cmd.Context(), flagInterval)
	if err != nil {
		return err
	}
	var utilPtr *float64
	if util >= 0 {
		utilPtr = &util
	}
	t1, err := scrapeOutputTokens(cmd.Context(), parser, flagEndpoint)
	if err != nil {
		return fmt.Errorf("scrape %s: %w", flagEndpoint, err)
	}

	tps := mfu.TokensPerSec(t0, t1, flagInterval)
	if tps <= 0 {
		return fmt.Errorf("no output-token throughput observed for engine %q over %s (idle, or %q does not export generation tokens)", flagEngine, flagInterval, flagEngine)
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
	})
	if err != nil {
		return err
	}
	fmt.Fprint(cmd.OutOrStdout(), mfu.Render(est, rateSource))
	return nil
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

// detectGPU reads the GPU model + count from nvidia-smi. Best-effort;
// the caller falls back to --gpu / --gpu-count.
func detectGPU(ctx context.Context) (name string, count int, err error) {
	cctx, cancel := context.WithTimeout(ctx, nvidiaSmiTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return "", 0, err
	}
	return parseNvidiaSmi(out)
}

// nvidiaSmiTimeout bounds each nvidia-smi call. scan targets GPUs under load,
// exactly when a wedged driver can make nvidia-smi hang for many seconds; a
// hung utilization poll is dropped rather than stalling the sampling window.
const nvidiaSmiTimeout = 5 * time.Second

// readUtil is the per-snapshot utilization reader sampleGPUUtil polls. It is a
// package var so tests can stub out the nvidia-smi call and exercise the
// averaging and zero-sample sentinel paths without a GPU.
var readUtil = readGPUUtil

// sampleGPUUtil blocks for window, polling nvidia-smi GPU utilization every
// ~2s, and returns the mean across samples and GPUs (0..100). It returns a
// negative value if utilization could not be read at all (GPU present but the
// query is unsupported), so the caller can omit the claim rather than fake one.
// Honors ctx cancellation.
func sampleGPUUtil(ctx context.Context, window time.Duration) (float64, error) {
	const step = 2 * time.Second
	deadline := time.NewTimer(window)
	defer deadline.Stop()
	ticker := time.NewTicker(step)
	defer ticker.Stop()

	var sum float64
	var n int
	take := func() {
		if u, ok := readUtil(ctx); ok {
			sum += u
			n++
		}
	}
	take() // one reading at the window start
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-deadline.C:
			if n == 0 {
				return -1, nil
			}
			return sum / float64(n), nil
		case <-ticker.C:
			take()
		}
	}
}

// readGPUUtil takes one nvidia-smi utilization snapshot across all GPUs,
// bounded by nvidiaSmiTimeout so a hung driver drops the sample instead of
// stalling the window.
func readGPUUtil(ctx context.Context) (float64, bool) {
	cctx, cancel := context.WithTimeout(ctx, nvidiaSmiTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi",
		"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false
	}
	return parseGPUUtil(out)
}

// parseGPUUtil averages the integer utilization percentages nvidia-smi prints
// (one line per GPU). Factored out of readGPUUtil so it is unit-testable
// without a GPU.
func parseGPUUtil(out []byte) (float64, bool) {
	var sum float64
	var n int
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		l := strings.TrimSpace(line)
		if l == "" {
			continue
		}
		v, err := strconv.ParseFloat(l, 64)
		if err != nil {
			continue
		}
		sum += v
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

// parseNvidiaSmi parses `nvidia-smi --query-gpu=name --format=csv,noheader`
// output into the first GPU's name and the GPU count. Factored out of
// detectGPU so it is unit-testable without a GPU.
func parseNvidiaSmi(out []byte) (name string, count int, err error) {
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		l := strings.TrimSpace(line)
		if l == "" {
			continue
		}
		if name == "" {
			name = l
		}
		count++
	}
	if name == "" {
		return "", 0, fmt.Errorf("nvidia-smi returned no GPUs")
	}
	return name, count, nil
}

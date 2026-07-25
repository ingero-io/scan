package main

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ingero-io/scan/internal/mfu"
)

// renderPrometheus turns one mfu.Estimate into Prometheus exposition text:
// HELP/TYPE/value lines for the public MFU-gap board. It is hand-rolled on
// purpose - scan stays cgo-free and pulls in no Prometheus client dependency.
//
// It exposes ONLY the gap and the public CEILING envelope already computed by
// mfu.Estimate. It NEVER renders a cause, a recoverable figure, sizing, or any
// fitted per-SKU band: scan does not import the packages that compute those.
// Labels are {model,gpu}; the healthy-MFU band is a public reference with no
// label. Each gauge is rendered as HELP, TYPE, then a single value line.
func renderPrometheus(est mfu.Estimate, rateKnown bool) string {
	labels := fmt.Sprintf(`{model=%q,gpu=%q}`, escapeLabel(est.Model), escapeLabel(est.GPU))

	var b strings.Builder
	gauge := func(name, help string, value float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n", name, help)
		fmt.Fprintf(&b, "# TYPE %s gauge\n", name)
		fmt.Fprintf(&b, "%s%s %s\n", name, labels, formatFloat(value))
	}

	// Vendor-reported utilization (0..1) - the dashboard number. Omitted
	// entirely when utilization could not be read, rather than published as a
	// fake 0. The series name stays vendor-neutral so a dashboard built on an
	// NVIDIA fleet keeps working when an AMD host reports into it.
	if est.GPUUtilPct != nil {
		gauge("scan_gpu_utilization",
			"live GPU utilization (0..1) as reported by the host's vendor tool, the number the dashboard shows",
			*est.GPUUtilPct/100)
	}

	gauge("scan_mfu",
		"modeled model FLOPs utilization (0..1), the real work behind the utilization",
		est.MFU)
	gauge("scan_achieved_tflops",
		"achieved tensor throughput in TFLOP/s (modeled from output tokens/sec)",
		est.AchievedTFLOPS)
	gauge("scan_peak_tflops",
		"aggregate published dense BF16/FP16 peak in TFLOP/s across the GPUs",
		est.PeakTFLOPS)
	gauge("scan_output_tokens_per_second",
		"achieved output (generation) tokens/sec aggregated across the GPUs",
		est.TokensPerSec)

	// Dollar lines only when a rate is known - otherwise they would all be 0
	// and imply a free workload.
	if rateKnown {
		gauge("scan_monthly_cost_usd",
			"estimated monthly USD for this workload at the configured GPU rate",
			est.MonthlyUSD)
		gauge("scan_headroom_monthly_usd_low",
			"low bound of the monthly consolidation-headroom envelope in USD (a gross gap, not a savings promise)",
			est.MonthlyHeadroomLowUSD)
		gauge("scan_headroom_monthly_usd_high",
			"high bound of the monthly consolidation-headroom envelope in USD (a gross gap, not a savings promise)",
			est.MonthlyHeadroomHighUSD)
	}

	gauge("scan_headroom_multiple",
		"how many times below the mid healthy-serving MFU band this workload runs",
		est.HeadroomMultiple)

	// The public 0.35 / 0.50 reference band - no per-workload label.
	fmt.Fprintf(&b, "# HELP scan_healthy_mfu_band_low %s\n",
		"low edge of the public healthy-serving MFU band for well-batched inference")
	fmt.Fprintf(&b, "# TYPE scan_healthy_mfu_band_low gauge\n")
	fmt.Fprintf(&b, "scan_healthy_mfu_band_low %s\n", formatFloat(mfu.HealthyServingMFULow))
	fmt.Fprintf(&b, "# HELP scan_healthy_mfu_band_high %s\n",
		"high edge of the public healthy-serving MFU band for well-batched inference")
	fmt.Fprintf(&b, "# TYPE scan_healthy_mfu_band_high gauge\n")
	fmt.Fprintf(&b, "scan_healthy_mfu_band_high %s\n", formatFloat(mfu.HealthyServingMFUHigh))

	return b.String()
}

// formatFloat renders a metric value with enough precision for the small
// fractions (MFU, the band) without scientific notation noise.
func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', 6, 64)
}

// escapeLabel escapes a Prometheus label value per the exposition format:
// backslash, double-quote, and newline.
func escapeLabel(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(s)
}

// snapshot holds the most recent rendered exposition text behind a lock so the
// /metrics handler always serves a whole, consistent body even mid-recompute.
type snapshot struct {
	mu   sync.RWMutex
	body string
}

func (s *snapshot) set(body string) {
	s.mu.Lock()
	s.body = body
	s.mu.Unlock()
}

func (s *snapshot) get() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.body
}

// serveMetrics starts the /metrics HTTP server on addr and blocks until ctx is
// done or the server fails. The handler renders the latest snapshot.
func serveMetrics(addr string, snap *snapshot) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		body := snap.get()
		if body == "" {
			// No snapshot yet (first sample still in flight).
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintln(w, "# scan: no snapshot yet")
			return
		}
		fmt.Fprint(w, body)
	})
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	return srv
}

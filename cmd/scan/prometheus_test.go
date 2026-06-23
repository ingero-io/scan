package main

import (
	"strings"
	"testing"

	"github.com/ingero-io/scan/internal/mfu"
)

// sampleEstimate builds a representative Estimate for the exposition tests.
func fixtureEstimate(t *testing.T, util *float64) mfu.Estimate {
	t.Helper()
	est, err := mfu.Compute(mfu.Input{
		Model: "qwen2-7b", GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 1,
		TokensPerSec: 600, HourlyUSDPerGPU: 1.10, GPUUtilPct: util,
	})
	if err != nil {
		t.Fatalf("Compute: %v", err)
	}
	return est
}

func TestRenderPrometheus_HasExpectedMetricNames(t *testing.T) {
	util := 90.0
	out := renderPrometheus(fixtureEstimate(t, &util), true)

	want := []string{
		"scan_gpu_utilization",
		"scan_mfu",
		"scan_achieved_tflops",
		"scan_peak_tflops",
		"scan_output_tokens_per_second",
		"scan_monthly_cost_usd",
		"scan_headroom_monthly_usd_low",
		"scan_headroom_monthly_usd_high",
		"scan_headroom_multiple",
		"scan_healthy_mfu_band_low",
		"scan_healthy_mfu_band_high",
	}
	for _, name := range want {
		// Each metric must appear as a HELP line, a TYPE line, and a value line.
		if strings.Count(out, name) < 3 {
			t.Errorf("metric %q: want >=3 occurrences (HELP/TYPE/value), got %d\n%s", name, strings.Count(out, name), out)
		}
		if !strings.Contains(out, "# TYPE "+name+" gauge") {
			t.Errorf("metric %q: missing TYPE gauge line", name)
		}
	}

	// Labels carry the model + gpu; the band lines carry no label.
	if !strings.Contains(out, `model="qwen2-7b"`) || !strings.Contains(out, `gpu="NVIDIA A100-SXM4-40GB"`) {
		t.Errorf("missing {model,gpu} labels:\n%s", out)
	}
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "scan_healthy_mfu_band_low ") || strings.HasPrefix(line, "scan_healthy_mfu_band_high ") {
			if strings.Contains(line, "{") {
				t.Errorf("healthy-band line must have no label, got: %q", line)
			}
		}
	}
	// The band values are the public 0.35 / 0.50 reference.
	if !strings.Contains(out, "scan_healthy_mfu_band_low 0.35") {
		t.Errorf("band low not 0.35:\n%s", out)
	}
	if !strings.Contains(out, "scan_healthy_mfu_band_high 0.5") {
		t.Errorf("band high not 0.5:\n%s", out)
	}
}

func TestRenderPrometheus_OmitsUtilWhenUnavailable(t *testing.T) {
	out := renderPrometheus(fixtureEstimate(t, nil), true)
	if strings.Contains(out, "scan_gpu_utilization") {
		t.Errorf("scan_gpu_utilization must be omitted when util is unavailable:\n%s", out)
	}
	// The rest of the board is still present.
	if !strings.Contains(out, "scan_mfu") {
		t.Errorf("scan_mfu missing in util-omitted case:\n%s", out)
	}
}

func TestRenderPrometheus_OmitsDollarLinesWithoutRate(t *testing.T) {
	util := 90.0
	out := renderPrometheus(fixtureEstimate(t, &util), false)
	for _, name := range []string{"scan_monthly_cost_usd", "scan_headroom_monthly_usd_low", "scan_headroom_monthly_usd_high"} {
		if strings.Contains(out, name) {
			t.Errorf("dollar metric %q must be omitted when no rate is known:\n%s", name, out)
		}
	}
}

// TestRenderPrometheus_NoForbiddenTokens guards the public carve-out: the board
// shows only the gap and the public ceiling envelope. It must never leak the
// moat vocabulary (cause attribution, recoverable hours, the word "ceiling",
// sizing, calibration).
func TestRenderPrometheus_NoForbiddenTokens(t *testing.T) {
	util := 90.0
	withRate := renderPrometheus(fixtureEstimate(t, &util), true)
	noRate := renderPrometheus(fixtureEstimate(t, nil), false)

	forbidden := []string{
		"cause", "recoverable", "ceiling", "sizing", "calibrat",
		"attribut", "receipt", "stall", "throttle", "straggler",
	}
	for _, out := range []string{withRate, noRate} {
		low := strings.ToLower(out)
		for _, tok := range forbidden {
			if strings.Contains(low, tok) {
				t.Errorf("forbidden token %q leaked into exposition:\n%s", tok, out)
			}
		}
	}
}

func TestEscapeLabel(t *testing.T) {
	got := escapeLabel(`a"b\c` + "\n" + "d")
	want := `a\"b\\c\nd`
	if got != want {
		t.Errorf("escapeLabel = %q, want %q", got, want)
	}
}

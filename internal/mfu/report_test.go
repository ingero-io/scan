package mfu

import (
	"strings"
	"testing"
)

func TestParserFor(t *testing.T) {
	for _, e := range []string{"vllm", "VLLM", "sglang", "tgi"} {
		if _, err := ParserFor(e); err != nil {
			t.Errorf("ParserFor(%q): %v", e, err)
		}
	}
	if _, err := ParserFor("ollama"); err == nil {
		t.Error("want error for unknown engine")
	}
}

func TestRender_IncludesGapAndDollar(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 8000, HourlyUSDPerGPU: 12.29,
	})
	if err != nil {
		t.Fatal(err)
	}
	out := Render(est, "ec2 list price")
	for _, want := range []string{"ESTIMATE", "MFU", "14.2%", "headroom", "below well-batched", "up to ~$", "CEILING", "$", "ec2 list price", "--rate", "agent recovers"} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q\n---\n%s", want, out)
		}
	}
	// The over-promising (1-MFU) savings/cause framing must be gone.
	if strings.Contains(out, "at risk") || strings.Contains(out, "on stalls") {
		t.Errorf("render must not use the old savings/cause framing:\n%s", out)
	}
}

func TestRender_ShowsLiveUtilWhenRead(t *testing.T) {
	est, err := Compute(Input{
		Model: "qwen2-7b", GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 1,
		TokensPerSec: 1000, HourlyUSDPerGPU: 1.10, GPUUtilPct: ptr(91),
	})
	if err != nil {
		t.Fatal(err)
	}
	out := Render(est, "--rate")
	if !strings.Contains(out, "util     : 91%") {
		t.Errorf("render should show the live nvidia-smi utilization:\n%s", out)
	}
	// The old hardcoded "~100%" boilerplate must be gone.
	if strings.Contains(out, "~100%") || strings.Contains(out, "DCGM/nvidia-smi report") {
		t.Errorf("render must not print the static ~100%% utilization claim:\n%s", out)
	}
}

func TestRender_OmitsUtilWhenUnread(t *testing.T) {
	est, err := Compute(Input{
		Model: "qwen2-7b", GPU: "NVIDIA A100-SXM4-40GB", GPUCount: 1,
		TokensPerSec: 1000, // GPUUtilPct nil: not read
	})
	if err != nil {
		t.Fatal(err)
	}
	out := Render(est, "")
	if strings.Contains(out, "util     :") {
		t.Errorf("no utilization line should render when util was not read:\n%s", out)
	}
	if !strings.Contains(out, "MFU") {
		t.Error("MFU line should still render")
	}
}

func ptr(v float64) *float64 { return &v }

func TestRender_OmitsDollarWhenNoRate(t *testing.T) {
	est, err := Compute(Input{Model: "mistral-7b", GPU: "L40S", GPUCount: 1, TokensPerSec: 2000})
	if err != nil {
		t.Fatal(err)
	}
	out := Render(est, "")
	if strings.Contains(out, "/mo") {
		t.Errorf("dollar lines (cost/envelope) should be omitted without a rate:\n%s", out)
	}
	if !strings.Contains(out, "MFU") {
		t.Error("MFU line should always render")
	}
	// The rate-independent headroom multiple still renders (it is MFU-only).
	if !strings.Contains(out, "below well-batched") {
		t.Errorf("rate-independent headroom multiple should render without a rate:\n%s", out)
	}
}

// Near-healthy replicas (MFU in/just-under the band) must NOT render a
// meaningless "~1x below", a "$0-" low envelope bound, or an envelope line with
// no headroom line above it. Sweep the danger zone.
func TestRender_NearHealthyNoDegenerateLines(t *testing.T) {
	// llama-3-70b on 8x H100 (peak 7912 TFLOP/s). Pick tok/s to land MFU across
	// the band: ~25000->0.40, ~28800->0.4548, ~22000->0.35.
	for _, toks := range []float64{22000, 25000, 28800, 31000} {
		est, err := Compute(Input{
			Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
			TokensPerSec: toks, HourlyUSDPerGPU: 2.49,
		})
		if err != nil {
			t.Fatal(err)
		}
		out := Render(est, "ec2 list price")
		if strings.Contains(out, "~1x below") {
			t.Errorf("MFU=%.3f tok/s=%g: rendered the nonsensical '~1x below':\n%s", est.MFU, toks, out)
		}
		if strings.Contains(out, "up to ~$0-") {
			t.Errorf("MFU=%.3f tok/s=%g: rendered a $0 envelope low bound:\n%s", est.MFU, toks, out)
		}
		if strings.Contains(out, "envelope") && !strings.Contains(out, "headroom") {
			t.Errorf("MFU=%.3f tok/s=%g: envelope line with no headroom line:\n%s", est.MFU, toks, out)
		}
	}
}

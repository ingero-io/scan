package mfu

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want, tol float64, what string) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %g, want %g (+/- %g)", what, got, want, tol)
	}
}

func TestCompute_Llama70BOnH100(t *testing.T) {
	// 70e9 params, H100 peak 989 TFLOP/s, 1 GPU, 1000 tok/s.
	// achieved = 1000 * 2 * 70e9 / 1e12 = 140 TFLOP/s; MFU = 140/989 = 0.1416.
	est, err := Compute(Input{Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1, TokensPerSec: 1000})
	if err != nil {
		t.Fatal(err)
	}
	approx(t, est.AchievedTFLOPS, 140, 0.5, "achieved")
	approx(t, est.PeakTFLOPS, 989, 0.5, "peak")
	approx(t, est.MFU, 0.1416, 0.001, "mfu")
	if !est.Plausible {
		t.Error("want plausible")
	}
}

func TestCompute_CostAndHeadroom(t *testing.T) {
	// 8 H100 at $2.49/hr (Lambda), llama-70b, 8000 tok/s aggregate.
	// achieved = 8000*2*70e9/1e12 = 1120 TFLOP/s; peak = 989*8 = 7912; mfu = 0.1416.
	// monthly = 2.49*8*730 = 14541.6. Headroom vs healthy band [0.35,0.50]:
	//   fracLow  = 1 - 0.1416/0.35 = 0.5954 -> $8658/mo
	//   fracHigh = 1 - 0.1416/0.50 = 0.7168 -> $10424/mo
	//   multiple = 0.425/0.1416 = 3.00x below well-batched serving
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 8000, HourlyUSDPerGPU: 2.49,
	})
	if err != nil {
		t.Fatal(err)
	}
	approx(t, est.MFU, 0.1416, 0.001, "mfu")
	approx(t, est.MonthlyUSD, 14541.6, 1, "monthly")
	approx(t, est.HeadroomMultiple, 3.00, 0.05, "headroom multiple")
	approx(t, est.HeadroomFracLow, 0.5954, 0.002, "headroom frac low")
	approx(t, est.HeadroomFracHigh, 0.7168, 0.002, "headroom frac high")
	approx(t, est.MonthlyHeadroomLowUSD, 8658, 5, "headroom low usd")
	approx(t, est.MonthlyHeadroomHighUSD, 10424, 5, "headroom high usd")
	// The healthy-anchored envelope must sit strictly below the old (1-MFU)
	// ceiling ($12484) that over-promised an unreachable 100% MFU.
	if est.MonthlyHeadroomHighUSD >= (1-est.MFU)*est.MonthlyUSD {
		t.Errorf("healthy-anchored envelope %g should be below the (1-MFU) ceiling %g",
			est.MonthlyHeadroomHighUSD, (1-est.MFU)*est.MonthlyUSD)
	}
}

func TestCompute_HealthyReplicaHasNoHeadroom(t *testing.T) {
	// A replica at/above the healthy band shows zero headroom - never invent an
	// envelope for an already-well-batched workload. ~32000 tok/s -> mfu ~0.57.
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 32000, HourlyUSDPerGPU: 2.49,
	})
	if err != nil {
		t.Fatal(err)
	}
	if est.HeadroomFracHigh != 0 || est.MonthlyHeadroomHighUSD != 0 {
		t.Errorf("healthy replica should have zero high-end headroom, got fracHigh=%g usd=%g",
			est.HeadroomFracHigh, est.MonthlyHeadroomHighUSD)
	}
}

func TestCompute_NoCostWhenRateZero(t *testing.T) {
	est, err := Compute(Input{Model: "mistral-7b", GPU: "A100-SXM4-80GB", GPUCount: 1, TokensPerSec: 2000})
	if err != nil {
		t.Fatal(err)
	}
	// No rate -> no dollar figures, but the rate-independent headroom fraction
	// and multiple are still populated from MFU alone.
	if est.MonthlyUSD != 0 || est.MonthlyHeadroomLowUSD != 0 || est.MonthlyHeadroomHighUSD != 0 {
		t.Errorf("want zero dollars with no rate, got monthly=%g low=%g high=%g",
			est.MonthlyUSD, est.MonthlyHeadroomLowUSD, est.MonthlyHeadroomHighUSD)
	}
	if est.HeadroomFracHigh <= 0 || est.HeadroomMultiple <= 0 {
		t.Errorf("headroom fraction/multiple are rate-independent and should be set, got frac=%g mult=%g",
			est.HeadroomFracHigh, est.HeadroomMultiple)
	}
}

func TestCompute_ParamsOverride(t *testing.T) {
	est, err := Compute(Input{Model: "my-custom-model", ParamsOverride: 13e9, GPU: "L40S", GPUCount: 1, TokensPerSec: 500})
	if err != nil {
		t.Fatalf("override should bypass the model table: %v", err)
	}
	if est.ParamsActive != 13e9 {
		t.Errorf("params = %g, want 13e9", est.ParamsActive)
	}
	// achieved = 500*2*13e9/1e12 = 13 TFLOP/s; peak L40S = 362; mfu ~0.0359.
	approx(t, est.MFU, 0.0359, 0.001, "mfu")
}

func TestCompute_UnknownModelErrors(t *testing.T) {
	_, err := Compute(Input{Model: "totally-unknown-7000b", GPU: "H100", GPUCount: 1, TokensPerSec: 100})
	if err == nil {
		t.Fatal("want error for unknown model with no override")
	}
}

func TestCompute_UnknownGPUErrors(t *testing.T) {
	_, err := Compute(Input{Model: "llama-3-8b", GPU: "NVIDIA RTX 4090", GPUCount: 1, TokensPerSec: 100})
	if err == nil {
		t.Fatal("want error for unknown GPU")
	}
}

func TestCompute_RejectsNonPositiveInputs(t *testing.T) {
	for _, in := range []Input{
		{Model: "llama-3-8b", GPU: "H100", GPUCount: 1, TokensPerSec: 0},
		{Model: "llama-3-8b", GPU: "H100", GPUCount: 0, TokensPerSec: 100},
	} {
		if _, err := Compute(in); err == nil {
			t.Errorf("want error for input %+v", in)
		}
	}
}

func TestCompute_ImplausibleMFUFlagged(t *testing.T) {
	// Absurd throughput drives MFU > 1; must be flagged, not silently shipped.
	est, err := Compute(Input{Model: "llama-3-70b", GPU: "NVIDIA L4 24GB", GPUCount: 1, TokensPerSec: 1_000_000})
	if err != nil {
		t.Fatal(err)
	}
	if est.Plausible {
		t.Errorf("MFU = %g should be flagged implausible", est.MFU)
	}
	found := false
	for _, c := range est.Caveats {
		if len(c) > 0 && (c[0] == 'M') { // "MFU > 100%..."
			found = true
		}
	}
	if !found {
		t.Errorf("want an implausible-MFU caveat, got %v", est.Caveats)
	}
}

func TestResolvePeakTFLOPS_NameVariants(t *testing.T) {
	cases := map[string]float64{
		"NVIDIA H100 80GB HBM3": 989,
		"NVIDIA A100-SXM4-80GB": 312,
		"NVIDIA A100-SXM4-40GB": 312,
		"NVIDIA L40S":           362,
		"NVIDIA L4 24GB":        121,
		"NVIDIA GH200 480GB":    989,
		"Tesla V100-SXM2-16GB":  112,
	}
	for name, want := range cases {
		got, err := ResolvePeakTFLOPS(name)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if got != want {
			t.Errorf("%s peak = %g, want %g", name, got, want)
		}
	}
}

func TestResolveParams_OverrideWins(t *testing.T) {
	got, err := ResolveParams("llama-3-70b", 5e9)
	if err != nil {
		t.Fatal(err)
	}
	if got != 5e9 {
		t.Errorf("override = %g, want 5e9 (override must win over the table)", got)
	}
}

func TestResolveParams_MoEUsesActive(t *testing.T) {
	got, err := ResolveParams("mixtral-8x7b-instruct", 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 13e9 {
		t.Errorf("mixtral-8x7b active params = %g, want 13e9 (active, not 47e9 total)", got)
	}
}

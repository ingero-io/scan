package mfu

import (
	"strings"
	"testing"
)

// Weight-only quantization compresses stored weights and dequantizes them to
// BF16 before the matmul, so it must NOT move the denominator. Getting this
// backwards understates MFU by roughly 4x, which is the same size of error as
// the FP8 overstatement this precision work exists to remove, pointing the other
// way. This is the single most important assertion in the file.
func TestResolveComputePrecision_WeightOnlySchemesStayBF16(t *testing.T) {
	for _, scheme := range []string{
		"awq",
		"AWQ",
		"awq_marlin",
		"gptq",
		"gptq_marlin",
		"bitsandbytes",
		"aqlm",
		"hqq",
		"gguf",
		"experts_int8",
	} {
		got, detected := ResolveComputePrecision(scheme, "", "")
		if !detected {
			t.Errorf("%s: detected = false, want a positive BF16 determination", scheme)
			continue
		}
		if got != PrecisionBF16 {
			t.Errorf("%s resolved to %s, want bf16: the scheme dequantizes before the GEMM, so compute stays BF16", scheme, got)
		}
	}
}

// The KV cache dtype changes how much memory the cache occupies, not what
// precision the matmuls run in, so it must never move the denominator on its own.
func TestResolveComputePrecision_KVCacheDtypeDoesNotMoveTheDenominator(t *testing.T) {
	withCache, _ := ResolveComputePrecision("", "bfloat16", "fp8")
	withoutCache, _ := ResolveComputePrecision("", "bfloat16", "")
	if withCache != withoutCache {
		t.Errorf("kv-cache-dtype fp8 changed precision from %s to %s; it is a storage format, not a compute format", withoutCache, withCache)
	}
	if withCache != PrecisionBF16 {
		t.Errorf("precision = %s, want bf16", withCache)
	}

	// The same must hold when the cache dtype is the ONLY low-precision flag on
	// the command line, which is the realistic misconfiguration.
	got, _ := ResolveComputePrecision("", "", "fp8_e5m2")
	if got == PrecisionFP8 {
		t.Error("kv-cache-dtype alone resolved to fp8 compute; the GEMMs never left BF16")
	}
}

func TestResolveComputePrecision_ActivationAndWeightSchemesMoveTheDenominator(t *testing.T) {
	for _, tc := range []struct {
		scheme string
		want   Precision
	}{
		{"fp8", PrecisionFP8},
		{"fp8_e4m3", PrecisionFP8},
		{"FP8_E5M2", PrecisionFP8},
		{"modelopt_fp8", PrecisionFP8},
		{"fbgemm_fp8", PrecisionFP8},
		{"nvfp4", PrecisionFP4},
		{"modelopt_fp4", PrecisionFP4},
		{"mxfp4", PrecisionFP4},
		{"w8a8", PrecisionINT8},
	} {
		got, detected := ResolveComputePrecision(tc.scheme, "", "")
		if !detected {
			t.Errorf("%s: detected = false, want a determination", tc.scheme)
			continue
		}
		if got != tc.want {
			t.Errorf("%s resolved to %s, want %s", tc.scheme, got, tc.want)
		}
	}
}

// compressed-tensors spells an FP8 activation-and-weight scheme with the INT8
// W8A8 token in its name. Resolving that to INT8 would pick a peak half the size
// of the right one, so the FP8 test has to win.
func TestResolveComputePrecision_W8A8FP8IsFP8NotINT8(t *testing.T) {
	got, _ := ResolveComputePrecision("w8a8_fp8", "", "")
	if got != PrecisionFP8 {
		t.Errorf("w8a8_fp8 resolved to %s, want fp8: the activations are FP8, not INT8", got)
	}
}

// An undetermined precision must report itself as undetermined so the caller
// labels the output. Silently returning BF16 with detected=true is what let a
// low-precision workload print a plausible, actionable and doubled MFU.
func TestResolveComputePrecision_UndeterminedIsReportedNotGuessed(t *testing.T) {
	for _, tc := range []struct{ quant, dtype string }{
		{"", ""},
		{"", "auto"},
		{"auto", "auto"},
		{"", "float32"},
		{"compressed-tensors", ""},
		{"some-future-scheme", ""},
	} {
		got, detected := ResolveComputePrecision(tc.quant, tc.dtype, "")
		if detected {
			t.Errorf("quant=%q dtype=%q: detected = true, want false so the output is labelled", tc.quant, tc.dtype)
		}
		if got != PrecisionUnknown {
			t.Errorf("quant=%q dtype=%q: precision = %s, want unknown", tc.quant, tc.dtype, got)
		}
	}
}

func TestResolveComputePrecision_ExplicitDtypeIsDetected(t *testing.T) {
	for _, dtype := range []string{"bfloat16", "float16", "half", "BFloat16"} {
		got, detected := ResolveComputePrecision("", dtype, "")
		if !detected || got != PrecisionBF16 {
			t.Errorf("dtype %s resolved to (%s, detected=%v), want (bf16, true)", dtype, got, detected)
		}
	}
}

// On H100 the BF16 SPARSE figure and the FP8 DENSE figure are both 1979. A
// lookup that arrived at 1979 by way of sparsity would be coincidentally right
// here and wrong on every other part, so this asserts both that FP8 resolves to
// 1979 and that the BF16 entry is the dense 989 rather than its sparse double.
func TestResolvePeakTFLOPS_H100FP8DoesNotArriveViaTheSparseFigure(t *testing.T) {
	bf16, supported, err := ResolvePeakTFLOPS("NVIDIA H100 80GB HBM3", PrecisionBF16)
	if err != nil {
		t.Fatal(err)
	}
	if !supported {
		t.Fatal("H100 reports no BF16 support")
	}
	if bf16 != 989 {
		t.Errorf("H100 BF16 peak = %v, want 989 dense (1979 would be the sparse figure)", bf16)
	}

	fp8, supported, err := ResolvePeakTFLOPS("NVIDIA H100 80GB HBM3", PrecisionFP8)
	if err != nil {
		t.Fatal(err)
	}
	if !supported {
		t.Fatal("H100 reports no FP8 support")
	}
	if fp8 != 1979 {
		t.Errorf("H100 FP8 peak = %v, want 1979 dense", fp8)
	}
	// Allow for the rounding in the table (989 and 1979 are rounded from 989.4 and
	// 1978.9) but not for a factor-of-two sparsity leak.
	if ratio := fp8 / bf16; ratio < 1.99 || ratio > 2.01 {
		t.Errorf("H100 FP8 %v / dense BF16 %v = %.3f, want ~2; a sparsity-inflated value may have leaked in", fp8, bf16, ratio)
	}
}

// No cell in the table may hold a sparsity-inflated figure. Structured sparsity
// doubles every precision it applies to, so a leaked sparse value shows up as an
// FP8 entry more than twice its own BF16 entry.
func TestGPUPeakTable_HoldsNoSparsityInflatedFigures(t *testing.T) {
	for _, e := range gpuPeakTFLOPS {
		if e.bf16 <= 0 {
			t.Errorf("%s has no dense BF16 figure; every row needs one as its fallback", e.token)
		}
		// Vendors round each precision independently (L40S publishes 362.05 dense
		// BF16 against a flat 733 dense FP8), so exact multiples are not available.
		// The tolerance is wide enough to absorb that rounding and nowhere near
		// wide enough to hide a sparsity leak, which is always a clean factor of 2.
		const roundingTolerance = 1.05
		if e.fp8 > 0 && e.fp8 > 2*e.bf16*roundingTolerance {
			t.Errorf("%s FP8 %v exceeds twice its dense BF16 %v, which is the signature of a sparse figure", e.token, e.fp8, e.bf16)
		}
		if e.fp4 > 0 && e.fp4 > 4*e.bf16*roundingTolerance {
			t.Errorf("%s FP4 %v exceeds four times its dense BF16 %v, which is the signature of a sparse figure", e.token, e.fp4, e.bf16)
		}
		if e.fp4 > 0 && e.fp8 <= 0 {
			t.Errorf("%s advertises FP4 but no FP8; no shipped part has FP4 tensor hardware without FP8", e.token)
		}
		if e.formFactor == "" {
			t.Errorf("%s states no form factor, so its convention has to be inferred from its neighbours", e.token)
		}
		if e.family == "" {
			t.Errorf("%s states no family, which would break the display and rate-join key", e.token)
		}
	}
}

// AMD's headline FP4 figure for MI355X assumes 2:4 sparsity and is four times its
// own dense FP8 number. The dense value is half that.
func TestResolvePeakTFLOPS_CDNA4FP4IsDenseNotTheSparseHeadline(t *testing.T) {
	fp8, _, err := ResolvePeakTFLOPS("Instinct MI355X", PrecisionFP8)
	if err != nil {
		t.Fatal(err)
	}
	fp4, supported, err := ResolvePeakTFLOPS("Instinct MI355X", PrecisionFP4)
	if err != nil {
		t.Fatal(err)
	}
	if !supported {
		t.Fatal("MI355X reports no FP4 support; CDNA4 has FP4 tensor hardware")
	}
	if fp4 != 2*fp8 {
		t.Errorf("MI355X FP4 %v is not twice its dense FP8 %v; the 4x headline is the sparse figure", fp4, fp8)
	}
}

// Asking for a precision the silicon does not implement must not divide by zero
// and must not silently pretend the hardware is there.
func TestResolvePeakTFLOPS_UnsupportedPrecisionFallsBackAndSaysSo(t *testing.T) {
	for _, tc := range []struct {
		gpu string
		p   Precision
	}{
		{"NVIDIA A100-SXM4-80GB", PrecisionFP8}, // Ampere has no FP8 tensor hardware
		{"NVIDIA H100 80GB HBM3", PrecisionFP4}, // Hopper has no FP4 tensor hardware
		{"Tesla T4", PrecisionFP8},
		{"Instinct MI300X", PrecisionFP4}, // CDNA3 has no FP4
	} {
		peak, supported, err := ResolvePeakTFLOPS(tc.gpu, tc.p)
		if err != nil {
			t.Errorf("%s %s: %v", tc.gpu, tc.p, err)
			continue
		}
		if supported {
			t.Errorf("%s reports %s support it does not have", tc.gpu, tc.p)
		}
		if peak <= 0 {
			t.Errorf("%s %s: peak = %v, want the BF16 fallback rather than a zero divisor", tc.gpu, tc.p, peak)
		}
	}
}

// The whole point of the precision term: the same workload on the same GPU must
// report half the MFU when it is known to serve in FP8, because the hardware can
// do twice the work.
func TestCompute_FP8ServingHalvesReportedMFUAgainstBF16(t *testing.T) {
	in := Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1,
		TokensPerSec: 500, DeviceAttribution: DevicesOperatorSet,
	}
	in.Precision = PrecisionBF16
	bf16Est, err := Compute(in)
	if err != nil {
		t.Fatal(err)
	}
	in.Precision = PrecisionFP8
	fp8Est, err := Compute(in)
	if err != nil {
		t.Fatal(err)
	}
	ratio := bf16Est.MFU / fp8Est.MFU
	if ratio < 1.99 || ratio > 2.01 {
		t.Errorf("BF16 MFU %.4f / FP8 MFU %.4f = %.4f, want ~2: FP8 peak is twice BF16 so the same work is half the MFU",
			bf16Est.MFU, fp8Est.MFU, ratio)
	}
}

// An undetected precision must be labelled in the output, not printed bare.
func TestCompute_UndetectedPrecisionIsLabelledInOutput(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1,
		TokensPerSec: 500, DeviceAttribution: DevicesOperatorSet,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !est.PrecisionAssumed {
		t.Error("PrecisionAssumed = false for an input carrying no precision")
	}
	if est.Precision != PrecisionBF16 {
		t.Errorf("fallback precision = %s, want bf16", est.Precision)
	}
	var labelled bool
	for _, c := range est.Caveats {
		if strings.Contains(c, "precision=assumed-bf16 (not detected)") {
			labelled = true
		}
	}
	if !labelled {
		t.Errorf("no assumed-precision label in caveats: %v", est.Caveats)
	}
	if out := Render(est, ""); !strings.Contains(out, "precision=assumed-bf16 (not detected)") {
		t.Errorf("rendered report does not carry the assumed-precision label:\n%s", out)
	}
}

func TestCompute_DetectedPrecisionIsNotLabelledAssumed(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1,
		TokensPerSec: 500, Precision: PrecisionFP8, DeviceAttribution: DevicesOperatorSet,
	})
	if err != nil {
		t.Fatal(err)
	}
	if est.PrecisionAssumed {
		t.Error("PrecisionAssumed = true for an explicitly supplied precision")
	}
	for _, c := range est.Caveats {
		if strings.Contains(c, "assumed-bf16") {
			t.Errorf("detected precision still carries an assumed label: %q", c)
		}
	}
}

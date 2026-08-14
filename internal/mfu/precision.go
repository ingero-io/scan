package mfu

import "strings"

// Precision is the numeric format an engine's matmuls actually execute in, which
// is what selects the hardware peak the MFU denominator must divide by.
//
// This is COMPUTE precision, not STORAGE precision, and the distinction is the
// whole point of the type. A weight-only quantized model stores 4-bit weights and
// dequantizes them to BF16 before the GEMM runs, so its compute precision is BF16
// and its denominator is the BF16 peak. Treating its storage format as its
// compute format would divide by roughly a quarter of the real peak and understate
// MFU by about 4x, an error just as large as the one this type exists to fix and
// pointing the other way.
type Precision string

const (
	// PrecisionUnknown means serving precision could not be determined from the
	// engine. Callers fall back to the BF16 peak AND must label the output as an
	// assumption: an unlabelled fallback is exactly what let low-precision serving
	// report a silently doubled MFU.
	PrecisionUnknown Precision = ""
	PrecisionBF16    Precision = "bf16"
	PrecisionFP8     Precision = "fp8"
	PrecisionFP4     Precision = "fp4"
	PrecisionINT8    Precision = "int8"
)

// String renders the precision for report output, naming the undetected case
// explicitly rather than printing an empty field.
func (p Precision) String() string {
	if p == PrecisionUnknown {
		return "unknown"
	}
	return string(p)
}

// weightOnlySchemes are quantization schemes that compress WEIGHTS in memory and
// dequantize to BF16/FP16 before the matmul. They reduce memory footprint and
// bandwidth, not GEMM precision, so every one of them keeps the BF16 peak as its
// denominator. Mapping any of these to a low-precision peak is the single most
// damaging mistake available here.
var weightOnlySchemes = []string{
	"awq",
	"gptq",
	"marlin",
	"bitsandbytes",
	"aqlm",
	"hqq",
	"gguf",
	"squeezellm",
	"deepspeedfp",
	"experts_int8",
}

// fp8Schemes run the matmul itself in FP8 (activations and weights both), which
// is available on Hopper and later NVIDIA parts and on CDNA3 and later AMD parts.
// These genuinely double the peak and so genuinely move the denominator.
var fp8Schemes = []string{
	"fp8",
	"fbgemm_fp8",
	"modelopt_fp8",
	"ptpc_fp8",
	"quark_fp8",
}

// fp4Schemes run the matmul in 4-bit float, available on Blackwell and CDNA4.
// Distinct from the 4-bit WEIGHT-ONLY schemes above, which do not.
var fp4Schemes = []string{
	"nvfp4",
	"modelopt_fp4",
	"mxfp4",
}

// int8ComputeSchemes run the matmul in INT8 with INT8 activations (W8A8). Note
// that W8A8 is the only INT8 case that moves the denominator: an INT8
// WEIGHT-ONLY scheme dequantizes and belongs in weightOnlySchemes.
var int8ComputeSchemes = []string{
	"w8a8",
	"int8_w8a8",
	"modelopt_int8",
}

// ResolveComputePrecision maps an engine's precision-determining command line
// flags onto the compute precision its matmuls run in.
//
// kvCacheDtype is accepted and deliberately IGNORED. The KV cache dtype changes
// how much memory the cache occupies, not what precision the GEMMs execute in, so
// letting it move the denominator would halve a reported MFU for a workload whose
// matmuls never left BF16. It is a parameter rather than an omission so that the
// contract is visible at the call site and testable.
//
// The second return reports whether precision was actually DETERMINED. False
// means the caller must label its output as an assumption rather than print a
// bare number.
func ResolveComputePrecision(quantization, dtype, kvCacheDtype string) (Precision, bool) {
	_ = kvCacheDtype // storage precision, not compute precision; see the doc comment

	q := normalizeScheme(quantization)

	// Weight-only is checked FIRST. Several weight-only scheme names contain a
	// bit-width token that would otherwise look like a compute precision, so an
	// order that tested the low-precision names first would misclassify them.
	if q != "" && matchesScheme(q, weightOnlySchemes) {
		return PrecisionBF16, true
	}
	if q != "" {
		// FP8 is tested before INT8 because compressed-tensors spells an FP8
		// activation-and-weight scheme "w8a8_fp8", which carries the INT8 W8A8
		// token as well; testing INT8 first would call an FP8 workload INT8 and
		// pick a peak that is half the right one.
		switch {
		case matchesScheme(q, fp4Schemes):
			return PrecisionFP4, true
		case matchesScheme(q, fp8Schemes):
			return PrecisionFP8, true
		case matchesScheme(q, int8ComputeSchemes):
			return PrecisionINT8, true
		}
	}

	// compressed-tensors is a container format whose compute precision is carried
	// in the checkpoint config rather than the command line, so the flag alone
	// cannot decide it. Report undetected rather than assume a favourable case.
	if q == "compressed-tensors" || q == "compressed_tensors" {
		return PrecisionUnknown, false
	}

	// No quantization scheme in play: the serving dtype is the compute precision.
	// Anything else, including an absent dtype, an explicit "auto" (which resolves
	// to the checkpoint's own dtype at load time and so is not knowable from the
	// command line) and float32 (whose peak is far below the tensor-core BF16
	// figure), is reported undetected so the caller labels it instead of printing
	// a bare number.
	if q == "" || q == "none" || q == "auto" {
		switch normalizeScheme(dtype) {
		case "bfloat16", "bf16", "float16", "fp16", "half":
			return PrecisionBF16, true
		}
	}

	return PrecisionUnknown, false
}

// normalizeScheme lowercases and trims a flag value so "FP8_E4M3" and " fp8 "
// compare equal to the table entries.
func normalizeScheme(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

// matchesScheme reports whether the normalized scheme name contains any of the
// listed tokens. Substring matching is deliberate: engines ship many decorated
// variants of one scheme ("awq_marlin", "fp8_e4m3", "gptq_marlin_24") that all
// share the compute behaviour of the base scheme.
func matchesScheme(scheme string, tokens []string) bool {
	for _, t := range tokens {
		if strings.Contains(scheme, t) {
			return true
		}
	}
	return false
}

package mfu

import (
	"strings"
	"testing"
)

// A replica narrower than the host must be measured against the GPUs it holds,
// not against every GPU the host enumerates. An 8-GPU box running a
// tensor-parallel-1 replica is the case that made this wrong: the denominator
// was eight boards' worth of peak for a workload confined to one.
func TestCompute_SingleReplicaOnMultiGPUHostUsesTheReplicaWidth(t *testing.T) {
	const perGPUPeak = 989.0

	replica, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1,
		TokensPerSec: 500, Precision: PrecisionBF16,
		DeviceAttribution: DevicesEngineDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	if replica.PeakTFLOPS != perGPUPeak {
		t.Errorf("peak = %v, want %v (the one GPU the replica declared)", replica.PeakTFLOPS, perGPUPeak)
	}

	// The same throughput attributed to the whole host divides by eight times the
	// hardware and so reports one eighth of the MFU.
	host, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 500, Precision: PrecisionBF16,
	})
	if err != nil {
		t.Fatal(err)
	}
	if ratio := replica.MFU / host.MFU; ratio < 7.99 || ratio > 8.01 {
		t.Errorf("replica MFU %.5f / host-wide MFU %.5f = %.3f, want ~8", replica.MFU, host.MFU, ratio)
	}
}

// An unattributed multi-GPU count must withhold the dollar envelope. Over-counting
// devices deflates MFU, which inflates the headroom fraction, which recommends
// consolidating capacity that was never idle. That recommendation is the thing an
// operator acts on, so it is the thing that must not be printed on a guess.
func TestCompute_UnattributedDeviceCountWithholdsTheDollarEnvelope(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 200, Precision: PrecisionBF16, HourlyUSDPerGPU: 6.88,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !est.DollarEnvelopeSuppressed {
		t.Error("DollarEnvelopeSuppressed = false for a host-wide device count")
	}
	if est.MonthlyHeadroomLowUSD != 0 || est.MonthlyHeadroomHighUSD != 0 {
		t.Errorf("headroom dollars = %v..%v, want 0 while the device attribution is unverified",
			est.MonthlyHeadroomLowUSD, est.MonthlyHeadroomHighUSD)
	}
	if est.MonthlyUSD <= 0 {
		t.Error("MonthlyUSD = 0; the spend line is what the counted GPUs cost and still stands")
	}

	// Assert the ENVELOPE LINE itself says withheld. Matching the bare word
	// anywhere in the report would also match the caveat text and would pass
	// even if the envelope line were silently dropped.
	out := Render(est, "ec2 list price")
	if !strings.Contains(out, "envelope : withheld") {
		t.Errorf("report has no withheld envelope line, so a dropped envelope reads as no headroom found:\n%s", out)
	}
	if strings.Contains(out, "consolidation headroom") {
		t.Errorf("report still prints a consolidation headroom figure despite an unattributed device count:\n%s", out)
	}
}

func TestCompute_AttributedDeviceCountKeepsTheDollarEnvelope(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 200, Precision: PrecisionBF16, HourlyUSDPerGPU: 6.88,
		DeviceAttribution: DevicesEngineDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	if est.DollarEnvelopeSuppressed {
		t.Error("DollarEnvelopeSuppressed = true for an engine-declared device count")
	}
	if est.MonthlyHeadroomHighUSD <= 0 {
		t.Error("headroom dollars withheld despite an attributed device count")
	}
}

// One counted GPU needs no attribution: a single device is the whole host, so
// there is no wider hardware the engine could be measured against.
func TestCompute_SingleGPUHostNeedsNoAttribution(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 1,
		TokensPerSec: 200, Precision: PrecisionBF16, HourlyUSDPerGPU: 6.88,
	})
	if err != nil {
		t.Fatal(err)
	}
	if est.DollarEnvelopeSuppressed {
		t.Error("DollarEnvelopeSuppressed = true on a single-GPU host, where the host-wide count is the engine's count")
	}
}

// On a MIG host the enumeration returns one row per PHYSICAL GPU while the engine
// holds a slice of one board's SMs, so the whole-board peak is the wrong
// denominator and no scaling of it is defensible without knowing the profile.
// The estimate is refused rather than reported against hardware the engine does
// not have.
func TestCompute_MIGHostDoesNotDivideByTheWholeBoardPeak(t *testing.T) {
	_, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA A100-SXM4-80GB", GPUCount: 8,
		TokensPerSec: 500, Precision: PrecisionBF16, MIGEnabled: true,
	})
	if err == nil {
		t.Fatal("MIG host produced an estimate against the whole-board peak; want a refusal")
	}
	if !strings.Contains(err.Error(), "MIG") {
		t.Errorf("error %q does not name MIG as the reason", err)
	}
}

// An operator who knows the slice can still measure it, so the MIG refusal must
// not be an unconditional dead end.
func TestCompute_MIGHostAcceptsAnExplicitlyAttributedCount(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA A100-SXM4-80GB", GPUCount: 1,
		TokensPerSec: 500, Precision: PrecisionBF16, MIGEnabled: true,
		DeviceAttribution: DevicesOperatorSet,
	})
	if err != nil {
		t.Fatalf("MIG host with an operator-set count was refused: %v", err)
	}
	if est.MFU <= 0 {
		t.Error("MFU = 0 for an explicitly attributed MIG slice")
	}
}

func TestCompute_UnattributedCountIsCaveatedAsALowerBound(t *testing.T) {
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "NVIDIA H100 80GB HBM3", GPUCount: 8,
		TokensPerSec: 200, Precision: PrecisionBF16,
	})
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, c := range est.Caveats {
		if strings.Contains(c, "not attributed to this engine") {
			found = true
		}
	}
	if !found {
		t.Errorf("no device-attribution caveat in %v", est.Caveats)
	}
}

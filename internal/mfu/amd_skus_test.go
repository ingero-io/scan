package mfu

import (
	"strings"
	"testing"
)

// Names as the AMD tools report them: amd-smi's asic.market_name and rocm-smi's
// "Card Series" both read "Instinct MI300X" rather than an "AMD " prefixed
// string, so the peak table has to match on the bare SKU token.
func TestResolvePeakTFLOPS_AMDInstinct(t *testing.T) {
	for _, tc := range []struct {
		name string
		want float64
	}{
		{"Instinct MI300X", 1307},
		{"AMD Instinct MI300X OAM", 1307},
		{"Instinct MI325X", 1307},
		{"Instinct MI350X", 2300},
		{"Instinct MI355X", 2500},
	} {
		got, err := ResolvePeakTFLOPS(tc.name)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("%s peak = %v TFLOP/s, want %v", tc.name, got, tc.want)
		}
	}
}

// MI350X and MI355X share a die but not a clock, so collapsing them would put a
// 9% error straight into the MFU denominator on a CDNA4 fleet.
func TestResolvePeakTFLOPS_CDNA4PartsAreNotCollapsed(t *testing.T) {
	air, err := ResolvePeakTFLOPS("Instinct MI350X")
	if err != nil {
		t.Fatal(err)
	}
	liquid, err := ResolvePeakTFLOPS("Instinct MI355X")
	if err != nil {
		t.Fatal(err)
	}
	if air == liquid {
		t.Errorf("MI350X and MI355X both resolve to %v; the air-cooled part has a lower published peak", air)
	}
}

func TestResolvePeakTFLOPS_UnlistedInstinctPartErrors(t *testing.T) {
	// The APU and older Instinct parts are deliberately absent: a GPU with no
	// verified published peak must fail loudly rather than borrow a neighbour's
	// number and quietly mis-scale every MFU on that host.
	if _, err := ResolvePeakTFLOPS("Instinct MI300A"); err == nil {
		t.Error("want an error for an Instinct part with no peak-table entry")
	}
}

func TestLookupRate_AMDInstinctUsesPublishedRate(t *testing.T) {
	rate, provider, ok := LookupRate("Instinct MI300X")
	if !ok {
		t.Fatal("ok = false, want a bundled rate for MI300X")
	}
	if provider != "hotaisle" {
		t.Errorf("provider = %q, want hotaisle (the only provider publishing a per-GPU MI300X rate)", provider)
	}
	if rate != 3.39 {
		t.Errorf("rate = %v, want 3.39 (the max published rate, consistent with the upper-bound framing)", rate)
	}
}

func TestLookupRate_UnpricedAMDPartAsksForAnExplicitRate(t *testing.T) {
	// MI325X and MI355X are rented under reservation with no published hourly
	// figure. No entry means the meter omits the dollar lines and the operator
	// passes --rate, which is the honest outcome; an invented number is not.
	if _, _, ok := LookupRate("Instinct MI355X"); ok {
		t.Error("ok = true, want false until a provider publishes an MI355X per-GPU rate")
	}
}

func TestLookupRate_AMDProviderDoesNotShadowNvidiaRates(t *testing.T) {
	// hotaisle sits last in the provider order precisely so an NVIDIA SKU still
	// resolves to its hyperscaler list price.
	_, provider, ok := LookupRate("NVIDIA H100 80GB HBM3")
	if !ok {
		t.Fatal("ok = false, want a bundled rate for H100")
	}
	if provider != "ec2" {
		t.Errorf("provider = %q, want ec2 (the first provider in the lookup order)", provider)
	}
}

func TestCompute_OnAMDHostNamesTheVendorTool(t *testing.T) {
	util := 96.0
	est, err := Compute(Input{
		Model: "llama-3-70b", GPU: "Instinct MI300X", GPUCount: 8,
		TokensPerSec: 2000, GPUUtilPct: &util, UtilTool: "amd-smi/rocm-smi",
	})
	if err != nil {
		t.Fatal(err)
	}
	if est.PeakTFLOPS != 1307*8 {
		t.Errorf("peak = %v, want %v (per-GPU peak times device count)", est.PeakTFLOPS, 1307*8)
	}
	if !est.Plausible {
		t.Errorf("MFU = %v is implausible for 2000 tok/s of a 70B model on 8 MI300X", est.MFU)
	}
	if est.UtilTool != "amd-smi/rocm-smi" {
		t.Errorf("UtilTool = %q, want it carried into the report so the number is attributed to the host's own tool", est.UtilTool)
	}
	out := Render(est, "hotaisle list price")
	if !strings.Contains(out, "amd-smi/rocm-smi") {
		t.Errorf("report does not name the AMD tool:\n%s", out)
	}
	if strings.Contains(out, "nvidia-smi") {
		t.Errorf("report names nvidia-smi on an AMD host:\n%s", out)
	}
}

package device

import (
	"os"
	"path/filepath"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return b
}

func TestParseAMDNames_AMDSMIStatic(t *testing.T) {
	name, count, err := parseAMDNames(readFixture(t, "amd-smi-static.json"))
	if err != nil {
		t.Fatal(err)
	}
	if name != "Instinct MI300X" {
		t.Errorf("name = %q, want %q (asic.market_name)", name, "Instinct MI300X")
	}
	if count != 2 {
		t.Errorf("count = %d, want 2 (one array entry per GPU)", count)
	}
}

func TestParseAMDNames_ROCmSMIFallback(t *testing.T) {
	// rocm-smi keys the whole document by card and titles the model "Card
	// Series", so the same parse has to work on a flat, human-titled shape.
	name, count, err := parseAMDNames(readFixture(t, "rocm-smi-productname.json"))
	if err != nil {
		t.Fatal(err)
	}
	if name != "Instinct MI325X" {
		t.Errorf("name = %q, want %q (Card Series, not the hex Card Model)", name, "Instinct MI325X")
	}
	if count != 2 {
		t.Errorf("count = %d, want 2", count)
	}
}

func TestParseAMDBusy_AMDSMIMetric(t *testing.T) {
	got, ok := parseAMDBusy(readFixture(t, "amd-smi-metric.json"))
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if got != 93 { // mean of 96 and 90
		t.Errorf("busy = %v, want 93 (mean gfx_activity across GPUs)", got)
	}
}

func TestParseAMDBusy_ROCmSMIPrefersPercentOverActivityCounter(t *testing.T) {
	// rocm-smi --showuse prints "GPU use (%)" AND a raw "GFX Activity"
	// accumulator. Reading the accumulator would report a busy percent in the
	// hundreds of millions, so the percent spelling must win.
	got, ok := parseAMDBusy(readFixture(t, "rocm-smi-showuse.json"))
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if got != 93 { // mean of 96 and 90
		t.Errorf("busy = %v, want 93 (the percent, not the GFX Activity counter)", got)
	}
}

func TestParseAMDBusy_UnsupportedSensorIsNotZero(t *testing.T) {
	// "N/A" must read as no sample rather than as an idle GPU, or an
	// unsupported sensor would silently mint a 0%-utilization claim.
	if _, ok := parseAMDBusy([]byte(`{"card0": {"GPU use (%)": "N/A"}}`)); ok {
		t.Error("ok = true, want false when the only reading is N/A")
	}
}

func TestParseAMDBusy_MixedReadingsAverageOnlyUsableOnes(t *testing.T) {
	got, ok := parseAMDBusy([]byte(`{"card0": {"GPU use (%)": "80"}, "card1": {"GPU use (%)": "N/A"}}`))
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if got != 80 {
		t.Errorf("busy = %v, want 80 (the unreadable device is skipped, not counted as 0)", got)
	}
}

func TestParseAMD_RejectsNonJSON(t *testing.T) {
	// An older tool without --json prints a text table; that must be an error,
	// not a silent zero-GPU or zero-percent answer.
	if _, _, err := parseAMDNames([]byte("GPU  Name\n0  MI300X\n")); err == nil {
		t.Error("want error on non-JSON output")
	}
	if _, ok := parseAMDBusy([]byte("GPU use: 96%\n")); ok {
		t.Error("ok = true, want false on non-JSON output")
	}
}

func TestParseAMDNames_UnknownShapeErrors(t *testing.T) {
	// Valid JSON with no recognizable model key must fail loudly: a GPU model
	// the peak table cannot match is better reported than guessed.
	if _, _, err := parseAMDNames([]byte(`[{"gpu": 0, "bus": {"bdf": "0000:0c:00.0"}}]`)); err == nil {
		t.Error("want error when no model-name key is present")
	}
}

func TestGPUEntries_OrdersCardKeysNumerically(t *testing.T) {
	// Lexicographic ordering would put card10 before card2 and make "the first
	// GPU" the wrong device on a 16-way box.
	root, err := decodeJSON([]byte(`{"card10": {"Card Series": "ten"}, "card2": {"Card Series": "two"}, "card0": {"Card Series": "zero"}}`))
	if err != nil {
		t.Fatal(err)
	}
	entries := gpuEntries(root)
	if len(entries) != 3 {
		t.Fatalf("entries = %d, want 3", len(entries))
	}
	first, ok := findValue(entries[0], amdNameKeys)
	if !ok || toText(first) != "zero" {
		t.Errorf("first entry = %v, want card0", first)
	}
	second, _ := findValue(entries[1], amdNameKeys)
	if toText(second) != "two" {
		t.Errorf("second entry = %v, want card2 before card10", second)
	}
}

func TestGPUEntries_SingleUnwrappedObject(t *testing.T) {
	root, err := decodeJSON([]byte(`{"asic": {"market_name": "Instinct MI355X"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if n := len(gpuEntries(root)); n != 1 {
		t.Errorf("entries = %d, want 1 (an unfamiliar wrapper still reads as one device)", n)
	}
}

func TestFindValue_IsDeterministicAcrossRuns(t *testing.T) {
	// Go randomizes map iteration, so a naive depth-first walk could return a
	// different sibling on each run. The alias order and the sorted descent
	// have to pin one answer.
	doc := []byte(`{"zzz": {"market_name": "last"}, "aaa": {"market_name": "first"}}`)
	root, err := decodeJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	want := ""
	for i := 0; i < 50; i++ {
		v, ok := findValue(root, amdNameKeys)
		if !ok {
			t.Fatal("value not found")
		}
		got := toText(v)
		if i == 0 {
			want = got
			continue
		}
		if got != want {
			t.Fatalf("run %d returned %q, first run returned %q: the walk is not deterministic", i, got, want)
		}
	}
	if want != "first" {
		t.Errorf("got %q, want %q (children are visited in sorted key order)", want, "first")
	}
}

func TestToNumber_UnwrapsValueAndRejectsText(t *testing.T) {
	if v, ok := toNumber(map[string]any{"value": float64(42), "unit": "%"}); !ok || v != 42 {
		t.Errorf("wrapped reading = %v ok=%v, want 42 true", v, ok)
	}
	if v, ok := toNumber("87"); !ok || v != 87 {
		t.Errorf("string reading = %v ok=%v, want 87 true", v, ok)
	}
	if _, ok := toNumber("N/A"); ok {
		t.Error("N/A parsed as a number")
	}
}

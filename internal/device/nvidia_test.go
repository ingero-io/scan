package device

import "testing"

func TestParseNvidiaBusy_AveragesAcrossGPUs(t *testing.T) {
	// `--format=csv,noheader,nounits` prints one integer percent per GPU.
	got, ok := parseNvidiaBusy([]byte("90\n92\n88\n90\n"))
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if got != 90 {
		t.Errorf("mean util = %v, want 90", got)
	}
}

func TestParseNvidiaBusy_IgnoresBlankAndJunkLines(t *testing.T) {
	got, ok := parseNvidiaBusy([]byte("  88  \n\nN/A\n"))
	if !ok {
		t.Fatal("ok = false, want true (one numeric line present)")
	}
	if got != 88 {
		t.Errorf("util = %v, want 88", got)
	}
}

func TestParseNvidiaBusy_NoNumbersNotOK(t *testing.T) {
	if _, ok := parseNvidiaBusy([]byte("\n[N/A]\n")); ok {
		t.Error("ok = true, want false when no utilization could be parsed")
	}
}

func TestParseNvidiaBusy_ToleratesDecimals(t *testing.T) {
	// nounits prints integers today, but tolerate a future format change.
	got, ok := parseNvidiaBusy([]byte("90.5\n89.5\n"))
	if !ok || got != 90 {
		t.Errorf("got %v ok=%v, want 90 true", got, ok)
	}
}

func TestParseNvidiaNames(t *testing.T) {
	name, n, err := parseNvidiaNames([]byte("NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\n\nNVIDIA H100 80GB HBM3\n"))
	if err != nil {
		t.Fatal(err)
	}
	if name != "NVIDIA H100 80GB HBM3" || n != 3 {
		t.Errorf("got %q/%d, want 'NVIDIA H100 80GB HBM3'/3 (blank lines skipped)", name, n)
	}
	if _, _, err := parseNvidiaNames([]byte("  \n\n")); err == nil {
		t.Error("empty output should error")
	}
}

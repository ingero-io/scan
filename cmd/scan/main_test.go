package main

import (
	"context"
	"testing"
	"time"
)

func TestSampleGPUUtil_ZeroSamplesReturnsSentinel(t *testing.T) {
	orig := readUtil
	defer func() { readUtil = orig }()
	readUtil = func(context.Context) (float64, bool) { return 0, false } // never reads

	got, err := sampleGPUUtil(context.Background(), 10*time.Millisecond)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got != -1 {
		t.Errorf("got %v, want -1 (sentinel: no utilization samples)", got)
	}
}

func TestSampleGPUUtil_AveragesReadings(t *testing.T) {
	orig := readUtil
	defer func() { readUtil = orig }()
	readUtil = func(context.Context) (float64, bool) { return 80, true }

	got, err := sampleGPUUtil(context.Background(), 10*time.Millisecond)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got != 80 {
		t.Errorf("mean = %v, want 80", got)
	}
}

func TestSampleGPUUtil_HonorsContextCancel(t *testing.T) {
	orig := readUtil
	defer func() { readUtil = orig }()
	readUtil = func(context.Context) (float64, bool) { return 50, true }

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	if _, err := sampleGPUUtil(ctx, time.Hour); err == nil {
		t.Error("want context error when ctx is cancelled, got nil")
	}
}

func TestParseNvidiaSmi_MultiGPU(t *testing.T) {
	out := []byte("NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3\n")
	name, count, err := parseNvidiaSmi(out)
	if err != nil {
		t.Fatal(err)
	}
	if name != "NVIDIA H100 80GB HBM3" {
		t.Errorf("name = %q, want NVIDIA H100 80GB HBM3", name)
	}
	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}

func TestParseNvidiaSmi_SingleGPUTrailingWhitespace(t *testing.T) {
	out := []byte("  NVIDIA A100-SXM4-80GB  \n\n")
	name, count, err := parseNvidiaSmi(out)
	if err != nil {
		t.Fatal(err)
	}
	if name != "NVIDIA A100-SXM4-80GB" {
		t.Errorf("name = %q, want NVIDIA A100-SXM4-80GB", name)
	}
	if count != 1 {
		t.Errorf("count = %d, want 1 (blank lines ignored)", count)
	}
}

func TestParseNvidiaSmi_NoGPUsErrors(t *testing.T) {
	if _, _, err := parseNvidiaSmi([]byte("\n  \n")); err == nil {
		t.Error("want error when nvidia-smi returns no GPUs")
	}
}

func TestParseGPUUtil_AveragesAcrossGPUs(t *testing.T) {
	// `--format=csv,noheader,nounits` prints one integer percent per GPU.
	got, ok := parseGPUUtil([]byte("90\n92\n88\n90\n"))
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if got != 90 {
		t.Errorf("mean util = %v, want 90", got)
	}
}

func TestParseGPUUtil_IgnoresBlankAndJunkLines(t *testing.T) {
	got, ok := parseGPUUtil([]byte("  88  \n\nN/A\n"))
	if !ok {
		t.Fatal("ok = false, want true (one numeric line present)")
	}
	if got != 88 {
		t.Errorf("util = %v, want 88", got)
	}
}

func TestParseGPUUtil_NoNumbersNotOK(t *testing.T) {
	if _, ok := parseGPUUtil([]byte("\n[N/A]\n")); ok {
		t.Error("ok = true, want false when no utilization could be parsed")
	}
}

func TestParseGPUUtil_ToleratesDecimals(t *testing.T) {
	// nounits prints integers today, but tolerate a future format change.
	got, ok := parseGPUUtil([]byte("90.5\n89.5\n"))
	if !ok || got != 90 {
		t.Errorf("got %v ok=%v, want 90 true", got, ok)
	}
}

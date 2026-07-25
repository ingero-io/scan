package device

import (
	"context"
	"testing"
	"time"
)

func stubBusy(t *testing.T, fn func(context.Context, Vendor) (float64, bool)) {
	t.Helper()
	orig := readBusy
	t.Cleanup(func() { readBusy = orig })
	readBusy = fn
}

func TestSampleBusy_ZeroSamplesReturnsSentinel(t *testing.T) {
	stubBusy(t, func(context.Context, Vendor) (float64, bool) { return 0, false })

	got, err := SampleBusy(context.Background(), NVIDIA, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got != -1 {
		t.Errorf("got %v, want -1 (sentinel: no utilization samples)", got)
	}
}

func TestSampleBusy_AveragesReadings(t *testing.T) {
	stubBusy(t, func(context.Context, Vendor) (float64, bool) { return 80, true })

	got, err := SampleBusy(context.Background(), AMD, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got != 80 {
		t.Errorf("mean = %v, want 80", got)
	}
}

func TestSampleBusy_HonorsContextCancel(t *testing.T) {
	stubBusy(t, func(context.Context, Vendor) (float64, bool) { return 50, true })

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	if _, err := SampleBusy(ctx, NVIDIA, time.Hour); err == nil {
		t.Error("want context error when ctx is cancelled, got nil")
	}
}

func TestSampleBusy_PassesVendorThrough(t *testing.T) {
	// One meter, two vendors: the sampling loop must read the vendor it was
	// told about rather than defaulting to nvidia-smi on an AMD host.
	var seen Vendor
	stubBusy(t, func(_ context.Context, v Vendor) (float64, bool) {
		seen = v
		return 10, true
	})
	if _, err := SampleBusy(context.Background(), AMD, 10*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	if seen != AMD {
		t.Errorf("vendor = %q, want %q", seen, AMD)
	}
}

func TestReadBusy_UnknownVendorIsNotASample(t *testing.T) {
	if _, ok := readBusyFor(context.Background(), Vendor("intel")); ok {
		t.Error("ok = true for an unsupported vendor, want false")
	}
}

func TestVendorTool(t *testing.T) {
	if got := NVIDIA.Tool(); got != "nvidia-smi" {
		t.Errorf("NVIDIA tool = %q", got)
	}
	if got := AMD.Tool(); got == "" {
		t.Error("AMD tool name is empty")
	}
}

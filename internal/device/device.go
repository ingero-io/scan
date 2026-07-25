// Package device reads a GPU host's own vendor CLI to answer the two questions
// the userspace meter cannot get from a serving engine's /metrics endpoint:
// which GPUs are on this host, and how busy the vendor's own tool reports them.
//
// It speaks nvidia-smi on NVIDIA hosts and amd-smi (with a rocm-smi fallback) on
// AMD/ROCm hosts, and returns the same shape for both, so one meter reads two
// vendors. That symmetry is the point: the busy percent here is the number the
// operator's dashboard already shows, which is exactly what the MFU estimate is
// contrasted against.
//
// The package reads only what the vendor tool already reports: model name,
// device count, busy percent. Explaining WHY a busy GPU is doing little work is
// the Ingero agent's job, not this one's.
package device

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Vendor is the GPU vendor whose CLI a host is read through.
type Vendor string

const (
	NVIDIA Vendor = "nvidia"
	AMD    Vendor = "amd"
)

// Tool names the command line tool a vendor's devices are read through. It is
// printed next to the utilization figure so an operator can reproduce the
// reading with a command they already run. AMD names both tools because either
// one may have produced the number, depending on what the host has installed.
func (v Vendor) Tool() string {
	switch v {
	case NVIDIA:
		return "nvidia-smi"
	case AMD:
		return "amd-smi/rocm-smi"
	default:
		return string(v)
	}
}

// Info identifies the GPUs present on this host. Name is the model string the
// vendor tool reports for the FIRST device (for example "NVIDIA H100 80GB HBM3"
// or "Instinct MI300X"); it is what the peak and rate tables are matched
// against, so it is passed through verbatim rather than normalized here.
type Info struct {
	Vendor Vendor
	Name   string
	Count  int
}

// toolTimeout bounds every vendor CLI call. The meter targets GPUs under load,
// exactly when a wedged driver can make nvidia-smi or amd-smi hang for many
// seconds; a hung poll is dropped rather than stalling the sampling window.
const toolTimeout = 5 * time.Second

// Detect returns the GPUs on this host, trying NVIDIA first and then AMD. A
// host with both is vanishingly rare in practice and the first vendor that
// reports devices wins; pass an explicit GPU model to the caller's override
// flag if that ever needs steering. The error names both attempts so an
// operator on a GPU host can see which tool was missing or failed.
func Detect(ctx context.Context) (Info, error) {
	nvInfo, nvErr := DetectNVIDIA(ctx)
	if nvErr == nil {
		return nvInfo, nil
	}
	amdInfo, amdErr := DetectAMD(ctx)
	if amdErr == nil {
		return amdInfo, nil
	}
	return Info{}, fmt.Errorf("no GPU found: %w", errors.Join(nvErr, amdErr))
}

// readBusy is the per-snapshot busy reader SampleBusy polls. It is a package
// var so tests can exercise the averaging and no-sample paths without a GPU.
var readBusy = readBusyFor

// readBusyFor takes one busy-percent reading averaged across this host's
// devices, using the vendor's own tool.
func readBusyFor(ctx context.Context, v Vendor) (float64, bool) {
	switch v {
	case NVIDIA:
		return readNVIDIABusy(ctx)
	case AMD:
		return readAMDBusy(ctx)
	default:
		return 0, false
	}
}

// ReadBusy takes one busy-percent snapshot (0..100) averaged across the host's
// GPUs. ok is false when the vendor tool is present but reports no usable
// reading, so a caller can omit the claim instead of printing a fabricated one.
func ReadBusy(ctx context.Context, v Vendor) (float64, bool) {
	return readBusy(ctx, v)
}

// SampleBusy blocks for window, polling the vendor tool every ~2s, and returns
// the mean busy percent across samples and devices (0..100). It returns a
// negative value when nothing could be read at all (GPUs present but the query
// is unsupported), so the caller can omit the utilization claim rather than
// fake one. Honors ctx cancellation.
func SampleBusy(ctx context.Context, v Vendor, window time.Duration) (float64, error) {
	const step = 2 * time.Second
	deadline := time.NewTimer(window)
	defer deadline.Stop()
	ticker := time.NewTicker(step)
	defer ticker.Stop()

	var sum float64
	var n int
	take := func() {
		if u, ok := readBusy(ctx, v); ok {
			sum += u
			n++
		}
	}
	take() // one reading at the window start
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-deadline.C:
			if n == 0 {
				return -1, nil
			}
			return sum / float64(n), nil
		case <-ticker.C:
			take()
		}
	}
}

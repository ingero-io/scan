package device

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// DetectNVIDIA reads the GPU model and count from nvidia-smi. The CSV name
// query is the presence test as well: a host with the driver but no visible
// device returns no rows and is reported as no GPU.
func DetectNVIDIA(ctx context.Context) (Info, error) {
	cctx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	if err != nil {
		return Info{}, fmt.Errorf("nvidia-smi: %w", err)
	}
	name, count, err := parseNvidiaNames(out)
	if err != nil {
		return Info{}, err
	}
	return Info{Vendor: NVIDIA, Name: name, Count: count, MIGEnabled: readNVIDIAMIG(ctx)}, nil
}

// readNVIDIABusy takes one nvidia-smi utilization snapshot across all GPUs,
// bounded by toolTimeout so a hung driver drops the sample instead of stalling
// the window.
func readNVIDIABusy(ctx context.Context) (float64, bool) {
	cctx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi",
		"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false
	}
	return parseNvidiaBusy(out)
}

// parseNvidiaBusy averages the integer utilization percentages nvidia-smi
// prints (one line per GPU). Factored out so it is unit-testable without a GPU.
func parseNvidiaBusy(out []byte) (float64, bool) {
	var sum float64
	var n int
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		l := strings.TrimSpace(line)
		if l == "" {
			continue
		}
		v, err := strconv.ParseFloat(l, 64)
		if err != nil {
			continue
		}
		sum += v
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

// parseNvidiaNames parses `nvidia-smi --query-gpu=name --format=csv,noheader`
// output into the first GPU's name and the GPU count.
func parseNvidiaNames(out []byte) (name string, count int, err error) {
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		l := strings.TrimSpace(line)
		if l == "" {
			continue
		}
		if name == "" {
			name = l
		}
		count++
	}
	if name == "" {
		return "", 0, fmt.Errorf("nvidia-smi returned no GPUs")
	}
	return name, count, nil
}

// readNVIDIAMIG reports whether any visible device has MIG mode enabled. It is a
// SEPARATE nvidia-smi call rather than an extra column on the name query so that
// a driver too old to know the mig.mode.current field cannot fail the device
// detection that the whole meter depends on.
//
// A failed or unparseable query is reported as not-MIG. That is sound rather than
// optimistic: MIG exists only on Ampere datacenter parts and later, and every
// driver new enough to partition a GPU is new enough to answer this query.
func readNVIDIAMIG(ctx context.Context) bool {
	cctx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi",
		"--query-gpu=mig.mode.current", "--format=csv,noheader").Output()
	if err != nil {
		return false
	}
	return parseNvidiaMIG(out)
}

// parseNvidiaMIG reports whether any line of the mig.mode.current query says
// Enabled. Non-MIG-capable devices report "N/A" and MIG-capable but unpartitioned
// ones report "Disabled"; either way only an explicit Enabled counts.
func parseNvidiaMIG(out []byte) bool {
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if strings.EqualFold(strings.TrimSpace(line), "enabled") {
			return true
		}
	}
	return false
}

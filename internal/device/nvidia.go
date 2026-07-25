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
	return Info{Vendor: NVIDIA, Name: name, Count: count}, nil
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

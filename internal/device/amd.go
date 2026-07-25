package device

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ROCmPresent reports whether this looks like a ROCm host at all. It is used
// only to sharpen the error an operator sees: a box with /opt/rocm but a broken
// amd-smi is a different problem from a box with no AMD GPU. The tool calls
// themselves are the real detection.
func ROCmPresent() bool {
	if _, err := exec.LookPath("amd-smi"); err == nil {
		return true
	}
	if _, err := exec.LookPath("rocm-smi"); err == nil {
		return true
	}
	// The kernel fusion driver node exists on any host with an AMD GPU bound to
	// amdgpu/KFD, whether or not the userspace tools are installed.
	for _, p := range []string{"/sys/class/kfd", "/opt/rocm"} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

// DetectAMD reads the GPU model and count from amd-smi, falling back to
// rocm-smi. Both tools ship with ROCm and both change their JSON shape across
// releases, so the parsing below matches documented key ALIASES rather than
// binding to one release's struct: a renamed field degrades to "not reported"
// instead of crashing, and the live capture on a real box pins which spelling
// the fleet in front of us actually emits.
func DetectAMD(ctx context.Context) (Info, error) {
	if out, err := runAMDTool(ctx, "amd-smi", "static", "--json"); err == nil {
		if name, count, perr := parseAMDNames(out); perr == nil {
			return Info{Vendor: AMD, Name: name, Count: count}, nil
		}
	}
	out, err := runAMDTool(ctx, "rocm-smi", "--showproductname", "--json")
	if err != nil {
		if ROCmPresent() {
			return Info{}, fmt.Errorf("ROCm is present but neither amd-smi nor rocm-smi reported a GPU: %w", err)
		}
		return Info{}, fmt.Errorf("amd-smi/rocm-smi: %w", err)
	}
	name, count, err := parseAMDNames(out)
	if err != nil {
		return Info{}, err
	}
	return Info{Vendor: AMD, Name: name, Count: count}, nil
}

// readAMDBusy takes one graphics-activity snapshot averaged across the host's
// AMD GPUs. gfx_activity is the AMD counterpart of nvidia-smi's
// utilization.gpu: the vendor's own busy percent, which is exactly the number
// the MFU estimate is contrasted against.
func readAMDBusy(ctx context.Context) (float64, bool) {
	if out, err := runAMDTool(ctx, "amd-smi", "metric", "--json"); err == nil {
		if busy, ok := parseAMDBusy(out); ok {
			return busy, ok
		}
	}
	out, err := runAMDTool(ctx, "rocm-smi", "--showuse", "--json")
	if err != nil {
		return 0, false
	}
	return parseAMDBusy(out)
}

// runAMDTool runs one ROCm CLI call bounded by toolTimeout.
func runAMDTool(ctx context.Context, name string, args ...string) ([]byte, error) {
	cctx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, name, args...).Output()
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w", name, strings.Join(args, " "), err)
	}
	return out, nil
}

// Key aliases, in priority order. amd-smi nests the model under asic.market_name
// and the busy percent under usage.gfx_activity; rocm-smi prints flat,
// human-titled keys like "Card Series" and "GPU use (%)". Normalization strips
// punctuation and case so both spellings hit the same alias.
// Busy alias order is load-bearing: rocm-smi --showuse prints BOTH "GPU use (%)"
// and a raw "GFX Activity" accumulator, and only the first is a percent, so the
// percent spellings must be tried before the activity ones. amd-smi has no
// "gpu_use" key, so its usage.gfx_activity percent is still reached.
var (
	amdNameKeys = []string{"marketname", "productname", "cardseries", "devicename", "cardmodel"}
	amdBusyKeys = []string{"gpuuse", "gpuutilization", "gpuutil", "gfxactivity", "graphicsactivity"}
)

// parseAMDNames returns the first GPU's model name and the device count from an
// amd-smi or rocm-smi JSON document.
func parseAMDNames(out []byte) (name string, count int, err error) {
	root, err := decodeJSON(out)
	if err != nil {
		return "", 0, err
	}
	entries := gpuEntries(root)
	if len(entries) == 0 {
		return "", 0, fmt.Errorf("no GPUs in AMD tool output")
	}
	for _, e := range entries {
		v, ok := findValue(e, amdNameKeys)
		if !ok {
			continue
		}
		if s := toText(v); s != "" && name == "" {
			name = s
		}
	}
	if name == "" {
		return "", 0, fmt.Errorf("AMD tool output carries no recognizable GPU model name")
	}
	return name, len(entries), nil
}

// parseAMDBusy averages the graphics-activity percent across the GPUs in an
// amd-smi or rocm-smi JSON document. ok is false when no device reported a
// usable number, which keeps an "N/A" reading from being averaged in as 0.
func parseAMDBusy(out []byte) (float64, bool) {
	root, err := decodeJSON(out)
	if err != nil {
		return 0, false
	}
	var sum float64
	var n int
	for _, e := range gpuEntries(root) {
		v, ok := findValue(e, amdBusyKeys)
		if !ok {
			continue
		}
		f, ok := toNumber(v)
		if !ok {
			continue
		}
		sum += f
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n), true
}

func decodeJSON(out []byte) (any, error) {
	var root any
	if err := json.Unmarshal(out, &root); err != nil {
		return nil, fmt.Errorf("parse AMD tool JSON: %w", err)
	}
	return root, nil
}

// cardKey matches the per-device keys rocm-smi uses for its top-level map
// ("card0", "card1", ...). amd-smi emits an array instead, handled separately.
var cardKey = regexp.MustCompile(`^(card|gpu)[0-9]+$`)

// gpuEntries splits a decoded AMD tool document into one node per GPU. amd-smi
// emits an array of device objects; rocm-smi emits an object keyed by card.
// Anything else is treated as a single device so a one-GPU box with an
// unfamiliar wrapper still reads rather than silently reporting zero GPUs.
func gpuEntries(root any) []any {
	switch n := root.(type) {
	case []any:
		return n
	case map[string]any:
		var keys []string
		for k := range n {
			if cardKey.MatchString(normalizeKey(k)) {
				keys = append(keys, k)
			}
		}
		if len(keys) == 0 {
			return []any{n}
		}
		sortCardKeys(keys)
		entries := make([]any, 0, len(keys))
		for _, k := range keys {
			entries = append(entries, n[k])
		}
		return entries
	default:
		return nil
	}
}

// sortCardKeys orders rocm-smi's per-device keys by their numeric suffix, so a
// 16-GPU host reads card2 before card10 and "the first GPU" means device 0.
func sortCardKeys(keys []string) {
	sort.Slice(keys, func(i, j int) bool {
		pi, ni := splitCardKey(keys[i])
		pj, nj := splitCardKey(keys[j])
		if pi != pj {
			return pi < pj
		}
		return ni < nj
	})
}

func splitCardKey(k string) (prefix string, index int) {
	norm := normalizeKey(k)
	cut := len(norm)
	for cut > 0 && norm[cut-1] >= '0' && norm[cut-1] <= '9' {
		cut--
	}
	n, err := strconv.Atoi(norm[cut:])
	if err != nil {
		return norm, 0
	}
	return norm[:cut], n
}

// findValue walks a decoded JSON tree for the first value whose key matches one
// of keys after normalization. Aliases are tried in priority order at every
// level before descending, and children are visited in sorted key order, so the
// result never depends on Go's randomized map iteration.
func findValue(node any, keys []string) (any, bool) {
	switch n := node.(type) {
	case map[string]any:
		byNorm := make(map[string]any, len(n))
		for k, v := range n {
			byNorm[normalizeKey(k)] = v
		}
		for _, want := range keys {
			if v, ok := byNorm[want]; ok {
				return v, true
			}
		}
		childKeys := make([]string, 0, len(n))
		for k := range n {
			childKeys = append(childKeys, k)
		}
		sort.Strings(childKeys)
		for _, k := range childKeys {
			if v, ok := findValue(n[k], keys); ok {
				return v, true
			}
		}
	case []any:
		for _, item := range n {
			if v, ok := findValue(item, keys); ok {
				return v, true
			}
		}
	}
	return nil, false
}

// normalizeKey lowercases a JSON key and drops everything that is not a letter
// or digit, so "GPU use (%)", "gfx_activity" and "Market Name" compare against
// the alias lists as "gpuuse", "gfxactivity" and "marketname".
func normalizeKey(k string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(k) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// toNumber reads a numeric leaf. amd-smi wraps readings as
// {"value": 42, "unit": "%"}, rocm-smi prints them as strings, and both use
// "N/A" for an unsupported sensor, which must not be read as zero.
func toNumber(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err != nil {
			return 0, false
		}
		return f, true
	case map[string]any:
		if inner, ok := findValue(t, []string{"value"}); ok {
			return toNumber(inner)
		}
	}
	return 0, false
}

// toText reads a string leaf, unwrapping the same {"value": ...} shape.
func toText(v any) string {
	switch t := v.(type) {
	case string:
		return strings.TrimSpace(t)
	case map[string]any:
		if inner, ok := findValue(t, []string{"value"}); ok {
			return toText(inner)
		}
	}
	return ""
}

package enginedetect

import "testing"

// The serving engine's Prometheus endpoint exports throughput, latency and queue
// counters and carries no dtype or quantization at all, so the command line is
// the only place the precision-determining flags can be read. These assert the
// reading, not the interpretation: mapping a scheme to a compute precision is the
// meter's job.
func TestDetect_VLLMPrecisionFlags(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 400,
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--model", "meta-llama/Llama-3-70B",
		"--quantization", "fp8",
		"--dtype", "auto",
		"--kv-cache-dtype", "fp8_e5m2",
		"--port", "8000",
	)
	d, ok := detectAt(dir, 400)
	if !ok {
		t.Fatal("vLLM not detected")
	}
	if d.Quantization != "fp8" {
		t.Errorf("Quantization = %q, want fp8", d.Quantization)
	}
	if d.Dtype != "auto" {
		t.Errorf("Dtype = %q, want auto", d.Dtype)
	}
	if d.KVCacheDtype != "fp8_e5m2" {
		t.Errorf("KVCacheDtype = %q, want fp8_e5m2", d.KVCacheDtype)
	}
}

// Both spellings appear in the wild and an engine started with the equals form
// must not read as an engine with no precision flags at all.
func TestDetect_PrecisionFlagsAcceptEqualsForm(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 401,
		"python3", "-m", "vllm.entrypoints.openai.api_server",
		"--quantization=awq", "--dtype=bfloat16",
	)
	d, _ := detectAt(dir, 401)
	if d.Quantization != "awq" {
		t.Errorf("Quantization = %q, want awq", d.Quantization)
	}
	if d.Dtype != "bfloat16" {
		t.Errorf("Dtype = %q, want bfloat16", d.Dtype)
	}
}

// TGI spells the same concept --quantize.
func TestDetect_TGIUsesQuantizeFlag(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 402,
		"text-generation-launcher", "--model-id", "bigscience/bloom",
		"--quantize", "gptq", "--num-shard", "4",
	)
	d, ok := detectAt(dir, 402)
	if !ok {
		t.Fatal("TGI not detected")
	}
	if d.Quantization != "gptq" {
		t.Errorf("Quantization = %q, want gptq from --quantize", d.Quantization)
	}
	if d.Devices != 4 {
		t.Errorf("Devices = %d, want 4 from --num-shard", d.Devices)
	}
}

// The replica width is what stops a narrow replica being measured against every
// GPU on the host. Tensor and pipeline widths multiply.
func TestDetect_VLLMDeviceWidthMultipliesParallelAxes(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want int
	}{
		{"tensor parallel only", []string{"--tensor-parallel-size", "4"}, 4},
		{"short tensor parallel flag", []string{"-tp", "2"}, 2},
		{"tensor times pipeline", []string{"--tensor-parallel-size", "2", "--pipeline-parallel-size", "2"}, 4},
		{"equals form", []string{"--tensor-parallel-size=8"}, 8},
		{"no width flag is unknown, not one", nil, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			args := append([]string{"python3", "-m", "vllm.entrypoints.openai.api_server"}, tc.args...)
			writeFakeCmdline(t, dir, 403, args...)
			d, ok := detectAt(dir, 403)
			if !ok {
				t.Fatal("vLLM not detected")
			}
			if d.Devices != tc.want {
				t.Errorf("Devices = %d, want %d", d.Devices, tc.want)
			}
		})
	}
}

func TestDetect_SGLangDeviceWidth(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 404,
		"python3", "-m", "sglang.launch_server",
		"--model-path", "meta-llama/Llama-3-8B",
		"--tp-size", "2", "--dp-size", "2",
	)
	d, ok := detectAt(dir, 404)
	if !ok {
		t.Fatal("SGLang not detected")
	}
	if d.Devices != 4 {
		t.Errorf("Devices = %d, want 4 (tensor-parallel 2 across 2 data-parallel replicas behind one endpoint)", d.Devices)
	}
}

// An engine started with no precision flags must report empty strings rather
// than a default, so the meter labels the estimate instead of assuming a
// favourable precision.
func TestDetect_AbsentPrecisionFlagsAreEmptyNotDefaulted(t *testing.T) {
	dir := t.TempDir()
	writeFakeCmdline(t, dir, 405,
		"python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "gpt2",
	)
	d, _ := detectAt(dir, 405)
	if d.Quantization != "" || d.Dtype != "" || d.KVCacheDtype != "" {
		t.Errorf("absent flags produced (%q, %q, %q), want all empty", d.Quantization, d.Dtype, d.KVCacheDtype)
	}
	if d.Devices != 0 {
		t.Errorf("Devices = %d, want 0 for an unknown width", d.Devices)
	}
}

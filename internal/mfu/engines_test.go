package mfu

import "testing"

// TestParserFor_EveryEngineProducesTokens guards the dead-engine class:
// every engine ParserFor accepts must yield a non-zero output-token
// total from a realistic /metrics body. This single test catches both
// the TGI histogram gap and any engine advertised without an
// output-token source.
func TestParserFor_EveryEngineProducesTokens(t *testing.T) {
	bodies := map[string]string{
		"vllm":   "vllm:generation_tokens_total 12345\n",
		"sglang": "sglang_generation_tokens_total 6789\n",
		"tgi":    "tgi_request_generated_tokens_sum 4242\ntgi_request_generated_tokens_count 10\n",
	}
	for engine, body := range bodies {
		p, err := ParserFor(engine)
		if err != nil {
			t.Fatalf("%s: ParserFor: %v", engine, err)
		}
		samples, err := p.Parse([]byte(body))
		if err != nil {
			t.Fatalf("%s: parse: %v", engine, err)
		}
		if got := SumOutputTokens(samples); got <= 0 {
			t.Errorf("%s: SumOutputTokens = %g, want > 0 (engine advertised but yields no tokens)", engine, got)
		}
	}
	if _, err := ParserFor("triton"); err == nil {
		t.Error("triton must be rejected by ParserFor (no output-token source)")
	}
}

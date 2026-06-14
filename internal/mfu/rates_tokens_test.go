package mfu

import (
	"testing"
	"time"

	"github.com/ingero-io/scan/internal/scrape"
)

func TestLookupRate_DefaultProviderIsDeterministic(t *testing.T) {
	// H100 is on every provider; ec2 is first in the default order, and
	// within a provider we take the max matching list price.
	rate, provider, ok := LookupRate("NVIDIA H100 80GB HBM3")
	if !ok {
		t.Fatal("expected a rate for H100")
	}
	if provider != "ec2" {
		t.Errorf("provider = %q, want ec2 (first in default order)", provider)
	}
	if rate != 6.88 {
		t.Errorf("rate = %g, want 6.88 (ec2 H100 post-2025-cut)", rate)
	}
	// Determinism: repeated lookups must agree (map iteration is randomized).
	for i := 0; i < 50; i++ {
		r2, p2, _ := LookupRate("NVIDIA H100 80GB HBM3")
		if r2 != rate || p2 != provider {
			t.Fatalf("non-deterministic lookup: got %g/%s, want %g/%s", r2, p2, rate, provider)
		}
	}
}

func TestLookupRate_FamilyTokenMatch(t *testing.T) {
	// A short name still resolves via the family token; ec2 lists A100,
	// max matching is the 80GB at 3.43.
	rate, provider, ok := LookupRate("A100")
	if !ok {
		t.Fatal("expected a rate for A100")
	}
	if provider != "ec2" || rate != 3.43 {
		t.Errorf("got %g/%s, want 3.43/ec2", rate, provider)
	}
}

func TestLookupRate_FallsThroughToProviderWithTheGPU(t *testing.T) {
	// L40S is only on lambda + coreweave; default order reaches
	// coreweave (1.50) before lambda (1.29).
	rate, provider, ok := LookupRate("NVIDIA L40S")
	if !ok {
		t.Fatal("expected a rate for L40S")
	}
	if provider != "coreweave" || rate != 1.50 {
		t.Errorf("got %g/%s, want 1.50/coreweave", rate, provider)
	}
}

func TestLookupRate_UnknownGPU(t *testing.T) {
	if _, _, ok := LookupRate("NVIDIA RTX 4090"); ok {
		t.Error("unknown GPU should not resolve a bundled rate")
	}
}

func TestTokensPerSec(t *testing.T) {
	cases := []struct {
		prev, curr float64
		interval   time.Duration
		want       float64
	}{
		{1000, 1300, 10 * time.Second, 30}, // normal
		{1000, 1000, 10 * time.Second, 0},  // idle
		{1300, 1000, 10 * time.Second, 0},  // counter reset -> 0, not negative
		{1000, 2000, 0, 0},                 // zero interval -> 0
	}
	for _, c := range cases {
		if got := TokensPerSec(c.prev, c.curr, c.interval); got != c.want {
			t.Errorf("TokensPerSec(%g,%g,%s) = %g, want %g", c.prev, c.curr, c.interval, got, c.want)
		}
	}
}

func TestSumOutputTokens(t *testing.T) {
	samples := []scrape.ScrapedSample{
		{CanonicalName: OutputTokenCanonical, Kind: scrape.SampleCounter, Value: 500},
		{CanonicalName: OutputTokenCanonical, Kind: scrape.SampleCounter, Value: 700}, // multiple workers
		{CanonicalName: "gen_ai.client.token.usage.input", Kind: scrape.SampleCounter, Value: 9999},
		{CanonicalName: OutputTokenCanonical, Kind: scrape.SampleGauge, Value: 1}, // wrong kind, ignored
	}
	if got := SumOutputTokens(samples); got != 1200 {
		t.Errorf("SumOutputTokens = %g, want 1200 (only output counters)", got)
	}
}

func TestLookupRate_GH200ResolvesToLambdaOnly(t *testing.T) {
	// GH200 is only in lambda; the longest-token sort must resolve
	// "gh200" (not the "h200" substring), and the provider fall-through
	// must reach lambda.
	rate, provider, ok := LookupRate("NVIDIA GH200 480GB")
	if !ok || provider != "lambda" || rate != 2.29 {
		t.Errorf("got %g/%s ok=%v, want 2.29/lambda", rate, provider, ok)
	}
	if familyToken("NVIDIA GH200 480GB") != "gh200" {
		t.Errorf("familyToken = %q, want gh200 (not the h200 substring)", familyToken("NVIDIA GH200 480GB"))
	}
}

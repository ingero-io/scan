package mfu

import (
	"time"

	"github.com/ingero-io/scan/internal/scrape"
)

// OutputTokenCanonical is the OTel semconv name the scrapers map the
// generation-token counter to (vllm:generation_tokens_total, sglang/tgi
// equivalents). The counter is read twice and the delta divided by the
// elapsed interval to get achieved output tokens/sec.
const OutputTokenCanonical = "gen_ai.client.token.usage.output"

// SumOutputTokens totals the output (generation) token counter across
// scraped samples. It is a monotonic counter, so a single reading is a
// running total; the rate comes from TokensPerSec over two readings.
func SumOutputTokens(samples []scrape.ScrapedSample) float64 {
	var total float64
	for _, s := range samples {
		if s.CanonicalName == OutputTokenCanonical &&
			(s.Kind == scrape.SampleCounter || (s.Kind == scrape.SampleHistogram && s.IsSum)) {
			total += s.Value
		}
	}
	return total
}

// TokensPerSec converts two counter readings taken `interval` apart
// into a per-second rate. Returns 0 for a non-positive interval or an
// apparent counter reset (curr < prev, e.g. the server restarted
// between readings) so a restart never produces a negative or nonsense
// throughput.
func TokensPerSec(prev, curr float64, interval time.Duration) float64 {
	if interval <= 0 || curr < prev {
		return 0
	}
	return (curr - prev) / interval.Seconds()
}

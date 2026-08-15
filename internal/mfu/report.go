package mfu

import (
	"fmt"
	"strings"

	"github.com/ingero-io/scan/internal/scrape"
)

// ParserFor returns the Prometheus exposition parser for a serving
// engine name. Accepts vllm, sglang, tgi (case-insensitive). Triton is
// intentionally unsupported here: it exposes no output-token metric to
// derive tokens/sec.
func ParserFor(engine string) (scrape.Parser, error) {
	switch strings.ToLower(strings.TrimSpace(engine)) {
	case "vllm":
		return scrape.VLLMParser{}, nil
	case "sglang":
		return scrape.SGLangParser{}, nil
	case "tgi":
		return scrape.TGIParser{}, nil
	default:
		return nil, fmt.Errorf("unknown engine %q (want vllm|sglang|tgi)", engine)
	}
}

// assumedSuffix marks a denominator that was defaulted rather than detected,
// so the estimate line distinguishes a measured divisor from a guessed one.
func assumedSuffix(est Estimate) string {
	if est.PrecisionAssumed {
		return ", assumed - serving precision not detected"
	}
	return ""
}

// Render formats an Estimate as a human-readable report. rateSource
// describes where the hourly rate came from (e.g. "ec2 list price" or
// "--rate"); empty when no rate was available, in which case the
// dollar lines are omitted.
func Render(est Estimate, rateSource string) string {
	var b strings.Builder
	fmt.Fprintln(&b, "GPU MFU scan (ESTIMATE - modeled, not measured)")
	fmt.Fprintf(&b, "  workload : %s on %dx %s\n", est.Model, est.GPUCount, est.GPU)
	fmt.Fprintf(&b, "  output   : %.0f tokens/sec  ->  %.0f of %.0f TFLOP/s used\n",
		est.TokensPerSec, est.AchievedTFLOPS, est.PeakTFLOPS)
	// Contrast MFU against the LIVE utilization scan just read from the host's
	// own vendor tool - the dashboard number - not a hardcoded "~100%". When
	// utilization could not be read, claim nothing rather than fake a figure.
	// The tool is named so an operator can reproduce the reading with the
	// command they already run.
	utilTool := est.UtilTool
	if utilTool == "" {
		utilTool = "vendor-reported"
	}
	if est.GPUUtilPct != nil {
		fmt.Fprintf(&b, "  util     : %.0f%%   (%s GPU utilization - what the dashboard shows)\n", *est.GPUUtilPct, utilTool)
		fmt.Fprintf(&b, "  MFU      : %.1f%%   (the real work behind that utilization)\n", est.MFU*100)
	} else {
		fmt.Fprintf(&b, "  MFU      : %.1f%%   (%s utilization unavailable)\n", est.MFU*100, utilTool)
	}
	// Headroom is shown as a CEILING vs a published healthy-serving band, never
	// as "$ you will save". The SLO-safe recoverable slice + a signed receipt are
	// the agent's job (footer below). No cause is ever named here.
	// Show headroom only when the replica is MATERIALLY below the healthy band:
	// near-healthy multiples round to a meaningless "~1x" and the envelope's low
	// bound collapses to $0, so gate both lines together on the same cutoff
	// (multiple >= 1.5 -> the printed "~Nx" is always >= 2x and the low $ bound > 0).
	showHeadroom := est.HeadroomMultiple >= 1.5
	if showHeadroom {
		fmt.Fprintf(&b, "  headroom : ~%.0fx below well-batched serving (~%.0f-%.0f%% MFU, public benchmark)\n",
			est.HeadroomMultiple, healthyServingMFULow*100, healthyServingMFUHigh*100)
	}
	if est.HourlyUSDPerGPU > 0 {
		src := rateSource
		if src != "" {
			src = " (" + src + ", upper bound)"
		}
		fmt.Fprintf(&b, "  cost     : ~$%.0f/mo at $%.2f/GPU/hr%s\n", est.MonthlyUSD, est.HourlyUSDPerGPU, src)
		switch {
		case est.DollarEnvelopeSuppressed:
			// Say the figure is being withheld and why. Printing nothing here
			// would read as "no headroom found", a different and wrong claim.
			fmt.Fprintln(&b, "  envelope : withheld - the GPU count behind this MFU was not attributed to")
			fmt.Fprintln(&b, "             this engine, and a dollar figure would imply a confidence the")
			fmt.Fprintln(&b, "             input does not support")
		case showHeadroom && est.MonthlyHeadroomHighUSD > 0:
			fmt.Fprintf(&b, "  envelope : up to ~$%.0f-$%.0f/mo of consolidation headroom (CEILING, not a promise)\n",
				est.MonthlyHeadroomLowUSD, est.MonthlyHeadroomHighUSD)
		}
	}
	// Name the divisor that was actually used. "dense BF16 peak" was printed
	// unconditionally here even when the workload served in a lower precision,
	// which described the wrong arithmetic to the reader.
	fmt.Fprintf(&b, "\n  Estimate: MFU is modeled (2 FLOPs/param/token, dense %s peak%s), not\n",
		strings.ToUpper(est.Precision.String()), assumedSuffix(est))
	fmt.Fprintln(&b, "  measured. Pass --rate <usd/hr> for your real GPU cost.")
	for _, c := range est.Caveats {
		fmt.Fprintf(&b, "  - %s\n", c)
	}
	fmt.Fprintln(&b, "\n  The envelope is GROSS headroom (idle + stalled). The SLO-safe slice you")
	fmt.Fprintln(&b, "  can actually reclaim - split from idle rightsizing, attributed to a cause,")
	fmt.Fprintln(&b, "  and proven by a signed receipt - is what the Ingero agent measures. This")
	fmt.Fprintln(&b, "  meter shows the gap; the agent recovers it.")
	return b.String()
}

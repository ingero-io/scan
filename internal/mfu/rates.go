package mfu

import (
	_ "embed"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed gpu_rates.yaml
var bundledRatesYAML []byte

type ratesCatalog struct {
	CurrencyName   string                        `yaml:"currency_name"`
	CurrencySymbol string                        `yaml:"currency_symbol"`
	Providers      map[string]map[string]float64 `yaml:"providers"`
	FallbackRate   float64                       `yaml:"fallback_rate"`
}

// defaultProviderOrder is the lookup order for the single auto rate.
// ec2 first: recognizable and on the higher end, which is acceptable
// because the number is explicitly a list-price UPPER BOUND and
// `--rate` overrides it. The AMD-only provider sits last so it never
// shadows a hyperscaler rate for an NVIDIA SKU; it is reached only when
// no earlier provider lists the family at all.
var defaultProviderOrder = []string{"ec2", "gcp", "azure", "coreweave", "lambda", "hotaisle"}

func loadBundledRates() (ratesCatalog, error) {
	var c ratesCatalog
	if err := yaml.Unmarshal(bundledRatesYAML, &c); err != nil {
		return ratesCatalog{}, fmt.Errorf("parse bundled gpu_rates: %w", err)
	}
	return c, nil
}

// LookupRate returns one hourly USD/GPU list-price estimate for a GPU
// model from the bundled catalog, plus the provider it came from. It
// scans providers in defaultProviderOrder and returns, from the first
// provider that lists the GPU's family (h100, a100, ...), the MAX
// matching list price (deterministic, and consistent with the
// upper-bound framing). ok=false when nothing matches, in which case
// the caller should require an explicit --rate.
//
// The result is a LIST-PRICE UPPER BOUND; real exposure
// (spot/reserved/committed-use) is lower. Override with --rate for the
// operator's real rate.
func LookupRate(gpu string) (usdPerHour float64, provider string, ok bool) {
	c, err := loadBundledRates()
	if err != nil {
		return 0, "", false
	}
	token := familyToken(gpu)
	if token == "" {
		return 0, "", false
	}
	for _, p := range defaultProviderOrder {
		models, has := c.Providers[p]
		if !has {
			continue
		}
		best := 0.0
		for name, rate := range models {
			if rate > 0 && rate > best && strings.Contains(strings.ToLower(name), token) {
				best = rate
			}
		}
		if best > 0 {
			return best, p, true
		}
	}
	return 0, "", false
}

// familyToken returns the GPU family token (h100, a100, l40s, l4,
// gh200, v100, ...) shared by the rate catalog and the peak table, or
// "" if unknown.
func familyToken(gpu string) string {
	g := strings.ToLower(gpu)
	for _, e := range gpuPeakTFLOPS {
		if strings.Contains(g, e.token) {
			return e.token
		}
	}
	return ""
}

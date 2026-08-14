package device

import "testing"

// nvidia-smi reports one row per PHYSICAL GPU even when MIG is enabled, so the
// count alone cannot reveal a partitioned host. This query is what does, and its
// answers have three shapes: Enabled, Disabled, and N/A on parts that cannot do
// MIG at all.
func TestParseNvidiaMIG(t *testing.T) {
	for _, tc := range []struct {
		name string
		out  string
		want bool
	}{
		{"all disabled", "Disabled\nDisabled\nDisabled\nDisabled\n", false},
		{"not MIG capable", "N/A\nN/A\n", false},
		{"all enabled", "Enabled\nEnabled\n", true},
		{"partially enabled host still counts", "Disabled\nEnabled\nDisabled\n", true},
		{"case tolerated", "enabled\n", true},
		{"padded", "  Enabled  \n", true},
		{"empty output", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseNvidiaMIG([]byte(tc.out)); got != tc.want {
				t.Errorf("parseNvidiaMIG(%q) = %v, want %v", tc.out, got, tc.want)
			}
		})
	}
}

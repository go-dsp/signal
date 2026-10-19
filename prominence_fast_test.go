package signal

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestPeakProminencesStackMatchesScan checks the monotonic-stack bases
// against the per-peak scan (which a single peak takes) at every sample of
// quantized random signals, so that ties and plateaus are frequent.
func TestPeakProminencesStackMatchesScan(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 9))
	for trial := 0; trial < 200; trial++ {
		n := 1 + r.IntN(60)
		levels := 1 + r.IntN(6)
		x := make([]float64, n)
		for i := range x {
			x[i] = float64(r.IntN(levels))
			if trial%7 == 0 {
				x[i] = math.Inf(1 - 2*r.IntN(2))
			}
		}
		all := make([]int, n)
		for i := range all {
			all[i] = i
		}
		prom, left, right, err := PeakProminences(x, all, 0)
		if err != nil {
			t.Fatal(err)
		}
		for i := range all {
			p1, l1, r1, err := PeakProminences(x, []int{i}, 0)
			if err != nil {
				t.Fatal(err)
			}
			same := math.Float64bits(p1[0]) == math.Float64bits(prom[i]) || (math.IsNaN(p1[0]) && math.IsNaN(prom[i]))
			if !same || l1[0] != left[i] || r1[0] != right[i] {
				t.Fatalf("x=%v peak %d: stack (%v, %d, %d), scan (%v, %d, %d)", x, i, prom[i], left[i], right[i], p1[0], l1[0], r1[0])
			}
		}
	}
	// NaN keeps the scan, where it stops a walk like a higher sample: the
	// base on the NaN side is the peak itself.
	x := []float64{0, 2, math.NaN(), 3, 1}
	prom, left, right, err := PeakProminences(x, []int{1, 3}, 0)
	if err != nil || prom[0] != 0 || prom[1] != 0 || left[0] != 0 || right[0] != 1 || left[1] != 3 || right[1] != 4 {
		t.Fatalf("NaN: %v %v %v %v", prom, left, right, err)
	}
}

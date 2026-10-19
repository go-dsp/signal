package signal

import (
	"math"
	"math/rand/v2"
	"testing"
)

// refFIR is the plain transposed direct form, the definition lfilterFIR's
// direct-chain path must reproduce bit for bit.
func refFIR(b, x, zi []float64) (y, zf []float64) {
	nb := len(b)
	z := make([]float64, nb)
	copy(z, zi)
	y = make([]float64, len(x))
	for n, xn := range x {
		y[n] = z[0] + b[0]*xn
		for i := 1; i < nb-1; i++ {
			z[i-1] = z[i] + b[i]*xn
		}
		if nb > 1 {
			z[nb-2] = b[nb-1] * xn
		}
	}
	return y, z[:nb-1]
}

// TestLFilterFIRDirectChainBitIdentical covers the four-at-a-time chain, its
// tail, the head that still sees the initial state, and the final state.
func TestLFilterFIRDirectChainBitIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, nb := range []int{1, 2, 8, 9, 37, 101} {
		for _, n := range []int{0, 1, nb - 1, 2*nb - 1, 2 * nb, 2*nb + 3, 513} {
			if n < 0 {
				continue
			}
			for _, withZi := range []bool{false, true} {
				b := make([]float64, nb)
				for i := range b {
					b[i] = r.NormFloat64()
				}
				x := make([]float64, n)
				for i := range x {
					x[i] = r.NormFloat64() * math.Exp(4*r.Float64())
				}
				var zi []float64
				if withZi {
					zi = make([]float64, nb-1)
					for i := range zi {
						zi[i] = r.NormFloat64()
					}
				}
				y, zf, err := LFilter(b, []float64{1}, x, zi)
				if err != nil {
					t.Fatal(err)
				}
				wy, wzf := refFIR(b, x, zi)
				for i := range wy {
					if math.Float64bits(y[i]) != math.Float64bits(wy[i]) {
						t.Fatalf("nb=%d n=%d zi=%v: y[%d] = %v, want %v", nb, n, withZi, i, y[i], wy[i])
					}
				}
				for i := range wzf {
					if math.Float64bits(zf[i]) != math.Float64bits(wzf[i]) {
						t.Fatalf("nb=%d n=%d zi=%v: zf[%d] = %v, want %v", nb, n, withZi, i, zf[i], wzf[i])
					}
				}
			}
		}
	}
}

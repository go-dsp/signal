package signal

import (
	"math"
	"math/big"
	"testing"
)

const kbdPrec = 320 // bits of the reference

// kaiserBig returns the Kaiser window of m points (symmetric) in extended
// precision. With alpha = (m-1)/2 and r = (i-alpha)/alpha, the argument of
// I0 is beta*sqrt(1-r^2), whose square over four is exactly
// beta^2 * i*(m-1-i) / (m-1)^2, so the power series
// I0(x) = sum (x^2/4)^k / (k!)^2 needs no square root and no rounding of the
// argument.
func kaiserBig(m int, beta float64) []*big.Float {
	i0 := func(t *big.Float) *big.Float { // sum t^k / (k!)^2
		sum := new(big.Float).SetPrec(kbdPrec).SetInt64(1)
		term := new(big.Float).SetPrec(kbdPrec).SetInt64(1)
		eps := new(big.Float).SetPrec(kbdPrec).SetMantExp(big.NewFloat(1), -kbdPrec-8)
		for k := 1; ; k++ {
			term.Mul(term, t)
			term.Quo(term, new(big.Float).SetPrec(kbdPrec).SetInt64(int64(k)*int64(k)))
			sum.Add(sum, term)
			if new(big.Float).Quo(term, sum).Cmp(eps) < 0 {
				return sum
			}
		}
	}
	b2 := new(big.Float).SetPrec(kbdPrec).SetFloat64(beta)
	b2.Mul(b2, b2)
	denom := new(big.Float).SetPrec(kbdPrec).SetInt64(int64(m-1) * int64(m-1))
	// I0(beta): x = beta, so the series variable is beta^2/4
	norm := i0(new(big.Float).SetPrec(kbdPrec).Quo(b2, new(big.Float).SetPrec(kbdPrec).SetInt64(4)))
	w := make([]*big.Float, m)
	for i := 0; i <= (m-1)/2; i++ {
		t := new(big.Float).SetPrec(kbdPrec).SetInt64(int64(i) * int64(m-1-i))
		t.Mul(t, b2)
		t.Quo(t, denom)
		v := i0(t)
		v.Quo(v, norm)
		w[i], w[m-1-i] = v, v
	}
	return w
}

// kbdBig is the Kaiser-Bessel-derived window of n points in extended precision.
func kbdBig(n int, beta float64) []*big.Float {
	k := kaiserBig(n/2+1, beta)
	total := new(big.Float).SetPrec(kbdPrec)
	for _, v := range k {
		total.Add(total, v)
	}
	w := make([]*big.Float, n)
	sum := new(big.Float).SetPrec(kbdPrec)
	for i := 0; i < n/2; i++ {
		sum.Add(sum, k[i])
		q := new(big.Float).SetPrec(kbdPrec).Quo(sum, total)
		w[i] = q.Sqrt(q)
		w[n-1-i] = w[i]
	}
	return w
}

func maxRelErr(t *testing.T, got []float64, want []*big.Float) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	var worst float64
	for i, g := range got {
		d := new(big.Float).SetPrec(kbdPrec).SetFloat64(g)
		d.Sub(d, want[i])
		d.Quo(d, want[i])
		e, _ := d.Float64()
		worst = math.Max(worst, math.Abs(e))
	}
	return worst
}

// TestKaiserBesselDerivedExtendedPrecision checks the window against the
// definition computed in 320-bit arithmetic, not against SciPy, whose summation is
// as naive as ours. The bounds are the measured relative errors with a margin: the
// error grows with n because the running sum of the Kaiser window adds n/2 rounded
// terms: measured 7e-17 at 2 points, 9e-16 at 2048 and 4e-15 at 65536.
func TestKaiserBesselDerivedExtendedPrecision(t *testing.T) {
	for _, c := range []struct {
		n     int
		beta  float64
		bound float64
	}{
		{2, 4, 2e-16}, {2048, 4, 3e-15}, {2048, 12, 3e-15}, {65536, 4, 1.5e-14}, {65536, 12, 1.5e-14},
	} {
		got := KaiserBesselDerived(c.beta)(c.n, true)
		e := maxRelErr(t, got, kbdBig(c.n, c.beta))
		t.Logf("n=%d beta=%g: largest relative error %.3g (bound %.3g)", c.n, c.beta, e, c.bound)
		if e > c.bound {
			t.Errorf("n=%d beta=%g: relative error %g exceeds %g", c.n, c.beta, e, c.bound)
		}
	}
}

// TestKaiserExtendedPrecision: the Kaiser window itself, in the same way.
func TestKaiserExtendedPrecision(t *testing.T) {
	for _, beta := range []float64{0, 4, 12} {
		got := Kaiser(beta)(2049, true)
		e := maxRelErr(t, got, kaiserBig(2049, beta))
		t.Logf("beta=%g: largest relative error %.3g", beta, e)
		if e > 2e-15 {
			t.Errorf("beta=%g: relative error %g exceeds 2e-15", beta, e)
		}
	}
}

// TestKaiserHugeBeta: above a beta of about 714, I0 overflows and the quotient of two
// infinities made the whole window NaN. The scaled form keeps it finite, in (0, 1],
// with the peak at one.
func TestKaiserHugeBeta(t *testing.T) {
	for _, beta := range []float64{700, 720, 1000, 1e5, -1000} {
		for _, w := range [][]float64{Kaiser(beta)(65, true), KaiserBesselDerived(math.Abs(beta))(64, true)} {
			for i, v := range w {
				if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
					t.Fatalf("beta %g: w[%d] = %v", beta, i, v)
				}
			}
		}
		if mid := Kaiser(beta)(65, true)[32]; mid != 1 {
			t.Errorf("beta %g: the peak is %v, want 1", beta, mid)
		}
	}
}

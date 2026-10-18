package signal

import (
	"math"
	"testing"
)

func protoCompare(t *testing.T, group string, got ZPK, o Out, tolRoots float64) {
	t.Helper()
	if d := matchRoots(got.Z, toC(o.Idx(0).Arr())); d > tolRoots {
		t.Errorf("%s zeros: %g", group, d)
	}
	if d := matchRoots(got.P, toC(o.Idx(1).Arr())); d > tolRoots {
		t.Errorf("%s poles: %g", group, d)
	}
	compareVec(t, group+" k", []float64{got.K}, []float64{o.Idx(2).X()}, tolRoots)
}

func TestButtapCheb1Cheb2Oracle(t *testing.T) {
	for n := 0; n <= 40; n++ {
		g, err := ButtAP(n)
		mustOK(t, err)
		protoCompare(t, "buttap", g, MustOracle(t, "signal.buttap", []any{n}, nil), 1e-14)
		for _, r := range []float64{0.1, 1, 3, 20} {
			g, err = Cheb1AP(n, r)
			mustOK(t, err)
			protoCompare(t, "cheb1ap", g, MustOracle(t, "signal.cheb1ap", []any{n, r}, nil), 1e-13)
			g, err = Cheb2AP(n, 10*r)
			mustOK(t, err)
			protoCompare(t, "cheb2ap", g, MustOracle(t, "signal.cheb2ap", []any{n, 10 * r}, nil), 1e-13)
		}
	}
	for _, f := range []func() error{
		func() error { _, e := ButtAP(-1); return e },
		func() error { _, e := Cheb1AP(-1, 1); return e },
		func() error { _, e := Cheb2AP(-1, 1); return e },
		func() error { _, e := EllipAP(-1, 1, 2); return e },
		func() error { _, e := BesselAP(-1, BesselPhase); return e },
	} {
		mustErr(t, "negative order", f())
	}
}

func TestEllipjOracle(t *testing.T) {
	rng := newRNG(20)
	var us, ms []float64
	for i := 0; i < 400; i++ {
		us = append(us, 6*rng.Float64()-3)
		switch i % 4 {
		case 0:
			ms = append(ms, rng.Float64())
		case 1:
			ms = append(ms, 1e-10*rng.Float64())
		case 2:
			ms = append(ms, 1-1e-11*rng.Float64())
		default:
			ms = append(ms, 0.99*rng.Float64())
		}
	}
	us = append(us, 0.3, 0.3, 0.3)
	ms = append(ms, 0, 1, 0.5)
	o := MustOracle(t, "special.ellipj", []any{F(us), F(ms)}, nil)
	var sn, cn, dn []float64
	for i := range us {
		s, c, d := Ellipj(us[i], ms[i])
		sn, cn, dn = append(sn, s), append(cn, c), append(dn, d)
	}
	compareVec(t, "ellipj sn", sn, o.Idx(0).F(), 1e-14)
	compareVec(t, "ellipj cn", cn, o.Idx(1).F(), 1e-14)
	compareVec(t, "ellipj dn", dn, o.Idx(2).F(), 1e-14)
	for _, c := range [][2]float64{{math.NaN(), 0.5}, {1, math.NaN()}, {1, -0.1}, {1, 1.1}} {
		s, _, _ := Ellipj(c[0], c[1])
		if !math.IsNaN(s) {
			t.Error("domain")
		}
	}
}

func TestEllipapOracle(t *testing.T) {
	// Above order ~16 the pole positions are so ill-conditioned that SciPy
	// itself misses the specified ripple by 1e-7 to 1e-2; compare below that.
	for n := 0; n <= 16; n++ {
		for _, c := range [][2]float64{{0.1, 40}, {1, 60}, {3, 20}, {0.01, 100}} {
			g, err := EllipAP(n, c[0], c[1])
			mustOK(t, err)
			protoCompare(t, "ellipap", g, MustOracle(t, "signal.ellipap", []any{n, c[0], c[1]}, nil), 1e-6)
		}
	}
	_, err := EllipAP(4, 0, 40)
	mustErr(t, "rp = 0", err)
	_, err = EllipAP(4, 3, 1) // rs below rp: modulus above one
	mustErr(t, "rs < rp", err)
	_, err = arcJacSC1(1, math.NaN())
	mustErr(t, "NaN parameter", err)
	v, err := arcJacSN(0.5, 1)
	mustOK(t, err)
	if math.Abs(real(v)-math.Atanh(0.5)) > 1e-15 {
		t.Error("m = 1")
	}
	v, _ = arcJacSN(0.5, 2)
	if !math.IsNaN(real(v)) {
		t.Error("m > 1")
	}
	_, err = arcJacSN(0.5, math.NaN())
	mustErr(t, "no convergence", err)
}

func TestBesselapOracle(t *testing.T) {
	for n := 0; n <= 25; n++ {
		for _, nm := range []struct {
			n BesselNorm
			s string
		}{{BesselPhase, "phase"}, {BesselDelay, "delay"}, {BesselMag, "mag"}} {
			g, err := BesselAP(n, nm.n)
			mustOK(t, err)
			o := MustOracle(t, "signal.besselap", []any{n}, KW{"norm": nm.s})
			tol := 1e-12
			if nm.s == "mag" {
				// SciPy locates the -3 dB point by a secant search that stops at 1.5e-8
				tol = 1e-6
			}
			protoCompare(t, "besselap "+nm.s, g, o, tol)
		}
	}
	_, err := BesselAP(3, BesselNorm(9))
	mustErr(t, "norm", err)
	oldA, oldN := aberthMaxIter, besselNewtonSteps
	aberthMaxIter, besselNewtonSteps = 0, 0
	_, err = BesselAP(4, BesselPhase)
	aberthMaxIter, besselNewtonSteps = oldA, oldN
	mustErr(t, "no convergence", err)
	if l := bigLog10(new(bigInt).Lsh(bigOne(), 3000)); math.Abs(l-3000*math.Log10(2)) > 1e-9 {
		t.Error("bigLog10", l)
	}
	if got := secant(func(x float64) float64 { return 1 }, 1, 1e-8); math.Abs(got-1.0001) > 1e-12 {
		t.Error("secant on a flat function", got)
	}
	// iteration limit: a function that never converges
	n := 0
	secant(func(x float64) float64 { n++; return float64(n*n) * x }, 1, 1e-300)
}

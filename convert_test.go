package signal

import (
	"math"
	"testing"
)

func TestRootsOracle(t *testing.T) {
	rng := newRNG(1)
	for n := 1; n <= 40; n++ {
		for rep := 0; rep < 3; rep++ {
			p := randVec(rng, n+1)
			p[0] += 2 // keep it off zero
			got, err := Roots(p)
			mustOK(t, err)
			want := toC(MustOracle(t, "np.roots", []any{F(p)}, nil).Arr())
			if d := matchRoots(got, want); d > 1e-6 {
				t.Errorf("n=%d: root mismatch %g", n, d)
			}
		}
	}
	// x^n - 1 stalls the plain double shift and needs the exceptional ones
	for _, n := range []int{6, 9, 16} {
		p := make([]float64, n+1)
		p[0], p[n] = 1, -1
		got, _ := Roots(p)
		want := toC(MustOracle(t, "np.roots", []any{F(p)}, nil).Arr())
		if d := matchRoots(got, want); d > 1e-12 {
			t.Errorf("unit roots n=%d: %g", n, d)
		}
	}
	// trailing and leading zeros
	z := []float64{0, 1, -3, 2, 0, 0}
	got, _ := Roots(z)
	want := toC(MustOracle(t, "np.roots", []any{F(z)}, nil).Arr())
	if d := matchRoots(got, want); d > 1e-12 {
		t.Errorf("zeros: %g", d)
	}
	if r, _ := Roots([]float64{5}); len(r) != 0 {
		t.Error("constant has no roots")
	}
	if r, _ := Roots(nil); len(r) != 0 {
		t.Error("empty has no roots")
	}
	old := hqrLimit
	hqrLimit = 0
	_, err := Roots([]float64{1, 2, 3, 4, 5})
	hqrLimit = old
	mustErr(t, "no convergence", err)
}

func TestPolyHelpers(t *testing.T) {
	if _, err := polyFromRoots([]complex128{1 + 1i}); err == nil {
		t.Error("lone complex root")
	}
	if _, err := polyFromRoots([]complex128{1 + 1i, 1 - 2i}); err == nil {
		t.Error("unmatched pair")
	}
	if _, err := polyFromRoots([]complex128{1 + 1i, 2 + 1i, 1 - 1i, 2 - 1i}); err != nil {
		t.Error(err)
	}
	if _, err := polyFromRoots([]complex128{1 + 1i, 1 - 1i, 1 - 1i, 1 + 2i}); err == nil {
		t.Error("count mismatch")
	}
	if polyMul(nil, []float64{1}) != nil {
		t.Error("polyMul empty")
	}
	if g := polyAdd([]float64{1, 2}, []float64{1, 2, 3, 4}); !eqSlice(g, []float64{1, 2, 4, 6}) {
		t.Error(g)
	}
	if polyval([]float64{1, 0, 1}, 1i) != 0 || polyvalReal([]float64{1, 0, -4}, 2) != 0 {
		t.Error("polyval")
	}
	_ = math.Pi
}

func TestNormalizeOracle(t *testing.T) {
	cases := [][2][]float64{
		{{1, 2, 3}, {2, 4, 6}},
		{{0, 0, 1, 2}, {0, 3, 1}},
		{{1e-15, 1e-16, 5}, {1, 2}},
		{{0, 0}, {1, 1}},
		{{4}, {2}},
	}
	for _, c := range cases {
		b, a, err := Normalize(c[0], c[1])
		mustOK(t, err)
		o := MustOracle(t, "signal.normalize", []any{F(c[0]), F(c[1])}, nil)
		compareVec(t, "convert/normalize", b, o.Idx(0).F(), 0)
		compareVec(t, "convert/normalize", a, o.Idx(1).F(), 0)
	}
	_, _, err := Normalize([]float64{1}, []float64{0, 0})
	mustErr(t, "zero denominator", err)
	_, _, err = Normalize(nil, []float64{1})
	mustErr(t, "empty numerator", err)
	_, err = TF2ZPK([]float64{1}, nil)
	mustErr(t, "tf2zpk denominator", err)
	_, err = TF2ZPK([]float64{0, 0}, []float64{1, 2})
	mustErr(t, "zero numerator", err)
}

func TestTF2ZPKOracle(t *testing.T) {
	rng := newRNG(2)
	for n := 1; n <= 12; n++ {
		b, a := randVec(rng, n+1), randVec(rng, n+1)
		a[0] += 2
		got, err := TF2ZPK(b, a)
		mustOK(t, err)
		o := MustOracle(t, "signal.tf2zpk", []any{F(b), F(a)}, nil)
		if d := matchRoots(got.Z, toC(o.Idx(0).Arr())); d > 1e-6 {
			t.Errorf("zeros n=%d: %g", n, d)
		}
		if d := matchRoots(got.P, toC(o.Idx(1).Arr())); d > 1e-6 {
			t.Errorf("poles n=%d: %g", n, d)
		}
		if got.K != o.Idx(2).X() {
			t.Errorf("k %v vs %v", got.K, o.Idx(2).X())
		}
	}
}

func TestZPK2TFOracle(t *testing.T) {
	rng := newRNG(3)
	for n := 0; n <= 30; n++ {
		z := ZPK{randRoots(rng, n, rng.IntN(3), 0.1, 1.2), randRoots(rng, n+rng.IntN(2), rng.IntN(3), 0.1, 0.99), 0.5 + rng.Float64()}
		b, a, err := ZPK2TF(z)
		mustOK(t, err)
		o := MustOracle(t, "signal.zpk2tf", []any{C(z.Z), C(z.P), z.K}, nil)
		compareVec(t, "convert/zpk2tf num", b, o.Idx(0).F(), 1e-13)
		compareVec(t, "convert/zpk2tf den", a, o.Idx(1).F(), 1e-13)
	}
	_, _, err := ZPK2TF(ZPK{Z: []complex128{1i}})
	mustErr(t, "unpaired zero", err)
	_, _, err = ZPK2TF(ZPK{P: []complex128{1i}})
	mustErr(t, "unpaired pole", err)
}

func TestCplxOracle(t *testing.T) {
	rng := newRNG(4)
	inputs := [][]complex128{
		{4, 3, 1, 2 - 2i, 2 + 2i, 2 - 1i, 2 + 1i, 2 - 1i, 2 + 1i, 1 + 1i, 1 - 1i},
		{1, 2, 3},
		{1i, -1i},
		{0.5 + 1e-17i, 0.2},
	}
	for n := 1; n < 25; n += 3 {
		inputs = append(inputs, randRoots(rng, n, rng.IntN(4), 0.2, 1))
	}
	for _, in := range inputs {
		zc, zr, err := cplxReal(in, cplxTol)
		mustOK(t, err)
		o := MustOracle(t, "signal._filter_design._cplxreal", []any{C(in)}, nil)
		compareC(t, "convert/cplxreal complex", zc, toC(o.Idx(0).Arr()), 0)
		compareVec(t, "convert/cplxreal real", zr, o.Idx(1).F(), 0)
		pr, err := cplxPair(in, cplxTol)
		mustOK(t, err)
		o = MustOracle(t, "signal._filter_design._cplxpair", []any{C(in)}, nil)
		compareC(t, "convert/cplxpair", pr, o.C(), 0)
	}
	zc, zr, err := cplxReal(nil, cplxTol)
	if err != nil || zc != nil || zr != nil {
		t.Error("empty")
	}
	_, _, err = cplxReal([]complex128{1i, 2 + 1i}, cplxTol)
	mustErr(t, "count", err)
	_, _, err = cplxReal([]complex128{1 + 1i, 1 - 2i}, cplxTol)
	mustErr(t, "mismatch", err)
	_, err = cplxPair([]complex128{1i}, cplxTol)
	mustErr(t, "cplxPair", err)
}

func TestZPK2SOSOracle(t *testing.T) {
	rng := newRNG(5)
	pairings := []struct {
		p    Pairing
		name string
	}{{PairNearest, "nearest"}, {PairKeepOdd, "keep_odd"}, {PairMinimal, "minimal"}}
	for _, pr := range pairings {
		for _, analog := range []bool{false, true} {
			if analog && pr.p != PairMinimal {
				continue
			}
			for n := 1; n <= 14; n++ {
				for rep := 0; rep < 6; rep++ {
					np := n
					nz := n - rng.IntN(3)
					if nz < 0 {
						nz = 0
					}
					if !analog && pr.p == PairMinimal && rep%2 == 0 {
						nz = np // digital minimal is legal in SciPy
					}
					rmax := 0.99
					if analog {
						rmax = 5
					}
					zp := ZPK{randRoots(rng, nz, rng.IntN(4), 0.05, 1.5), randRoots(rng, np, rng.IntN(4), 0.05, rmax), 0.3 + rng.Float64()}
					if analog {
						// stable analog poles
						for i, v := range zp.P {
							zp.P[i] = complex(-math.Abs(real(v))-0.01, imag(v))
						}
					}
					o, oerr := Oracle(t, "signal.zpk2sos", []any{C(zp.Z), C(zp.P), zp.K}, KW{"pairing": pr.name, "analog": analog})
					got, err := ZPK2SOS(zp, pr.p, analog)
					if oerr != nil || err != nil {
						if (oerr == nil) != (err == nil) {
							t.Errorf("%s analog=%v n=%d: error disagreement: go=%v scipy=%v", pr.name, analog, n, err, oerr)
						}
						continue
					}
					compareVec(t, "convert/zpk2sos", sosFlat(got), o.F(), 1e-13)
				}
			}
		}
	}
	for _, analog := range []bool{false, true} {
		got, err := ZPK2SOS(ZPK{K: 2}, PairMinimal, analog)
		mustOK(t, err)
		o := MustOracle(t, "signal.zpk2sos", []any{C(nil), C(nil), 2.0}, KW{"pairing": "minimal", "analog": analog})
		compareVec(t, "convert/zpk2sos", sosFlat(got), o.F(), 0)
	}
	_, err := ZPK2SOS(ZPK{K: 1, P: []complex128{0.5}}, PairNearest, true)
	mustErr(t, "analog pairing", err)
	_, err = ZPK2SOS(ZPK{K: 1, P: []complex128{0.5}}, Pairing(7), false)
	mustErr(t, "bad pairing", err)
	_, err = ZPK2SOS(ZPK{K: 1, Z: []complex128{0.1, 0.2}, P: []complex128{0.5}}, PairMinimal, true)
	mustErr(t, "more zeros than poles", err)
	_, err = ZPK2SOS(ZPK{K: 1, P: []complex128{0.5i}}, PairNearest, false)
	mustErr(t, "unpaired pole", err)
	_, err = ZPK2SOS(ZPK{K: 1, Z: []complex128{0.5i}, P: []complex128{0.5}}, PairNearest, false)
	mustErr(t, "unpaired zero", err)
	// a lone complex zero cannot be paired with real poles
	_, err = ZPK2SOS(ZPK{K: 1, Z: []complex128{0.5 + 0.5i, 0.5 - 0.5i, 0.3}, P: []complex128{0.5, 0.2, 0.1}}, PairNearest, false)
	if err != nil {
		t.Log("pairing result:", err)
	}
}

func TestTF2SOSAndBackOracle(t *testing.T) {
	rng := newRNG(6)
	for n := 1; n <= 10; n++ {
		z := ZPK{randRoots(rng, n, 1, 0.1, 0.9), randRoots(rng, n, 1, 0.1, 0.9), 1.3}
		b, a, err := ZPK2TF(z)
		mustOK(t, err)
		sos, err := TF2SOS(b, a, PairNearest, false)
		mustOK(t, err)
		o := MustOracle(t, "signal.tf2sos", []any{F(b), F(a)}, nil)
		compareVec(t, "convert/tf2sos", sosFlat(sos), o.F(), 1e-9)
		bb, aa := SOS2TF(sos)
		ob := MustOracle(t, "signal.sos2tf", []any{F2(sosRows(sos))}, nil)
		compareVec(t, "convert/sos2tf num", bb, ob.Idx(0).F(), 1e-14)
		compareVec(t, "convert/sos2tf den", aa, ob.Idx(1).F(), 1e-14)
		zp, err := SOS2ZPK(sos)
		mustOK(t, err)
		oz := MustOracle(t, "signal.sos2zpk", []any{F2(sosRows(sos))}, nil)
		if d := matchRoots(zp.Z, toC(oz.Idx(0).Arr())); d > 1e-7 {
			t.Errorf("sos2zpk zeros %g", d)
		}
		if d := matchRoots(zp.P, toC(oz.Idx(1).Arr())); d > 1e-7 {
			t.Errorf("sos2zpk poles %g", d)
		}
		if math.Abs(zp.K-oz.Idx(2).X()) > 1e-13*math.Abs(zp.K) {
			t.Errorf("sos2zpk k %g vs %g", zp.K, oz.Idx(2).X())
		}
	}
	_, err := TF2SOS([]float64{1}, nil, PairNearest, false)
	mustErr(t, "tf2sos", err)
	_, err = SOS2ZPK(SOS{{0, 0, 0, 1, 0, 0}})
	mustErr(t, "sos2zpk zero section", err)
}

func sosRows(s SOS) [][]float64 {
	r := make([][]float64, len(s))
	for i := range s {
		r[i] = s[i][:]
	}
	return r
}

func TestTF2ZPKRootFailure(t *testing.T) {
	old := hqrLimit
	defer func() { hqrLimit = old }()
	hqrLimit = 0
	_, err := TF2ZPK([]float64{1, 2, 3, 4, 5}, []float64{1})
	mustErr(t, "zeros", err)
	_, err = TF2ZPK([]float64{1}, []float64{1, 2, 3, 4, 5})
	mustErr(t, "poles", err)
	if _, err := polyFromRoots([]complex128{1 + 2i, 1 + 1i, 1 - 1i, 1 - 2i}); err != nil {
		t.Error(err)
	}
}

package signal

import (
	"math"
	"testing"
)

func butterBA(t *testing.T, n int, wn float64) (b, a []float64) {
	t.Helper()
	z, err := Butter(n, []float64{wn}, Lowpass, nil)
	mustOK(t, err)
	b, a, err = z.TF()
	mustOK(t, err)
	return b, a
}

func butterSOS(t *testing.T, n int, wn float64) SOS {
	t.Helper()
	z, err := Butter(n, []float64{wn}, Lowpass, nil)
	mustOK(t, err)
	s, err := z.SOS(false)
	mustOK(t, err)
	return s
}

func TestLFilterOracle(t *testing.T) {
	rng := newRNG(60)
	for _, n := range []int{1, 2, 3, 5, 8} {
		b, a := butterBA(t, n, 0.3)
		for _, ln := range []int{1, 7, 100, 1000} {
			x := randVec(rng, ln)
			y, zf, err := LFilter(b, a, x, nil)
			mustOK(t, err)
			o := MustOracle(t, "signal.lfilter", []any{F(b), F(a), F(x)}, nil)
			compareVec(t, "lfilter", y, o.F(), 1e-12)
			zi := randVec(rng, n)
			y, zf, err = LFilter(b, a, x, zi)
			mustOK(t, err)
			o = MustOracle(t, "signal.lfilter", []any{F(b), F(a), F(x)}, KW{"zi": F(zi)})
			compareVec(t, "lfilter zi", y, o.Idx(0).F(), 1e-12)
			compareVec(t, "lfilter zf", zf, o.Idx(1).F(), 1e-12)
		}
	}
	// unnormalised a, FIR with and without state, b longer than a
	x := randVec(rng, 50)
	for _, c := range [][2][]float64{{{1, 2, 3}, {2, 0.5, 0.1}}, {{1, 0.5, 0.25, 0.125}, {1}}, {{3}, {2}}, {{1, 2, 3, 4, 5}, {1, 0.2}}} {
		y, _, err := LFilter(c[0], c[1], x, nil)
		mustOK(t, err)
		o := MustOracle(t, "signal.lfilter", []any{F(c[0]), F(c[1]), F(x)}, nil)
		compareVec(t, "lfilter coefficients", y, o.F(), 1e-13)
		zi := randVec(rng, max(len(c[0]), len(c[1]))-1)
		if len(zi) > 0 {
			y, zf, err := LFilter(c[0], c[1], x, zi)
			mustOK(t, err)
			o := MustOracle(t, "signal.lfilter", []any{F(c[0]), F(c[1]), F(x)}, KW{"zi": F(zi)})
			compareVec(t, "lfilter zi", y, o.Idx(0).F(), 1e-13)
			compareVec(t, "lfilter zf", zf, o.Idx(1).F(), 1e-13)
		}
	}
	_, _, err := LFilter(nil, []float64{1}, x, nil)
	mustErr(t, "empty b", err)
	_, _, err = LFilter([]float64{1}, []float64{0, 1}, x, nil)
	mustErr(t, "a0", err)
	_, _, err = LFilter([]float64{1, 1}, []float64{1}, x, []float64{1, 2})
	mustErr(t, "zi length", err)
}

func TestLFilterZIAndICOracle(t *testing.T) {
	rng := newRNG(61)
	for _, n := range []int{1, 2, 4, 6} {
		b, a := butterBA(t, n, 0.2)
		zi, err := LFilterZI(b, a)
		mustOK(t, err)
		o := MustOracle(t, "signal.lfilter_zi", []any{F(b), F(a)}, nil)
		compareVec(t, "lfilter_zi", zi, o.F(), 1e-10)
		y, x := randVec(rng, n), randVec(rng, n)
		ic, err := LFilterIC(b, a, y, x)
		mustOK(t, err)
		o = MustOracle(t, "signal.lfiltic", []any{F(b), F(a), F(y)}, KW{"x": F(x)})
		compareVec(t, "lfiltic", ic, o.F(), 1e-12)
		ic, err = LFilterIC(b, a, y[:1], nil)
		mustOK(t, err)
		o = MustOracle(t, "signal.lfiltic", []any{F(b), F(a), F(y[:1])}, nil)
		compareVec(t, "lfiltic no x", ic, o.F(), 1e-12)
	}
	// unnormalised and unequal lengths
	b, a := []float64{2, 1}, []float64{2, -1, 0.5}
	zi, err := LFilterZI(b, a)
	mustOK(t, err)
	o := MustOracle(t, "signal.lfilter_zi", []any{F(b), F(a)}, nil)
	compareVec(t, "lfilter_zi", zi, o.F(), 1e-14)
	ic, err := LFilterIC(b, a, []float64{1, 2}, []float64{3})
	mustOK(t, err)
	o = MustOracle(t, "signal.lfiltic", []any{F(b), F(a), F([]float64{1, 2})}, KW{"x": F([]float64{3})})
	compareVec(t, "lfiltic", ic, o.F(), 1e-14)
	_, err = LFilterZI(nil, a)
	mustErr(t, "empty", err)
	_, err = LFilterZI(b, []float64{0, 1})
	mustErr(t, "a0", err)
	_, err = LFilterZI(b, []float64{1, -1})
	mustErr(t, "pole at 1", err)
	_, err = LFilterIC(b, nil, nil, nil)
	mustErr(t, "ic a empty", err)
	_, err = LFilterIC(b, []float64{0, 1}, nil, nil)
	mustErr(t, "ic a0", err)
}

func TestFiltFiltOracle(t *testing.T) {
	rng := newRNG(62)
	padNames := map[PadType]any{PadOdd: "odd", PadEven: "even", PadConstant: "constant", PadNone: nil}
	for _, n := range []int{2, 4, 6} {
		b, a := butterBA(t, n, 0.25)
		x := randVec(rng, 200)
		for _, pt := range sortedKeys(padNames) {
			name := padNames[pt]
			for _, pl := range []int{0, 5, 50} {
				got, err := FiltFilt(b, a, x, &FiltFiltOptions{PadType: pt, PadLen: pl})
				mustOK(t, err)
				kw := KW{"padtype": name}
				if pl > 0 && pt != PadNone {
					kw["padlen"] = pl
				}
				o := MustOracle(t, "signal.filtfilt", []any{F(b), F(a), F(x)}, kw)
				compareVec(t, "filtfilt", got, o.F(), 1e-10)
			}
		}
		for _, ir := range []int{0, 30} {
			got, err := FiltFilt(b, a, x, &FiltFiltOptions{Method: FiltGust, IRLen: ir})
			mustOK(t, err)
			kw := KW{"method": "gust"}
			if ir > 0 {
				kw["irlen"] = ir
			}
			o := MustOracle(t, "signal.filtfilt", []any{F(b), F(a), F(x)}, kw)
			compareVec(t, "filtfilt gust", got, o.F(), 1e-8)
		}
	}
	// a pure gain, and a short FIR
	got, err := FiltFilt([]float64{2}, []float64{4}, []float64{1, 2, 3}, &FiltFiltOptions{Method: FiltGust})
	mustOK(t, err)
	o := MustOracle(t, "signal.filtfilt", []any{F([]float64{2}), F([]float64{4}), F([]float64{1, 2, 3})}, KW{"method": "gust"})
	compareVec(t, "filtfilt gust gain", got, o.F(), 1e-15)
	x := randVec(rng, 40)
	got, err = FiltFilt([]float64{0.25, 0.5, 0.25}, []float64{1}, x, nil)
	mustOK(t, err)
	o = MustOracle(t, "signal.filtfilt", []any{F([]float64{0.25, 0.5, 0.25}), F([]float64{1}), F(x)}, nil)
	compareVec(t, "filtfilt fir", got, o.F(), 1e-13)

	b, a := butterBA(t, 2, 0.2)
	for i, f := range []func() error{
		func() error { _, e := FiltFilt(nil, a, x, nil); return e },
		func() error { _, e := FiltFilt(b, a, x, &FiltFiltOptions{Method: FiltMethod(5)}); return e },
		func() error { _, e := FiltFilt(b, a, x, &FiltFiltOptions{PadType: PadType(8)}); return e },
		func() error { _, e := FiltFilt(b, a, x, &FiltFiltOptions{PadLen: -2}); return e },
		func() error { _, e := FiltFilt(b, a, x[:5], nil); return e },
		func() error { _, e := FiltFilt(b, []float64{1, -1, 0}, x, nil); return e },
		func() error { _, e := FiltFilt(b, a, nil, &FiltFiltOptions{Method: FiltGust}); return e },
		func() error { _, e := FiltFilt(b, []float64{0, 1}, x, &FiltFiltOptions{Method: FiltGust}); return e },
	} {
		mustErr(t, "filtfilt error "+string(rune('a'+i)), f())
	}
}

func TestLstsq(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 1}, {1, 1}}
	x, err := lstsq(a, []float64{1, 2, 3.1})
	mustOK(t, err)
	if math.Abs(x[0]-1.0333333333333334) > 1e-12 || math.Abs(x[1]-2.0333333333333334) > 1e-12 {
		t.Errorf("lstsq = %v", x)
	}
	_, err = lstsq([][]float64{{0, 1}, {0, 2}}, []float64{1, 2})
	mustErr(t, "rank deficient", err)
	_, err = lstsq([][]float64{{1, 1}, {1, 1}, {1, 1}}, []float64{1, 2, 3})
	mustErr(t, "rank deficient 2", err)
	if _, err := lstsq([][]float64{{1}, {0}}, []float64{3, 1}); err != nil {
		t.Error(err)
	}
}

func TestSOSFilterOracle(t *testing.T) {
	rng := newRNG(63)
	for _, n := range []int{1, 2, 5, 9, 16} {
		sos := butterSOS(t, n, 0.2)
		x := randVec(rng, 500)
		y, _, err := SOSFilt(sos, x, nil)
		mustOK(t, err)
		o := MustOracle(t, "signal.sosfilt", []any{F2(sosRows(sos)), F(x)}, nil)
		compareVec(t, "sosfilt", y, o.F(), 1e-12)
		zi := make([][2]float64, len(sos))
		zrows := make([][]float64, len(sos))
		for i := range zi {
			zi[i] = [2]float64{rng.Float64(), rng.Float64()}
			zrows[i] = zi[i][:]
		}
		y, zf, err := SOSFilt(sos, x, zi)
		mustOK(t, err)
		o = MustOracle(t, "signal.sosfilt", []any{F2(sosRows(sos)), F(x)}, KW{"zi": F2(zrows)})
		compareVec(t, "sosfilt zi", y, o.Idx(0).F(), 1e-12)
		var zff []float64
		for _, z := range zf {
			zff = append(zff, z[0], z[1])
		}
		compareVec(t, "sosfilt zf", zff, o.Idx(1).F(), 1e-12)

		z0, err := SOSFiltZI(sos)
		mustOK(t, err)
		var zz []float64
		for _, z := range z0 {
			zz = append(zz, z[0], z[1])
		}
		o = MustOracle(t, "signal.sosfilt_zi", []any{F2(sosRows(sos))}, nil)
		compareVec(t, "sosfilt_zi", zz, o.F(), 1e-10)

		for _, pt := range []PadType{PadOdd, PadEven, PadConstant, PadNone} {
			got, err := SOSFiltFilt(sos, x, &FiltFiltOptions{PadType: pt})
			mustOK(t, err)
			kw := KW{"padtype": map[PadType]any{PadOdd: "odd", PadEven: "even", PadConstant: "constant", PadNone: nil}[pt]}
			o = MustOracle(t, "signal.sosfiltfilt", []any{F2(sosRows(sos)), F(x)}, kw)
			compareVec(t, "sosfiltfilt", got, o.F(), 1e-10)
		}
	}
	sos := butterSOS(t, 2, 0.2)
	x := randVec(rng, 50)
	got, err := SOSFiltFilt(sos, x, &FiltFiltOptions{PadLen: 10})
	mustOK(t, err)
	o := MustOracle(t, "signal.sosfiltfilt", []any{F2(sosRows(sos)), F(x)}, KW{"padlen": 10})
	compareVec(t, "sosfiltfilt padlen", got, o.F(), 1e-10)
	// a section with a zero b2 and a2 shortens the default padding
	sh := SOS{{1, 1, 0, 1, -0.5, 0}}
	got, err = SOSFiltFilt(sh, x, nil)
	mustOK(t, err)
	o = MustOracle(t, "signal.sosfiltfilt", []any{F2(sosRows(sh)), F(x)}, nil)
	compareVec(t, "sosfiltfilt short", got, o.F(), 1e-12)

	bad := SOS{{1, 0, 0, 2, 0, 0}}
	for i, f := range []func() error{
		func() error { _, _, e := SOSFilt(nil, x, nil); return e },
		func() error { _, _, e := SOSFilt(bad, x, nil); return e },
		func() error { _, _, e := SOSFilt(sos, x, make([][2]float64, 5)); return e },
		func() error { _, e := SOSFiltZI(nil); return e },
		func() error { _, e := SOSFiltZI(SOS{{1, 0, 0, 1, -1, 0}}); return e },
		func() error { _, e := SOSFiltFilt(nil, x, nil); return e },
		func() error { _, e := SOSFiltFilt(sos, x[:3], nil); return e },
		func() error { _, e := SOSFiltFilt(SOS{{1, 0, 0, 1, -1, 0}}, x, nil); return e },
	} {
		mustErr(t, "sos error "+string(rune('a'+i)), f())
	}
}

package signal

import (
	"math"
	"testing"
)

func TestFreqzOracle(t *testing.T) {
	rng := newRNG(10)
	type cs struct {
		b, a []float64
		o    FreqOptions
		kw   KW
		arg  any
	}
	var cases []cs
	for _, n := range []int{1, 7, 8, 64, 100} {
		for _, whole := range []bool{false, true} {
			for _, nyq := range []bool{false, true} {
				b := randVec(rng, 1+rng.IntN(9))
				a := randVec(rng, 1+rng.IntN(4))
				a[0] += 3
				for _, fir := range []bool{true, false} {
					aa := a
					if fir {
						aa = []float64{2}
					}
					kw := KW{"worN": n, "whole": whole, "include_nyquist": nyq}
					cases = append(cases, cs{b, aa, FreqOptions{N: n, Whole: whole, IncludeNyquist: nyq}, kw, nil})
				}
			}
		}
	}
	// explicit frequencies and a sampling rate
	w := []float64{0, 100, 250.5, 1000}
	b, a := randVec(rng, 5), []float64{1, 0.2, 0.1}
	cases = append(cases, cs{b, a, FreqOptions{W: w, Fs: 2000}, KW{"worN": F(w), "fs": 2000.0}, nil})
	cases = append(cases, cs{b, a, FreqOptions{N: 33, Fs: 1000}, KW{"worN": 33, "fs": 1000.0}, nil})
	cases = append(cases, cs{[]float64{1, 2}, []float64{1}, FreqOptions{N: 3}, KW{"worN": 3}, nil}) // nfft < len(b)? no: 6>=2
	cases = append(cases, cs{[]float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1}, FreqOptions{N: 3}, KW{"worN": 3}, nil})
	cases = append(cases, cs{[]float64{1, 2}, []float64{1}, FreqOptions{N: 0}, KW{"worN": 512}, nil}) // default N
	for i, c := range cases {
		oc := c.o
		gw, gh, err := Freqz(c.b, c.a, &oc)
		mustOK(t, err)
		o := MustOracle(t, "signal.freqz", []any{F(c.b), F(c.a)}, c.kw)
		compareVec(t, "freqz w", gw, o.Idx(0).F(), 1e-15)
		compareC(t, "freqz h", gh, o.Idx(1).C(), 1e-13)
		if len(gh) != len(gw) {
			t.Errorf("case %d: lengths %d %d", i, len(gh), len(gw))
		}
	}
	gw, gh, err := Freqz([]float64{1, 2}, []float64{1}, nil)
	mustOK(t, err)
	if len(gw) != 512 || len(gh) != 512 {
		t.Error("default points")
	}
	_, _, err = Freqz(nil, []float64{1}, nil)
	mustErr(t, "empty b", err)
	_, _, err = Freqz([]float64{1}, nil, nil)
	mustErr(t, "empty a", err)
	_, _, err = Freqz([]float64{1}, []float64{1}, &FreqOptions{N: -1})
	mustErr(t, "negative N", err)
	_, _, err = Freqz([]float64{1}, []float64{1}, &FreqOptions{Fs: -1})
	mustErr(t, "negative Fs", err)
	_, _, err = Freqz([]float64{1}, []float64{1}, &FreqOptions{Fs: math.Inf(1)})
	mustErr(t, "infinite Fs", err)
	if w, h, _ := Freqz([]float64{1}, []float64{1}, &FreqOptions{W: []float64{}}); len(w) != 0 || len(h) != 0 {
		t.Error("empty W")
	}
}

func TestFreqzZPKAndSOSOracle(t *testing.T) {
	rng := newRNG(11)
	for n := 1; n <= 9; n += 2 {
		s := ZPK{randRoots(rng, n, 1, 0.2, 1.1), randRoots(rng, n, 1, 0.2, 0.95), 0.8}
		for _, whole := range []bool{false, true} {
			gw, gh, err := FreqzZPK(s, &FreqOptions{N: 40, Whole: whole})
			mustOK(t, err)
			o := MustOracle(t, "signal.freqz_zpk", []any{C(s.Z), C(s.P), s.K}, KW{"worN": 40, "whole": whole})
			compareVec(t, "freqz_zpk w", gw, o.Idx(0).F(), 1e-15)
			compareC(t, "freqz_zpk h", gh, o.Idx(1).C(), 1e-13)
		}
		w := []float64{0.1, 0.5, 2.5}
		gw, gh, err := FreqzZPK(s, &FreqOptions{W: w, Fs: 10})
		mustOK(t, err)
		o := MustOracle(t, "signal.freqz_zpk", []any{C(s.Z), C(s.P), s.K}, KW{"worN": F(w), "fs": 10.0})
		compareVec(t, "freqz_zpk w", gw, o.Idx(0).F(), 1e-15)
		compareC(t, "freqz_zpk h", gh, o.Idx(1).C(), 1e-13)

		sos, err := ZPK2SOS(s, PairNearest, false)
		mustOK(t, err)
		gw, gh, err = SOSFreqz(sos, &FreqOptions{N: 50})
		mustOK(t, err)
		o = MustOracle(t, "signal.sosfreqz", []any{F2(sosRows(sos))}, KW{"worN": 50})
		compareVec(t, "sosfreqz w", gw, o.Idx(0).F(), 1e-15)
		compareC(t, "sosfreqz h", gh, o.Idx(1).C(), 1e-12)
	}
	_, _, err := FreqzZPK(ZPK{}, &FreqOptions{Fs: -2})
	mustErr(t, "zpk Fs", err)
	_, _, err = SOSFreqz(nil, nil)
	mustErr(t, "no sections", err)
	_, _, err = SOSFreqz(SOS{{1, 0, 0, 1, 0, 0}}, &FreqOptions{N: -1})
	mustErr(t, "sos N", err)
}

func TestGroupDelayOracle(t *testing.T) {
	rng := newRNG(12)
	for n := 1; n <= 8; n++ {
		b, a := randVec(rng, n+1), randVec(rng, n+1)
		a[0] += 3
		for _, whole := range []bool{false, true} {
			gw, gd, err := GroupDelay(b, a, &FreqOptions{N: 64, Whole: whole})
			mustOK(t, err)
			o := MustOracle(t, "signal.group_delay", []any{Tup(F(b), F(a))}, KW{"w": 64, "whole": whole})
			compareVec(t, "group_delay w", gw, o.Idx(0).F(), 1e-15)
			compareVec(t, "group_delay gd", gd, o.Idx(1).F(), 1e-12)
		}
	}
	// a singular point: zero of A on the unit circle at w = pi/2
	b, a := []float64{1, 0.5}, []float64{1, 0, 1}
	w := []float64{0.3, math.Pi / 2, 1}
	gw, gd, err := GroupDelay(b, a, &FreqOptions{W: w, Fs: 2 * math.Pi})
	mustOK(t, err)
	o := MustOracle(t, "signal.group_delay", []any{Tup(F(b), F(a))}, KW{"w": F(w)})
	compareVec(t, "group_delay w", gw, o.Idx(0).F(), 1e-15)
	compareVec(t, "group_delay singular", gd, o.Idx(1).F(), 1e-9)
	_, _, err = GroupDelay(nil, a, nil)
	mustErr(t, "empty", err)
	_, _, err = GroupDelay(b, a, &FreqOptions{N: -3})
	mustErr(t, "N", err)
}

func TestFreqsOracle(t *testing.T) {
	rng := newRNG(13)
	for n := 1; n <= 7; n++ {
		b, a := randVec(rng, n), randVec(rng, n+1)
		a[0] += 2
		for i := range a {
			a[i] = math.Abs(a[i]) + 0.1 // positive coefficients: stable
		}
		gw, gh, err := Freqs(b, a, nil)
		mustOK(t, err)
		o := MustOracle(t, "signal.freqs", []any{F(b), F(a)}, nil)
		compareVec(t, "freqs w", gw, o.Idx(0).F(), 1e-9)
		compareC(t, "freqs h", gh, o.Idx(1).C(), 1e-8)
		w := []float64{0.1, 1, 10}
		gw, gh, err = Freqs(b, a, &FreqOptions{W: w})
		mustOK(t, err)
		o = MustOracle(t, "signal.freqs", []any{F(b), F(a)}, KW{"worN": F(w)})
		compareVec(t, "freqs w", gw, o.Idx(0).F(), 0)
		compareC(t, "freqs h", gh, o.Idx(1).C(), 1e-13)

		s := ZPK{randRoots(rng, n, 1, 0.5, 30), randRoots(rng, n+1, 1, 0.5, 30), 2}
		for i, v := range s.P {
			s.P[i] = complex(-math.Abs(real(v))-0.1, imag(v))
		}
		gw, gh, err = FreqsZPK(s, &FreqOptions{N: 30})
		mustOK(t, err)
		o = MustOracle(t, "signal.freqs_zpk", []any{C(s.Z), C(s.P), s.K}, KW{"worN": 30})
		compareVec(t, "freqs_zpk w", gw, o.Idx(0).F(), 1e-12)
		compareC(t, "freqs_zpk h", gh, o.Idx(1).C(), 1e-12)
		gw, gh, err = FreqsZPK(s, &FreqOptions{W: w})
		mustOK(t, err)
		o = MustOracle(t, "signal.freqs_zpk", []any{C(s.Z), C(s.P), s.K}, KW{"worN": F(w)})
		compareC(t, "freqs_zpk h", gh, o.Idx(1).C(), 1e-13)
		_ = gw
	}
	// a pure integrator and a constant exercise the 1e-10 and no-pole paths
	for _, c := range [][2][]float64{{{1}, {1, 0}}, {{1, 2}, {1}}} {
		gw, gh, err := Freqs(c[0], c[1], nil)
		mustOK(t, err)
		o := MustOracle(t, "signal.freqs", []any{F(c[0]), F(c[1])}, nil)
		compareVec(t, "freqs w", gw, o.Idx(0).F(), 1e-9)
		compareC(t, "freqs h", gh, o.Idx(1).C(), 1e-8)
	}
	_, _, err := Freqs(nil, []float64{1}, nil)
	mustErr(t, "empty", err)
	_, _, err = Freqs([]float64{1}, []float64{1}, &FreqOptions{N: -1})
	mustErr(t, "N", err)
	old := hqrLimit
	hqrLimit = 0
	_, _, err = Freqs([]float64{1, 2, 3, 4, 5}, []float64{1}, nil)
	mustErr(t, "num roots", err)
	_, _, err = Freqs([]float64{1}, []float64{1, 2, 3, 4, 5}, nil)
	mustErr(t, "den roots", err)
	hqrLimit = old
	_, _, err = FreqsZPK(ZPK{}, &FreqOptions{N: -1})
	mustErr(t, "zpk N", err)
	if l := linspace(0, 1, 0, true); len(l) != 0 {
		t.Error("linspace 0")
	}
	if l := linspace(2, 3, 1, true); len(l) != 1 || l[0] != 2 {
		t.Error("linspace 1")
	}
	if polyvalLow(nil, 1) != 0 {
		t.Error("polyvalLow empty")
	}
}

func TestGroupDelayExactSingularity(t *testing.T) {
	b, a := []float64{1, -1}, []float64{1}
	w := []float64{0, 1}
	gw, gd, err := GroupDelay(b, a, &FreqOptions{W: w, Fs: 2 * math.Pi})
	mustOK(t, err)
	o := MustOracle(t, "signal.group_delay", []any{Tup(F(b), F(a))}, KW{"w": F(w)})
	compareVec(t, "group_delay w", gw, o.Idx(0).F(), 1e-15)
	compareVec(t, "group_delay gd", gd, o.Idx(1).F(), 1e-12)
	if gd[0] != 0 {
		t.Error("singular delay should be 0")
	}
}

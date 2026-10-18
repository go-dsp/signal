package signal

import (
	"math"
	"testing"
)

func TestUpfirdnOracle(t *testing.T) {
	rng := newRNG(90)
	modes := []PadMode{PadModeConstant, PadModeEdge, PadModeWrap, PadModeSymmetric, PadModeReflect, PadModeAntisymmetric, PadModeAntireflect, PadModeSmooth, PadModeLine}
	for _, c := range [][4]int{{1, 5, 1, 1}, {7, 20, 1, 1}, {7, 20, 3, 2}, {15, 30, 5, 7}, {40, 12, 2, 5}, {9, 3, 4, 1}, {25, 100, 3, 3}, {3, 50, 1, 4}, {64, 33, 7, 3}} {
		h, x := randVec(rng, c[0]), randVec(rng, c[1])
		for _, mode := range modes {
			got, err := Upfirdn(h, x, c[2], c[3], mode, 1.5)
			mustOK(t, err)
			o := MustOracle(t, "signal.upfirdn", []any{F(h), F(x), c[2], c[3]}, KW{"mode": string(mode), "cval": 1.5})
			if len(got) != len(o.F()) {
				t.Fatalf("%v %v: length %d vs %d", c, mode, len(got), len(o.F()))
			}
			compareVec(t, "upfirdn "+string(mode), got, o.F(), 1e-12)
		}
	}
	got, err := Upfirdn([]float64{1, 2}, []float64{1, 2, 3}, 2, 3, "", 0)
	mustOK(t, err)
	if len(got) != upfirdnLen(2, 3, 2, 3) {
		t.Error("default mode")
	}
	if g, _ := Upfirdn([]float64{1}, nil, 1, 1, "", 0); len(g) != 0 {
		t.Error("empty x")
	}
	// a filter much longer than the signal reaches far into the extension
	for _, n := range []int{1, 3} {
		x := []float64{2, -1, 4}[:n]
		for _, mode := range modes {
			g, err := Upfirdn(make1(20), x, 1, 1, mode, 5)
			mustOK(t, err)
			if n == 1 && (mode == PadModeReflect || mode == PadModeAntireflect || mode == PadModeSmooth || mode == PadModeLine) {
				// SciPy's extension of a one-sample signal is not defined
				// for these modes; ours repeats the sample.
				for _, v := range g {
					if v != 0 && math.IsNaN(v) {
						t.Errorf("NaN from %s on one sample", mode)
					}
				}
				continue
			}
			o := MustOracle(t, "signal.upfirdn", []any{F(make1(20)), F(x)}, KW{"mode": string(mode), "cval": 5.0})
			compareVec(t, "upfirdn far "+string(mode), g, o.F(), 1e-12)
		}
	}
	for i, f := range []func() error{
		func() error { _, e := Upfirdn(nil, []float64{1}, 1, 1, "", 0); return e },
		func() error { _, e := Upfirdn([]float64{1}, []float64{1}, 0, 1, "", 0); return e },
		func() error { _, e := Upfirdn([]float64{1}, []float64{1}, 1, 0, "", 0); return e },
		func() error { _, e := Upfirdn([]float64{1}, []float64{1}, 1, 1, "bogus", 0); return e },
		func() error { _, e := Upfirdn([]float64{1}, make([]float64, 10), 1<<30, 1, "", 0); return e },
	} {
		mustErr(t, "upfirdn error "+string(rune('a'+i)), f())
	}
}

func make1(n int) []float64 {
	h := make([]float64, n)
	for i := range h {
		h[i] = float64(i%3) - 1
	}
	return h
}

func TestResamplePolyOracle(t *testing.T) {
	rng := newRNG(91)
	x := testSignal(rng, 200)
	for _, c := range [][2]int{{3, 2}, {2, 3}, {1, 4}, {5, 1}, {7, 5}, {4, 6}, {1, 1}} {
		got, err := ResamplePoly(x, c[0], c[1], nil)
		mustOK(t, err)
		o := MustOracle(t, "signal.resample_poly", []any{F(x), c[0], c[1]}, nil)
		compareVec(t, "resample_poly", got, o.F(), 1e-12)
	}
	for _, pt := range []string{"mean", "median", "minimum", "maximum", "edge", "reflect", "line", "constant"} {
		got, err := ResamplePoly(x, 3, 2, &ResamplePolyOptions{PadType: pt})
		mustOK(t, err)
		o := MustOracle(t, "signal.resample_poly", []any{F(x), 3, 2}, KW{"padtype": pt})
		compareVec(t, "resample_poly "+pt, got, o.F(), 1e-12)
	}
	got, err := ResamplePoly(x, 5, 3, &ResamplePolyOptions{Window: Hann})
	mustOK(t, err)
	o := MustOracle(t, "signal.resample_poly", []any{F(x), 5, 3}, KW{"window": "hann"})
	compareVec(t, "resample_poly hann", got, o.F(), 1e-12)
	taps := Make3Taps()
	got, err = ResamplePoly(x, 2, 1, &ResamplePolyOptions{Taps: taps})
	mustOK(t, err)
	o = MustOracle(t, "signal.resample_poly", []any{F(x), 2, 1}, KW{"window": F(taps)})
	compareVec(t, "resample_poly taps", got, o.F(), 1e-12)
	got, err = ResamplePoly(x, 2, 1, &ResamplePolyOptions{Cval: 3})
	mustOK(t, err)
	o = MustOracle(t, "signal.resample_poly", []any{F(x), 2, 1}, KW{"cval": 3.0})
	compareVec(t, "resample_poly cval", got, o.F(), 1e-12)
	for i, f := range []func() error{
		func() error { _, e := ResamplePoly(x, 0, 1, nil); return e },
		func() error { _, e := ResamplePoly(x, 1, 0, nil); return e },
		func() error { _, e := ResamplePoly(x, 2, 1, &ResamplePolyOptions{PadType: "bogus"}); return e },
		func() error { _, e := ResamplePoly(x, 2, 1, &ResamplePolyOptions{PadType: "edge", Cval: 1}); return e },
		func() error {
			_, e := ResamplePoly(x, 2, 1, &ResamplePolyOptions{Window: func(int, bool) []float64 { return nil }})
			return e
		},
	} {
		mustErr(t, "resample_poly error "+string(rune('a'+i)), f())
	}
	if g, _ := ResamplePoly(nil, 3, 2, nil); len(g) != 0 {
		t.Error("empty input")
	}
}

// Make3Taps returns a short symmetric FIR for the explicit-filter path.
func Make3Taps() []float64 { return []float64{0.1, 0.2, 0.4, 0.2, 0.1} }

func TestResampleOracle(t *testing.T) {
	rng := newRNG(92)
	for _, n := range []int{64, 65, 100, 7} {
		x := testSignal(rng, n)
		for _, num := range []int{n, n / 2, n + 1, 2 * n, 2*n + 1, (n + 1) / 2, 3} {
			got, err := Resample(x, num, nil)
			mustOK(t, err)
			o := MustOracle(t, "signal.resample", []any{F(x), num}, nil)
			compareVec(t, "resample", got, o.F(), 1e-12)
		}
		got, err := Resample(x, n*3/2, &ResampleOptions{Window: Hann})
		mustOK(t, err)
		o := MustOracle(t, "signal.resample", []any{F(x), n * 3 / 2}, KW{"window": "hann"})
		compareVec(t, "resample window", got, o.F(), 1e-12)
		w := make([]float64, n)
		for i := range w {
			w[i] = 1 / (1 + float64(min(i, n-i)))
		}
		got, err = Resample(x, n*2, &ResampleOptions{WindowFreq: w})
		mustOK(t, err)
		o = MustOracle(t, "signal.resample", []any{F(x), n * 2}, KW{"window": F(w)})
		compareVec(t, "resample window array", got, o.F(), 1e-12)
	}
	x := testSignal(rng, 32)
	for i, f := range []func() error{
		func() error { _, e := Resample(nil, 4, nil); return e },
		func() error { _, e := Resample(x, 0, nil); return e },
		func() error { _, e := Resample(x, 1<<40, nil); return e },
		func() error { _, e := Resample(x, 8, &ResampleOptions{WindowFreq: []float64{1}}); return e },
		func() error {
			_, e := Resample(x, 8, &ResampleOptions{Window: func(int, bool) []float64 { return nil }})
			return e
		},
	} {
		mustErr(t, "resample error "+string(rune('a'+i)), f())
	}
}

func TestDecimateOracle(t *testing.T) {
	rng := newRNG(93)
	x := testSignal(rng, 400)
	for _, q := range []int{2, 3, 5, 8} {
		for _, o := range []DecimateOptions{{}, {NoZeroPhase: true}, {FIR: true}, {FIR: true, NoZeroPhase: true}, {N: 4}, {N: 30, FIR: true}} {
			got, err := Decimate(x, q, &o)
			mustOK(t, err)
			kw := KW{"zero_phase": !o.NoZeroPhase}
			if o.FIR {
				kw["ftype"] = "fir"
			}
			if o.N != 0 {
				kw["n"] = o.N
			}
			ref := MustOracle(t, "signal.decimate", []any{F(x), q}, kw)
			compareVec(t, "decimate", got, ref.F(), 1e-10)
		}
	}
	for i, f := range []func() error{
		func() error { _, e := Decimate(x, 0, nil); return e },
		func() error { _, e := Decimate(x, 2, &DecimateOptions{N: -1}); return e },
		func() error { _, e := Decimate(x[:5], 2, nil); return e },
		func() error { _, e := Decimate(x, 1, &DecimateOptions{FIR: true}); return e },
	} {
		mustErr(t, "decimate error "+string(rune('a'+i)), f())
	}
}

func TestHilbertOracle(t *testing.T) {
	rng := newRNG(94)
	for _, n := range []int{8, 9, 100, 1} {
		x := testSignal(rng, n)
		for _, nn := range []int{0, n, n + 3, max(n-3, 1)} {
			got, err := Hilbert(x, nn)
			mustOK(t, err)
			kw := KW{}
			if nn != 0 {
				kw["N"] = nn
			}
			o := MustOracle(t, "signal.hilbert", []any{F(x)}, kw)
			compareC(t, "hilbert", got, o.C(), 1e-12)
		}
	}
	_, err := Hilbert([]float64{1}, -1)
	mustErr(t, "negative N", err)
	_, err = Hilbert(nil, 0)
	mustErr(t, "empty", err)
	_, err = Hilbert([]float64{1}, 1<<40)
	mustErr(t, "huge N", err)
}

func peakSignal(rng interface{ Float64() float64 }) []float64 {
	x := make([]float64, 300)
	for i := range x {
		t := float64(i)
		x[i] = math.Sin(t/9) + 0.6*math.Sin(t/3.1) + 0.2*rng.Float64()
	}
	// plateaus
	for i := 100; i < 106; i++ {
		x[i] = 3
	}
	for i := 200; i < 203; i++ {
		x[i] = 2.5
	}
	return x
}

func ints(o Out) []int {
	var r []int
	for _, v := range o.F() {
		r = append(r, int(v))
	}
	return r
}

func TestFindPeaksOracle(t *testing.T) {
	rng := newRNG(95)
	x := peakSignal(rng)
	type pc struct {
		o  FindPeaksOptions
		kw KW
	}
	cases := []pc{
		{FindPeaksOptions{}, KW{}},
		{FindPeaksOptions{Height: AtLeast(0.5)}, KW{"height": 0.5}},
		{FindPeaksOptions{Height: Between(0.5, 1.5)}, KW{"height": Tup(0.5, 1.5)}},
		{FindPeaksOptions{Height: AtMost(1)}, KW{"height": Tup(nil, 1.0)}},
		{FindPeaksOptions{Threshold: AtLeast(0.05)}, KW{"threshold": 0.05}},
		{FindPeaksOptions{Distance: 12}, KW{"distance": 12}},
		{FindPeaksOptions{Distance: 12.5}, KW{"distance": 12.5}},
		{FindPeaksOptions{Prominence: AtLeast(0.5)}, KW{"prominence": 0.5}},
		{FindPeaksOptions{Prominence: AtLeast(0.2), Wlen: 20}, KW{"prominence": 0.2, "wlen": 20}},
		{FindPeaksOptions{Prominence: AtLeast(0.2), Wlen: 20.5}, KW{"prominence": 0.2, "wlen": 20.5}},
		{FindPeaksOptions{Width: AtLeast(3)}, KW{"width": 3}},
		{FindPeaksOptions{Width: Between(2, 30), RelHeight: 0.8}, KW{"width": Tup(2, 30), "rel_height": 0.8}},
		{FindPeaksOptions{PlateauSize: AtLeast(2)}, KW{"plateau_size": 2}},
		{FindPeaksOptions{Height: AtLeast(0), Distance: 5, Prominence: AtLeast(0.1), Width: AtLeast(1), PlateauSize: AtLeast(1), Threshold: AtLeast(0)},
			KW{"height": 0, "distance": 5, "prominence": 0.1, "width": 1, "plateau_size": 1, "threshold": 0}},
		{FindPeaksOptions{Height: AtLeast(100)}, KW{"height": 100}},
	}
	for i, c := range cases {
		o := c.o
		peaks, props, err := FindPeaks(x, &o)
		mustOK(t, err)
		ref := MustOracle(t, "signal.find_peaks", []any{F(x)}, c.kw)
		want := ints(ref.Idx(0))
		if len(peaks) != len(want) {
			t.Errorf("case %d: %d peaks, scipy %d", i, len(peaks), len(want))
			continue
		}
		for j := range peaks {
			if peaks[j] != want[j] {
				t.Errorf("case %d: peak %d is %d, scipy %d", i, j, peaks[j], want[j])
			}
		}
		pr := ref.Idx(1)
		checkF := func(name string, got []float64) {
			if pr.Has(name) {
				compareVec(t, "find_peaks "+name, got, pr.Get(name).F(), 1e-13)
			}
		}
		checkI := func(name string, got []int) {
			if pr.Has(name) {
				w := ints(pr.Get(name))
				if len(w) != len(got) {
					t.Errorf("case %d %s: length %d vs %d", i, name, len(got), len(w))
					return
				}
				for j := range w {
					if w[j] != got[j] {
						t.Errorf("case %d %s[%d]: %d vs %d", i, name, j, got[j], w[j])
					}
				}
			}
		}
		checkF("peak_heights", props.PeakHeights)
		checkF("left_thresholds", props.LeftThresholds)
		checkF("right_thresholds", props.RightThresholds)
		checkF("prominences", props.Prominences)
		checkF("widths", props.Widths)
		checkF("width_heights", props.WidthHeights)
		checkF("left_ips", props.LeftIPs)
		checkF("right_ips", props.RightIPs)
		checkI("plateau_sizes", props.PlateauSizes)
		checkI("left_edges", props.LeftEdges)
		checkI("right_edges", props.RightEdges)
		checkI("left_bases", props.LeftBases)
		checkI("right_bases", props.RightBases)
	}
	for i, f := range []func() error{
		func() error { _, _, e := FindPeaks(x, &FindPeaksOptions{Distance: 0.5}); return e },
		func() error { _, _, e := FindPeaks(x, &FindPeaksOptions{Prominence: AtLeast(0), Wlen: 0.5}); return e },
		func() error { _, _, e := FindPeaks(x, &FindPeaksOptions{Width: AtLeast(0), RelHeight: -1}); return e },
	} {
		mustErr(t, "find_peaks error "+string(rune('a'+i)), f())
	}
	// no peaks at all
	if p, _, err := FindPeaks([]float64{1, 1, 1}, nil); err != nil || len(p) != 0 || p == nil {
		t.Error("flat signal")
	}
}

func TestPeakProminenceWidthOracle(t *testing.T) {
	rng := newRNG(96)
	x := peakSignal(rng)
	peaks, _, err := FindPeaks(x, nil)
	mustOK(t, err)
	pf := I(peaks)
	for _, wlen := range []float64{0, 15, 40.5} {
		prom, l, r, err := PeakProminences(x, peaks, wlen)
		mustOK(t, err)
		kw := KW{}
		if wlen != 0 {
			kw["wlen"] = wlen
		}
		o := MustOracle(t, "signal.peak_prominences", []any{F(x), pf}, kw)
		compareVec(t, "peak_prominences", prom, o.Idx(0).F(), 1e-14)
		wi, wh, li, ri, err := PeakWidths(x, peaks, 0.5, prom, l, r)
		mustOK(t, err)
		kw["rel_height"] = 0.5
		o = MustOracle(t, "signal.peak_widths", []any{F(x), pf}, kw)
		compareVec(t, "peak_widths", wi, o.Idx(0).F(), 1e-13)
		compareVec(t, "peak_widths heights", wh, o.Idx(1).F(), 1e-14)
		compareVec(t, "peak_widths left", li, o.Idx(2).F(), 1e-13)
		compareVec(t, "peak_widths right", ri, o.Idx(3).F(), 1e-13)
	}
	_, _, _, err = PeakProminences(x, []int{-1}, 0)
	mustErr(t, "bad peak", err)
	_, _, _, err = PeakProminences(x, []int{1000}, 0)
	mustErr(t, "bad peak 2", err)
	prom, l, r, _ := PeakProminences(x, peaks, 0)
	_, _, _, _, err = PeakWidths(x, peaks, 0.5, prom[:1], l, r)
	mustErr(t, "short prominence data", err)
	_, _, _, _, err = PeakWidths(x, peaks, -1, prom, l, r)
	mustErr(t, "negative rel height", err)
	bad := append([]int(nil), l...)
	bad[0] = peaks[0] + 1
	_, _, _, _, err = PeakWidths(x, peaks, 0.5, prom, bad, r)
	mustErr(t, "invalid bases", err)
}

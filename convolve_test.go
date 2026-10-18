package signal

import (
	"testing"
)

var modeNames = map[ConvMode]string{ConvFull: "full", ConvSame: "same", ConvValid: "valid"}
var methodNames = map[ConvMethod]string{ConvAuto: "auto", ConvDirect: "direct", ConvFFT: "fft"}

func TestConvolveOracle(t *testing.T) {
	rng := newRNG(70)
	sizes := [][2]int{{1, 1}, {5, 3}, {3, 5}, {16, 16}, {100, 7}, {7, 100}, {1000, 31}, {257, 258}, {2000, 2000}, {40, 1}}
	for _, sz := range sizes {
		a, b := randVec(rng, sz[0]), randVec(rng, sz[1])
		for mode := ConvFull; mode <= ConvValid; mode++ {
			for method := ConvAuto; method <= ConvFFT; method++ {
				got, err := Convolve(a, b, mode, method)
				mustOK(t, err)
				o := MustOracle(t, "signal.convolve", []any{F(a), F(b)}, KW{"mode": modeNames[mode], "method": methodNames[method]})
				compareVec(t, "convolve "+methodNames[method], got, o.F(), 1e-12)
				got, err = Correlate(a, b, mode, method)
				mustOK(t, err)
				o = MustOracle(t, "signal.correlate", []any{F(a), F(b)}, KW{"mode": modeNames[mode], "method": methodNames[method]})
				compareVec(t, "correlate "+methodNames[method], got, o.F(), 1e-12)
			}
			got, err := FFTConvolve(a, b, mode)
			mustOK(t, err)
			o := MustOracle(t, "signal.fftconvolve", []any{F(a), F(b)}, KW{"mode": modeNames[mode]})
			compareVec(t, "fftconvolve", got, o.F(), 1e-13)
			got, err = OAConvolve(a, b, mode)
			mustOK(t, err)
			o = MustOracle(t, "signal.oaconvolve", []any{F(a), F(b)}, KW{"mode": modeNames[mode]})
			compareVec(t, "oaconvolve", got, o.F(), 1e-13)
			if m := ChooseConvMethod(sz[0], sz[1], mode); true {
				o := MustOracle(t, "signal.choose_conv_method", []any{F(a), F(b)}, KW{"mode": modeNames[mode]})
				want := ConvDirect
				if o.v.(string) == "fft" {
					want = ConvFFT
				}
				// the 'valid' cost model is evaluated with the longer input first
				if mode != ConvValid && m != want {
					t.Errorf("choose_conv_method(%d,%d,%v) = %v, scipy %v", sz[0], sz[1], mode, m, want)
				}
			}
		}
	}
	// long signal against short kernel: many overlap-add blocks
	a, b := randVec(rng, 50000), randVec(rng, 300)
	for _, swap := range []bool{false, true} {
		x, h := a, b
		if swap {
			x, h = b, a
		}
		got, err := OAConvolve(x, h, ConvFull)
		mustOK(t, err)
		o := MustOracle(t, "signal.oaconvolve", []any{F(x), F(h)}, nil)
		compareVec(t, "oaconvolve long", got, o.F(), 1e-12)
	}
	// empty inputs and errors
	for _, f := range []func() ([]float64, error){
		func() ([]float64, error) { return Convolve(nil, []float64{1}, ConvFull, ConvAuto) },
		func() ([]float64, error) { return FFTConvolve(nil, []float64{1}, ConvFull) },
		func() ([]float64, error) { return OAConvolve([]float64{1}, nil, ConvFull) },
	} {
		if r, err := f(); err != nil || len(r) != 0 {
			t.Error("empty input should give an empty result")
		}
	}
	if r, _ := FFTConvolve([]float64{3}, []float64{4}, ConvFull); len(r) != 1 || r[0] != 12 {
		t.Error("scalar fftconvolve")
	}
	_, err := Convolve([]float64{1}, []float64{1}, ConvMode(7), ConvAuto)
	mustErr(t, "mode", err)
	_, err = Convolve([]float64{1}, []float64{1}, ConvFull, ConvMethod(7))
	mustErr(t, "method", err)
	_, err = FFTConvolve([]float64{1}, []float64{1}, ConvMode(7))
	mustErr(t, "fft mode", err)
	_, err = OAConvolve([]float64{1}, []float64{1}, ConvMode(7))
	mustErr(t, "oa mode", err)
	_, err = CorrelationLags(3, 3, ConvMode(7))
	mustErr(t, "lags mode", err)
}

func TestCorrelationLagsOracle(t *testing.T) {
	for _, sz := range [][2]int{{5, 3}, {3, 5}, {4, 4}, {6, 2}, {7, 7}, {1, 1}} {
		for mode := ConvFull; mode <= ConvValid; mode++ {
			got, err := CorrelationLags(sz[0], sz[1], mode)
			mustOK(t, err)
			o := MustOracle(t, "signal.correlation_lags", []any{sz[0], sz[1]}, KW{"mode": modeNames[mode]})
			want := o.F()
			if len(got) != len(want) {
				t.Fatalf("lags %v %v: length %d vs %d", sz, mode, len(got), len(want))
			}
			for i := range got {
				if float64(got[i]) != want[i] {
					t.Errorf("lags %v %v differ at %d", sz, mode, i)
				}
			}
		}
	}
}

func TestDeconvolveOracle(t *testing.T) {
	rng := newRNG(71)
	for _, sz := range [][2]int{{10, 3}, {5, 5}, {3, 5}, {30, 1}} {
		num, den := randVec(rng, sz[0]), randVec(rng, sz[1])
		den[0] += 2
		q, r, err := Deconvolve(num, den)
		mustOK(t, err)
		o := MustOracle(t, "signal.deconvolve", []any{F(num), F(den)}, nil)
		if sz[1] > sz[0] { // SciPy returns an empty list for the quotient
			if len(q) != 0 || o.Idx(0).Len() != 0 {
				t.Error("quotient should be empty")
			}
		} else {
			compareVec(t, "deconvolve quot", q, o.Idx(0).F(), 1e-12)
		}
		compareVec(t, "deconvolve rem", r, o.Idx(1).F(), 1e-12)
	}
	_, _, err := Deconvolve(nil, []float64{1})
	mustErr(t, "empty signal", err)
	_, _, err = Deconvolve([]float64{1, 2}, []float64{0, 1})
	mustErr(t, "zero leading divisor", err)
}

func TestLambertW(t *testing.T) {
	for _, x := range []float64{-0.36, -0.3, -0.2, -0.1, -0.01, -1e-4} {
		w := lambertWm1(x)
		if got := w * expf(w); got < x-1e-13 || got > x+1e-13 {
			t.Errorf("W(%g) = %g gives %g", x, w, got)
		}
	}
}

func TestOALensLayouts(t *testing.T) {
	for s1 := 1; s1 <= 300; s1 += 7 {
		for s2 := 1; s2 <= 300; s2 += 11 {
			block, overlap, st1, st2 := calcOALens(s1, s2)
			if overlap == 0 {
				if st1 != s1 || st2 != s2 || block != s1+s2-1 {
					t.Errorf("fallback layout for %d,%d: %d %d %d", s1, s2, block, st1, st2)
				}
				continue
			}
			short, long := st2, st1
			if s2 > s1 {
				short, long = st1, st2
			}
			if short != min(s1, s2) || long != block-min(s1, s2)+1 || overlap != min(s1, s2)-1 {
				t.Errorf("overlap-add layout for %d,%d: %d %d %d %d", s1, s2, block, overlap, st1, st2)
			}
		}
	}
}

func TestConvolveZerosInKernel(t *testing.T) {
	got := convolveDirect([]float64{1, 2, 3}, []float64{0, 1, 0})
	want := []float64{0, 1, 2, 3, 0}
	if !eqSlice(got, want) {
		t.Errorf("got %v", got)
	}
	// rank deficient Gustafsson matrix: a filter that is a pure delay-free gain
	_, err := FiltFilt([]float64{1, 0}, []float64{1, 0}, []float64{1, 2, 3, 4, 5, 6}, &FiltFiltOptions{Method: FiltGust})
	mustErr(t, "rank deficient gust", err)
}

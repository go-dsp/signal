package signal

import (
	"math"
	"testing"
)

func TestKaiserOrdOracle(t *testing.T) {
	for _, a := range []float64{10, 20, 21.5, 40, 50, 60, 100} {
		for _, w := range []float64{0.01, 0.1, 0.3} {
			n, beta, err := KaiserOrd(a, w)
			mustOK(t, err)
			o := MustOracle(t, "signal.kaiserord", []any{a, w}, nil)
			if float64(n) != o.Idx(0).X() {
				t.Errorf("kaiserord(%g,%g): %d vs %v", a, w, n, o.Idx(0).X())
			}
			compareVec(t, "kaiserord beta", []float64{beta}, []float64{o.Idx(1).X()}, 1e-15)
			at := KaiserAtten(n, w)
			oa := MustOracle(t, "signal.kaiser_atten", []any{n, w}, nil)
			compareVec(t, "kaiser_atten", []float64{at}, []float64{oa.X()}, 1e-15)
		}
	}
	_, _, err := KaiserOrd(5, 0.1)
	mustErr(t, "small ripple", err)
	_, _, err = KaiserOrd(40, 0)
	mustErr(t, "width", err)
	if KaiserBeta(10) != 0 {
		t.Error("beta below 21 dB")
	}
}

func TestFirwinOracle(t *testing.T) {
	type fc struct {
		n      int
		cutoff []float64
		o      FirwinOptions
		kw     KW
	}
	passNames := map[FirPass]any{FirPassZeroTrue: true, FirPassZeroFalse: false, FirLowpass: "lowpass", FirHighpass: "highpass", FirBandpass: "bandpass", FirBandstop: "bandstop"}
	var cases []fc
	add := func(n int, cut []float64, o FirwinOptions) {
		kw := KW{"pass_zero": passNames[o.PassZero], "scale": !o.NoScale}
		if o.Width > 0 {
			kw["width"] = o.Width
		}
		if o.Fs > 0 {
			kw["fs"] = o.Fs
		}
		cases = append(cases, fc{n, cut, o, kw})
	}
	add(21, []float64{0.3}, FirwinOptions{})
	add(22, []float64{0.3}, FirwinOptions{})
	add(21, []float64{0.3}, FirwinOptions{PassZero: FirHighpass})
	add(31, []float64{0.2, 0.5}, FirwinOptions{PassZero: FirBandpass})
	add(31, []float64{0.2, 0.5}, FirwinOptions{PassZero: FirBandstop})
	add(32, []float64{0.2, 0.5}, FirwinOptions{PassZero: FirBandpass})
	add(41, []float64{0.1, 0.3, 0.6, 0.8}, FirwinOptions{PassZero: FirPassZeroFalse})
	add(41, []float64{0.1, 0.3, 0.6}, FirwinOptions{PassZero: FirPassZeroTrue})
	add(21, []float64{0.3}, FirwinOptions{NoScale: true})
	add(51, []float64{0.25}, FirwinOptions{Width: 0.1})
	add(51, []float64{1000}, FirwinOptions{Fs: 8000})
	add(1, []float64{0.3}, FirwinOptions{})
	for _, c := range cases {
		o := c.o
		if c.o.Width == 0 {
			o.Window = Hamming
		}
		got, err := Firwin(c.n, c.cutoff, &o)
		mustOK(t, err)
		want := MustOracle(t, "signal.firwin", []any{c.n, F(c.cutoff)}, c.kw)
		compareVec(t, "firwin", got, want.F(), 1e-14)
	}
	// windows by name
	for _, w := range []struct {
		win  Window
		spec any
	}{{Kaiser(5), Tup("kaiser", 5.0)}, {Hann, "hann"}, {Blackman, "blackman"}, {Tukey(0.3), Tup("tukey", 0.3)}, {Chebwin(70), Tup("chebwin", 70.0)}} {
		got, err := Firwin(33, []float64{0.4}, &FirwinOptions{Window: w.win})
		mustOK(t, err)
		want := MustOracle(t, "signal.firwin", []any{33, F([]float64{0.4})}, KW{"window": w.spec})
		compareVec(t, "firwin window", got, want.F(), 1e-12)
	}
	for _, c := range []struct {
		n   int
		cut []float64
		o   FirwinOptions
	}{
		{21, []float64{0.3}, FirwinOptions{Fs: -1}},
		{0, []float64{0.3}, FirwinOptions{}},
		{21, nil, FirwinOptions{}},
		{21, []float64{0}, FirwinOptions{}},
		{21, []float64{1}, FirwinOptions{}},
		{21, []float64{0.5, 0.4}, FirwinOptions{}},
		{21, []float64{0.3, 0.5}, FirwinOptions{PassZero: FirLowpass}},
		{21, []float64{0.3}, FirwinOptions{PassZero: FirBandpass}},
		{21, []float64{0.3}, FirwinOptions{PassZero: FirPass(9)}},
		{20, []float64{0.3}, FirwinOptions{PassZero: FirHighpass}},
		{21, []float64{0.3}, FirwinOptions{Window: func(int, bool) []float64 { return nil }}},
	} {
		_, err := Firwin(c.n, c.cut, &c.o)
		mustErr(t, "firwin", err)
	}
}

func TestFirwin2Oracle(t *testing.T) {
	type fc struct {
		n          int
		freq, gain []float64
		o          Firwin2Options
		kw         KW
	}
	cases := []fc{
		{31, []float64{0, 0.3, 0.6, 1}, []float64{1, 1, 0, 0}, Firwin2Options{}, KW{}},
		{32, []float64{0, 0.3, 0.6, 1}, []float64{1, 1, 0.2, 0}, Firwin2Options{}, KW{}},
		{33, []float64{0, 0.3, 0.3, 0.6, 1}, []float64{0, 0, 1, 1, 0}, Firwin2Options{}, KW{}},
		{31, []float64{0, 0.5, 1}, []float64{0, 1, 0}, Firwin2Options{Antisymmetric: true}, KW{"antisymmetric": true}},
		{32, []float64{0, 0.5, 1}, []float64{0, 1, 1}, Firwin2Options{Antisymmetric: true}, KW{"antisymmetric": true}},
		{31, []float64{0, 0.4, 1}, []float64{1, 0.5, 0}, Firwin2Options{Nfreqs: 513, Window: Hann}, KW{"nfreqs": 513, "window": "hann"}},
		{31, []float64{0, 0.4, 1}, []float64{1, 0.5, 0}, Firwin2Options{NoWindow: true}, KW{"window": nil}},
		{31, []float64{0, 2000, 4000}, []float64{1, 0.5, 0}, Firwin2Options{Fs: 8000}, KW{"fs": 8000.0}},
	}
	for _, c := range cases {
		got, err := Firwin2(c.n, c.freq, c.gain, &c.o)
		mustOK(t, err)
		want := MustOracle(t, "signal.firwin2", []any{c.n, F(c.freq), F(c.gain)}, c.kw)
		compareVec(t, "firwin2", got, want.F(), 1e-13)
	}
	bad := []struct {
		n          int
		freq, gain []float64
		o          Firwin2Options
	}{
		{31, []float64{0, 1}, []float64{1}, Firwin2Options{}},
		{31, []float64{0, 1}, []float64{1, 0}, Firwin2Options{Fs: -1}},
		{0, []float64{0, 1}, []float64{1, 0}, Firwin2Options{}},
		{31, []float64{0, 1}, []float64{1, 0}, Firwin2Options{Nfreqs: 16}},
		{31, []float64{0.1, 1}, []float64{1, 0}, Firwin2Options{}},
		{31, []float64{0}, []float64{1}, Firwin2Options{}},
		{31, []float64{0, 0.6, 0.5, 1}, []float64{1, 1, 0, 0}, Firwin2Options{}},
		{31, []float64{0, 0.5, 0.5, 0.5, 1}, []float64{1, 1, 0, 0, 0}, Firwin2Options{}},
		{31, []float64{0, 0, 0.5, 1}, []float64{1, 1, 0, 0}, Firwin2Options{}},
		{31, []float64{0, 0.5, 1, 1}, []float64{1, 1, 0, 0}, Firwin2Options{}},
		{32, []float64{0, 1}, []float64{1, 1}, Firwin2Options{}},
		{31, []float64{0, 1}, []float64{1, 0}, Firwin2Options{Antisymmetric: true}},
		{32, []float64{0, 1}, []float64{1, 1}, Firwin2Options{Antisymmetric: true}},
		{31, []float64{0, 0.5, 1}, []float64{0, 1, 0}, Firwin2Options{Window: func(int, bool) []float64 { return nil }}},
		{31, []float64{0, 5e-17, 5e-17, 1}, []float64{0, 0, 1, 1}, Firwin2Options{}},
	}
	for i, c := range bad {
		_, err := Firwin2(c.n, c.freq, c.gain, &c.o)
		if err == nil {
			t.Errorf("firwin2 bad case %d: expected an error", i)
		}
	}
}

func TestFirlsOracle(t *testing.T) {
	type lc struct {
		n                  int
		bands, desired, wt []float64
		fs                 float64
	}
	cases := []lc{
		{21, []float64{0, 0.3, 0.4, 1}, []float64{1, 1, 0, 0}, nil, 0},
		{31, []float64{0, 0.2, 0.3, 0.6, 0.7, 1}, []float64{0, 0, 1, 1, 0, 0}, []float64{1, 5, 1}, 0},
		{41, []float64{0, 0.5, 0.6, 1}, []float64{0, 1, 0, 0}, nil, 0},
		{15, []float64{0, 1000, 1500, 4000}, []float64{1, 1, 0, 0}, []float64{1, 10}, 8000},
		{1, []float64{0, 1}, []float64{1, 1}, nil, 0},
	}
	for _, c := range cases {
		got, err := Firls(c.n, c.bands, c.desired, c.wt, c.fs)
		mustOK(t, err)
		kw := KW{}
		if c.wt != nil {
			kw["weight"] = F(c.wt)
		}
		if c.fs != 0 {
			kw["fs"] = c.fs
		}
		want := MustOracle(t, "signal.firls", []any{c.n, F(c.bands), F(c.desired)}, kw)
		compareVec(t, "firls", got, want.F(), 1e-10)
	}
	b2, d2 := []float64{0, 0.3, 0.4, 1}, []float64{1, 1, 0, 0}
	for i, f := range []func() error{
		func() error { _, e := Firls(21, b2, d2, nil, -1); return e },
		func() error { _, e := Firls(20, b2, d2, nil, 0); return e },
		func() error { _, e := Firls(21, b2[:3], d2[:3], nil, 0); return e },
		func() error { _, e := Firls(21, []float64{0, 0.3, 0.4, 1.5}, d2, nil, 0); return e },
		func() error { _, e := Firls(21, b2, d2[:3], nil, 0); return e },
		func() error { _, e := Firls(21, []float64{0, 0, 0.4, 1}, d2, nil, 0); return e },
		func() error { _, e := Firls(21, []float64{0, 0.3, 0.2, 1}, d2, nil, 0); return e },
		func() error { _, e := Firls(21, []float64{0, 0.5, 0.4, 1}, d2, nil, 0); return e },
		func() error { _, e := Firls(21, []float64{0.2, 0.3, 0.1, 0.4}, d2, nil, 0); return e },
		func() error { _, e := Firls(21, b2, []float64{1, 1, -1, 0}, nil, 0); return e },
		func() error { _, e := Firls(21, b2, d2, []float64{1}, 0); return e },
		func() error { _, e := Firls(21, b2, d2, []float64{1, -1}, 0); return e },
		func() error { _, e := Firls(21, b2, d2, []float64{0, 0}, 0); return e },
	} {
		mustErr(t, "firls error "+string(rune('a'+i)), f())
	}
}

func TestRemezOracle(t *testing.T) {
	type rc struct {
		n       int
		bands   []float64
		desired []float64
		o       RemezOptions
		kw      KW
	}
	typeNames := map[RemezType]string{RemezBandpass: "bandpass", RemezDifferentiator: "differentiator", RemezHilbert: "hilbert"}
	cases := []rc{
		{31, []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}, RemezOptions{}, KW{}},
		{32, []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}, RemezOptions{}, KW{}},
		{41, []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5}, []float64{0, 1, 0}, RemezOptions{Weight: []float64{10, 1, 10}}, KW{"weight": F([]float64{10, 1, 10})}},
		{21, []float64{0, 0.2, 0.3, 0.5}, []float64{0, 1}, RemezOptions{}, KW{}},
		{22, []float64{0.1, 0.2, 0.3, 0.4}, []float64{1, 0}, RemezOptions{}, KW{}},
		{25, []float64{0, 0.4}, []float64{1}, RemezOptions{Type: RemezDifferentiator}, KW{}},
		{26, []float64{0, 0.45}, []float64{1}, RemezOptions{Type: RemezDifferentiator}, KW{}},
		{25, []float64{0.05, 0.45}, []float64{1}, RemezOptions{Type: RemezHilbert}, KW{}},
		{26, []float64{0.05, 0.5}, []float64{1}, RemezOptions{Type: RemezHilbert}, KW{}},
		{51, []float64{0, 1000, 1500, 4000}, []float64{1, 0}, RemezOptions{Fs: 8000, GridDensity: 32}, KW{"fs": 8000.0, "grid_density": 32}},
	}
	for i, c := range cases {
		c.kw["type"] = typeNames[c.o.Type]
		got, err := Remez(c.n, c.bands, c.desired, &c.o)
		mustOK(t, err)
		want := MustOracle(t, "signal.remez", []any{c.n, F(c.bands), F(c.desired)}, c.kw)
		d := compareVec(t, "remez", got, want.F(), 1e-3)
		t.Logf("case %d: normwise difference from SciPy %.3g", i, d)
	}
}

func TestRemezErrors(t *testing.T) {
	b, d := []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}
	for i, f := range []func() error{
		func() error { _, e := Remez(31, b, d, &RemezOptions{Fs: -1}); return e },
		func() error { _, e := Remez(31, b, d, &RemezOptions{Type: RemezType(7)}); return e },
		func() error { _, e := Remez(1, b, d, nil); return e },
		func() error { _, e := Remez(31, b[:3], d, nil); return e },
		func() error { _, e := Remez(31, nil, nil, nil); return e },
		func() error { _, e := Remez(31, b, d, &RemezOptions{MaxIter: -1}); return e },
		func() error { _, e := Remez(31, b, d, &RemezOptions{Weight: []float64{1}}); return e },
		func() error { _, e := Remez(31, []float64{0, 0.1, 0.05, 0.5}, d, nil); return e },
		func() error { _, e := Remez(31, []float64{0, 0.1, 0.15, 0.6}, d, nil); return e },
		func() error { _, e := Remez(31, b, d, &RemezOptions{Weight: []float64{1, 0}}); return e },
		func() error { _, e := Remez(31, []float64{0, 0.0001, 0.0002, 0.0003}, d, nil); return e },
	} {
		mustErr(t, "remez error "+string(rune('a'+i)), f())
	}
	// not converged in one iteration
	h, err := Remez(41, b, d, &RemezOptions{MaxIter: 1})
	if err != ErrNotConverged || len(h) != 41 {
		t.Errorf("expected ErrNotConverged with taps, got %v", err)
	}
	_ = math.Pi
}

func TestRemezExchange(t *testing.T) {
	cases := []struct {
		e    []float64
		want int
		ok   bool
		idx  []int
	}{
		{[]float64{1, 2, 1}, 3, false, nil},
		{[]float64{2, 1, 3, -1}, 2, true, []int{2, 3}},
		{[]float64{3, 1, 2, -1}, 2, true, []int{0, 3}},
		{[]float64{3, -1, 2, -4}, 2, true, []int{2, 3}},
		{[]float64{5, -1, 2, -4}, 3, true, []int{0, 1, 2}},
		{[]float64{-1, 1}, 2, true, []int{0, 1}},
	}
	for i, c := range cases {
		got, ok := exchange(c.e, c.want)
		if ok != c.ok || (ok && !equalInts(got, c.idx)) {
			t.Errorf("case %d: got %v %v, want %v %v", i, got, ok, c.idx, c.ok)
		}
	}
	old := exchangeFn
	exchangeFn = func([]float64, int) ([]int, bool) { return nil, false }
	_, err := Remez(31, []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}, nil)
	exchangeFn = old
	mustErr(t, "exchange failure", err)
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

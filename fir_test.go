package signal

import (
	"math"
	"math/rand/v2"
	"slices"
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

// TestRemezOracleSweep compares a seeded sweep of designs with SciPy: two
// and three bands, narrow and wide transitions, weights, even and odd
// lengths, differentiators and Hilbert transformers. Between them they take
// the exchange's rarer branches (an extremum gained at either end of the
// grid, the set shifted by one, the deviation no longer growing).
func TestRemezOracleSweep(t *testing.T) {
	r := rand.New(rand.NewPCG(17, 29))
	typeNames := map[RemezType]string{RemezBandpass: "bandpass", RemezDifferentiator: "differentiator", RemezHilbert: "hilbert"}
	worst, stalls, judged := 0.0, 0, 0
	for i := 0; i < 40; i++ {
		n := 11 + r.IntN(60)
		var bands, desired, weight []float64
		typ := RemezBandpass
		switch i % 8 {
		case 6:
			typ = RemezDifferentiator
			bands, desired, weight = []float64{0, 0.3 + 0.15*r.Float64()}, []float64{1}, []float64{1}
		case 7:
			typ = RemezHilbert
			lo := 0.02 + 0.05*r.Float64()
			bands, desired, weight = []float64{lo, 0.5 - lo}, []float64{1}, []float64{1}
		default:
			nb := 2 + r.IntN(2)
			edges := make([]float64, 2*nb)
			for k := range edges {
				edges[k] = r.Float64()
			}
			slices.Sort(edges)
			// spread the edges over [0, 0.5] keeping transitions >= 0.02
			for k := range edges {
				edges[k] = 0.5 * (edges[k]*float64(2*nb-1) + float64(k)) / float64(4*nb-2)
			}
			edges[0], edges[len(edges)-1] = 0, 0.5
			for k := 0; k < nb; k++ {
				desired = append(desired, float64((k+i)%2))
				weight = append(weight, 1+9*r.Float64())
			}
			bands = edges
		}
		o := RemezOptions{Type: typ, Weight: weight}
		got, err := Remez(n, bands, desired, &o)
		if err != nil && err != ErrNotConverged {
			t.Fatalf("design %d (n=%d bands=%v): %v", i, n, bands, err)
		}
		kw := KW{"type": typeNames[typ], "weight": F(weight)}
		want, oerr := Oracle(t, "signal.remez", []any{n, F(bands), F(desired)}, kw)
		if oerr != nil { // SciPy: "Failure to converge"
			// The stall test (the deviation no longer grows) compares two
			// nearly equal numbers, so it turns on last bits that fused
			// multiply-adds change between architectures: SciPy's
			// recording (arm64) may stall where amd64 converges. Either
			// outcome is accepted; the filter must be finite.
			for _, v := range got {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("design %d: non-finite tap %v (%v)", i, v, err)
				}
			}
			stalls++
			continue
		}
		h := want.F()
		if d := normDiff(got, h); d <= 1e-3 {
			worst = math.Max(worst, d)
			continue
		}
		// Coefficients can differ widely when the minimax error is near
		// rounding level (the transition bands are unconstrained); then
		// the criterion itself judges: the weighted band error.
		eg := remezBandError(got, bands, desired, weight, typ)
		es := remezBandError(h, bands, desired, weight, typ)
		if !(eg <= 1.25*es+1e-12) {
			t.Errorf("design %d (n=%d bands=%.4v): band error %.4g, SciPy %.4g", i, n, bands, eg, es)
		}
		judged++
	}
	t.Logf("worst normwise difference from SciPy %.3g; %d designs judged by band error; %d fail to converge in both", worst, judged, stalls)
}

// normDiff is max|a-b| / max|b|.
func normDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var d, m float64
	for i := range b {
		d = math.Max(d, math.Abs(a[i]-b[i]))
		m = math.Max(m, math.Abs(b[i]))
	}
	return d / m
}

// remezBandError is the Remez criterion for h: the largest weighted
// deviation of the amplitude from the desired response over the bands,
// sampled at 4000 points per band (the differentiator's desired response
// and weight scale with frequency, as in Remez).
func remezBandError(h, bands, desired, weight []float64, typ RemezType) float64 {
	worst := 0.0
	for b := range desired {
		lo, hi := bands[2*b], bands[2*b+1]
		for k := 0; k <= 4000; k++ {
			f := lo + (hi-lo)*float64(k)/4000
			var re, im float64
			for n, v := range h {
				s, c := math.Sincos(2 * math.Pi * f * float64(n))
				re += v * c
				im -= v * s
			}
			d, w := desired[b], weight[b]
			if typ == RemezDifferentiator {
				d *= f
				if desired[b] >= 0.0001 && f > 0 {
					w /= f
				}
			}
			worst = math.Max(worst, w*math.Abs(math.Hypot(re, im)-d))
		}
	}
	return worst
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
		func() error { _, e := Remez(1<<20, b, d, &RemezOptions{GridDensity: 4096}); return e },
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

// TestRemezTypeIIIBandShort is the regression for NaN taps: a type III
// differentiator whose band stops short of 0.5 has a reduced response with a
// pole at 0.5, so sampling the interpolant there (as the transform did before
// SciPy's Chebyshev mapping was ported) gave -Inf and NaN taps. It also
// stalls on arm64, where the previous pass's filter must be returned.
func TestRemezTypeIIIBandShort(t *testing.T) {
	for _, n := range []int{33, 65, 101} {
		h, err := Remez(n, []float64{0, 0.370209387678557}, []float64{1}, &RemezOptions{Type: RemezDifferentiator})
		if err != nil && err != ErrNotConverged {
			t.Fatal(err)
		}
		for i, v := range h {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("n=%d: tap %d is %v (%v)", n, i, v, err)
			}
		}
	}
}

// TestRemezBaryAtNode checks that the interpolant takes the node's value at
// a node (adjacent bands put two grid points on the same frequency).
func TestRemezBaryAtNode(t *testing.T) {
	x := []float64{-1, 0, 1}
	ad := []float64{1, 1, 1}
	if v := baryEval(0, x, ad, []float64{3, 7, 5}); v != 7 {
		t.Fatalf("at a node: %v, want 7", v)
	}
	if v := baryEval(1, x, ad, []float64{3, 7, 5}); v != 5 {
		t.Fatalf("at the odd last node: %v, want 5", v)
	}
	h, err := Remez(31, []float64{0, 0.2, 0.2, 0.5}, []float64{1, 0}, nil)
	if len(h) != 31 && err == nil {
		t.Fatalf("adjacent bands: %d taps, %v", len(h), err)
	}
}

package signal

import (
	"math"
	"testing"
)

var detNames = map[DetrendType]any{DetrendConstant: "constant", DetrendLinear: "linear", DetrendNone: false}
var scaleNames = map[Scaling]string{ScaleDensity: "density", ScaleSpectrum: "spectrum"}
var avgNames = map[Average]string{AverageMean: "mean", AverageMedian: "median"}

func testSignal(rng interface{ Float64() float64 }, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(0.3*float64(i)) + 0.5*math.Sin(1.1*float64(i)+1) + 0.3*(rng.Float64()-0.5) + 0.01*float64(i)
	}
	return x
}

func TestDetrendOracle(t *testing.T) {
	rng := newRNG(80)
	x := testSignal(rng, 100)
	for _, typ := range []DetrendType{DetrendConstant, DetrendLinear} {
		for _, bp := range [][]int{nil, {30}, {10, 50, 50, 100}} {
			got, err := Detrend(x, typ, bp)
			mustOK(t, err)
			kw := KW{"type": detNames[typ]}
			if bp != nil {
				kw["bp"] = bp
			}
			o := MustOracle(t, "signal.detrend", []any{F(x)}, kw)
			compareVec(t, "detrend", got, o.F(), 1e-12)
		}
	}
	got, err := Detrend([]float64{5}, DetrendLinear, nil)
	mustOK(t, err)
	if got[0] != 0 {
		t.Error("single point detrend")
	}
	if g, _ := Detrend(nil, DetrendConstant, nil); len(g) != 0 {
		t.Error("empty detrend")
	}
	_, err = Detrend(x, DetrendNone, nil)
	mustErr(t, "type", err)
	_, err = Detrend(x, DetrendLinear, []int{500})
	mustErr(t, "breakpoint", err)
	_, err = Detrend(x, DetrendLinear, []int{-1})
	mustErr(t, "negative breakpoint", err)
}

func TestWelchFamilyOracle(t *testing.T) {
	rng := newRNG(81)
	x, y := testSignal(rng, 1000), testSignal(rng, 1000)
	for i := range y {
		y[i] = 0.7*x[i] + 0.4*y[i]
	}
	type wc struct {
		o  SpectralOptions
		kw KW
	}
	cases := []wc{
		{SpectralOptions{}, KW{"nperseg": 256}},
		{SpectralOptions{Fs: 100, Nperseg: 128}, KW{"fs": 100.0, "nperseg": 128}},
		{SpectralOptions{Nperseg: 100, Noverlap: Overlap(0)}, KW{"nperseg": 100, "noverlap": 0}},
		{SpectralOptions{Nperseg: 100, Noverlap: Overlap(99)}, KW{"nperseg": 100, "noverlap": 99}},
		{SpectralOptions{Nperseg: 101, Nfft: 256}, KW{"nperseg": 101, "nfft": 256}},
		{SpectralOptions{Nperseg: 64, Window: Hamming}, KW{"nperseg": 64, "window": "hamming"}},
		{SpectralOptions{Nperseg: 64, Window: Kaiser(6)}, KW{"nperseg": 64, "window": Tup("kaiser", 6.0)}},
		{SpectralOptions{Nperseg: 64, Detrend: DetrendLinear}, KW{"nperseg": 64, "detrend": "linear"}},
		{SpectralOptions{Nperseg: 64, Detrend: DetrendNone}, KW{"nperseg": 64, "detrend": false}},
		{SpectralOptions{Nperseg: 64, Scaling: ScaleSpectrum}, KW{"nperseg": 64, "scaling": "spectrum"}},
		{SpectralOptions{Nperseg: 64, TwoSided: true}, KW{"nperseg": 64, "return_onesided": false}},
		{SpectralOptions{Nperseg: 63, Average: AverageMedian}, KW{"nperseg": 63, "average": "median"}},
		{SpectralOptions{Nperseg: 2000}, KW{"nperseg": 2000}},
		{SpectralOptions{Nperseg: 1000}, KW{"nperseg": 1000}},
	}
	for _, c := range cases {
		o := c.o
		f, p, err := Welch(x, &o)
		mustOK(t, err)
		ref := MustOracle(t, "signal.welch", []any{F(x)}, c.kw)
		compareVec(t, "welch f", f, ref.Idx(0).F(), 1e-14)
		compareVec(t, "welch p", p, ref.Idx(1).F(), 1e-12)

		f, cp, err := CSD(x, y, &o)
		mustOK(t, err)
		ref = MustOracle(t, "signal.csd", []any{F(x), F(y)}, c.kw)
		compareVec(t, "csd f", f, ref.Idx(0).F(), 1e-14)
		compareC(t, "csd p", cp, ref.Idx(1).C(), 1e-12)
		if c.o.Scaling == ScaleDefault && !c.o.TwoSided && c.o.Average == AverageMean {
			f, coh, err := Coherence(x, y, &o)
			mustOK(t, err)
			kw := KW{}
			for k, v := range c.kw {
				if k != "scaling" && k != "return_onesided" && k != "average" {
					kw[k] = v
				}
			}
			ref = MustOracle(t, "signal.coherence", []any{F(x), F(y)}, kw)
			compareVec(t, "coherence f", f, ref.Idx(0).F(), 1e-14)
			if d := compareVec(t, "coherence", coh, ref.Idx(1).F(), 1e-9); d > 1e-9 {
				t.Logf("  ^ case %+v", c.kw)
			}
		}
	}
	// uneven lengths are zero padded
	f, cp, err := CSD(x[:700], y, &SpectralOptions{Nperseg: 128})
	mustOK(t, err)
	ref := MustOracle(t, "signal.csd", []any{F(x[:700]), F(y)}, KW{"nperseg": 128})
	compareVec(t, "csd uneven f", f, ref.Idx(0).F(), 1e-14)
	compareC(t, "csd uneven", cp, ref.Idx(1).C(), 1e-12)
	// explicit window taps
	taps := Hann(50, false)
	f, p, err := Welch(x, &SpectralOptions{WindowTaps: taps})
	mustOK(t, err)
	ref = MustOracle(t, "signal.welch", []any{F(x)}, KW{"window": F(taps)})
	compareVec(t, "welch taps f", f, ref.Idx(0).F(), 1e-14)
	compareVec(t, "welch taps", p, ref.Idx(1).F(), 1e-12)
	// the default Fs of the zero options
	f, _, err = Welch(x, nil)
	mustOK(t, err)
	if f[1] != 1.0/256 {
		t.Errorf("default frequency step %g", f[1])
	}
}

func TestPeriodogramOracle(t *testing.T) {
	rng := newRNG(82)
	x := testSignal(rng, 300)
	type pc struct {
		o  SpectralOptions
		kw KW
	}
	taps := Hamming(300, true)
	cases := []pc{
		{SpectralOptions{}, KW{}},
		{SpectralOptions{Nfft: 512}, KW{"nfft": 512}},
		{SpectralOptions{Nfft: 300}, KW{"nfft": 300}},
		{SpectralOptions{Nfft: 100}, KW{"nfft": 100}},
		{SpectralOptions{Window: Hann, Fs: 1000}, KW{"window": "hann_symmetric", "fs": 1000.0}},
		{SpectralOptions{Detrend: DetrendLinear, Scaling: ScaleSpectrum}, KW{"detrend": "linear", "scaling": "spectrum"}},
		{SpectralOptions{WindowTaps: taps}, KW{"window": F(taps)}},
	}
	for i, c := range cases {
		o := c.o
		if i == 4 {
			o.Window = func(n int, _ bool) []float64 { return Hann(n, true) }
		}
		f, p, err := Periodogram(x, &o)
		mustOK(t, err)
		ref := MustOracle(t, "signal.periodogram", []any{F(x)}, c.kw)
		compareVec(t, "periodogram f", f, ref.Idx(0).F(), 1e-14)
		compareVec(t, "periodogram", p, ref.Idx(1).F(), 1e-12)
	}
	_, _, err := Periodogram(nil, nil)
	mustErr(t, "empty", err)
	_, _, err = Periodogram(x, &SpectralOptions{WindowTaps: taps[:10]})
	mustErr(t, "taps size", err)
}

func TestSpectrogramOracle(t *testing.T) {
	rng := newRNG(83)
	x := testSignal(rng, 1000)
	modes := map[SpectrogramMode]string{SpecPSD: "psd", SpecMagnitude: "magnitude", SpecAngle: "angle", SpecPhase: "phase"}
	for mode, name := range modes {
		for _, c := range []struct {
			o  SpectralOptions
			kw KW
		}{
			{SpectralOptions{}, KW{}},
			{SpectralOptions{Nperseg: 128, Noverlap: Overlap(100), Fs: 1000}, KW{"nperseg": 128, "noverlap": 100, "fs": 1000.0}},
			{SpectralOptions{Nperseg: 65, Scaling: ScaleSpectrum, Detrend: DetrendLinear}, KW{"nperseg": 65, "scaling": "spectrum", "detrend": "linear"}},
			{SpectralOptions{Nperseg: 64, TwoSided: true}, KW{"nperseg": 64, "return_onesided": false}},
		} {
			o := c.o
			f, tt, s, err := Spectrogram(x, &o, mode)
			mustOK(t, err)
			kw := KW{"mode": name}
			for k, v := range c.kw {
				kw[k] = v
			}
			ref := MustOracle(t, "signal.spectrogram", []any{F(x)}, kw)
			compareVec(t, "spectrogram f", f, ref.Idx(0).F(), 1e-14)
			compareVec(t, "spectrogram t", tt, ref.Idx(1).F(), 1e-14)
			var flat []float64
			for _, row := range s {
				flat = append(flat, row...)
			}
			if mode == SpecAngle || mode == SpecPhase {
				// The phase of a bin that is zero up to rounding is arbitrary
				// (and +-pi for a real one): compare modulo 2*pi where the
				// magnitude is meaningful.
				mkw := KW{}
				for k, v := range kw {
					mkw[k] = v
				}
				mkw["mode"] = "magnitude"
				mag := MustOracle(t, "signal.spectrogram", []any{F(x)}, mkw).Idx(2).F()
				want := ref.Idx(2).F()
				big := 0.0
				for _, m := range mag {
					big = math.Max(big, m)
				}
				worst := 0.0
				for i := range flat {
					if mag[i] < 1e-6*big {
						continue
					}
					d := flat[i] - want[i]
					d -= 2 * math.Pi * math.Round(d/(2*math.Pi))
					worst = math.Max(worst, math.Abs(d))
				}
				if worst > 1e-9 {
					t.Errorf("spectrogram %s: phase differs by %g", name, worst)
				}
				continue
			}
			compareVec(t, "spectrogram "+name, flat, ref.Idx(2).F(), 1e-12)
		}
	}
	_, _, _, err := Spectrogram(x, nil, SpectrogramMode(9))
	mustErr(t, "mode", err)
	_, _, _, err = Spectrogram(nil, nil, SpecPSD)
	mustErr(t, "empty", err)
	_, _, _, err = Spectrogram(x, &SpectralOptions{Nperseg: 64, Noverlap: Overlap(64)}, SpecPSD)
	mustErr(t, "overlap", err)
}

func TestUnwrapOracle(t *testing.T) {
	p := []float64{0, 3, -3, 0.1, 3.1, 6, -6, 0, math.Pi, -math.Pi, 2 * math.Pi}
	got := unwrap(p)
	o := MustOracle(t, "np.unwrap", []any{F(p)}, nil)
	compareVec(t, "unwrap", got, o.F(), 1e-14)
}

func TestSTFTOracle(t *testing.T) {
	rng := newRNG(84)
	x := testSignal(rng, 1000)
	flatten := func(z [][]complex128) []complex128 {
		var out []complex128
		for _, r := range z {
			out = append(out, r...)
		}
		return out
	}
	for _, c := range []struct {
		o  STFTOptions
		kw KW
	}{
		{STFTOptions{}, KW{}},
		{STFTOptions{SpectralOptions: SpectralOptions{Nperseg: 100, Noverlap: Overlap(75), Fs: 8}}, KW{"nperseg": 100, "noverlap": 75, "fs": 8.0}},
		{STFTOptions{SpectralOptions: SpectralOptions{Nperseg: 64, Scaling: ScaleDensity}}, KW{"nperseg": 64, "scaling": "psd"}},
		{STFTOptions{Boundary: "even", SpectralOptions: SpectralOptions{Nperseg: 64}}, KW{"nperseg": 64, "boundary": "even"}},
		{STFTOptions{Boundary: "odd", SpectralOptions: SpectralOptions{Nperseg: 64}}, KW{"nperseg": 64, "boundary": "odd"}},
		{STFTOptions{Boundary: "constant", NoPad: true, SpectralOptions: SpectralOptions{Nperseg: 63}}, KW{"nperseg": 63, "boundary": "constant", "padded": false}},
		{STFTOptions{NoBoundary: true, SpectralOptions: SpectralOptions{Nperseg: 64, Detrend: DetrendLinear}}, KW{"nperseg": 64, "boundary": nil, "detrend": "linear"}},
		{STFTOptions{SpectralOptions: SpectralOptions{Nperseg: 64, TwoSided: true}}, KW{"nperseg": 64, "return_onesided": false}},
	} {
		o := c.o
		f, tt, z, err := STFT(x, &o)
		mustOK(t, err)
		ref := MustOracle(t, "signal.stft", []any{F(x)}, c.kw)
		compareVec(t, "stft f", f, ref.Idx(0).F(), 1e-14)
		compareVec(t, "stft t", tt, ref.Idx(1).F(), 1e-14)
		compareC(t, "stft", flatten(z), ref.Idx(2).C(), 1e-12)

		// and back
		io := ISTFTOptions{SpectralOptions: o.SpectralOptions, TwoSidedInput: o.TwoSided, NoBoundary: o.NoBoundary}
		if o.Boundary != "" && o.Boundary != "zeros" || o.NoBoundary {
			// reconstruction is exact only with the zero boundary; compare to SciPy instead
		}
		_, xr, err := ISTFT(z, &io)
		mustOK(t, err)
		ikw := KW{"boundary": !o.NoBoundary}
		for k, v := range c.kw {
			if k == "nperseg" || k == "noverlap" || k == "fs" || k == "scaling" {
				ikw[k] = v
			}
		}
		if o.TwoSided {
			ikw["input_onesided"] = false
		}
		ref2 := MustOracle(t, "signal.istft", []any{C2(z)}, ikw)
		want := ref2.Idx(1).F()
		if o.TwoSided { // SciPy keeps a complex result; the real part is ours
			for _, v := range ref2.Idx(1).C() {
				want = append(want, real(v))
			}
		}
		compareVec(t, "istft", xr, want, 1e-10)
	}
	// a zero padded FFT length in the inverse
	_, _, z0, _ := STFT(x, nil)
	_, xr512, err := ISTFT(z0, &ISTFTOptions{SpectralOptions: SpectralOptions{Nperseg: 256, Nfft: 512}})
	mustOK(t, err)
	ref3 := MustOracle(t, "signal.istft", []any{C2(z0)}, KW{"nperseg": 256, "nfft": 512})
	compareVec(t, "istft nfft", xr512, ref3.Idx(1).F(), 1e-10)
	if g, err := Detrend([]float64{1, 2, 4}, DetrendDefault, nil); err != nil || len(g) != 3 {
		t.Error("default detrend is linear")
	}
	// perfect reconstruction with the default parameters
	_, _, z, err := STFT(x, nil)
	mustOK(t, err)
	tt, xr, err := ISTFT(z, nil)
	mustOK(t, err)
	compareVec(t, "istft roundtrip", xr[:len(x)], x, 1e-12)
	if len(tt) != len(xr) || tt[1] != 1 {
		t.Error("istft time axis")
	}

	for i, f := range []func() error{
		func() error { _, _, _, e := STFT(nil, nil); return e },
		func() error { _, _, _, e := STFT(x, &STFTOptions{Boundary: "bogus"}); return e },
		func() error {
			_, _, _, e := STFT(x, &STFTOptions{SpectralOptions: SpectralOptions{Nperseg: -1}})
			return e
		},
		func() error { _, _, e := ISTFT(nil, nil); return e },
		func() error { _, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Fs: -1}}); return e },
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Nperseg: -3}})
			return e
		},
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Nperseg: 256, Nfft: 100}})
			return e
		},
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Noverlap: Overlap(300)}})
			return e
		},
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{WindowTaps: make([]float64, 100)}})
			return e
		},
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Window: func(int, bool) []float64 { return nil }}})
			return e
		},
		func() error {
			_, _, e := ISTFT(z, &ISTFTOptions{SpectralOptions: SpectralOptions{Scaling: Scaling(5)}})
			return e
		},
	} {
		mustErr(t, "stft error "+string(rune('a'+i)), f())
	}
}

func TestCOLAOracle(t *testing.T) {
	for _, c := range []struct {
		name         string
		w            Window
		nperseg, ovl int
	}{
		{"hann", Hann, 64, 32}, {"hann", Hann, 64, 16}, {"hamming", Hamming, 50, 25}, {"boxcar", Boxcar, 40, 10},
		{"hann", Hann, 33, 11}, {"blackman", Blackman, 64, 48},
	} {
		win := c.w(c.nperseg, false)
		got, err := CheckCOLA(win, c.nperseg, c.ovl, 1e-10)
		mustOK(t, err)
		o := MustOracle(t, "signal.check_COLA", []any{c.name, c.nperseg, c.ovl}, nil)
		if got != o.Bool() {
			t.Errorf("COLA %s %d %d: %v vs %v", c.name, c.nperseg, c.ovl, got, o.Bool())
		}
		got, err = CheckNOLA(win, c.nperseg, c.ovl, 1e-10)
		mustOK(t, err)
		o = MustOracle(t, "signal.check_NOLA", []any{c.name, c.nperseg, c.ovl}, nil)
		if got != o.Bool() {
			t.Errorf("NOLA %s %d %d: %v vs %v", c.name, c.nperseg, c.ovl, got, o.Bool())
		}
	}
	w := Hann(8, false)
	for i, f := range []func() error{
		func() error { _, e := CheckCOLA(w, 0, 0, 1e-10); return e },
		func() error { _, e := CheckCOLA(w, 8, 8, 1e-10); return e },
		func() error { _, e := CheckCOLA(w, 8, -1, 1e-10); return e },
		func() error { _, e := CheckCOLA(w, 9, 1, 1e-10); return e },
		func() error { _, e := CheckNOLA(w, 8, -1, 1e-10); return e },
		func() error { _, e := CheckNOLA(w, 7, 1, 1e-10); return e },
	} {
		mustErr(t, "cola error "+string(rune('a'+i)), f())
	}
}

func TestLombscargleOracle(t *testing.T) {
	rng := newRNG(85)
	n := 120
	x := make([]float64, n)
	y := make([]float64, n)
	w := make([]float64, n)
	for i := range x {
		x[i] = float64(i)*0.1 + 0.05*rng.Float64()
		y[i] = 2*math.Sin(2.5*x[i]) + 0.3*rng.Float64() + 1
		w[i] = 0.5 + rng.Float64()
	}
	freqs := linspace(0.1, 10, 80, true)
	for _, c := range []struct {
		o  LombOptions
		kw KW
	}{
		{LombOptions{}, KW{}},
		{LombOptions{Normalize: true}, KW{"normalize": true}},
		{LombOptions{FloatingMean: true}, KW{"floating_mean": true}},
		{LombOptions{Weights: w, FloatingMean: true, Normalize: true}, KW{"weights": F(w), "floating_mean": true, "normalize": true}},
	} {
		o := c.o
		got, err := Lombscargle(x, y, freqs, &o)
		mustOK(t, err)
		ref := MustOracle(t, "signal.lombscargle", []any{F(x), F(y), F(freqs)}, c.kw)
		compareVec(t, "lombscargle", got, ref.F(), 1e-11)
	}
	amp, err := LombscargleAmplitude(x, y, freqs, &LombOptions{FloatingMean: true})
	mustOK(t, err)
	ref := MustOracle(t, "signal.lombscargle", []any{F(x), F(y), F(freqs)}, KW{"normalize": "amplitude", "floating_mean": true})
	compareC(t, "lombscargle amplitude", amp, ref.C(), 1e-11)
	for i, f := range []func() error{
		func() error { _, e := Lombscargle(nil, nil, freqs, nil); return e },
		func() error { _, e := Lombscargle(x, y[:5], freqs, nil); return e },
		func() error { _, e := Lombscargle(x, y, nil, nil); return e },
		func() error { _, e := Lombscargle(x, y, freqs, &LombOptions{Weights: w[:3]}); return e },
		func() error { _, e := Lombscargle(x, y, freqs, &LombOptions{Weights: make([]float64, n)}); return e },
		func() error {
			neg := append([]float64(nil), w...)
			neg[0] = -1
			_, e := Lombscargle(x, y, freqs, &LombOptions{Weights: neg})
			return e
		},
		func() error { _, e := LombscargleAmplitude(nil, nil, freqs, nil); return e },
	} {
		mustErr(t, "lomb error "+string(rune('a'+i)), f())
	}
}

func TestSpectralOptionErrors(t *testing.T) {
	x := make([]float64, 100)
	for i := range x {
		x[i] = float64(i % 7)
	}
	for i, o := range []SpectralOptions{
		{Fs: -1}, {Nperseg: -4}, {WindowTaps: make([]float64, 200)}, {WindowTaps: make([]float64, 50), Nperseg: 40},
		{Nperseg: 32, Nfft: 16}, {Nperseg: 32, Noverlap: Overlap(40)}, {Nperseg: 32, Noverlap: Overlap(-2)},
		{Detrend: DetrendType(9)}, {Scaling: Scaling(9)}, {Window: func(int, bool) []float64 { return nil }},
		{Average: Average(5), Nperseg: 16},
	} {
		o := o
		if _, _, err := Welch(x, &o); err == nil {
			t.Errorf("welch option error %d: expected an error", i)
		}
	}
	_, _, err := Welch(nil, nil)
	mustErr(t, "empty", err)
	_, _, err = CSD(nil, x, nil)
	mustErr(t, "csd empty", err)
	_, _, err = CSD(x, x, &SpectralOptions{Fs: -1})
	mustErr(t, "csd fs", err)
	_, _, err = CSD(x, x, &SpectralOptions{Average: Average(5), Nperseg: 16})
	mustErr(t, "csd average", err)
	_, _, err = Coherence(x, nil, nil)
	mustErr(t, "coherence y", err)
	_, _, err = Coherence(nil, x, nil)
	mustErr(t, "coherence x", err)
	_, _, err = Coherence(x, x, &SpectralOptions{Fs: -1})
	mustErr(t, "coherence fs", err)
	if medianBias(2) != 1 || median([]float64{3, 1, 2, 4}) != 2.5 {
		t.Error("median helpers")
	}
}

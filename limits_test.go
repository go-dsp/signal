package signal

import (
	"math"
	"testing"
)

func TestAllocationLimits(t *testing.T) {
	huge := 1 << 30 // above every limit, and still an int on 32-bit systems
	x := make([]float64, 100)
	for i, f := range []func() error{
		func() error { _, e := ButtAP(huge); return e },
		func() error { _, e := Cheb1AP(huge, 1); return e },
		func() error { _, e := Cheb2AP(huge, 40); return e },
		func() error { _, e := EllipAP(huge, 1, 40); return e },
		func() error { _, e := BesselAP(huge, BesselPhase); return e },
		func() error { _, e := BesselAP(151, BesselPhase); return e },
		func() error { _, e := Firwin(huge, []float64{0.3}, nil); return e },
		func() error { _, e := Firwin2(huge, []float64{0, 1}, []float64{1, 0}, nil); return e },
		func() error {
			_, e := Firwin2(21, []float64{0, 1}, []float64{1, 0}, &Firwin2Options{Nfreqs: huge})
			return e
		},
		func() error { _, e := Firls(1<<20+1, []float64{0, 1}, []float64{1, 1}, nil, 0); return e },
		func() error { _, e := Remez(huge, []float64{0, 0.1, 0.2, 0.5}, []float64{1, 0}, nil); return e },
		func() error {
			_, e := Remez(21, []float64{0, 0.1, 0.2, 0.5}, []float64{1, 0}, &RemezOptions{GridDensity: huge})
			return e
		},
		func() error { _, _, e := Freqz([]float64{1}, []float64{1}, &FreqOptions{N: huge}); return e },
		func() error { _, _, e := Welch(x, &SpectralOptions{Nfft: huge}); return e },
		func() error { _, e := ResamplePoly(x, huge, 1, nil); return e },
		func() error { _, e := ResamplePoly(x, 1, huge, nil); return e },
		func() error { _, e := Upfirdn([]float64{1}, x, huge, 1, "", 0); return e },
		func() error { _, e := Resample(x, huge, nil); return e },
		func() error { _, e := Hilbert(x, huge); return e },
		func() error { _, e := Decimate(x, 2, &DecimateOptions{N: huge}); return e },
	} {
		mustErr(t, "limit "+string(rune('a'+i)), f())
	}
}

// TestResamplePolyPostPad finds factor/filter combinations that need the
// extra zeros after the filter and checks the output length contract.
func TestResamplePolyPostPad(t *testing.T) {
	hits := 0
	for nin := 1; nin <= 9; nin++ {
		for up := 1; up <= 5; up++ {
			for down := 1; down <= 9; down++ {
				for ntaps := 1; ntaps <= 9; ntaps += 2 {
					taps := make([]float64, ntaps)
					for i := range taps {
						taps[i] = 1
					}
					g := gcd(up, down)
					u, d := up/g, down/g
					if u == 1 && d == 1 {
						continue
					}
					y, err := ResamplePoly(make([]float64, nin), up, down, &ResamplePolyOptions{Taps: taps})
					mustOK(t, err)
					want := nin * u
					want = want/d + b2i(want%d != 0)
					if len(y) != want {
						t.Fatalf("nin=%d %d/%d taps=%d: length %d, want %d", nin, up, down, ntaps, len(y), want)
					}
					half := (ntaps - 1) / 2
					prePad := d - half%d
					if upfirdnLen(ntaps+prePad, nin, u, d) < want+(half+prePad)/d {
						hits++
					}
				}
			}
		}
	}
	if hits == 0 {
		t.Log("no combination needed post padding; the loop is a safeguard")
	}
}

func TestUpfirdnSparsePhases(t *testing.T) {
	// up larger than the filter: some polyphase branches have no taps
	h, x := []float64{1, 2}, []float64{1, 2, 3, 4}
	got, err := Upfirdn(h, x, 7, 3, "", 0)
	mustOK(t, err)
	o := MustOracle(t, "signal.upfirdn", []any{F(h), F(x), 7, 3}, nil)
	compareVec(t, "upfirdn sparse", got, o.F(), 1e-14)
}

func FuzzFirwin(f *testing.F) {
	f.Add(21, 0.3, 0.0, 0)
	f.Add(22, 0.9, 0.5, 2)
	f.Fuzz(func(t *testing.T, n int, c1, c2 float64, pz int) {
		if n > 5000 || n < -5 {
			t.Skip()
		}
		cut := []float64{c1}
		if c2 != 0 {
			cut = append(cut, c2)
		}
		h, err := Firwin(n, cut, &FirwinOptions{PassZero: FirPass(pz)})
		if err == nil {
			if len(h) != n {
				t.Fatalf("length %d for %d taps", len(h), n)
			}
			for _, v := range h {
				if math.IsInf(v, 0) {
					t.Fatalf("infinite tap")
				}
			}
		}
	})
}

func FuzzIIRFilter(f *testing.F) {
	f.Add(4, 0.3, 0.0, 0, 0)
	f.Add(7, 0.2, 0.5, 2, 3)
	f.Fuzz(func(t *testing.T, n int, w1, w2 float64, bt, ft int) {
		if n > 40 || n < -2 {
			t.Skip()
		}
		wn := []float64{w1}
		if w2 != 0 {
			wn = append(wn, w2)
		}
		z, err := IIRFilter(n, wn, BandType(bt), FilterType(ft), &IIROptions{Rp: 1, Rs: 40})
		if err == nil && (len(z.P) < n) {
			t.Fatalf("%d poles for order %d", len(z.P), n)
		}
	})
}

func FuzzWelchOptions(f *testing.F) {
	f.Add(200, 64, 10, 0, 0)
	f.Add(50, 70, -1, 100, 5)
	f.Fuzz(func(t *testing.T, n, nperseg, noverlap, nfft, det int) {
		if n < 0 || n > 3000 || nfft > 4096 {
			t.Skip()
		}
		x := make([]float64, n)
		for i := range x {
			x[i] = math.Sin(float64(i))
		}
		o := &SpectralOptions{Nperseg: nperseg, Noverlap: Overlap(noverlap), Nfft: nfft, Detrend: DetrendType(det)}
		f, p, err := Welch(x, o)
		if err == nil && len(f) != len(p) {
			t.Fatal("frequency and power lengths differ")
		}
	})
}

func FuzzFindPeaks(f *testing.F) {
	f.Add([]byte{1, 3, 2, 5, 5, 2, 8, 1}, 2.0, 1.0)
	f.Fuzz(func(t *testing.T, data []byte, dist, wlen float64) {
		x := make([]float64, len(data))
		for i, b := range data {
			x[i] = float64(b)
		}
		peaks, _, err := FindPeaks(x, &FindPeaksOptions{Distance: dist, Prominence: AtLeast(0), Wlen: wlen, Width: AtLeast(0)})
		if err == nil {
			for _, p := range peaks {
				if p < 1 || p >= len(x)-1 {
					t.Fatalf("peak %d outside the interior", p)
				}
			}
		}
	})
}

func FuzzConvolve(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{4, 5}, 0, 0)
	f.Fuzz(func(t *testing.T, a, b []byte, mode, method int) {
		x, h := make([]float64, len(a)), make([]float64, len(b))
		for i, v := range a {
			x[i] = float64(v)
		}
		for i, v := range b {
			h[i] = float64(v)
		}
		got, err := Convolve(x, h, ConvMode(mode), ConvMethod(method))
		if err == nil && len(a) > 0 && len(b) > 0 && ConvMode(mode) == ConvFull && len(got) != len(a)+len(b)-1 {
			t.Fatalf("full convolution length %d", len(got))
		}
	})
}

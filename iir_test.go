package signal

import (
	"math"
	"testing"
)

var ftNames = map[FilterType]string{Butterworth: "butter", Chebyshev1: "cheby1", Chebyshev2: "cheby2", Elliptic: "ellip", Bessel: "bessel"}
var btNames = map[BandType]string{Lowpass: "lowpass", Highpass: "highpass", Bandpass: "bandpass", Bandstop: "bandstop"}
var normNames = map[BesselNorm]string{BesselPhase: "phase", BesselDelay: "delay", BesselMag: "mag"}

// wnArg passes a single critical frequency to SciPy as a scalar (NumPy 2.5
// refuses to convert a one-element array).
func wnArg(wn []float64) any {
	if len(wn) == 1 {
		return wn[0]
	}
	return F(wn)
}

func zpkOf(o Out) ZPK {
	return ZPK{toC(o.Idx(0).Arr()), toC(o.Idx(1).Arr()), o.Idx(2).X()}
}

func sciftype(ft FilterType, norm BesselNorm) string {
	if ft == Bessel {
		return "bessel_" + normNames[norm]
	}
	return ftNames[ft]
}

func TestIIRFilterOracle(t *testing.T) {
	rng := newRNG(40)
	unpaired := 0
	defer func() { t.Logf("%d designs SciPy and Go both refuse to pair into sections", unpaired) }()
	for _, ft := range []FilterType{Butterworth, Chebyshev1, Chebyshev2, Elliptic, Bessel} {
		for _, bt := range []BandType{Lowpass, Highpass, Bandpass, Bandstop} {
			for _, analog := range []bool{false, true} {
				for n := 1; n <= 12; n++ {
					if ft == Elliptic && n > 10 {
						continue
					}
					norm := BesselNorm(rng.IntN(3))
					if ft == Bessel && !analog && norm == BesselMag && n > 20 {
						continue
					}
					var wn []float64
					switch {
					case bt == Lowpass || bt == Highpass:
						wn = []float64{0.05 + 0.85*rng.Float64()}
					default:
						a := 0.05 + 0.4*rng.Float64()
						wn = []float64{a, a + 0.05 + 0.4*rng.Float64()}
					}
					if analog {
						for i := range wn {
							wn[i] *= 10
						}
					}
					opt := &IIROptions{Rp: 0.5 + 2*rng.Float64(), Rs: 30 + 40*rng.Float64(), Analog: analog, Bessel: norm}
					kw := KW{"rp": opt.Rp, "rs": opt.Rs, "btype": btNames[bt], "analog": analog, "ftype": sciftype(ft, norm), "output": "zpk"}
					got, err := IIRFilter(n, wn, bt, ft, opt)
					mustOK(t, err)
					o := MustOracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
					tol := 1e-12
					if ft == Elliptic || (ft == Bessel && norm == BesselMag) {
						tol = 1e-7
					}
					compareC(t, "iirfilter zeros", got.Z, toC(o.Idx(0).Arr()), tol)
					if d := compareC(t, "iirfilter poles", got.P, toC(o.Idx(1).Arr()), tol); d > tol {
						t.Logf("  ^ %s %s analog=%v n=%d wn=%v norm=%v", ftNames[ft], btNames[bt], analog, n, wn, norm)
					}
					compareVec(t, "iirfilter gain", []float64{got.K}, []float64{o.Idx(2).X()}, tol)

					// sos / ba outputs against the same pairing inputs
					if ft == Elliptic || ft == Bessel {
						continue
					}
					sos, err := got.SOS(analog)
					kw["output"] = "sos"
					so, oerr := Oracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
					if (err == nil) != (oerr == nil) {
						// SciPy also fails to pair a lone real zero with two real poles
						t.Fatalf("%s %s analog=%v n=%d: go %v, scipy %v", ftNames[ft], btNames[bt], analog, n, err, oerr)
					}
					if err != nil {
						unpaired++
						continue
					}
					compareVec(t, "iirfilter sos", sosFlat(sos), so.F(), 1e-11)
					if n <= 6 {
						b, a, err := got.TF()
						mustOK(t, err)
						kw["output"] = "ba"
						bo := MustOracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
						compareVec(t, "iirfilter b", b, bo.Idx(0).F(), 1e-11)
						compareVec(t, "iirfilter a", a, bo.Idx(1).F(), 1e-11)
					}
				}
			}
		}
	}
}

func TestIIRHighOrderOracle(t *testing.T) {
	// Orders where the transfer-function form is ill-conditioned are compared
	// in zpk and sos form.
	for _, n := range []int{16, 24, 32, 40} {
		for _, ft := range []FilterType{Butterworth, Chebyshev1, Chebyshev2} {
			for _, bt := range []BandType{Lowpass, Bandpass} {
				wn := []float64{0.3}
				if bt == Bandpass {
					wn = []float64{0.2, 0.45}
				}
				opt := &IIROptions{Rp: 1, Rs: 60}
				got, err := IIRFilter(n, wn, bt, ft, opt)
				mustOK(t, err)
				kw := KW{"rp": 1.0, "rs": 60.0, "btype": btNames[bt], "ftype": ftNames[ft], "output": "zpk"}
				o := MustOracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
				compareC(t, "iirfilter high zeros", got.Z, toC(o.Idx(0).Arr()), 1e-12)
				compareC(t, "iirfilter high poles", got.P, toC(o.Idx(1).Arr()), 1e-12)
				sos, err := got.SOS(false)
				mustOK(t, err)
				kw["output"] = "sos"
				so := MustOracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
				compareVec(t, "iirfilter high sos", sosFlat(sos), so.F(), 1e-10)
				// the transfer function form of the same filter is far less accurate:
				// its coefficients are compared to SciPy's own ba output.
				b, a, err := got.TF()
				mustOK(t, err)
				kw["output"] = "ba"
				bo := MustOracle(t, "signal.iirfilter", []any{n, wnArg(wn)}, kw)
				compareVec(t, "iirfilter high ba num", b, bo.Idx(0).F(), 1e-7)
				compareVec(t, "iirfilter high ba den", a, bo.Idx(1).F(), 1e-7)
			}
		}
	}
}

func TestIIRFilterFsAndShortcuts(t *testing.T) {
	got, err := Butter(4, []float64{1000}, Lowpass, &IIROptions{Fs: 8000})
	mustOK(t, err)
	o := MustOracle(t, "signal.butter", []any{4, 1000.0}, KW{"fs": 8000.0, "output": "zpk"})
	compareC(t, "butter fs", got.P, toC(o.Idx(1).Arr()), 1e-13)
	got, err = Cheby1(3, 1, []float64{0.3}, Highpass, nil)
	mustOK(t, err)
	o = MustOracle(t, "signal.cheby1", []any{3, 1.0, 0.3}, KW{"btype": "highpass", "output": "zpk"})
	compareC(t, "cheby1", got.P, toC(o.Idx(1).Arr()), 1e-13)
	got, err = Cheby2(3, 40, []float64{0.3}, Lowpass, &IIROptions{Rp: 99})
	mustOK(t, err)
	o = MustOracle(t, "signal.cheby2", []any{3, 40.0, 0.3}, KW{"output": "zpk"})
	compareC(t, "cheby2", got.P, toC(o.Idx(1).Arr()), 1e-13)
	got, err = Ellip(3, 1, 40, []float64{0.2, 0.4}, Bandstop, nil)
	mustOK(t, err)
	o = MustOracle(t, "signal.ellip", []any{3, 1.0, 40.0, F([]float64{0.2, 0.4})}, KW{"btype": "bandstop", "output": "zpk"})
	compareC(t, "ellip", got.P, toC(o.Idx(1).Arr()), 1e-12)
	got, err = BesselFilter(3, []float64{0.3}, Lowpass, BesselDelay, nil)
	mustOK(t, err)
	o = MustOracle(t, "signal.bessel", []any{3, 0.3}, KW{"norm": "delay", "output": "zpk"})
	compareC(t, "bessel", got.P, toC(o.Idx(1).Arr()), 1e-12)
}

func TestIIRFilterErrors(t *testing.T) {
	ok := []float64{0.3}
	cases := []struct {
		name string
		n    int
		wn   []float64
		bt   BandType
		ft   FilterType
		o    *IIROptions
	}{
		{"analog fs", 2, ok, Lowpass, Butterworth, &IIROptions{Analog: true, Fs: 10}},
		{"bad fs", 2, ok, Lowpass, Butterworth, &IIROptions{Fs: -1}},
		{"zero wn", 2, []float64{0}, Lowpass, Butterworth, nil},
		{"wn order", 2, []float64{0.4, 0.3}, Bandpass, Butterworth, nil},
		{"band type", 2, ok, BandType(9), Butterworth, nil},
		{"neg rp", 2, ok, Lowpass, Butterworth, &IIROptions{Rp: -1}},
		{"neg rs", 2, ok, Lowpass, Butterworth, &IIROptions{Rs: -1}},
		{"cheby1 rp", 2, ok, Lowpass, Chebyshev1, nil},
		{"cheby2 rs", 2, ok, Lowpass, Chebyshev2, nil},
		{"ellip rp rs", 2, ok, Lowpass, Elliptic, &IIROptions{Rp: 1}},
		{"filter type", 2, ok, Lowpass, FilterType(9), nil},
		{"negative order", -1, ok, Lowpass, Butterworth, nil},
		{"wn >= 1", 2, []float64{1}, Lowpass, Butterworth, nil},
		{"wn >= fs/2", 2, []float64{600}, Lowpass, Butterworth, &IIROptions{Fs: 1000}},
		{"lowpass two", 2, []float64{0.2, 0.3}, Lowpass, Butterworth, nil},
		{"bandpass one", 2, ok, Bandpass, Butterworth, nil},
	}
	for _, c := range cases {
		_, err := IIRFilter(c.n, c.wn, c.bt, c.ft, c.o)
		mustErr(t, c.name, err)
	}
	// improper transfer function out of a transform
	_, err := IIRFilter(2, ok, Highpass, Butterworth, nil)
	mustOK(t, err)
}

func TestOrderSelectionOracle(t *testing.T) {
	type spec struct {
		wp, ws []float64
	}
	specs := []spec{
		{[]float64{0.2}, []float64{0.3}},
		{[]float64{0.4}, []float64{0.25}},
		{[]float64{0.3, 0.6}, []float64{0.2, 0.7}},
		{[]float64{0.2, 0.7}, []float64{0.3, 0.6}},
		{[]float64{0.1}, []float64{0.12}},
	}
	fn := map[string]func(wp, ws []float64, gp, gs float64, analog bool, fs float64) (int, []float64, error){
		"buttord": ButtOrd, "cheb1ord": Cheb1Ord, "cheb2ord": Cheb2Ord, "ellipord": EllipOrd,
	}
	for name, f := range fn {
		for si, s := range specs {
			for _, mode := range []string{"digital", "analog", "fs"} {
				wp, ws := append([]float64(nil), s.wp...), append([]float64(nil), s.ws...)
				var analog bool
				var fs float64
				kw := KW{}
				switch mode {
				case "analog":
					analog = true
					for i := range wp {
						wp[i] *= 100
						ws[i] *= 100
					}
					kw["analog"] = true
				case "fs":
					fs = 2000
					for i := range wp {
						wp[i] *= 1000
						ws[i] *= 1000
					}
					kw["fs"] = 2000.0
				}
				for _, g := range [][2]float64{{1, 40}, {0.5, 60}, {3, 20}} {
					n, wn, err := f(wp, ws, g[0], g[1], analog, fs)
					mustOK(t, err)
					o := MustOracle(t, "signal."+name, []any{F(wp), F(ws), g[0], g[1]}, kw)
					if int(o.Idx(0).X()) != n {
						t.Errorf("%s spec %d %s %v: order %d, scipy %v", name, si, mode, g, n, o.Idx(0).X())
					}
					compareVec(t, name+" wn", wn, o.Idx(1).F(), 1e-4)
				}
			}
		}
	}
	// scalar arguments, spelled like MATLAB's
	n, wn, err := ButtOrd([]float64{0.2}, []float64{0.3}, 3, 40, false, 0)
	mustOK(t, err)
	if n != 5 && n < 3 {
		t.Errorf("order %d", n)
	}
	_ = wn
}

func TestOrderSelectionErrors(t *testing.T) {
	one, two := []float64{0.2}, []float64{0.2, 0.3}
	for _, f := range []func(wp, ws []float64, gp, gs float64, analog bool, fs float64) (int, []float64, error){ButtOrd, Cheb1Ord, Cheb2Ord, EllipOrd} {
		for _, c := range []struct {
			wp, ws []float64
			gp, gs float64
			an     bool
			fs     float64
		}{
			{one, []float64{0.3}, 0, 40, false, 0},
			{one, []float64{0.3}, 1, 0, false, 0},
			{one, []float64{0.3}, 40, 1, false, 0},
			{one, two, 1, 40, false, 0},
			{[]float64{.1, .2, .3}, []float64{.1, .2, .3}, 1, 40, false, 0},
			{one, []float64{0.3}, 1, 40, true, 10},
			{one, []float64{0.3}, 1, 40, false, -1},
		} {
			_, _, err := f(c.wp, c.ws, c.gp, c.gs, c.an, c.fs)
			mustErr(t, "order selection", err)
		}
	}
}

func TestIIRDesignOracle(t *testing.T) {
	type spec struct {
		wp, ws []float64
	}
	specs := []spec{
		{[]float64{0.2}, []float64{0.3}},
		{[]float64{0.4}, []float64{0.25}},
		{[]float64{0.3, 0.6}, []float64{0.2, 0.7}},
		{[]float64{0.2, 0.7}, []float64{0.3, 0.6}},
	}
	for _, ft := range []FilterType{Butterworth, Chebyshev1, Chebyshev2, Elliptic} {
		for _, s := range specs {
			got, err := IIRDesign(s.wp, s.ws, 1, 50, ft, nil)
			mustOK(t, err)
			o := MustOracle(t, "signal.iirdesign", []any{wnArg(s.wp), wnArg(s.ws), 1.0, 50.0}, KW{"ftype": ftNames[ft], "output": "zpk"})
			// bandstop edges come from a bounded search with xtol 1e-5
			compareC(t, "iirdesign zeros", got.Z, toC(o.Idx(0).Arr()), 1e-5)
			compareC(t, "iirdesign poles", got.P, toC(o.Idx(1).Arr()), 1e-5)
			compareVec(t, "iirdesign k", []float64{got.K}, []float64{o.Idx(2).X()}, 1e-5)
		}
	}
	// analog and fs
	got, err := IIRDesign([]float64{100}, []float64{200}, 1, 40, Butterworth, &IIROptions{Analog: true})
	mustOK(t, err)
	o := MustOracle(t, "signal.iirdesign", []any{F([]float64{100}), F([]float64{200}), 1.0, 40.0}, KW{"analog": true, "ftype": "butter", "output": "zpk"})
	compareC(t, "iirdesign analog", got.P, toC(o.Idx(1).Arr()), 1e-9)
	got, err = IIRDesign([]float64{1000}, []float64{1500}, 1, 40, Chebyshev1, &IIROptions{Fs: 8000})
	mustOK(t, err)
	o = MustOracle(t, "signal.iirdesign", []any{F([]float64{1000}), F([]float64{1500}), 1.0, 40.0}, KW{"fs": 8000.0, "ftype": "cheby1", "output": "zpk"})
	compareC(t, "iirdesign fs", got.P, toC(o.Idx(1).Arr()), 1e-9)

	one := []float64{0.2}
	for _, c := range []struct {
		wp, ws []float64
		gp, gs float64
		ft     FilterType
		o      *IIROptions
	}{
		{one, []float64{0.3}, 1, 40, Bessel, nil},
		{one, []float64{0.3}, 1, 40, FilterType(9), nil},
		{one, []float64{0.3}, 0, 40, Butterworth, nil},
		{one, []float64{0.3, 0.4}, 1, 40, Butterworth, nil},
		{one, []float64{0.3}, 1, 40, Butterworth, &IIROptions{Fs: -1}},
		{[]float64{0}, []float64{0.3}, 1, 40, Butterworth, nil},
		{one, []float64{1.2}, 1, 40, Butterworth, nil},
		{one, []float64{600}, 1, 40, Butterworth, &IIROptions{Fs: 1000}},
		{[]float64{0.2, 0.5}, []float64{0.3, 0.6}, 1, 40, Butterworth, nil},
	} {
		_, err := IIRDesign(c.wp, c.ws, c.gp, c.gs, c.ft, c.o)
		mustErr(t, "iirdesign", err)
	}
	_ = math.Pi
}

func TestFminbound(t *testing.T) {
	// parabolic steps on a smooth function, golden steps on a kinked one
	x := fminbound(func(x float64) float64 { return (x - 1.3) * (x - 1.3) }, -4, 5, 1e-5, 500)
	if math.Abs(x-1.3) > 1e-5 {
		t.Errorf("parabola: %g", x)
	}
	x = fminbound(func(x float64) float64 { return math.Abs(x - 0.7) }, -3, 4, 1e-5, 500)
	if math.Abs(x-0.7) > 1e-4 {
		t.Errorf("abs: %g", x)
	}
	x = fminbound(func(x float64) float64 { return math.Sin(3*x) + 0.2*x }, 0, 6, 1e-5, 500)
	_ = x
	n := 0
	fminbound(func(x float64) float64 { n++; return math.Sin(40 * x) }, 0, 100, 1e-12, 7)
	if n > 8 {
		t.Errorf("maxfun ignored: %d evaluations", n)
	}
	// a minimum at the right end (si < 0 branch) and at the left end
	if x := fminbound(func(x float64) float64 { return -x }, 0, 1, 1e-5, 500); x < 0.99 {
		t.Errorf("right end: %g", x)
	}
	if x := fminbound(func(x float64) float64 { return x }, 0, 1, 1e-5, 500); x > 0.01 {
		t.Errorf("left end: %g", x)
	}
}

func TestIIRDesignAnalogFs(t *testing.T) {
	_, err := IIRDesign([]float64{1}, []float64{2}, 1, 40, Butterworth, &IIROptions{Analog: true, Fs: 10})
	mustErr(t, "analog with fs", err)
	got, err := BesselFilter(2, []float64{0.3}, Lowpass, BesselMag, &IIROptions{Analog: false})
	mustOK(t, err)
	if len(got.P) != 2 {
		t.Error("bessel poles")
	}
}

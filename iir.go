package signal

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/go-numerics/special"
)

// FilterType is the family of an IIR design.
type FilterType int

const (
	Butterworth FilterType = iota
	Chebyshev1
	Chebyshev2
	Elliptic
	Bessel
)

// BandType is the band layout of a filter.
type BandType int

const (
	Lowpass BandType = iota
	Highpass
	Bandpass
	Bandstop
)

// IIROptions holds the optional parameters of an IIR design.
type IIROptions struct {
	// Rp is the maximum passband ripple in dB (Chebyshev I, elliptic).
	Rp float64
	// Rs is the minimum stopband attenuation in dB (Chebyshev II, elliptic).
	Rs float64
	// Analog designs an analog filter, with Wn in rad/s.
	Analog bool
	// Fs is the sampling frequency of a digital design: Wn is then in the
	// same units. Zero means Wn is normalised to the Nyquist frequency (1).
	Fs float64
	// Bessel selects the Bessel normalisation (default BesselPhase).
	Bessel BesselNorm
}

// TF returns the transfer-function coefficients of the filter.
func (z ZPK) TF() (b, a []float64, err error) { return ZPK2TF(z) }

// SOS returns the filter as second-order sections with SciPy's default
// pairing: nearest for digital filters, minimal for analog ones.
func (z ZPK) SOS(analog bool) (SOS, error) {
	if analog {
		return ZPK2SOS(z, PairMinimal, true)
	}
	return ZPK2SOS(z, PairNearest, false)
}

func validFs(fs float64) error {
	if !(fs > 0) || math.IsInf(fs, 0) {
		return errors.New("signal: fs must be a positive finite number")
	}
	return nil
}

// IIRFilter designs an order-n IIR filter of the given family and band
// layout (scipy.signal.iirfilter). wn holds one critical frequency for a
// lowpass or highpass filter and two for a bandpass or bandstop filter; for
// Chebyshev I the frequency is the edge of the ripple band, for Chebyshev II
// the start of the stopband. The result is zeros, poles and gain; use TF or
// SOS to convert (prefer SOS above order 8).
func IIRFilter(n int, wn []float64, bt BandType, ft FilterType, o *IIROptions) (ZPK, error) {
	var opt IIROptions
	if o != nil {
		opt = *o
	}
	w := slices.Clone(wn)
	if opt.Fs != 0 {
		if opt.Analog {
			return ZPK{}, errors.New("signal: fs cannot be specified for an analog filter")
		}
		if err := validFs(opt.Fs); err != nil {
			return ZPK{}, err
		}
		for i := range w {
			w[i] /= opt.Fs / 2
		}
	}
	for _, v := range w {
		if !(v > 0) {
			return ZPK{}, errors.New("signal: filter critical frequencies must be greater than 0")
		}
	}
	if len(w) > 1 && !(w[0] < w[1]) {
		return ZPK{}, errors.New("signal: Wn[0] must be less than Wn[1]")
	}
	if bt < Lowpass || bt > Bandstop {
		return ZPK{}, fmt.Errorf("signal: invalid band type %d", bt)
	}
	if opt.Rp < 0 {
		return ZPK{}, errors.New("signal: passband ripple (rp) must be positive")
	}
	if opt.Rs < 0 {
		return ZPK{}, errors.New("signal: stopband attenuation (rs) must be positive")
	}
	var proto ZPK
	var err error
	switch ft {
	case Butterworth:
		proto, err = ButtAP(n)
	case Bessel:
		proto, err = BesselAP(n, opt.Bessel)
	case Chebyshev1:
		if opt.Rp == 0 {
			return ZPK{}, errors.New("signal: passband ripple (rp) must be provided to design a Chebyshev I filter")
		}
		proto, err = Cheb1AP(n, opt.Rp)
	case Chebyshev2:
		if opt.Rs == 0 {
			return ZPK{}, errors.New("signal: stopband attenuation (rs) must be provided to design a Chebyshev II filter")
		}
		proto, err = Cheb2AP(n, opt.Rs)
	case Elliptic:
		if opt.Rp == 0 || opt.Rs == 0 {
			return ZPK{}, errors.New("signal: both rp and rs must be provided to design an elliptic filter")
		}
		proto, err = EllipAP(n, opt.Rp, opt.Rs)
	default:
		return ZPK{}, fmt.Errorf("signal: invalid filter type %d", ft)
	}
	if err != nil {
		return ZPK{}, err
	}
	warped := w
	fsd := 2.0
	if !opt.Analog {
		warped = make([]float64, len(w))
		for i, v := range w {
			if v >= 1 {
				if opt.Fs != 0 {
					return ZPK{}, fmt.Errorf("signal: digital filter critical frequencies must be 0 < Wn < fs/2 (fs=%g -> fs/2=%g)", opt.Fs, opt.Fs/2)
				}
				return ZPK{}, errors.New("signal: digital filter critical frequencies must be 0 < Wn < 1")
			}
			warped[i] = 2 * fsd * math.Tan(math.Pi*v/fsd)
		}
	}
	// The prototypes are proper (no more zeros than poles), the only way the
	// transforms can fail, so their errors are not examined.
	var z ZPK
	switch bt {
	case Lowpass, Highpass:
		if len(w) != 1 {
			return ZPK{}, errors.New("signal: must specify a single critical frequency Wn for lowpass or highpass filter")
		}
		if bt == Lowpass {
			z, _ = LP2LPZPK(proto, warped[0])
		} else {
			z, _ = LP2HPZPK(proto, warped[0])
		}
	default:
		if len(w) != 2 {
			return ZPK{}, errors.New("signal: Wn must specify start and stop frequencies for bandpass or bandstop filter")
		}
		bw := warped[1] - warped[0]
		wo := math.Sqrt(warped[0] * warped[1])
		if bt == Bandpass {
			z, _ = LP2BPZPK(proto, wo, bw)
		} else {
			z, _ = LP2BSZPK(proto, wo, bw)
		}
	}
	if !opt.Analog {
		z, _ = BilinearZPK(z, fsd)
	}
	return z, nil
}

// Butter designs a Butterworth filter (scipy.signal.butter).
func Butter(n int, wn []float64, bt BandType, o *IIROptions) (ZPK, error) {
	return IIRFilter(n, wn, bt, Butterworth, o)
}

// Cheby1 designs a Chebyshev type I filter with rp dB of passband ripple.
func Cheby1(n int, rp float64, wn []float64, bt BandType, o *IIROptions) (ZPK, error) {
	return IIRFilter(n, wn, bt, Chebyshev1, withRipple(o, rp, 0))
}

// Cheby2 designs a Chebyshev type II filter with rs dB of stopband attenuation.
func Cheby2(n int, rs float64, wn []float64, bt BandType, o *IIROptions) (ZPK, error) {
	return IIRFilter(n, wn, bt, Chebyshev2, withRipple(o, 0, rs))
}

// Ellip designs an elliptic (Cauer) filter.
func Ellip(n int, rp, rs float64, wn []float64, bt BandType, o *IIROptions) (ZPK, error) {
	return IIRFilter(n, wn, bt, Elliptic, withRipple(o, rp, rs))
}

// BesselFilter designs a Bessel/Thomson filter with the given normalisation.
func BesselFilter(n int, wn []float64, bt BandType, norm BesselNorm, o *IIROptions) (ZPK, error) {
	var opt IIROptions
	if o != nil {
		opt = *o
	}
	opt.Bessel = norm
	return IIRFilter(n, wn, bt, Bessel, &opt)
}

func withRipple(o *IIROptions, rp, rs float64) *IIROptions {
	var opt IIROptions
	if o != nil {
		opt = *o
	}
	if rp != 0 {
		opt.Rp = rp
	}
	if rs != 0 {
		opt.Rs = rs
	}
	return &opt
}

// ---- order selection --------------------------------------------------------

func validateGpassGstop(gpass, gstop float64) error {
	switch {
	case gpass <= 0:
		return errors.New("signal: gpass should be larger than 0.0")
	case gstop <= 0:
		return errors.New("signal: gstop should be larger than 0.0")
	case gpass > gstop:
		return errors.New("signal: gpass should be smaller than gstop")
	}
	return nil
}

type ordSpec struct {
	passb, stopb []float64
	kind         int // 1 low, 2 high, 3 stop, 4 pass
	analog       bool
	fs           float64
}

func newOrdSpec(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (ordSpec, error) {
	if err := validateGpassGstop(gpass, gstop); err != nil {
		return ordSpec{}, err
	}
	if len(wp) != len(ws) || (len(wp) != 1 && len(wp) != 2) {
		return ordSpec{}, errors.New("signal: wp and ws must have one or two elements each, and the same length")
	}
	if fs != 0 {
		if analog {
			return ordSpec{}, errors.New("signal: fs cannot be specified for an analog filter")
		}
		if err := validFs(fs); err != nil {
			return ordSpec{}, err
		}
	}
	sp := ordSpec{analog: analog, fs: fs, kind: 2*(len(wp)-1) + 1}
	sp.passb, sp.stopb = slices.Clone(wp), slices.Clone(ws)
	if fs != 0 {
		for i := range wp {
			sp.passb[i] = 2 * wp[i] / fs
			sp.stopb[i] = 2 * ws[i] / fs
		}
	}
	if sp.passb[0] >= sp.stopb[0] {
		sp.kind++
	}
	if !analog {
		for i := range sp.passb {
			sp.passb[i] = math.Tan(math.Pi * sp.passb[i] / 2)
			sp.stopb[i] = math.Tan(math.Pi * sp.stopb[i] / 2)
		}
	}
	return sp, nil
}

// bandStopObj is scipy's band_stop_obj: the (negative-of-)order estimate used
// to place the passband edge of a bandstop design.
func bandStopObj(wp float64, ind int, passb, stopb []float64, gpass, gstop float64, kind string) float64 {
	pc := slices.Clone(passb)
	pc[ind] = wp
	nat := math.Inf(1)
	for _, s := range stopb {
		nat = math.Min(nat, math.Abs(s*(pc[0]-pc[1])/(s*s-pc[0]*pc[1])))
	}
	switch kind {
	case "butter":
		gs, gp := math.Pow(10, 0.1*math.Abs(gstop)), math.Pow(10, 0.1*math.Abs(gpass))
		return math.Log10((gs-1)/(gp-1)) / (2 * math.Log10(nat))
	case "cheby":
		gs, gp := math.Pow(10, 0.1*math.Abs(gstop)), math.Pow(10, 0.1*math.Abs(gpass))
		return math.Acosh(math.Sqrt((gs-1)/(gp-1))) / math.Acosh(nat)
	}
	gs, gp := math.Pow(10, 0.1*gstop), math.Pow(10, 0.1*gpass)
	arg1 := math.Sqrt((gp - 1) / (gs - 1))
	arg0 := 1 / nat
	d00, d01 := special.EllipK(arg0*arg0), special.EllipK(1-arg0*arg0)
	d10, d11 := special.EllipK(arg1*arg1), special.EllipK(1-arg1*arg1)
	return d00 * d11 / (d01 * d10)
}

// fminbound minimises f on [x1, x2] by Brent's method (scipy.optimize.fminbound).
func fminbound(f func(float64) float64, x1, x2, xatol float64, maxfun int) float64 {
	sqrtEps := math.Sqrt(2.2e-16)
	golden := 0.5 * (3 - math.Sqrt(5))
	a, b := x1, x2
	fulc := a + golden*(b-a)
	nfc, xf := fulc, fulc
	rat, e := 0.0, 0.0
	x := xf
	fx := f(x)
	num := 1
	ffulc, fnfc := fx, fx
	xm := 0.5 * (a + b)
	tol1 := sqrtEps*math.Abs(xf) + xatol/3
	tol2 := 2 * tol1
	for math.Abs(xf-xm) > tol2-0.5*(b-a) {
		gold := true
		if math.Abs(e) > tol1 {
			gold = false
			r := (xf - nfc) * (fx - ffulc)
			q := (xf - fulc) * (fx - fnfc)
			p := (xf-fulc)*q - (xf-nfc)*r
			q = 2 * (q - r)
			if q > 0 {
				p = -p
			}
			q = math.Abs(q)
			r = e
			e = rat
			if math.Abs(p) < math.Abs(0.5*q*r) && p > q*(a-xf) && p < q*(b-xf) {
				rat = p / q
				x = xf + rat
				if (x-a) < tol2 || (b-x) < tol2 {
					si := 1.0
					if xm-xf < 0 {
						si = -1
					}
					rat = tol1 * si
				}
			} else {
				gold = true
			}
		}
		if gold {
			if xf >= xm {
				e = a - xf
			} else {
				e = b - xf
			}
			rat = golden * e
		}
		si := 1.0
		if rat < 0 {
			si = -1
		}
		x = xf + si*math.Max(math.Abs(rat), tol1)
		fu := f(x)
		num++
		if fu <= fx {
			if x >= xf {
				a = xf
			} else {
				b = xf
			}
			fulc, ffulc = nfc, fnfc
			nfc, fnfc = xf, fx
			xf, fx = x, fu
		} else {
			if x < xf {
				a = x
			} else {
				b = x
			}
			if fu <= fnfc || nfc == xf {
				fulc, ffulc = nfc, fnfc
				nfc, fnfc = x, fu
			} else if fu <= ffulc || fulc == xf || fulc == nfc {
				fulc, ffulc = x, fu
			}
		}
		xm = 0.5 * (a + b)
		tol1 = sqrtEps*math.Abs(xf) + xatol/3
		tol2 = 2 * tol1
		if num >= maxfun {
			break
		}
	}
	return xf
}

// natFreq is scipy's _find_nat_freq: the natural frequency of the lowpass
// prototype, and the (possibly adjusted) passband edges.
func (sp ordSpec) natFreq(gpass, gstop float64, kind string) (float64, []float64) {
	passb, stopb := sp.passb, sp.stopb
	var nat []float64
	switch sp.kind {
	case 1:
		nat = []float64{stopb[0] / passb[0]}
	case 2:
		nat = []float64{passb[0] / stopb[0]}
	case 3:
		obj := func(ind int) func(float64) float64 {
			return func(w float64) float64 { return bandStopObj(w, ind, passb, stopb, gpass, gstop, kind) }
		}
		wp0 := fminbound(obj(0), passb[0], stopb[0]-1e-12, 1e-5, 500)
		wp1 := fminbound(obj(1), stopb[1]+1e-12, passb[1], 1e-5, 500)
		passb = []float64{wp0, wp1}
		for _, s := range stopb {
			nat = append(nat, s*(passb[0]-passb[1])/(s*s-passb[0]*passb[1]))
		}
	default:
		for _, s := range stopb {
			nat = append(nat, (s*s-passb[0]*passb[1])/(s*(passb[0]-passb[1])))
		}
	}
	m := math.Inf(1)
	for _, v := range nat {
		m = math.Min(m, math.Abs(v))
	}
	return m, passb
}

func (sp ordSpec) postprocess(w []float64) []float64 {
	out := make([]float64, len(w))
	for i, v := range w {
		if !sp.analog {
			v = math.Atan(v) * 2 / math.Pi
		}
		if sp.fs != 0 {
			v = v * sp.fs / 2
		}
		out[i] = v
	}
	return out
}

// ButtOrd returns the minimum order and the natural (-3 dB) frequencies of a
// Butterworth filter meeting the passband (wp, gpass dB) and stopband (ws,
// gstop dB) specification (scipy.signal.buttord). wp and ws hold one
// frequency (lowpass/highpass) or two (bandpass/bandstop); with fs == 0 they
// are normalised to the Nyquist frequency.
func ButtOrd(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (int, []float64, error) {
	sp, err := newOrdSpec(wp, ws, gpass, gstop, analog, fs)
	if err != nil {
		return 0, nil, err
	}
	nat, passb := sp.natFreq(gpass, gstop, "butter")
	gs, gp := math.Pow(10, 0.1*math.Abs(gstop)), math.Pow(10, 0.1*math.Abs(gpass))
	ord := int(math.Ceil(math.Log10((gs-1)/(gp-1)) / (2 * math.Log10(nat))))
	w0 := 1.0
	if ord != 0 {
		w0 = math.Pow(gp-1, -1/(2*float64(ord)))
	}
	var wn []float64
	switch sp.kind {
	case 1:
		wn = []float64{w0 * passb[0]}
	case 2:
		wn = []float64{passb[0] / w0}
	case 3:
		discr := math.Sqrt(math.Pow(passb[1]-passb[0], 2) + 4*w0*w0*passb[0]*passb[1])
		wn = []float64{((passb[1] - passb[0]) + discr) / (2 * w0), ((passb[1] - passb[0]) - discr) / (2 * w0)}
		sortAbs(wn)
	default:
		d := passb[1] - passb[0]
		for _, s := range []float64{-w0, w0} {
			wn = append(wn, -s*d/2+math.Sqrt(s*s/4*d*d+passb[0]*passb[1]))
		}
		sortAbs(wn)
	}
	return ord, sp.postprocess(wn), nil
}

func sortAbs(w []float64) {
	for i := range w {
		w[i] = math.Abs(w[i])
	}
	slices.Sort(w)
}

// Cheb1Ord returns the minimum order and the passband edge frequencies of a
// Chebyshev type I filter (scipy.signal.cheb1ord).
func Cheb1Ord(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (int, []float64, error) {
	sp, err := newOrdSpec(wp, ws, gpass, gstop, analog, fs)
	if err != nil {
		return 0, nil, err
	}
	nat, passb := sp.natFreq(gpass, gstop, "cheby")
	gs, gp := math.Pow(10, 0.1*math.Abs(gstop)), math.Pow(10, 0.1*math.Abs(gpass))
	v := math.Acosh(math.Sqrt((gs - 1) / (gp - 1)))
	return int(math.Ceil(v / math.Acosh(nat))), sp.postprocess(passb), nil
}

// Cheb2Ord returns the minimum order and the stopband-start frequencies of a
// Chebyshev type II filter (scipy.signal.cheb2ord).
func Cheb2Ord(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (int, []float64, error) {
	sp, err := newOrdSpec(wp, ws, gpass, gstop, analog, fs)
	if err != nil {
		return 0, nil, err
	}
	nat, passb := sp.natFreq(gpass, gstop, "cheby")
	gs, gp := math.Pow(10, 0.1*math.Abs(gstop)), math.Pow(10, 0.1*math.Abs(gpass))
	v := math.Acosh(math.Sqrt((gs - 1) / (gp - 1)))
	ord := int(math.Ceil(v / math.Acosh(nat)))
	nf := 1 / math.Cosh(1/float64(ord)*v)
	var w []float64
	switch sp.kind {
	case 1:
		w = []float64{passb[0] / nf}
	case 2:
		w = []float64{passb[0] * nf}
	case 3:
		n0 := nf/2*(passb[0]-passb[1]) + math.Sqrt(nf*nf*(passb[1]-passb[0])*(passb[1]-passb[0])/4+passb[1]*passb[0])
		w = []float64{n0, passb[1] * passb[0] / n0}
	default:
		n0 := 1/(2*nf)*(passb[0]-passb[1]) + math.Sqrt((passb[1]-passb[0])*(passb[1]-passb[0])/(4*nf*nf)+passb[1]*passb[0])
		w = []float64{n0, passb[0] * passb[1] / n0}
	}
	return ord, sp.postprocess(w), nil
}

// EllipOrd returns the minimum order and the passband edge frequencies of an
// elliptic filter (scipy.signal.ellipord).
func EllipOrd(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (int, []float64, error) {
	sp, err := newOrdSpec(wp, ws, gpass, gstop, analog, fs)
	if err != nil {
		return 0, nil, err
	}
	nat, passb := sp.natFreq(gpass, gstop, "ellip")
	arg1sq := pow10m1(0.1*gpass) / pow10m1(0.1*gstop)
	arg0 := 1 / nat
	d00, d01 := special.EllipK(arg0*arg0), special.EllipKm1(arg0*arg0)
	d10, d11 := special.EllipK(arg1sq), special.EllipKm1(arg1sq)
	return int(math.Ceil(d00 * d11 / (d01 * d10))), sp.postprocess(passb), nil
}

// IIRDesign designs the lowest-order filter of the given family meeting a
// passband/stopband specification (scipy.signal.iirdesign): wp and ws are the
// passband and stopband edges (one or two each), gpass the maximum passband
// loss and gstop the minimum stopband attenuation in dB. Bessel has no order
// selection.
func IIRDesign(wp, ws []float64, gpass, gstop float64, ft FilterType, o *IIROptions) (ZPK, error) {
	var opt IIROptions
	if o != nil {
		opt = *o
	}
	var ordf func(wp, ws []float64, gpass, gstop float64, analog bool, fs float64) (int, []float64, error)
	switch ft {
	case Butterworth:
		ordf = ButtOrd
	case Chebyshev1:
		ordf = Cheb1Ord
	case Chebyshev2:
		ordf = Cheb2Ord
	case Elliptic:
		ordf = EllipOrd
	case Bessel:
		return ZPK{}, errors.New("signal: bessel does not have order selection; use IIRFilter")
	default:
		return ZPK{}, fmt.Errorf("signal: invalid IIR filter type %d", ft)
	}
	if err := validateGpassGstop(gpass, gstop); err != nil {
		return ZPK{}, err
	}
	if len(wp) != len(ws) || (len(wp) != 1 && len(wp) != 2) {
		return ZPK{}, errors.New("signal: wp and ws must have one or two elements each, and the same length")
	}
	if opt.Fs != 0 {
		if err := validFs(opt.Fs); err != nil {
			return ZPK{}, err
		}
	}
	for i := range wp {
		if wp[i] <= 0 || ws[i] <= 0 {
			return ZPK{}, errors.New("signal: values for wp, ws must be greater than 0")
		}
		if !opt.Analog {
			lim := 1.0
			if opt.Fs != 0 {
				lim = opt.Fs / 2
			}
			if wp[i] >= lim || ws[i] >= lim {
				return ZPK{}, fmt.Errorf("signal: values for wp, ws must be less than %g", lim)
			}
		}
	}
	if len(wp) == 2 && !((ws[0] < wp[0] && wp[1] < ws[1]) || (wp[0] < ws[0] && ws[1] < wp[1])) {
		return ZPK{}, errors.New("signal: passband must lie strictly inside stopband or vice versa")
	}
	// a passband that starts above the stopband edge is high (one edge) or
	// inside it (two edges)
	bt := [2][2]BandType{{Lowpass, Highpass}, {Bandstop, Bandpass}}[len(wp)-1][b2i(wp[0] >= ws[0])]
	n, wn, err := ordf(wp, ws, gpass, gstop, opt.Analog, opt.Fs)
	if err != nil {
		return ZPK{}, err
	}
	opt.Rp, opt.Rs = gpass, gstop
	return IIRFilter(n, wn, bt, ft, &opt)
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// IIRNotch designs a second-order notch (band-reject) filter at frequency w0
// with quality factor q (scipy.signal.iirnotch; MATLAB iirnotch). fs == 0
// means 2, so that w0 is relative to the Nyquist frequency.
func IIRNotch(w0, q, fs float64) (b, a []float64, err error) { return notchPeak(w0, q, fs, true) }

// IIRPeak designs a second-order peaking (band-pass) filter
// (scipy.signal.iirpeak; MATLAB iirpeak).
func IIRPeak(w0, q, fs float64) (b, a []float64, err error) { return notchPeak(w0, q, fs, false) }

func notchPeak(w0, q, fs float64, notch bool) ([]float64, []float64, error) {
	if fs == 0 {
		fs = 2
	}
	if err := validFs(fs); err != nil {
		return nil, nil, err
	}
	w0 = 2 * w0 / fs
	if w0 > 1 || w0 < 0 {
		return nil, nil, errors.New("signal: w0 should be such that 0 < w0 < 1")
	}
	bw := w0 / q * math.Pi
	w0 *= math.Pi
	gain := 1 / (1 + math.Tan(bw/2))
	var b []float64
	if notch {
		b = []float64{gain, gain * -2 * math.Cos(w0), gain}
	} else {
		b = []float64{1 - gain, 0, -(1 - gain)}
	}
	return b, []float64{1, -2 * gain * math.Cos(w0), 2*gain - 1}, nil
}

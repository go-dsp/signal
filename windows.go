package signal

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"

	"github.com/go-fft/fft"
	"github.com/go-numerics/special"
)

// Window generates an n-point window: symmetric when sym is true (for filter
// design), periodic otherwise (for spectral analysis, the DFT-even form of
// get_window's default). A window that cannot be generated for the given
// parameters returns nil.
type Window func(n int, sym bool) []float64

func ones(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 1
	}
	return w
}

// build generates a window of m points by calling f with the extended length
// of the periodic form and truncating, like scipy's _extend/_truncate.
func build(m int, sym bool, f func(m int) []float64) []float64 {
	if m <= 1 {
		return ones(max(m, 0))
	}
	if sym {
		return f(m)
	}
	return f(m + 1)[:m]
}

// Hann is the Hann window (go-fft's, extended to the periodic form).
func Hann(n int, sym bool) []float64 { return build(n, sym, fft.Hann) }

// Hamming is the Hamming window.
func Hamming(n int, sym bool) []float64 { return build(n, sym, fft.Hamming) }

// Blackman is the Blackman window.
func Blackman(n int, sym bool) []float64 { return build(n, sym, fft.Blackman) }

// BlackmanHarris is the 4-term Blackman–Harris window.
func BlackmanHarris(n int, sym bool) []float64 { return build(n, sym, fft.BlackmanHarris) }

// Bartlett is the triangular window with zero end points.
func Bartlett(n int, sym bool) []float64 { return build(n, sym, fft.Bartlett) }

// Boxcar is the rectangular window.
func Boxcar(n int, sym bool) []float64 { return build(n, sym, ones) }

// Triang is the triangular window with nonzero end points.
func Triang(n int, sym bool) []float64 {
	return build(n, sym, func(m int) []float64 {
		w := make([]float64, m)
		for i := 0; i < (m+1)/2; i++ {
			k := float64(i + 1)
			var v float64
			if m%2 == 0 {
				v = (2*k - 1) / float64(m)
			} else {
				v = 2 * k / (float64(m) + 1)
			}
			w[i], w[m-1-i] = v, v
		}
		return w
	})
}

// Parzen is the Parzen (de la Vallée Poussin) window.
func Parzen(n int, sym bool) []float64 {
	return build(n, sym, func(m int) []float64 {
		w := make([]float64, m)
		for i := range w {
			x := math.Abs(-(float64(m)-1)/2 + float64(i))
			if x <= (float64(m)-1)/4 {
				w[i] = 1 - 6*math.Pow(x/(float64(m)/2), 2) + 6*math.Pow(x/(float64(m)/2), 3)
			} else {
				w[i] = 2 * math.Pow(1-x/(float64(m)/2), 3)
			}
		}
		return w
	})
}

// Bohman is the Bohman window.
func Bohman(n int, sym bool) []float64 {
	return build(n, sym, func(m int) []float64 {
		w := make([]float64, m)
		fac := linspace(-1, 1, m, true)
		for i := 1; i < m-1; i++ {
			f := math.Abs(fac[i])
			w[i] = (1-f)*math.Cos(math.Pi*f) + 1/math.Pi*math.Sin(math.Pi*f)
		}
		return w
	})
}

func generalCosine(a []float64) Window {
	return func(n int, sym bool) []float64 {
		return build(n, sym, func(m int) []float64 {
			fac := linspace(-math.Pi, math.Pi, m, true)
			w := make([]float64, m)
			for k, ak := range a {
				for i := range w {
					w[i] += ak * math.Cos(float64(k)*fac[i])
				}
			}
			return w
		})
	}
}

// GeneralCosine is the sum of cosines sum_k a[k] cos(k*x), x in [-π, π]
// (scipy.signal.windows.general_cosine).
func GeneralCosine(a []float64) Window { return generalCosine(a) }

// GeneralHamming is alpha - (1-alpha) cos(2πn/(M-1)); alpha 0.54 is Hamming and
// 0.5 Hann.
func GeneralHamming(alpha float64) Window { return generalCosine([]float64{alpha, 1 - alpha}) }

// Nuttall is the minimum 4-term Blackman–Harris window of Nuttall.
func Nuttall(n int, sym bool) []float64 {
	return generalCosine([]float64{0.3635819, 0.4891775, 0.1365995, 0.0106411})(n, sym)
}

// Flattop is the flat top window.
func Flattop(n int, sym bool) []float64 {
	return generalCosine([]float64{0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368})(n, sym)
}

// Cosine is the half-cosine (sine) window.
func Cosine(n int, sym bool) []float64 {
	return build(n, sym, func(m int) []float64 {
		w := make([]float64, m)
		for i := range w {
			w[i] = math.Sin(math.Pi / float64(m) * (float64(i) + 0.5))
		}
		return w
	})
}

// Barthann is the modified Bartlett–Hann window.
func Barthann(n int, sym bool) []float64 {
	return build(n, sym, func(m int) []float64 {
		w := make([]float64, m)
		for i := range w {
			fac := math.Abs(float64(i)/(float64(m)-1) - 0.5)
			w[i] = 0.62 - 0.48*fac + 0.38*math.Cos(2*math.Pi*fac)
		}
		return w
	})
}

// Kaiser is the Kaiser window with shape parameter beta.
func Kaiser(beta float64) Window {
	return func(n int, sym bool) []float64 {
		return build(n, sym, func(m int) []float64 {
			alpha := float64(m-1) / 2
			w := make([]float64, m)
			i0b := special.I0(beta)
			for i := range w {
				r := (float64(i) - alpha) / alpha
				w[i] = special.I0(beta*math.Sqrt(1-r*r)) / i0b
			}
			return w
		})
	}
}

// Tukey is the tapered cosine window; alpha is the fraction of the window
// inside the cosine tapered regions (0 is rectangular, 1 is Hann).
func Tukey(alpha float64) Window {
	return func(n int, sym bool) []float64 {
		if n <= 1 || alpha <= 0 {
			return ones(max(n, 0))
		}
		if alpha >= 1 {
			return Hann(n, sym)
		}
		return build(n, sym, func(m int) []float64 {
			w := make([]float64, m)
			width := int(math.Floor(alpha * float64(m-1) / 2))
			for i := range w {
				x := float64(i)
				switch {
				case i <= width:
					w[i] = 0.5 * (1 + math.Cos(math.Pi*(-1+2*x/alpha/float64(m-1))))
				case i < m-width-1:
					w[i] = 1
				default:
					w[i] = 0.5 * (1 + math.Cos(math.Pi*(-2/alpha+1+2*x/alpha/float64(m-1))))
				}
			}
			return w
		})
	}
}

// Gaussian is the Gaussian window with standard deviation std (in samples).
func Gaussian(std float64) Window {
	return func(n int, sym bool) []float64 {
		return build(n, sym, func(m int) []float64 {
			w := make([]float64, m)
			sig2 := 2 * std * std
			for i := range w {
				x := float64(i) - (float64(m)-1)/2
				w[i] = math.Exp(-x * x / sig2)
			}
			return w
		})
	}
}

// GeneralGaussian is exp(-0.5 |n/sig|^(2p)); p = 1 is Gaussian.
func GeneralGaussian(p, sig float64) Window {
	return func(n int, sym bool) []float64 {
		return build(n, sym, func(m int) []float64 {
			w := make([]float64, m)
			for i := range w {
				x := float64(i) - (float64(m)-1)/2
				w[i] = math.Exp(-0.5 * math.Pow(math.Abs(x/sig), 2*p))
			}
			return w
		})
	}
}

// Exponential is the Poisson window exp(-|n - center|/tau). A NaN center
// means the middle of the window; as in SciPy an explicit center is only
// allowed for a periodic window (sym false), otherwise nil is returned.
func Exponential(center, tau float64) Window {
	return func(n int, sym bool) []float64 {
		if sym && !math.IsNaN(center) {
			return nil
		}
		return build(n, sym, func(m int) []float64 {
			w := make([]float64, m)
			c := float64(m-1) / 2
			if !math.IsNaN(center) {
				c = center
			}
			for i := range w {
				w[i] = math.Exp(-math.Abs(float64(i)-c) / tau)
			}
			return w
		})
	}
}

// Chebwin is the Dolph–Chebyshev window with at dB of sidelobe attenuation.
func Chebwin(at float64) Window {
	return func(n int, sym bool) []float64 {
		return build(n, sym, func(m int) []float64 {
			order := float64(m) - 1
			beta := math.Cosh(1 / order * math.Acosh(math.Pow(10, math.Abs(at)/20)))
			p := make([]complex128, m)
			for k := range p {
				x := beta * math.Cos(math.Pi*float64(k)/float64(m))
				var v float64
				switch {
				case x > 1:
					v = math.Cosh(order * math.Acosh(x))
				case x < -1:
					v = float64(2*(m%2)-1) * math.Cosh(order*math.Acosh(-x))
				default:
					v = math.Cos(order * math.Acos(x))
				}
				p[k] = complex(v, 0)
			}
			var w []float64
			if m%2 == 1 {
				spec := fft.FFT(p)
				h := (m + 1) / 2
				w = make([]float64, 0, m)
				for i := h - 1; i >= 1; i-- {
					w = append(w, real(spec[i]))
				}
				for i := 0; i < h; i++ {
					w = append(w, real(spec[i]))
				}
			} else {
				for k := range p {
					p[k] *= cmplx.Exp(complex(0, math.Pi/float64(m)*float64(k)))
				}
				spec := fft.FFT(p)
				h := m/2 + 1
				w = make([]float64, 0, m)
				for i := h - 1; i >= 1; i-- {
					w = append(w, real(spec[i]))
				}
				for i := 1; i < h; i++ {
					w = append(w, real(spec[i]))
				}
			}
			mx := w[0]
			for _, v := range w {
				mx = math.Max(mx, v)
			}
			for i := range w {
				w[i] /= mx
			}
			return w
		})
	}
}

// DPSSNorm selects the normalisation of discrete prolate spheroidal
// sequences.
type DPSSNorm int

const (
	// DPSSApproximate (the default for a single window) scales the peak to 1
	// and corrects for even lengths by an analytic approximation.
	DPSSApproximate DPSSNorm = iota
	// DPSSSubsample scales to 1 at the interpolated centre of even lengths.
	DPSSSubsample
	// DPSSNorm2 normalises to unit energy (the default for several windows).
	DPSSNorm2
)

// DPSS returns the first kmax discrete prolate spheroidal (Slepian) sequences
// of length m and time-half-bandwidth product nw, the multitaper windows of
// Thomson, computed from the symmetric tridiagonal eigenproblem of Percival
// and Walden. ratios holds the spectral concentration ratios.
func DPSS(m int, nw float64, kmax int, sym bool, norm DPSSNorm) (windows [][]float64, ratios []float64, err error) {
	if norm < DPSSApproximate || norm > DPSSNorm2 {
		return nil, nil, errors.New("signal: unknown DPSS normalisation")
	}
	if m <= 1 {
		w := ones(max(m, 0))
		out := make([][]float64, max(kmax, 1))
		for i := range out {
			out[i] = w
		}
		return out, ones(len(out)), nil
	}
	if kmax <= 0 || kmax > m {
		return nil, nil, errors.New("signal: kmax must be greater than 0 and not more than m")
	}
	if nw >= float64(m)/2 {
		return nil, nil, errors.New("signal: NW must be less than m/2")
	}
	if nw <= 0 {
		return nil, nil, errors.New("signal: NW must be positive")
	}
	trunc := false
	if !sym {
		m++
		trunc = true
	}
	w := nw / float64(m)
	d := make([]float64, m)
	e := make([]float64, m-1)
	c := math.Cos(2 * math.Pi * w)
	for i := range d {
		x := (float64(m-1) - 2*float64(i)) / 2
		d[i] = x * x * c
	}
	for i := range e {
		t := float64(i + 1)
		e[i] = t * (float64(m) - t) / 2
	}
	_, vecs := tridiagTopEigen(d, e, kmax)
	// vecs[0] belongs to the largest eigenvalue
	win := vecs
	for i := 0; i < len(win); i += 2 { // even tapers have a positive mean
		var s float64
		for _, v := range win[i] {
			s += v
		}
		if s < 0 {
			negate(win[i])
		}
	}
	thresh := math.Max(1e-7, 1/float64(m))
	for i := 1; i < len(win); i += 2 { // odd tapers start with a positive lobe
		for _, v := range win[i] {
			if v*v > thresh {
				if v < 0 {
					negate(win[i])
				}
				break
			}
		}
	}
	ratios = dpssRatios(win, w)
	if norm != DPSSNorm2 {
		mx := win[0][0]
		for _, wv := range win {
			for _, v := range wv {
				mx = math.Max(mx, v)
			}
		}
		for _, wv := range win {
			for i := range wv {
				wv[i] /= mx
			}
		}
		if m%2 == 0 {
			var corr float64
			if norm == DPSSApproximate {
				corr = float64(m) * float64(m) / (float64(m)*float64(m) + nw)
			} else {
				corr = subsampleCorrection(win[0])
			}
			for _, wv := range win {
				for i := range wv {
					wv[i] *= corr
				}
			}
		}
	}
	if trunc {
		for i := range win {
			win[i] = win[i][:m-1]
		}
	}
	return win, ratios, nil
}

func negate(v []float64) {
	for i := range v {
		v[i] = -v[i]
	}
}

// subsampleCorrection is scipy's 'subsample' normalisation for even m: the
// first window's value at the (half-sample) centre via its spectrum.
func subsampleCorrection(w []float64) float64 {
	m := len(w)
	s := fft.RFFT(w)
	var sum float64 = real(s[0])
	for k := 1; k <= m/2; k++ {
		shift := -(1 - 1/float64(m)) * float64(k)
		sum += real(2 * s[k] * cmplx.Exp(complex(0, -math.Pi*shift)))
	}
	return float64(m) / sum
}

// dpssRatios returns the spectral concentration ratios of the sequences by
// the autocorrelation technique of Percival and Walden (p. 390).
func dpssRatios(win [][]float64, w float64) []float64 {
	m := len(win[0])
	r := make([]float64, m)
	r[0] = 2 * w
	for i := 1; i < m; i++ {
		x := math.Pi * 2 * w * float64(i)
		r[i] = 4 * w * math.Sin(x) / x
	}
	out := make([]float64, len(win))
	for k, v := range win {
		// autocorrelation by FFT, as scipy does, for the same rounding
		n := nextFastLen(2*m - 1)
		pad := make([]float64, n)
		copy(pad, v)
		spec := fft.RFFT(pad)
		for i := range spec {
			spec[i] *= cmplx.Conj(spec[i])
		}
		ac := fft.IRFFT(spec, n)
		var s float64
		for i := 0; i < m; i++ {
			s += ac[i] * r[i]
		}
		out[k] = s
	}
	return out
}

// nextFastLen returns the smallest 5-smooth length (factors 2, 3, 5) not
// below n: scipy.fft.next_fast_len(n, real=True), the sizes real transforms
// are fastest at.
func nextFastLen(n int) int {
	if n <= 6 {
		return max(n, 1)
	}
	best := math.MaxInt
	for f5 := 1; f5 < 2*n; f5 *= 5 {
		for f35 := f5; f35 < 2*n; f35 *= 3 {
			x := f35
			for x < n {
				x *= 2
			}
			best = min(best, x)
		}
	}
	return best
}

// GetWindow returns the named window (scipy.signal.get_window): the name is
// one of boxcar, triang, bartlett, hann, hamming, blackman, blackmanharris,
// nuttall, flattop, cosine, bohman, parzen, barthann, kaiser(beta),
// gaussian(std), general_gaussian(p, sig), general_hamming(alpha),
// general_cosine(a...), tukey(alpha), exponential(tau), chebwin(at) and
// dpss(NW), with the parameters in params. fftbins selects the periodic
// (true) form; a "_symmetric" or "_periodic" suffix overrides it.
func GetWindow(name string, n int, fftbins bool, params ...float64) ([]float64, error) {
	if n <= 0 {
		return nil, errors.New("signal: window length is not a positive integer")
	}
	sym := !fftbins
	switch {
	case strings.HasSuffix(name, "_symmetric"):
		sym, name = true, strings.TrimSuffix(name, "_symmetric")
	case strings.HasSuffix(name, "_periodic"):
		sym, name = false, strings.TrimSuffix(name, "_periodic")
	}
	type entry struct {
		fixed Window
		make  func(p []float64) Window
		np    int // required parameters; -1 for any number (>= 1)
	}
	table := map[string]entry{
		"boxcar": {fixed: Boxcar}, "box": {fixed: Boxcar}, "ones": {fixed: Boxcar}, "rect": {fixed: Boxcar}, "rectangular": {fixed: Boxcar},
		"triang": {fixed: Triang}, "triangle": {fixed: Triang}, "tri": {fixed: Triang},
		"bartlett": {fixed: Bartlett}, "bart": {fixed: Bartlett}, "brt": {fixed: Bartlett},
		"hann": {fixed: Hann}, "han": {fixed: Hann},
		"hamming": {fixed: Hamming}, "hamm": {fixed: Hamming}, "ham": {fixed: Hamming},
		"blackman": {fixed: Blackman}, "black": {fixed: Blackman}, "blk": {fixed: Blackman},
		"blackmanharris": {fixed: BlackmanHarris}, "blackharr": {fixed: BlackmanHarris}, "bkh": {fixed: BlackmanHarris},
		"nuttall": {fixed: Nuttall}, "nutl": {fixed: Nuttall}, "nut": {fixed: Nuttall},
		"flattop": {fixed: Flattop}, "flat": {fixed: Flattop}, "flt": {fixed: Flattop},
		"cosine": {fixed: Cosine}, "halfcosine": {fixed: Cosine},
		"bohman": {fixed: Bohman}, "bman": {fixed: Bohman}, "bmn": {fixed: Bohman},
		"parzen": {fixed: Parzen}, "parz": {fixed: Parzen}, "par": {fixed: Parzen},
		"barthann": {fixed: Barthann}, "brthan": {fixed: Barthann}, "bth": {fixed: Barthann},
		"kaiser":           {make: func(p []float64) Window { return Kaiser(p[0]) }, np: 1},
		"ksr":              {make: func(p []float64) Window { return Kaiser(p[0]) }, np: 1},
		"gaussian":         {make: func(p []float64) Window { return Gaussian(p[0]) }, np: 1},
		"gauss":            {make: func(p []float64) Window { return Gaussian(p[0]) }, np: 1},
		"gss":              {make: func(p []float64) Window { return Gaussian(p[0]) }, np: 1},
		"general_gaussian": {make: func(p []float64) Window { return GeneralGaussian(p[0], p[1]) }, np: 2},
		"general_hamming":  {make: func(p []float64) Window { return GeneralHamming(p[0]) }, np: 1},
		"general_cosine":   {make: func(p []float64) Window { return GeneralCosine(p) }, np: -1},
		"tukey":            {make: func(p []float64) Window { return Tukey(p[0]) }, np: 1},
		"tuk":              {make: func(p []float64) Window { return Tukey(p[0]) }, np: 1},
		"exponential":      {make: func(p []float64) Window { return Exponential(p[0], p[1]) }, np: 2},
		"poisson":          {make: func(p []float64) Window { return Exponential(p[0], p[1]) }, np: 2},
		"chebwin":          {make: func(p []float64) Window { return Chebwin(p[0]) }, np: 1},
		"cheb":             {make: func(p []float64) Window { return Chebwin(p[0]) }, np: 1},
	}
	if name == "dpss" {
		if len(params) != 1 {
			return nil, errors.New("signal: window dpss must have one parameter")
		}
		w, _, err := DPSS(n, params[0], 1, sym, DPSSApproximate)
		if err != nil {
			return nil, err
		}
		return w[0], nil
	}
	e, ok := table[name]
	switch {
	case !ok:
		return nil, fmt.Errorf("signal: invalid window name %q", name)
	case e.fixed != nil && len(params) > 0:
		return nil, fmt.Errorf("signal: %q does not allow parameters", name)
	case e.fixed != nil:
		return e.fixed(n, sym), nil
	case e.np == -1 && len(params) == 0, e.np > 0 && len(params) != e.np:
		return nil, fmt.Errorf("signal: %q requires a different number of parameters than %d", name, len(params))
	}
	w := e.make(params)(n, sym)
	if w == nil {
		return nil, fmt.Errorf("signal: window %q cannot be generated with these parameters", name)
	}
	return w, nil
}

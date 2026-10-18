package signal

import (
	"errors"
	"math"
	"math/cmplx"

	"github.com/go-fft/fft"
)

// FreqOptions selects the frequencies of a response. The zero value means
// the defaults of scipy.signal.freqz: 512 points on [0, π) in rad/sample.
type FreqOptions struct {
	// N is the number of frequencies when W is nil (default 512 for digital
	// responses, 200 for analog ones).
	N int
	// W lists the frequencies to evaluate, in the units of Fs (rad/s for
	// analog responses, where Fs does not apply).
	W []float64
	// Whole evaluates around the whole unit circle, [0, 2π), instead of
	// the upper half [0, π).
	Whole bool
	// Fs is the sampling frequency; zero means 2π, so that frequencies are
	// in rad/sample.
	Fs float64
	// IncludeNyquist includes the Nyquist frequency when Whole is false
	// and N points are generated.
	IncludeNyquist bool
}

func (o *FreqOptions) resolve(defN int) (FreqOptions, error) {
	var r FreqOptions
	if o != nil {
		r = *o
	}
	if r.Fs == 0 {
		r.Fs = 2 * math.Pi
	}
	if !(r.Fs > 0) || math.IsInf(r.Fs, 0) {
		return r, errors.New("signal: Fs must be a positive finite number")
	}
	if r.N < 0 || r.N > maxAlloc {
		return r, errors.New("signal: N must be nonnegative and at most 2^28")
	}
	if r.W == nil && r.N == 0 {
		r.N = defN
	}
	return r, nil
}

// linspace is numpy.linspace.
func linspace(start, stop float64, n int, endpoint bool) []float64 {
	y := make([]float64, n)
	if n == 0 {
		return y
	}
	div := n
	if endpoint {
		div = n - 1
	}
	step := 0.0
	if div > 0 {
		step = (stop - start) / float64(div)
	}
	for i := range y {
		y[i] = float64(i)*step + start
	}
	if endpoint && n > 1 {
		y[n-1] = stop
	}
	return y
}

// polyvalLow evaluates c[0] + c[1]x + ... by Horner's rule (numpy.polynomial.polyval).
func polyvalLow(c []float64, x complex128) complex128 {
	if len(c) == 0 {
		return 0
	}
	s := complex(c[len(c)-1], 0)
	for i := len(c) - 2; i >= 0; i-- {
		s = complex(c[i], 0) + s*x
	}
	return s
}

// Freqz returns the frequency response H(e^jw) = B(e^-jw)/A(e^-jw) of the
// digital filter b/a at the frequencies w (in the units of o.Fs), like
// scipy.signal.freqz (MATLAB freqz).
func Freqz(b, a []float64, o *FreqOptions) (w []float64, h []complex128, err error) {
	opt, err := o.resolve(512)
	if err != nil {
		return nil, nil, err
	}
	if len(b) == 0 || len(a) == 0 {
		return nil, nil, errors.New("signal: empty filter coefficients")
	}
	if opt.W == nil {
		n := opt.N
		last := math.Pi
		if opt.Whole {
			last = 2 * math.Pi
		}
		w = linspace(0, last, n, opt.IncludeNyquist && !opt.Whole)
		nfft := 2 * n
		if opt.Whole {
			nfft = n
		} else if opt.IncludeNyquist {
			nfft = 2 * (n - 1)
		}
		if len(a) == 1 && nfft >= len(b) && nfft > 0 {
			h = freqzFFT(b, a[0], n, nfft, opt.Whole)
		}
	} else {
		w = make([]float64, len(opt.W))
		for i, f := range opt.W {
			w[i] = 2 * math.Pi * f / opt.Fs
		}
	}
	if h == nil {
		h = make([]complex128, len(w))
		for i, wi := range w {
			zm1 := cmplx.Exp(complex(0, -wi))
			h[i] = polyvalLow(b, zm1) / polyvalLow(a, zm1)
		}
	}
	for i := range w {
		w[i] *= opt.Fs / (2 * math.Pi)
	}
	return w, h, nil
}

// freqzFFT evaluates a pure-FIR response with a zero-padded real FFT.
func freqzFFT(b []float64, a0 float64, n, nfft int, whole bool) []complex128 {
	pad := make([]float64, nfft)
	copy(pad, b)
	spec := fft.RFFT(pad)
	h := spec[:min(n, len(spec))]
	for i := range h {
		h[i] /= complex(a0, 0)
	}
	if whole {
		// the upper half is the conjugate mirror, without DC (and Nyquist)
		stop := len(h)
		if nfft%2 == 0 {
			stop--
		}
		for i := stop - 1; i >= 1; i-- {
			h = append(h, cmplx.Conj(h[i]))
		}
	}
	return h
}

// FreqzZPK is Freqz for a filter given as zeros, poles and gain.
func FreqzZPK(s ZPK, o *FreqOptions) (w []float64, h []complex128, err error) {
	opt, err := o.resolve(512)
	if err != nil {
		return nil, nil, err
	}
	if opt.W == nil {
		last := math.Pi
		if opt.Whole {
			last = 2 * math.Pi
		}
		w = linspace(0, last, opt.N, false)
	} else {
		w = make([]float64, len(opt.W))
		for i, f := range opt.W {
			w[i] = 2 * math.Pi * f / opt.Fs
		}
	}
	h = make([]complex128, len(w))
	for i, wi := range w {
		zm1 := cmplx.Exp(complex(0, wi))
		h[i] = complex(s.K, 0) * prodDiff(zm1, s.Z) / prodDiff(zm1, s.P)
		w[i] = wi * (opt.Fs / (2 * math.Pi))
	}
	return w, h, nil
}

func prodDiff(x complex128, roots []complex128) complex128 {
	p := complex(1, 0)
	for _, r := range roots {
		p *= x - r
	}
	return p
}

// SOSFreqz returns the frequency response of a cascade of sections
// (scipy.signal.sosfreqz).
func SOSFreqz(sos SOS, o *FreqOptions) (w []float64, h []complex128, err error) {
	if len(sos) == 0 {
		return nil, nil, errors.New("signal: cannot compute frequencies with no sections")
	}
	for j, s := range sos {
		var rowh []complex128
		w, rowh, err = Freqz(s[:3], s[3:], o)
		if err != nil {
			return nil, nil, err
		}
		if j == 0 {
			h = make([]complex128, len(rowh))
			for i := range h {
				h[i] = 1
			}
		}
		for i := range h {
			h[i] *= rowh[i]
		}
	}
	return w, h, nil
}

// GroupDelay returns the group delay, in samples, of the digital filter b/a
// (scipy.signal.group_delay). Where the delay is singular it is set to zero.
func GroupDelay(b, a []float64, o *FreqOptions) (w, gd []float64, err error) {
	opt, err := o.resolve(512)
	if err != nil {
		return nil, nil, err
	}
	if len(b) == 0 || len(a) == 0 {
		return nil, nil, errors.New("signal: empty filter coefficients")
	}
	if opt.W == nil {
		last := math.Pi
		if opt.Whole {
			last = 2 * math.Pi
		}
		w = linspace(0, last, opt.N, false)
	} else {
		w = make([]float64, len(opt.W))
		for i, f := range opt.W {
			w[i] = 2 * math.Pi * f / opt.Fs
		}
	}
	// c = b * reversed(a): the cross-correlation polynomial
	ra := make([]float64, len(a))
	for i, v := range a {
		ra[len(a)-1-i] = v
	}
	c := polyMul(b, ra)
	cr := make([]float64, len(c))
	for i, v := range c {
		cr[i] = v * float64(i)
	}
	gd = make([]float64, len(w))
	for i, wi := range w {
		z := cmplx.Exp(complex(0, -wi))
		num := polyvalLow(cr, z)
		den := polyvalLow(c, z)
		g := real(num/den) - float64(len(a)) + 1
		if math.IsNaN(g) || math.IsInf(g, 0) {
			g = 0
		}
		gd[i] = g
		w[i] = wi * (opt.Fs / (2 * math.Pi))
	}
	return w, gd, nil
}

// findFreqs chooses analog evaluation frequencies that cover the interesting
// part of the response, from the poles and zeros (scipy's findfreqs).
func findFreqs(zeros, poles []complex128, n int) []float64 {
	ep := poles
	if len(ep) == 0 {
		ep = []complex128{-1000}
	}
	var ez []complex128
	for _, v := range ep {
		if imag(v) >= 0 {
			ez = append(ez, v)
		}
	}
	for _, v := range zeros {
		if cmplx.Abs(v) < 1e5 && imag(v) >= 0 {
			ez = append(ez, v)
		}
	}
	hmax, lmin := math.Inf(-1), math.Inf(1)
	for _, v := range ez {
		integ := 0.0
		if cmplx.Abs(v) < 1e-10 {
			integ = 1
		}
		hmax = math.Max(hmax, 3*math.Abs(real(v)+integ)+1.5*imag(v))
		lmin = math.Min(lmin, math.Abs(real(v)+integ)+2*imag(v))
	}
	hfreq := math.RoundToEven(log10(hmax) + 0.5)
	lfreq := math.RoundToEven(log10(0.1*lmin) - 0.5)
	w := linspace(lfreq, hfreq, n, true)
	for i, v := range w {
		w[i] = math.Pow(10, v)
	}
	return w
}

// Freqs returns the frequency response H(jw) = B(jw)/A(jw) of the analog
// filter b/a (scipy.signal.freqs); o.W is in rad/s and o.N defaults to 200
// logarithmically spaced points chosen from the poles and zeros.
func Freqs(b, a []float64, o *FreqOptions) (w []float64, h []complex128, err error) {
	opt, err := o.resolve(200)
	if err != nil {
		return nil, nil, err
	}
	if len(b) == 0 || len(a) == 0 {
		return nil, nil, errors.New("signal: empty filter coefficients")
	}
	w = opt.W
	if w == nil {
		var zs, ps []complex128
		if zs, err = Roots(b); err != nil {
			return nil, nil, err
		}
		if ps, err = Roots(a); err != nil {
			return nil, nil, err
		}
		w = findFreqs(zs, ps, opt.N)
	}
	h = make([]complex128, len(w))
	for i, wi := range w {
		s := complex(0, wi)
		h[i] = polyval(b, s) / polyval(a, s)
	}
	return w, h, nil
}

// FreqsZPK is Freqs for a filter given as zeros, poles and gain.
func FreqsZPK(s ZPK, o *FreqOptions) (w []float64, h []complex128, err error) {
	opt, err := o.resolve(200)
	if err != nil {
		return nil, nil, err
	}
	w = opt.W
	if w == nil {
		w = findFreqs(s.Z, s.P, opt.N)
	}
	h = make([]complex128, len(w))
	for i, wi := range w {
		x := complex(0, wi)
		h[i] = complex(s.K, 0) * prodDiff(x, s.Z) / prodDiff(x, s.P)
	}
	return w, h, nil
}

// log10 is math.Log10 made exact at the powers of ten, like libm's: math.Log10
// computes log2(x)*(ln2/ln10), which is off by an ulp at 0.1, and the
// rounding of findFreqs hinges on such half-integers.
func log10(x float64) float64 {
	l := math.Log10(x)
	if r := math.Round(l); math.Abs(l-r) < 1e-9 && math.Pow(10, r) == x {
		return r
	}
	return l
}

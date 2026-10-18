package signal

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/go-fft/fft"
)

// sinc is numpy.sinc: sin(πx)/(πx), with the same guard at zero.
func sinc(x float64) float64 {
	if x == 0 {
		x = 1e-20
	}
	y := math.Pi * x
	return math.Sin(y) / y
}

// KaiserBeta returns the Kaiser window shape parameter for a given stopband
// attenuation a in dB (scipy.signal.kaiser_beta).
func KaiserBeta(a float64) float64 {
	switch {
	case a > 50:
		return 0.1102 * (a - 8.7)
	case a > 21:
		return 0.5842*math.Pow(a-21, 0.4) + 0.07886*(a-21)
	}
	return 0
}

// KaiserAtten returns the attenuation in dB of a Kaiser-window FIR filter
// with numtaps taps and a transition width relative to Nyquist
// (scipy.signal.kaiser_atten).
func KaiserAtten(numtaps int, width float64) float64 {
	return 2.285*float64(numtaps-1)*math.Pi*width + 7.95
}

// KaiserOrd returns the number of taps and the Kaiser beta needed for a
// ripple (attenuation) of the given dB and a transition width relative to
// Nyquist (scipy.signal.kaiserord).
func KaiserOrd(ripple, width float64) (numtaps int, beta float64, err error) {
	a := math.Abs(ripple)
	if a < 8 {
		return 0, 0, fmt.Errorf("signal: requested maximum ripple attenuation %f is too small for the Kaiser formula", a)
	}
	if !(width > 0) {
		return 0, 0, errors.New("signal: transition width must be positive")
	}
	n := (a-7.95)/2.285/(math.Pi*width) + 1
	return int(math.Ceil(n)), KaiserBeta(a), nil
}

// FirPass selects the passband layout of Firwin, like the pass_zero argument
// of scipy.signal.firwin.
type FirPass int

const (
	// FirPassZeroTrue: the response at DC is passed (pass_zero=True).
	FirPassZeroTrue FirPass = iota
	// FirPassZeroFalse: the response at DC is blocked (pass_zero=False).
	FirPassZeroFalse
	// FirLowpass: a single cutoff, DC passed.
	FirLowpass
	// FirHighpass: a single cutoff, DC blocked.
	FirHighpass
	// FirBandpass: at least two cutoffs, DC blocked.
	FirBandpass
	// FirBandstop: at least two cutoffs, DC passed.
	FirBandstop
)

// FirwinOptions are the optional parameters of Firwin.
type FirwinOptions struct {
	// Width, if positive, is the approximate transition width in the units of
	// Fs; it selects a Kaiser window and overrides Window.
	Width float64
	// Window defaults to Hamming.
	Window Window
	// PassZero selects how the cutoffs are read (default FirPassZeroTrue).
	PassZero FirPass
	// NoScale disables scaling the passband gain to exactly 1.
	NoScale bool
	// Fs is the sampling frequency (default 2: cutoffs relative to Nyquist=1).
	Fs float64
}

// Firwin designs a linear-phase FIR filter with the window method
// (scipy.signal.firwin; MATLAB fir1). cutoff holds one or more strictly
// increasing frequencies between 0 and Fs/2.
func Firwin(numtaps int, cutoff []float64, o *FirwinOptions) ([]float64, error) {
	var opt FirwinOptions
	if o != nil {
		opt = *o
	}
	if opt.Fs == 0 {
		opt.Fs = 2
	}
	if err := validFs(opt.Fs); err != nil {
		return nil, err
	}
	if numtaps < 1 {
		return nil, errors.New("signal: numtaps must be positive")
	}
	nyq := 0.5 * opt.Fs
	cut := make([]float64, len(cutoff))
	for i, c := range cutoff {
		cut[i] = c / nyq
	}
	if len(cut) == 0 {
		return nil, errors.New("signal: at least one cutoff frequency must be given")
	}
	for i, c := range cut {
		if !(c > 0 && c < 1) {
			return nil, errors.New("signal: invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2")
		}
		if i > 0 && !(c-cut[i-1] > 0) {
			return nil, errors.New("signal: invalid cutoff frequencies: the frequencies must be strictly increasing")
		}
	}
	win := opt.Window
	if win == nil {
		win = Hamming
	}
	if opt.Width > 0 {
		win = Kaiser(KaiserBeta(KaiserAtten(numtaps, opt.Width/nyq)))
	}
	passZero := false
	switch opt.PassZero {
	case FirPassZeroTrue:
		passZero = true
	case FirPassZeroFalse:
	case FirLowpass, FirHighpass:
		if len(cut) != 1 {
			return nil, errors.New("signal: cutoff must have one element for lowpass or highpass")
		}
		passZero = opt.PassZero == FirLowpass
	case FirBandpass, FirBandstop:
		if len(cut) <= 1 {
			return nil, errors.New("signal: cutoff must have at least two elements for bandpass or bandstop")
		}
		passZero = opt.PassZero == FirBandstop
	default:
		return nil, fmt.Errorf("signal: invalid PassZero %d", opt.PassZero)
	}
	passNyquist := (len(cut)%2 == 0) == passZero
	if passNyquist && numtaps%2 == 0 {
		return nil, errors.New("signal: a filter with an even number of coefficients must have zero response at the Nyquist frequency")
	}
	edges := make([]float64, 0, len(cut)+2)
	if passZero {
		edges = append(edges, 0)
	}
	edges = append(edges, cut...)
	if passNyquist {
		edges = append(edges, 1)
	}
	alpha := 0.5 * float64(numtaps-1)
	h := make([]float64, numtaps)
	m := make([]float64, numtaps)
	for i := range m {
		m[i] = float64(i) - alpha
	}
	for j := 0; j < len(edges); j += 2 {
		left, right := edges[j], edges[j+1]
		for i := range h {
			h[i] += right * sinc(right*m[i])
			h[i] -= left * sinc(left*m[i])
		}
	}
	w := win(numtaps, true)
	if w == nil {
		return nil, errors.New("signal: the window cannot be generated")
	}
	for i := range h {
		h[i] *= w[i]
	}
	if !opt.NoScale {
		left, right := edges[0], edges[1]
		sf := 0.5 * (left + right)
		if left == 0 {
			sf = 0
		} else if right == 1 {
			sf = 1
		}
		var s float64
		for i := range h {
			s += h[i] * math.Cos(math.Pi*m[i]*sf)
		}
		for i := range h {
			h[i] /= s
		}
	}
	return h, nil
}

// interp is numpy.interp for a strictly increasing xp.
func interp(x float64, xp, fp []float64) float64 {
	n := len(xp)
	if x <= xp[0] {
		return fp[0]
	}
	if x >= xp[n-1] {
		return fp[n-1]
	}
	j := sort.SearchFloat64s(xp, x)
	if xp[j] == x {
		return fp[j]
	}
	j--
	slope := (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
	return slope*(x-xp[j]) + fp[j]
}

// Firwin2Options are the optional parameters of Firwin2.
type Firwin2Options struct {
	// Nfreqs is the size of the interpolation grid (default 1 + 2^ceil(log2(numtaps))).
	Nfreqs int
	// Window defaults to Hamming; NoWindow disables windowing.
	Window   Window
	NoWindow bool
	// Antisymmetric designs a type III/IV filter (Hilbert transformers,
	// differentiators).
	Antisymmetric bool
	// Fs is the sampling frequency (default 2).
	Fs float64
}

// Firwin2 designs an FIR filter by frequency sampling: freq and gain describe
// a piecewise-linear desired response from 0 to Fs/2 (scipy.signal.firwin2;
// MATLAB fir2).
func Firwin2(numtaps int, freq, gain []float64, o *Firwin2Options) ([]float64, error) {
	var opt Firwin2Options
	if o != nil {
		opt = *o
	}
	if opt.Fs == 0 {
		opt.Fs = 2
	}
	if err := validFs(opt.Fs); err != nil {
		return nil, err
	}
	nyq := 0.5 * opt.Fs
	if len(freq) != len(gain) {
		return nil, errors.New("signal: freq and gain must be of same length")
	}
	if numtaps < 1 {
		return nil, errors.New("signal: numtaps must be positive")
	}
	if opt.Nfreqs != 0 && numtaps >= opt.Nfreqs {
		return nil, fmt.Errorf("signal: numtaps must be less than nfreqs, got numtaps=%d and nfreqs=%d", numtaps, opt.Nfreqs)
	}
	if len(freq) < 2 || freq[0] != 0 || freq[len(freq)-1] != nyq {
		return nil, errors.New("signal: freq must start with 0 and end with fs/2")
	}
	for i := 0; i+1 < len(freq); i++ {
		if freq[i+1]-freq[i] < 0 {
			return nil, errors.New("signal: the values in freq must be nondecreasing")
		}
		if i+2 < len(freq) && (freq[i+1]-freq[i])+(freq[i+2]-freq[i+1]) == 0 {
			return nil, errors.New("signal: a value in freq must not occur more than twice")
		}
	}
	if freq[1] == 0 {
		return nil, errors.New("signal: value 0 must not be repeated in freq")
	}
	if freq[len(freq)-2] == nyq {
		return nil, errors.New("signal: value fs/2 must not be repeated in freq")
	}
	var ftype int
	switch {
	case opt.Antisymmetric && numtaps%2 == 0:
		ftype = 4
	case opt.Antisymmetric:
		ftype = 3
	case numtaps%2 == 0:
		ftype = 2
	default:
		ftype = 1
	}
	last := gain[len(gain)-1]
	switch {
	case ftype == 2 && last != 0:
		return nil, errors.New("signal: a type II filter must have zero gain at the Nyquist frequency")
	case ftype == 3 && (gain[0] != 0 || last != 0):
		return nil, errors.New("signal: a type III filter must have zero gain at zero and Nyquist frequencies")
	case ftype == 4 && gain[0] != 0:
		return nil, errors.New("signal: a type IV filter must have zero gain at zero frequency")
	}
	nfreqs := opt.Nfreqs
	if nfreqs == 0 {
		nfreqs = 1 + 1<<int(math.Ceil(math.Log2(float64(numtaps))))
	}
	fr := append([]float64(nil), freq...)
	repeated := false
	for i := 0; i+1 < len(fr); i++ {
		if fr[i] == fr[i+1] {
			repeated = true
		}
	}
	if repeated {
		eps := 2.220446049250313e-16 * nyq
		for k := 0; k+1 < len(fr); k++ {
			if fr[k] == fr[k+1] {
				fr[k] -= eps
				fr[k+1] += eps
			}
		}
		for k := 0; k+1 < len(fr); k++ {
			if fr[k+1]-fr[k] <= 0 {
				return nil, errors.New("signal: freq cannot contain numbers that are too close to a repeated value")
			}
		}
	}
	x := linspace(0, nyq, nfreqs, true)
	spec := make([]complex128, nfreqs)
	for i, xi := range x {
		fx := interp(xi, fr, gain)
		ph := -(float64(numtaps-1) / 2) * math.Pi * xi / nyq
		s := complex(math.Cos(ph), math.Sin(ph))
		if ftype > 2 {
			s *= 1i
		}
		spec[i] = complex(fx, 0) * s
	}
	full := fft.IRFFT(spec, 2*(nfreqs-1))
	out := full[:numtaps]
	if !opt.NoWindow {
		win := opt.Window
		if win == nil {
			win = Hamming
		}
		w := win(numtaps, true)
		if w == nil {
			return nil, errors.New("signal: the window cannot be generated")
		}
		for i := range out {
			out[i] *= w[i]
		}
	}
	out = append([]float64(nil), out...)
	if ftype == 3 {
		out[len(out)/2] = 0
	}
	return out, nil
}

// Firls designs a linear-phase FIR filter by least squares
// (scipy.signal.firls; MATLAB firls). numtaps must be odd. bands holds pairs
// of band edges (increasing, between 0 and fs/2), desired the gain at each
// edge, and weight (optional, one per band) the relative weight of each
// band. fs defaults to 2.
func Firls(numtaps int, bands, desired, weight []float64, fs float64) ([]float64, error) {
	if fs == 0 {
		fs = 2
	}
	if err := validFs(fs); err != nil {
		return nil, err
	}
	if numtaps%2 == 0 || numtaps < 1 {
		return nil, errors.New("signal: numtaps must be odd and >= 1")
	}
	mm := (numtaps - 1) / 2
	nyq := 0.5 * fs
	if len(bands)%2 != 0 {
		return nil, errors.New("signal: bands must contain frequency pairs")
	}
	nb := len(bands) / 2
	bn := make([]float64, len(bands))
	for i, b := range bands {
		bn[i] = b / nyq
		if bn[i] < 0 || bn[i] > 1 {
			return nil, errors.New("signal: bands must be between 0 and 1 relative to Nyquist")
		}
	}
	if len(desired) != len(bands) {
		return nil, fmt.Errorf("signal: desired must have one entry per frequency, got %d gains for %d frequencies", len(desired), len(bands))
	}
	for i := 0; i < nb; i++ {
		if !(bn[2*i+1]-bn[2*i] > 0) || (i > 0 && bn[2*i]-bn[2*i-2] < 0) {
			return nil, errors.New("signal: bands must be monotonically nondecreasing and have width > 0")
		}
		if i > 0 && bn[2*i-1] > bn[2*i] {
			return nil, errors.New("signal: bands must not overlap")
		}
	}
	for _, d := range desired {
		if d < 0 {
			return nil, errors.New("signal: desired must be non-negative")
		}
	}
	w := weight
	if w == nil {
		w = ones(nb)
	}
	if len(w) != nb {
		return nil, fmt.Errorf("signal: weight must be the same size as the number of band pairs (%d)", nb)
	}
	for _, v := range w {
		if v < 0 {
			return nil, errors.New("signal: weight must be non-negative")
		}
	}
	q := make([]float64, numtaps)
	for n := range q {
		for k := 0; k < nb; k++ {
			lo, hi := bn[2*k], bn[2*k+1]
			q[n] += (sinc(hi*float64(n))*hi - sinc(lo*float64(n))*lo) * w[k]
		}
	}
	a := make([][]float64, mm+1)
	for i := range a {
		a[i] = make([]float64, mm+1)
		for j := range a[i] {
			a[i][j] = q[abs(i-j)] + q[i+j]
		}
	}
	b := make([]float64, mm+1)
	for n := 0; n <= mm; n++ {
		nf := float64(n)
		for k := 0; k < nb; k++ {
			lo, hi := bn[2*k], bn[2*k+1]
			d0, d1 := desired[2*k], desired[2*k+1]
			m := (d1 - d0) / (hi - lo)
			c := d0 - lo*m
			edge := func(f float64) float64 {
				v := f * (m*f + c) * sinc(f*nf)
				if n == 0 {
					return v - m*f*f/2
				}
				return v + m*math.Cos(nf*math.Pi*f)/((math.Pi*nf)*(math.Pi*nf))
			}
			b[n] += (edge(hi) - edge(lo)) * w[k]
		}
	}
	if err := solveDense(a, b); err != nil {
		return nil, fmt.Errorf("signal: firls normal equations are singular: %w", err)
	}
	coeffs := make([]float64, numtaps)
	for i := 1; i <= mm; i++ {
		coeffs[mm-i] = b[i]
		coeffs[mm+i] = b[i]
	}
	coeffs[mm] = 2 * b[0]
	return coeffs, nil
}

func abs(i int) int {
	if i < 0 {
		return -i
	}
	return i
}

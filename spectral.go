package signal

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"github.com/go-fft/fft"
	"github.com/go-simd/floats"
)

// DetrendType selects what is removed from each segment before analysis.
type DetrendType int

const (
	// DetrendDefault is the default of the function it is passed to:
	// constant for Welch, CSD, Periodogram and Spectrogram, none for STFT.
	DetrendDefault DetrendType = iota
	// DetrendConstant removes the mean.
	DetrendConstant
	// DetrendLinear removes the least-squares line.
	DetrendLinear
	// DetrendNone leaves the data alone.
	DetrendNone
)

// Scaling selects the units of a spectrum.
type Scaling int

const (
	// ScaleDefault is the default of the function it is passed to: density for
	// Welch, CSD, Periodogram and Spectrogram, spectrum for STFT and ISTFT.
	ScaleDefault Scaling = iota
	// ScaleDensity gives a power spectral density (units^2/Hz).
	ScaleDensity
	// ScaleSpectrum gives a power spectrum (units^2).
	ScaleSpectrum
)

// Average selects how Welch-type estimators combine segments.
type Average int

const (
	// AverageMean takes the mean (Welch's method).
	AverageMean Average = iota
	// AverageMedian takes the median, corrected for bias (robust to outliers).
	AverageMedian
)

// SpectralOptions are the optional parameters of the spectral estimators.
// The zero value is SciPy's defaults for the function it is passed to.
type SpectralOptions struct {
	// Fs is the sampling frequency (default 1).
	Fs float64
	// Window generates the window (periodic form) of Nperseg samples;
	// WindowTaps, if not nil, is used instead and sets Nperseg.
	Window     Window
	WindowTaps []float64
	// Nperseg is the length of each segment (default 256, or the signal
	// length if shorter).
	Nperseg int
	// Noverlap is the number of overlapping samples (default depends on the
	// estimator: Nperseg/2 for Welch, Nperseg/8 for Spectrogram).
	Noverlap *int
	// Nfft is the FFT length, at least Nperseg (default Nperseg).
	Nfft int
	// Detrend is applied to each segment.
	Detrend DetrendType
	// Scaling is the density (default) or spectrum.
	Scaling Scaling
	// TwoSided returns both halves of the spectrum instead of 0..Fs/2.
	TwoSided bool
	// Average is used by Welch, CSD and Coherence.
	Average Average
}

// Overlap returns a pointer to n, for SpectralOptions.Noverlap.
func Overlap(n int) *int { return &n }

func (o *SpectralOptions) get() SpectralOptions {
	if o == nil {
		return SpectralOptions{Fs: 1}
	}
	r := *o
	if r.Fs == 0 {
		r.Fs = 1
	}
	return r
}

// segSetup is the resolved segmentation of a signal.
type segSetup struct {
	win                     []float64
	nperseg, noverlap, nfft int
	fs                      float64
	det                     DetrendType
	scaling                 Scaling
	onesided                bool
}

// segDefaults are the defaults of one estimator.
type segDefaults struct {
	win     Window
	overlap func(nperseg int) int
	det     DetrendType
	scaling Scaling
}

var (
	welchDefaults = segDefaults{hannPeriodic, ovHalf, DetrendConstant, ScaleDensity}
	stftDefaults  = segDefaults{hannPeriodic, ovHalf, DetrendNone, ScaleSpectrum}
	specDefaults  = segDefaults{Tukey(0.25), func(n int) int { return n / 8 }, DetrendConstant, ScaleDensity}
)

func (o SpectralOptions) resolve(n int, d segDefaults) (segSetup, error) {
	defWin, defOverlap := d.win, d.overlap
	var s segSetup
	if err := validFs(o.Fs); err != nil {
		return s, err
	}
	s.fs = o.Fs
	if o.Nperseg < 0 {
		return s, errors.New("signal: nperseg must be a positive integer")
	}
	if o.WindowTaps != nil {
		s.win = o.WindowTaps
		if n < len(s.win) {
			return s, errors.New("signal: window is longer than input signal")
		}
		if o.Nperseg != 0 && o.Nperseg != len(s.win) {
			return s, errors.New("signal: value specified for nperseg is different from length of window")
		}
		s.nperseg = len(s.win)
	} else {
		s.nperseg = o.Nperseg
		if s.nperseg == 0 {
			s.nperseg = 256
		}
		if s.nperseg > n {
			s.nperseg = n
		}
		w := o.Window
		if w == nil {
			w = defWin
		}
		s.win = w(s.nperseg, false)
		if s.win == nil {
			return s, errors.New("signal: the window cannot be generated")
		}
	}
	s.nfft = o.Nfft
	if s.nfft == 0 {
		s.nfft = s.nperseg
	} else if s.nfft < s.nperseg || s.nfft > maxAlloc/4 {
		return s, errors.New("signal: nfft must be at least nperseg and at most 2^26")
	}
	if o.Noverlap == nil {
		s.noverlap = defOverlap(s.nperseg)
	} else {
		s.noverlap = *o.Noverlap
	}
	if s.noverlap >= s.nperseg || s.noverlap < 0 {
		return s, errors.New("signal: noverlap must be non-negative and less than nperseg")
	}
	if o.Detrend < DetrendDefault || o.Detrend > DetrendNone {
		return s, errors.New("signal: unknown detrend type")
	}
	if o.Scaling < ScaleDefault || o.Scaling > ScaleSpectrum {
		return s, errors.New("signal: unknown scaling")
	}
	s.det, s.scaling = o.Detrend, o.Scaling
	if s.det == DetrendDefault {
		s.det = d.det
	}
	if s.scaling == ScaleDefault {
		s.scaling = d.scaling
	}
	s.onesided = !o.TwoSided
	return s, nil
}

func hannPeriodic(n int, _ bool) []float64 { return Hann(n, false) }

// detrendSegment removes the trend of seg in place.
func detrendSegment(seg []float64, t DetrendType) {
	switch t {
	case DetrendConstant:
		// SIMD lane-blocked sum (go-simd/floats): no serial add chain, and
		// closer to numpy's pairwise mean than a left-to-right loop.
		m := floats.Sum(seg) / float64(len(seg))
		for i := range seg {
			seg[i] -= m
		}
	case DetrendLinear:
		detrendLinear(seg)
	}
}

// detrendLinear subtracts the least-squares line (x versus (1..n)/n).
func detrendLinear(seg []float64) {
	n := len(seg)
	if n == 1 {
		seg[0] = 0
		return
	}
	var tm, ym float64
	for i, v := range seg {
		tm += float64(i+1) / float64(n)
		ym += v
	}
	tm /= float64(n)
	ym /= float64(n)
	var sty, stt float64
	for i, v := range seg {
		dt := float64(i+1)/float64(n) - tm
		sty += dt * (v - ym)
		stt += dt * dt
	}
	slope := sty / stt
	icpt := ym - slope*tm
	for i := range seg {
		seg[i] -= slope*(float64(i+1)/float64(n)) + icpt
	}
}

// Detrend removes a constant or piecewise-linear trend from x, with
// breakpoints bp between the linear pieces (scipy.signal.detrend).
func Detrend(x []float64, t DetrendType, bp []int) ([]float64, error) {
	if t == DetrendDefault {
		t = DetrendLinear
	}
	if t < DetrendConstant || t > DetrendLinear {
		return nil, errors.New("signal: trend type must be linear or constant")
	}
	out := slices.Clone(x)
	if t == DetrendConstant {
		if len(out) > 0 {
			detrendSegment(out, DetrendConstant)
		}
		return out, nil
	}
	pts := []int{0, len(x)}
	for _, b := range bp {
		if b > len(x) || b < 0 {
			return nil, errors.New("signal: breakpoints must be less than length of data")
		}
		pts = append(pts, b)
	}
	sort.Ints(pts)
	pts = slices.Compact(pts)
	for m := 0; m+1 < len(pts); m++ {
		detrendLinear(out[pts[m]:pts[m+1]])
	}
	return out, nil
}

// segmentSpectra returns the FFT of every windowed, detrended segment of x:
// result[k][f].
func segmentSpectra(x []float64, s segSetup) [][]complex128 {
	step := s.nperseg - s.noverlap
	nseg := (len(x)-s.nperseg)/step + 1
	out := make([][]complex128, nseg)
	buf := make([]float64, s.nfft)
	var rp *fft.RealPlan
	var cp *fft.Plan
	if s.onesided {
		rp = fft.NewRealPlan(s.nfft)
	} else {
		cp = fft.NewPlan(s.nfft)
	}
	cbuf := make([]complex128, s.nfft)
	for k := range out {
		seg := buf[:s.nperseg]
		copy(seg, x[k*step:k*step+s.nperseg])
		detrendSegment(seg, s.det)
		for i := range seg {
			seg[i] *= s.win[i]
		}
		clear(buf[s.nperseg:])
		if s.onesided {
			out[k] = rp.RFFT(make([]complex128, s.nfft/2+1), buf)
		} else {
			for i, v := range buf {
				cbuf[i] = complex(v, 0)
			}
			out[k] = cp.FFT(make([]complex128, s.nfft), cbuf)
		}
	}
	return out
}

func (s segSetup) freqs() []float64 {
	if s.onesided {
		return fft.RFFTFreq(s.nfft, 1/s.fs)
	}
	return fft.FFTFreq(s.nfft, 1/s.fs)
}

func (s segSetup) scale() float64 {
	sc := s.scaling
	var sw, sw2 float64
	for _, w := range s.win {
		sw += w
		sw2 += w * w
	}
	if sc == ScaleDensity {
		return 1 / (s.fs * sw2)
	}
	return 1 / (sw * sw)
}

// doubleOneSided doubles the interior of a one-sided spectrum.
func doubleOneSided(p []complex128, nfft int) {
	end := len(p)
	if nfft%2 == 0 {
		end--
	}
	for i := 1; i < end; i++ {
		p[i] *= 2
	}
}

// crossSpectra returns the per-segment cross-spectra conj(X)*Y, scaled.
func crossSpectra(x, y []float64, s segSetup) [][]complex128 {
	sx := segmentSpectra(x, s)
	sy := sx
	if y != nil {
		sy = segmentSpectra(y, s)
	}
	scale := complex(s.scale(), 0)
	for k := range sx {
		res := sx[k] // in place: element i reads only sx[k][i] and sy[k][i]
		for i := range res {
			res[i] = cmplx.Conj(sx[k][i]) * sy[k][i] * scale
		}
		if s.onesided {
			doubleOneSided(res, s.nfft)
		}
		sx[k] = res
		if y != nil {
			sy[k] = nil
		}
	}
	return sx
}

// psdColumns is real(crossSpectra(x, nil, s)) transposed to [f][segment],
// computed with the same complex operations per element (bit-identical) but
// through one reused spectrum buffer and one output block, instead of a
// spectrum, a scaled copy and a transposed copy per segment.
func psdColumns(x []float64, s segSetup) [][]float64 {
	step := s.nperseg - s.noverlap
	nseg := (len(x)-s.nperseg)/step + 1
	nf := s.nfft
	if s.onesided {
		nf = s.nfft/2 + 1
	}
	block := make([]float64, nf*nseg)
	out := make([][]float64, nf)
	for i := range out {
		out[i] = block[i*nseg : (i+1)*nseg : (i+1)*nseg]
	}
	buf := make([]float64, s.nfft)
	spec := make([]complex128, nf)
	var rp *fft.RealPlan
	var cp *fft.Plan
	var cbuf []complex128
	if s.onesided {
		rp = fft.NewRealPlan(s.nfft)
	} else {
		cp = fft.NewPlan(s.nfft)
		cbuf = make([]complex128, s.nfft)
	}
	// doubleOneSided's range: the interior of a one-sided spectrum.
	dlo, dhi := 0, 0
	if s.onesided {
		dlo, dhi = 1, nf
		if s.nfft%2 == 0 {
			dhi--
		}
	}
	scale := complex(s.scale(), 0)
	for k := 0; k < nseg; k++ {
		seg := buf[:s.nperseg]
		copy(seg, x[k*step:k*step+s.nperseg])
		detrendSegment(seg, s.det)
		for i := range seg {
			seg[i] *= s.win[i]
		}
		clear(buf[s.nperseg:])
		if s.onesided {
			rp.RFFT(spec, buf)
		} else {
			for i, v := range buf {
				cbuf[i] = complex(v, 0)
			}
			cp.FFT(spec, cbuf)
		}
		for i, c := range spec {
			v := cmplx.Conj(c) * c * scale
			if i >= dlo && i < dhi {
				v *= 2
			}
			out[i][k] = real(v)
		}
	}
	return out
}

// medianBias is the bias of the median of n exponential variates.
func medianBias(n int) float64 {
	b := 1.0
	for i := 1; i <= (n-1)/2; i++ {
		ii := float64(2 * i)
		b += 1/(ii+1) - 1/ii
	}
	return b
}

func median(v []float64) float64 {
	s := slices.Clone(v)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// average combines per-segment spectra [seg][freq].
func average(p [][]complex128, a Average) ([]complex128, error) {
	nseg, nf := len(p), len(p[0])
	out := make([]complex128, nf)
	if nseg == 1 {
		copy(out, p[0])
		return out, nil
	}
	switch a {
	case AverageMean:
		for _, seg := range p {
			for i, v := range seg {
				out[i] += v
			}
		}
		for i := range out {
			out[i] /= complex(float64(nseg), 0)
		}
	case AverageMedian:
		bias := medianBias(nseg)
		re, im := make([]float64, nseg), make([]float64, nseg)
		for i := range out {
			for k := range p {
				re[k], im[k] = real(p[k][i]), imag(p[k][i])
			}
			out[i] = complex(median(re)/bias, median(im)/bias)
		}
	default:
		return nil, errors.New("signal: average must be mean or median")
	}
	return out, nil
}

func ovHalf(n int) int { return n / 2 }

// CSD estimates the cross power spectral density of x and y by Welch's method
// (scipy.signal.csd; MATLAB cpsd). The shorter signal is zero padded.
func CSD(x, y []float64, o *SpectralOptions) (f []float64, p []complex128, err error) {
	opt := o.get()
	if len(x) == 0 || len(y) == 0 {
		return nil, nil, errors.New("signal: empty input")
	}
	n := max(len(x), len(y))
	s, err := opt.resolve(n, welchDefaults)
	if err != nil {
		return nil, nil, err
	}
	xp, yp := padTo(x, n), padTo(y, n)
	p, err = average(crossSpectra(xp, yp, s), opt.Average)
	if err != nil {
		return nil, nil, err
	}
	return s.freqs(), p, nil
}

func padTo(x []float64, n int) []float64 {
	if len(x) == n {
		return x
	}
	return append(slices.Clone(x), make([]float64, n-len(x))...)
}

// Welch estimates the power spectral density of x by Welch's method of
// averaged, windowed, overlapping periodograms (scipy.signal.welch; MATLAB
// pwelch).
func Welch(x []float64, o *SpectralOptions) (f, p []float64, err error) {
	opt := o.get()
	if len(x) == 0 {
		return nil, nil, errors.New("signal: empty input")
	}
	s, err := opt.resolve(len(x), welchDefaults)
	if err != nil {
		return nil, nil, err
	}
	pc, err := average(crossSpectra(x, nil, s), opt.Average)
	if err != nil {
		return nil, nil, err
	}
	p = make([]float64, len(pc))
	for i, v := range pc {
		p[i] = real(v)
	}
	return s.freqs(), p, nil
}

// Periodogram estimates the power spectral density of x from one windowed FFT
// (scipy.signal.periodogram; MATLAB periodogram). The default window is
// rectangular.
func Periodogram(x []float64, o *SpectralOptions) (f, p []float64, err error) {
	opt := o.get()
	if len(x) == 0 {
		return nil, nil, errors.New("signal: empty input")
	}
	if opt.Window == nil && opt.WindowTaps == nil {
		opt.Window = Boxcar
	}
	n := len(x)
	switch {
	case opt.Nfft == 0 || opt.Nfft == n:
		opt.Nperseg = n
	case opt.Nfft > n:
		opt.Nperseg = n
	default:
		x = x[:opt.Nfft]
		opt.Nperseg, opt.Nfft = opt.Nfft, 0
	}
	if opt.WindowTaps != nil && len(opt.WindowTaps) != opt.Nperseg {
		return nil, nil, errors.New("signal: the size of the window must be the same size of the input")
	}
	opt.Noverlap = Overlap(0)
	return Welch(x, &opt)
}

// Coherence estimates the magnitude-squared coherence of x and y,
// |Pxy|^2/(Pxx Pyy) (scipy.signal.coherence; MATLAB mscohere).
func Coherence(x, y []float64, o *SpectralOptions) (f, c []float64, err error) {
	opt := o.get()
	opt.TwoSided = false
	opt.Scaling = ScaleDensity
	opt.Detrend = max(opt.Detrend, DetrendConstant)
	f, pxx, err := Welch(x, &opt)
	if err != nil {
		return nil, nil, err
	}
	_, pyy, err := Welch(y, &opt)
	if err != nil {
		return nil, nil, err
	}
	_, pxy, _ := CSD(x, y, &opt) // the same options were accepted by Welch
	c = make([]float64, len(f))
	for i := range c {
		a := cmplx.Abs(pxy[i])
		c[i] = a * a / pxx[i] / pyy[i]
	}
	return f, c, nil
}

// SpectrogramMode selects what Spectrogram returns.
type SpectrogramMode int

const (
	// SpecPSD is the power spectral density or spectrum of each segment.
	SpecPSD SpectrogramMode = iota
	// SpecMagnitude is the magnitude of the STFT.
	SpecMagnitude
	// SpecAngle is the angle of the STFT in radians.
	SpecAngle
	// SpecPhase is the unwrapped angle along time.
	SpecPhase
)

// Spectrogram computes a spectrogram of x from consecutive Fourier transforms
// (scipy.signal.spectrogram; MATLAB spectrogram): S[f][t], with the default
// Tukey window (alpha 0.25) and Nperseg/8 overlap.
func Spectrogram(x []float64, o *SpectralOptions, mode SpectrogramMode) (f, t []float64, s [][]float64, err error) {
	opt := o.get()
	if len(x) == 0 {
		return nil, nil, nil, errors.New("signal: empty input")
	}
	if mode < SpecPSD || mode > SpecPhase {
		return nil, nil, nil, errors.New("signal: unknown spectrogram mode")
	}
	seg, err := opt.resolve(len(x), specDefaults)
	if err != nil {
		return nil, nil, nil, err
	}
	step := seg.nperseg - seg.noverlap
	if mode == SpecPSD {
		return seg.freqs(), segTimes(len(x), seg, step, false), psdColumns(x, seg), nil
	}
	f, t, z := stftCore(x, seg, "", false)
	switch mode {
	case SpecMagnitude:
		s = transposeComplex(z, cmplx.Abs)
	default:
		s = transposeComplex(z, cmplx.Phase)
		if mode == SpecPhase {
			for i := range s {
				s[i] = unwrap(s[i])
			}
		}
	}
	return f, t, s, nil
}

// unwrap corrects jumps larger than π in a phase sequence by multiples of 2π.
func unwrap(p []float64) []float64 {
	out := slices.Clone(p)
	var off float64
	for i := 1; i < len(p); i++ {
		d := p[i] - p[i-1]
		dd := math.Mod(d+math.Pi, 2*math.Pi)
		if dd < 0 {
			dd += 2 * math.Pi
		}
		dd -= math.Pi
		if dd == -math.Pi && d > 0 {
			dd = math.Pi
		}
		corr := dd - d
		if math.Abs(d) < math.Pi {
			corr = 0
		}
		off += corr
		out[i] = p[i] + off
	}
	return out
}

func segTimes(n int, s segSetup, step int, boundary bool) []float64 {
	var t []float64
	half := float64(s.nperseg) / 2
	for v := half; v < float64(n)-half+1; v += float64(step) {
		tv := v
		if boundary {
			tv -= half
		}
		t = append(t, tv/s.fs)
	}
	return t
}

func transposeReal(p [][]complex128, f func(complex128) float64) [][]float64 {
	nf := len(p[0])
	out := make([][]float64, nf)
	for i := range out {
		out[i] = make([]float64, len(p))
		for k := range p {
			out[i][k] = f(p[k][i])
		}
	}
	return out
}

func transposeComplex(p [][]complex128, f func(complex128) float64) [][]float64 {
	return transposeReal(p, f)
}

// boundaryExtend extends x by n samples at both ends: "zeros", "even", "odd",
// or "constant" (anything else extends with zeros).
func boundaryExtend(x []float64, n int, kind string) []float64 {
	switch kind {
	case "even":
		return extend(x, n, PadEven)
	case "odd":
		return extend(x, n, PadOdd)
	case "constant":
		return extend(x, n, PadConstant)
	}
	out := make([]float64, 0, len(x)+2*n) // "zeros"
	out = append(out, make([]float64, n)...)
	out = append(out, x...)
	return append(out, make([]float64, n)...)
}

// stftCore is scipy's _spectral_helper in 'stft' mode. It returns
// z[segment][frequency] (one spectrum per segment).
func stftCore(x []float64, s segSetup, boundary string, padded bool) (f, t []float64, z [][]complex128) {
	step := s.nperseg - s.noverlap
	if boundary != "" {
		x = boundaryExtend(x, s.nperseg/2, boundary)
	}
	if padded {
		// zero pad to nperseg + k*step samples
		nadd := (step - (len(x)-s.nperseg)%step) % step
		x = append(slices.Clone(x), make([]float64, nadd)...)
	}
	z = segmentSpectra(x, s)
	scale := complex(math.Sqrt(s.scale()), 0)
	for k := range z {
		for i := range z[k] {
			z[k][i] *= scale
		}
	}
	return s.freqs(), segTimes(len(x), s, step, boundary != ""), z
}

// STFTOptions are the parameters of STFT beyond SpectralOptions.
type STFTOptions struct {
	SpectralOptions
	// Boundary extends the signal at both ends so that the first window is
	// centred on the first sample: "zeros" (the default when empty and
	// NoBoundary is false), "even", "odd" or "constant".
	Boundary string
	// NoBoundary disables the extension.
	NoBoundary bool
	// NoPad disables zero padding to a whole number of segments.
	NoPad bool
}

// STFT computes the short-time Fourier transform of x (scipy.signal.stft;
// legacy API): Z[f][t] with the default Hann window of 256 samples, 50%
// overlap, zero boundary extension and padding, scaled as a spectrum.
func STFT(x []float64, o *STFTOptions) (f, t []float64, z [][]complex128, err error) {
	var opt STFTOptions
	if o != nil {
		opt = *o
	}
	so := opt.SpectralOptions.get()
	if len(x) == 0 {
		return nil, nil, nil, errors.New("signal: empty input")
	}
	b := opt.Boundary
	if b == "" && !opt.NoBoundary {
		b = "zeros"
	}
	if opt.NoBoundary {
		b = ""
	}
	if b != "" && b != "zeros" && b != "even" && b != "odd" && b != "constant" {
		return nil, nil, nil, fmt.Errorf("signal: unknown boundary option %q", b)
	}
	seg, err := so.resolve(len(x), stftDefaults)
	if err != nil {
		return nil, nil, nil, err
	}
	f, t, zz := stftCore(x, seg, b, !opt.NoPad)
	nf := len(zz[0])
	z = make([][]complex128, nf)
	for i := range z {
		z[i] = make([]complex128, len(zz))
		for k := range zz {
			z[i][k] = zz[k][i]
		}
	}
	return f, t, z, nil
}

// ISTFTOptions are the parameters of ISTFT.
type ISTFTOptions struct {
	SpectralOptions
	// TwoSidedInput says z holds the full spectrum rather than 0..Fs/2.
	TwoSidedInput bool
	// NoBoundary says the forward transform used no boundary extension.
	NoBoundary bool
}

// ISTFT inverts STFT by weighted overlap-add (scipy.signal.istft).
func ISTFT(z [][]complex128, o *ISTFTOptions) (t, x []float64, err error) {
	var opt ISTFTOptions
	if o != nil {
		opt = *o
	}
	so := opt.SpectralOptions.get()
	if err := validFs(so.Fs); err != nil {
		return nil, nil, err
	}
	if len(z) == 0 || len(z[0]) == 0 {
		return nil, nil, errors.New("signal: empty STFT")
	}
	nseg := len(z[0])
	nDefault := len(z)
	if !opt.TwoSidedInput {
		nDefault = 2 * (len(z) - 1)
	}
	nperseg := nDefault
	if so.Nperseg != 0 {
		nperseg = so.Nperseg
	}
	if nperseg < 1 {
		return nil, nil, errors.New("signal: nperseg must be a positive integer")
	}
	nfft := so.Nfft
	switch {
	case nfft == 0 && !opt.TwoSidedInput && nperseg == nDefault+1:
		nfft = nperseg
	case nfft == 0:
		nfft = nDefault
	case nfft < nperseg:
		return nil, nil, errors.New("signal: nfft must be greater than or equal to nperseg")
	}
	noverlap := nperseg / 2
	if so.Noverlap != nil {
		noverlap = *so.Noverlap
	}
	if noverlap >= nperseg || noverlap < 0 {
		return nil, nil, errors.New("signal: noverlap must be non-negative and less than nperseg")
	}
	step := nperseg - noverlap
	win := so.WindowTaps
	if win == nil {
		w := so.Window
		if w == nil {
			w = hannPeriodic
		}
		win = w(nperseg, false)
		if win == nil {
			return nil, nil, errors.New("signal: the window cannot be generated")
		}
	}
	if len(win) != nperseg {
		return nil, nil, fmt.Errorf("signal: window must have length of %d", nperseg)
	}
	var wsum, wsq float64
	for _, w := range win {
		wsum += w
		wsq += w * w
	}
	var amp float64
	switch so.Scaling {
	case ScaleDefault, ScaleSpectrum:
		amp = wsum
	case ScaleDensity:
		amp = math.Sqrt(so.Fs * wsq)
	default:
		return nil, nil, errors.New("signal: unknown scaling")
	}
	n := nperseg + (nseg-1)*step
	out := make([]complex128, n)
	norm := make([]float64, n)
	cp := fft.NewPlan(nfft)
	rp := fft.NewRealPlan(nfft)
	col := make([]complex128, len(z))
	for k := 0; k < nseg; k++ {
		for i := range z {
			col[i] = z[i][k]
		}
		var seg []complex128
		if opt.TwoSidedInput {
			seg = cp.IFFT(make([]complex128, nfft), padComplex(col, nfft))
		} else {
			r := rp.IRFFT(make([]float64, nfft), padComplex(col, nfft/2+1))
			seg = make([]complex128, nfft)
			for i, v := range r {
				seg[i] = complex(v, 0)
			}
		}
		for i := 0; i < nperseg; i++ {
			out[k*step+i] += seg[i] * complex(amp*win[i], 0)
			norm[k*step+i] += win[i] * win[i]
		}
	}
	lo, hi := 0, n
	if !opt.NoBoundary {
		lo, hi = nperseg/2, n-nperseg/2
	}
	x = make([]float64, hi-lo)
	for i := range x {
		d := norm[lo+i]
		if !(d > 1e-10) {
			d = 1
		}
		x[i] = real(out[lo+i]) / d
	}
	t = make([]float64, len(x))
	for i := range t {
		t[i] = float64(i) / so.Fs
	}
	return t, x, nil
}

func padComplex(c []complex128, n int) []complex128 {
	if len(c) >= n {
		return c[:n]
	}
	out := make([]complex128, n)
	copy(out, c)
	return out
}

func coverage(win []float64, nperseg, noverlap int, square bool) ([]float64, error) {
	if nperseg < 1 {
		return nil, errors.New("signal: nperseg must be a positive integer")
	}
	if noverlap >= nperseg {
		return nil, errors.New("signal: noverlap must be less than nperseg")
	}
	if noverlap < 0 {
		return nil, errors.New("signal: noverlap must be a nonnegative integer")
	}
	if len(win) != nperseg {
		return nil, errors.New("signal: window must have length of nperseg")
	}
	step := nperseg - noverlap
	sums := make([]float64, step)
	for ii := 0; ii < nperseg/step; ii++ {
		for j := 0; j < step; j++ {
			v := win[ii*step+j]
			if square {
				v *= v
			}
			sums[j] += v
		}
	}
	if r := nperseg % step; r != 0 {
		for j := 0; j < r; j++ {
			v := win[nperseg-r+j]
			if square {
				v *= v
			}
			sums[j] += v
		}
	}
	return sums, nil
}

// CheckCOLA reports whether the window and overlap satisfy the constant
// overlap-add condition, so that STFT with those parameters is invertible and
// the overlapped windows sum to a constant (scipy.signal.check_COLA).
func CheckCOLA(win []float64, nperseg, noverlap int, tol float64) (bool, error) {
	if noverlap < 0 {
		return false, errors.New("signal: noverlap must be a nonnegative integer")
	}
	sums, err := coverage(win, nperseg, noverlap, false)
	if err != nil {
		return false, err
	}
	med := median(sums)
	var dev float64
	for _, s := range sums {
		dev = math.Max(dev, math.Abs(s-med))
	}
	return dev < tol, nil
}

// CheckNOLA reports whether the window and overlap satisfy the nonzero
// overlap-add condition: the squared windows sum to a nonzero value
// everywhere, the minimum requirement for ISTFT (scipy.signal.check_NOLA).
func CheckNOLA(win []float64, nperseg, noverlap int, tol float64) (bool, error) {
	sums, err := coverage(win, nperseg, noverlap, true)
	if err != nil {
		return false, err
	}
	return slices.Min(sums) > tol, nil
}

// LombOptions are the optional parameters of Lombscargle.
type LombOptions struct {
	// Weights are non-negative sample weights (default uniform).
	Weights []float64
	// Normalize scales the periodogram by the signal variance (range [0, 1]).
	Normalize bool
	// FloatingMean fits a constant offset together with the sinusoid.
	FloatingMean bool
}

// Lombscargle computes the Lomb–Scargle periodogram of the unevenly sampled
// y(x) at the angular frequencies freqs (scipy.signal.lombscargle).
func Lombscargle(x, y, freqs []float64, o *LombOptions) ([]float64, error) {
	_, _, _, pg, err := lombCore(x, y, freqs, o)
	return pg, err
}

// LombscargleAmplitude is Lombscargle returning the complex best-fit
// amplitude and phase a + ib, rotated by the phase offset (normalize='amplitude').
func LombscargleAmplitude(x, y, freqs []float64, o *LombOptions) ([]complex128, error) {
	a, b, tau, _, err := lombCore(x, y, freqs, o)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, len(a))
	for i := range out {
		out[i] = complex(a[i], b[i]) * cmplx.Exp(complex(0, tau[i]))
	}
	return out, nil
}

func lombCore(x, y, freqs []float64, o *LombOptions) (a, b, tau, pgram []float64, err error) {
	var opt LombOptions
	if o != nil {
		opt = *o
	}
	if len(x) == 0 || len(x) != len(y) {
		return nil, nil, nil, nil, errors.New("signal: x and y must be non-empty and of equal length")
	}
	if len(freqs) == 0 {
		return nil, nil, nil, nil, errors.New("signal: freqs must be non-empty")
	}
	w := opt.Weights
	if w == nil {
		w = ones(len(x))
	}
	if len(w) != len(x) {
		return nil, nil, nil, nil, errors.New("signal: weights must have the same length as x")
	}
	var sw float64
	for _, v := range w {
		if v < 0 || math.IsNaN(v) {
			return nil, nil, nil, nil, errors.New("signal: weights must be non-negative")
		}
		sw += v
	}
	if !(sw > 0) {
		return nil, nil, nil, nil, errors.New("signal: weights must sum to a positive value")
	}
	wn := make([]float64, len(w))
	for i, v := range w {
		wn[i] = v * (1 / sw)
	}
	var yy, ymean float64
	for i := range y {
		ymean += wn[i] * y[i]
		yy += wn[i] * y[i] * y[i]
	}
	if opt.FloatingMean {
		yy -= ymean * ymean
	}
	const epsneg = 1.1102230246251565e-16
	a, b, tau = make([]float64, len(freqs)), make([]float64, len(freqs)), make([]float64, len(freqs))
	pgram = make([]float64, len(freqs))
	cs, sn := make([]float64, len(x)), make([]float64, len(x))
	for fi, fr := range freqs {
		var cc, c, s, cS float64
		for i := range x {
			sinv, cosv := math.Sincos(fr * x[i])
			cs[i], sn[i] = cosv, sinv
			cc += wn[i] * cosv * cosv
			cS += wn[i] * cosv * sinv
			c += wn[i] * cosv
			s += wn[i] * sinv
		}
		ss := 1 - cc
		if opt.FloatingMean {
			cc -= c * c
			ss -= s * s
			cS -= c * s
		}
		tau[fi] = 0.5 * math.Atan2(2*cS, cc-ss)
		var yc, ys float64
		cc, c, s = 0, 0, 0
		for i := range x {
			sinv, cosv := math.Sincos(fr*x[i] - tau[fi])
			yc += wn[i] * y[i] * cosv
			ys += wn[i] * y[i] * sinv
			cc += wn[i] * cosv * cosv
			c += wn[i] * cosv
			s += wn[i] * sinv
		}
		ss = 1 - cc
		if opt.FloatingMean {
			yc -= ymean * c
			ys -= ymean * s
			cc -= c * c
			ss -= s * s
		}
		cc = math.Max(cc, epsneg)
		ss = math.Max(ss, epsneg)
		a[fi], b[fi] = yc/cc, ys/ss
		pgram[fi] = 2 * (a[fi]*yc + b[fi]*ys)
		if opt.Normalize {
			pgram[fi] *= 0.5 / yy
		} else {
			pgram[fi] *= float64(len(x)) / 4
		}
	}
	return a, b, tau, pgram, nil
}

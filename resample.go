package signal

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/go-fft/fft"
	"github.com/go-simd/floats"
)

// PadMode is the signal extension used by Upfirdn and ResamplePoly beyond the
// ends of the input.
type PadMode string

// Extension modes; the first group are those of numpy.pad.
const (
	PadModeConstant      PadMode = "constant"
	PadModeEdge          PadMode = "edge"
	PadModeWrap          PadMode = "wrap"
	PadModeSymmetric     PadMode = "symmetric"
	PadModeReflect       PadMode = "reflect"
	PadModeAntisymmetric PadMode = "antisymmetric"
	PadModeAntireflect   PadMode = "antireflect"
	// PadModeSmooth extends by the slope of the last two points at each end.
	PadModeSmooth PadMode = "smooth"
	// PadModeLine extends by the line through the first and last points.
	PadModeLine PadMode = "line"
)

// extValue returns x extended to index j by the given mode.
func extValue(x []float64, j int, mode PadMode, cval float64) float64 {
	n := len(x)
	if j >= 0 && j < n {
		return x[j]
	}
	switch mode { // constant extension is filled in by the caller
	case PadModeEdge:
		if j < 0 {
			return x[0]
		}
		return x[n-1]
	case PadModeWrap:
		return x[((j%n)+n)%n]
	case PadModeSymmetric, PadModeAntisymmetric:
		jj := ((j % (2 * n)) + 2*n) % (2 * n)
		if jj >= n {
			jj = 2*n - 1 - jj
		}
		if mode == PadModeAntisymmetric {
			fl := j / n
			if j < 0 && j%n != 0 {
				fl--
			}
			if fl%2 != 0 {
				return -x[jj]
			}
		}
		return x[jj]
	case PadModeReflect:
		if n == 1 {
			return x[0]
		}
		jj := ((j % (2*n - 2)) + 2*n - 2) % (2*n - 2)
		if jj >= n {
			jj = 2*n - 2 - jj
		}
		return x[jj]
	case PadModeAntireflect:
		if n == 1 {
			return x[0]
		}
		if j < 0 {
			return 2*x[0] - extValue(x, -j, mode, cval)
		}
		return 2*x[n-1] - extValue(x, 2*(n-1)-j, mode, cval)
	case PadModeSmooth:
		if n == 1 {
			return x[0]
		}
		if j < 0 {
			return x[0] + float64(-j)*(x[0]-x[1])
		}
		return x[n-1] + float64(j-(n-1))*(x[n-1]-x[n-2])
	}
	// PadModeLine
	if n == 1 {
		return x[0]
	}
	slope := (x[n-1] - x[0]) / float64(n-1)
	return x[0] + slope*float64(j)
}

func validPadMode(m PadMode) bool {
	switch m {
	case PadModeConstant, PadModeEdge, PadModeWrap, PadModeSymmetric, PadModeReflect,
		PadModeAntisymmetric, PadModeAntireflect, PadModeSmooth, PadModeLine:
		return true
	}
	return false
}

// simdDotMin is the vector length from which the assembly dot kernel of
// go-simd/floats beats the scalar loop (see docs/perf.md).
const simdDotMin = 24

// dot is the inner product of equal-length vectors.
func dot(a, b []float64) float64 {
	if len(a) >= simdDotMin {
		return floats.Dot(a, b)
	}
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// upfirdnLen64 is the output length of Upfirdn, in 64-bit arithmetic so that
// absurd arguments are rejected instead of overflowing a 32-bit int.
func upfirdnLen64(lh, nx, up, down int) int64 {
	return (int64(nx-1)*int64(up)+int64(lh)-1)/int64(down) + 1
}

func upfirdnLen(lh, nx, up, down int) int { return int(upfirdnLen64(lh, nx, up, down)) }

// maxAlloc bounds the size of any single output a constructor or filter will
// allocate (in elements), so that absurd arguments fail instead of exhausting
// memory.
const maxAlloc = 1 << 28

// Upfirdn upsamples x by up (inserting zeros), filters with h, and
// downsamples by down, computing only the samples that are kept
// (scipy.signal.upfirdn). mode and cval define the signal beyond its ends.
func Upfirdn(h, x []float64, up, down int, mode PadMode, cval float64) ([]float64, error) {
	if len(h) == 0 {
		return nil, errors.New("signal: h must be 1-D with non-zero length")
	}
	if up < 1 || down < 1 {
		return nil, errors.New("signal: both up and down must be >= 1")
	}
	if mode == "" {
		mode = PadModeConstant
	}
	if !validPadMode(mode) {
		return nil, fmt.Errorf("signal: unknown mode %q", mode)
	}
	if len(x) == 0 {
		return []float64{}, nil
	}
	if upfirdnLen64(len(h), len(x), up, down) > maxAlloc || up > 1<<20 {
		return nil, errors.New("signal: output too large")
	}
	nout := upfirdnLen(len(h), len(x), up, down)
	// polyphase subfilters, each reversed for a forward dot product
	hrev := make([][]float64, up)
	for r := 0; r < up && r < len(h); r++ {
		for k := r; k < len(h); k += up {
			hrev[r] = append(hrev[r], h[k])
		}
		slices.Reverse(hrev[r])
	}
	pad := (len(h)+up-1)/up + 1
	xe := make([]float64, len(x)+2*pad)
	copy(xe[pad:], x)
	if mode == PadModeConstant {
		for i := 0; i < pad; i++ {
			xe[i], xe[len(xe)-1-i] = cval, cval
		}
	} else {
		for i := 0; i < pad; i++ {
			xe[pad-1-i] = extValue(x, -1-i, mode, cval)
			xe[pad+len(x)+i] = extValue(x, len(x)+i, mode, cval)
		}
	}
	y := make([]float64, nout)
	for n := range y {
		m := n * down
		r := m % up
		if r >= len(h) {
			continue
		}
		base := m / up
		hr := hrev[r]
		lo := base + pad - (len(hr) - 1)
		y[n] = dot(hr, xe[lo:lo+len(hr)])
	}
	return y, nil
}

// ResamplePolyOptions are the optional parameters of ResamplePoly.
type ResamplePolyOptions struct {
	// Window designs the anti-aliasing filter (default Kaiser with beta 5).
	Window Window
	// Taps, if not nil, is the FIR filter to use instead.
	Taps []float64
	// PadType is how the signal is extended: "mean", "median", "minimum" or
	// "maximum" (the signal minus that value is filtered, then it is added
	// back), or one of the PadMode values (default PadModeConstant).
	PadType string
	// Cval is the value for PadModeConstant.
	Cval float64
}

func outputLenPoly(lh, nin, up, down int) int { return upfirdnLen(lh, nin, up, down) }

// ResamplePoly resamples x by the rational factor up/down with a polyphase
// FIR filter (scipy.signal.resample_poly; MATLAB resample). The filter is
// linear phase and the output is aligned with the input.
func ResamplePoly(x []float64, up, down int, o *ResamplePolyOptions) ([]float64, error) {
	var opt ResamplePolyOptions
	if o != nil {
		opt = *o
	}
	if up < 1 || down < 1 {
		return nil, errors.New("signal: up and down must be >= 1")
	}
	g := gcd(up, down)
	up, down = up/g, down/g
	if up == 1 && down == 1 {
		return slices.Clone(x), nil
	}
	nin := len(x)
	if nin == 0 {
		return []float64{}, nil
	}
	if up > maxAlloc/(nin+20) || down > maxAlloc/20 {
		return nil, errors.New("signal: resampling factors too large")
	}
	nout := nin * up
	nout = nout/down + b2i(nout%down != 0)
	var h []float64
	var halfLen int
	if opt.Taps != nil {
		h = slices.Clone(opt.Taps)
		halfLen = (len(h) - 1) / 2
	} else {
		maxRate := max(up, down)
		halfLen = 10 * maxRate
		w := opt.Window
		if w == nil {
			w = Kaiser(5)
		}
		var err error
		h, err = Firwin(2*halfLen+1, []float64{1 / float64(maxRate)}, &FirwinOptions{Window: w})
		if err != nil {
			return nil, err
		}
	}
	for i := range h {
		h[i] *= float64(up)
	}
	nPrePad := down - halfLen%down
	nPostPad := 0
	nPreRemove := (halfLen + nPrePad) / down
	for outputLenPoly(len(h)+nPrePad+nPostPad, nin, up, down) < nout+nPreRemove {
		nPostPad++
	}
	h = append(append(make([]float64, nPrePad), h...), make([]float64, nPostPad)...)

	pt := opt.PadType
	if pt == "" {
		pt = string(PadModeConstant)
	}
	mode, cval := PadModeConstant, 0.0
	var background float64
	xs := x
	switch pt {
	case "mean", "median", "minimum", "maximum":
		if nin > 0 {
			switch pt {
			case "mean":
				for _, v := range x {
					background += v
				}
				background /= float64(nin)
			case "median":
				background = median(x)
			case "minimum":
				background = slices.Min(x)
			default:
				background = slices.Max(x)
			}
		}
		xs = make([]float64, nin)
		for i, v := range x {
			xs[i] = v - background
		}
	default:
		mode = PadMode(pt)
		if !validPadMode(mode) {
			return nil, fmt.Errorf("signal: padtype must be one of: maximum, mean, median, minimum, or a pad mode, got %q", pt)
		}
		if mode == PadModeConstant {
			cval = opt.Cval
		} else if opt.Cval != 0 {
			return nil, fmt.Errorf("signal: cval has no effect when padtype is %s", pt)
		}
	}
	y, _ := Upfirdn(h, xs, up, down, mode, cval) // sizes and mode were validated above
	keep := y[nPreRemove : nPreRemove+nout]
	if background != 0 {
		for i := range keep {
			keep[i] += background
		}
	}
	return slices.Clone(keep), nil
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// ResampleOptions are the optional parameters of Resample.
type ResampleOptions struct {
	// Window, if not nil, weights the spectrum (a frequency-domain window,
	// generated for len(x) points and centred on DC like SciPy's).
	Window Window
	// WindowFreq, if not nil, is the spectral weighting in FFT order.
	WindowFreq []float64
}

// Resample resamples x to num samples using the FFT, assuming x is periodic
// (scipy.signal.resample). It is best for periodic signals; use ResamplePoly
// otherwise.
func Resample(x []float64, num int, o *ResampleOptions) ([]float64, error) {
	var opt ResampleOptions
	if o != nil {
		opt = *o
	}
	nx := len(x)
	if nx == 0 || num < 1 {
		return nil, errors.New("signal: x must be non-empty and num positive")
	}
	if num > maxAlloc {
		return nil, errors.New("signal: output too large")
	}
	var win []float64
	switch {
	case opt.WindowFreq != nil:
		if len(opt.WindowFreq) != nx {
			return nil, fmt.Errorf("signal: window length %d is not equal to the number of frequency bins %d", len(opt.WindowFreq), nx)
		}
		win = slices.Clone(opt.WindowFreq)
	case opt.Window != nil:
		w := opt.Window(nx, false)
		if w == nil {
			return nil, errors.New("signal: the window cannot be generated")
		}
		// fftshift: the window's centre goes to index 0
		win = make([]float64, nx)
		sh := nx / 2
		for i := range w {
			win[(i+nx-sh)%nx] = w[i]
		}
	}
	sFac := float64(nx) / float64(num)
	m := min(num, nx)
	m2 := m/2 + 1
	spec := fft.RFFT(x)
	if win != nil {
		// fold the window: W1[l] = (W[l] + W[-l]) / 2 for l > 0
		orig := slices.Clone(win)
		for i := 1; i < len(spec); i++ {
			win[i] = (orig[i] + orig[nx-i]) / 2
		}
		for i := range spec {
			spec[i] *= complex(win[i], 0)
		}
	}
	if m2 < len(spec) {
		spec = spec[:m2]
	}
	if m%2 == 0 && num != nx {
		f := 0.5
		if num < nx {
			f = 2
		}
		spec[m/2] *= complex(f, 0)
	}
	for i := range spec {
		spec[i] /= complex(sFac, 0)
	}
	return fft.IRFFT(spec, num), nil
}

// DecimateOptions are the optional parameters of Decimate.
type DecimateOptions struct {
	// N is the filter order (default 8 for IIR, 20*q for FIR, where it is the
	// number of taps minus one).
	N int
	// FIR uses a Hamming-window FIR filter instead of a Chebyshev type I IIR.
	FIR bool
	// NoZeroPhase filters once instead of forward and backward.
	NoZeroPhase bool
}

// Decimate downsamples x by the integer factor q after low-pass filtering to
// prevent aliasing (scipy.signal.decimate; MATLAB decimate).
func Decimate(x []float64, q int, o *DecimateOptions) ([]float64, error) {
	var opt DecimateOptions
	if o != nil {
		opt = *o
	}
	if q < 1 {
		return nil, errors.New("signal: q must be a positive integer")
	}
	if opt.N < 0 {
		return nil, errors.New("signal: filter order must not be negative")
	}
	if opt.FIR {
		n := opt.N
		if n == 0 {
			n = 20 * q
		}
		b, err := Firwin(n+1, []float64{1 / float64(q)}, &FirwinOptions{Window: Hamming})
		if err != nil {
			return nil, err
		}
		if !opt.NoZeroPhase {
			return ResamplePoly(x, 1, q, &ResamplePolyOptions{Taps: b})
		}
		y, _ := Upfirdn(b, x, 1, q, PadModeConstant, 0) // sizes were checked by Firwin
		return y[:len(x)/q+b2i(len(x)%q != 0)], nil
	}
	n := opt.N
	if n == 0 {
		n = 8
	}
	z, err := Cheby1(n, 0.05, []float64{0.8 / float64(q)}, Lowpass, nil)
	if err != nil {
		return nil, err
	}
	sos, _ := z.SOS(false) // a Chebyshev lowpass always pairs
	var y []float64
	if opt.NoZeroPhase {
		y, _, err = SOSFilt(sos, x, nil)
	} else {
		y, err = SOSFiltFilt(sos, x, nil)
	}
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(y)/q+1)
	for i := 0; i < len(y); i += q {
		out = append(out, y[i])
	}
	return out, nil
}

// Hilbert returns the analytic signal x + i*H(x), computed in the frequency
// domain (scipy.signal.hilbert; MATLAB hilbert). n is the FFT length (0 means
// len(x)).
func Hilbert(x []float64, n int) ([]complex128, error) {
	if n == 0 {
		n = len(x)
	}
	if n <= 0 {
		return nil, errors.New("signal: N must be positive")
	}
	if n > maxAlloc {
		return nil, errors.New("signal: N too large")
	}
	buf := make([]complex128, n)
	for i := 0; i < min(n, len(x)); i++ {
		buf[i] = complex(x[i], 0)
	}
	spec := fft.FFT(buf)
	if n%2 == 0 {
		for i := 1; i < n/2; i++ {
			spec[i] *= 2
		}
		clear(spec[n/2+1:])
	} else {
		for i := 1; i < (n+1)/2; i++ {
			spec[i] *= 2
		}
		clear(spec[(n+1)/2:])
	}
	return fft.IFFT(spec), nil
}

// ---- peaks -----------------------------------------------------------------

// Interval is an optional lower and upper bound of a peak property.
type Interval struct {
	Min, Max       float64
	HasMin, HasMax bool
}

// AtLeast is the interval [v, inf).
func AtLeast(v float64) *Interval { return &Interval{Min: v, HasMin: true} }

// AtMost is the interval (-inf, v].
func AtMost(v float64) *Interval { return &Interval{Max: v, HasMax: true} }

// Between is the interval [lo, hi].
func Between(lo, hi float64) *Interval {
	return &Interval{Min: lo, Max: hi, HasMin: true, HasMax: true}
}

func (iv *Interval) keep(v float64) bool {
	return (!iv.HasMin || iv.Min <= v) && (!iv.HasMax || v <= iv.Max)
}

// FindPeaksOptions are the optional conditions of FindPeaks; a nil condition
// is not applied.
type FindPeaksOptions struct {
	// Height is the required peak height.
	Height *Interval
	// Threshold is the required vertical distance to the neighbouring samples.
	Threshold *Interval
	// Distance is the required minimal horizontal distance (in samples, >= 1)
	// between neighbouring peaks; the smaller peaks are removed first.
	Distance float64
	// Prominence is the required prominence of the peaks.
	Prominence *Interval
	// Width is the required width at RelHeight.
	Width *Interval
	// Wlen limits the window (in samples, > 1) in which prominence is evaluated.
	Wlen float64
	// RelHeight is the fraction of the prominence at which widths are measured
	// (default 0.5; widths at 0 are not available, use a tiny value).
	RelHeight float64
	// PlateauSize is the required size of a flat peak.
	PlateauSize *Interval
}

// PeakProperties holds the properties computed for the peaks found; a slice
// is nil when the condition that computes it was not requested.
type PeakProperties struct {
	PlateauSizes, LeftEdges, RightEdges []int
	PeakHeights                         []float64
	LeftThresholds, RightThresholds     []float64
	Prominences                         []float64
	LeftBases, RightBases               []int
	Widths, WidthHeights                []float64
	LeftIPs, RightIPs                   []float64
}

func localMaxima(x []float64) (mid, left, right []int) {
	i, imax := 1, len(x)-1
	for i < imax {
		if x[i-1] < x[i] {
			ahead := i + 1
			for ahead < imax && x[ahead] == x[i] {
				ahead++
			}
			if x[ahead] < x[i] {
				left = append(left, i)
				right = append(right, ahead-1)
				mid = append(mid, (i+ahead-1)/2)
				i = ahead
			}
		}
		i++
	}
	return mid, left, right
}

func selectByDistance(peaks []int, priority []float64, distance float64) []bool {
	dist := int(math.Ceil(distance))
	keep := make([]bool, len(peaks))
	for i := range keep {
		keep[i] = true
	}
	order := make([]int, len(peaks))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return priority[order[a]] < priority[order[b]] })
	for i := len(peaks) - 1; i >= 0; i-- {
		j := order[i]
		if !keep[j] {
			continue
		}
		for k := j - 1; k >= 0 && peaks[j]-peaks[k] < dist; k-- {
			keep[k] = false
		}
		for k := j + 1; k < len(peaks) && peaks[k]-peaks[j] < dist; k++ {
			keep[k] = false
		}
	}
	return keep
}

// PeakProminences calculates the prominence of each peak: its height above the
// higher of the two lowest contours towards the next higher peaks
// (scipy.signal.peak_prominences). wlen <= 1 means the whole signal.
func PeakProminences(x []float64, peaks []int, wlen float64) (prom []float64, left, right []int, err error) {
	w := -1
	if wlen != 0 {
		if !(wlen > 1) {
			return nil, nil, nil, fmt.Errorf("signal: wlen must be larger than 1, was %v", wlen)
		}
		w = int(math.Ceil(wlen))
	}
	prom = make([]float64, len(peaks))
	left, right = make([]int, len(peaks)), make([]int, len(peaks))
	for n, peak := range peaks {
		if peak < 0 || peak >= len(x) {
			return nil, nil, nil, fmt.Errorf("signal: peak %d is not a valid index for x", peak)
		}
		imin, imax := 0, len(x)-1
		if w >= 2 {
			imin = max(peak-w/2, imin)
			imax = min(peak+w/2, imax)
		}
		i := peak
		left[n] = peak
		lmin := x[peak]
		for imin <= i && x[i] <= x[peak] {
			if x[i] < lmin {
				lmin, left[n] = x[i], i
			}
			i--
		}
		i = peak
		right[n] = peak
		rmin := x[peak]
		for i <= imax && x[i] <= x[peak] {
			if x[i] < rmin {
				rmin, right[n] = x[i], i
			}
			i++
		}
		prom[n] = x[peak] - math.Max(lmin, rmin)
	}
	return prom, left, right, nil
}

// PeakWidths calculates the width of each peak at relHeight times its
// prominence below the peak (scipy.signal.peak_widths). The prominence data
// are those of PeakProminences.
func PeakWidths(x []float64, peaks []int, relHeight float64, prom []float64, left, right []int) (widths, heights, leftIP, rightIP []float64, err error) {
	if relHeight < 0 {
		return nil, nil, nil, nil, errors.New("signal: rel_height must be greater or equal to 0.0")
	}
	if len(prom) != len(peaks) || len(left) != len(peaks) || len(right) != len(peaks) {
		return nil, nil, nil, nil, errors.New("signal: prominence data must have one entry per peak")
	}
	widths, heights = make([]float64, len(peaks)), make([]float64, len(peaks))
	leftIP, rightIP = make([]float64, len(peaks)), make([]float64, len(peaks))
	for p, peak := range peaks {
		imin, imax := left[p], right[p]
		if !(0 <= imin && imin <= peak && peak <= imax && imax < len(x)) {
			return nil, nil, nil, nil, fmt.Errorf("signal: prominence data is invalid for peak %d", peak)
		}
		h := x[peak] - prom[p]*relHeight
		heights[p] = h
		i := peak
		for imin < i && h < x[i] {
			i--
		}
		lip := float64(i)
		if x[i] < h {
			lip += (h - x[i]) / (x[i+1] - x[i])
		}
		i = peak
		for i < imax && h < x[i] {
			i++
		}
		rip := float64(i)
		if x[i] < h {
			rip -= (h - x[i]) / (x[i-1] - x[i])
		}
		widths[p], leftIP[p], rightIP[p] = rip-lip, lip, rip
	}
	return widths, heights, leftIP, rightIP, nil
}

func pick[T any](v []T, keep []bool) []T {
	if v == nil {
		return nil
	}
	out := make([]T, 0, len(v))
	for i, k := range keep {
		if k {
			out = append(out, v[i])
		}
	}
	return out
}

// FindPeaks finds the local maxima of x satisfying the conditions of o
// (scipy.signal.find_peaks; MATLAB findpeaks). A flat peak is reported at its
// middle sample. Conditions apply in the order plateau size, height,
// threshold, distance, prominence, width.
func FindPeaks(x []float64, o *FindPeaksOptions) (peaks []int, props PeakProperties, err error) {
	var opt FindPeaksOptions
	if o != nil {
		opt = *o
	}
	if opt.Distance != 0 && opt.Distance < 1 {
		return nil, props, errors.New("signal: distance must be greater or equal to 1")
	}
	if opt.RelHeight == 0 {
		opt.RelHeight = 0.5
	}
	peaks, left, right := localMaxima(x)
	apply := func(keep []bool) {
		peaks = pick(peaks, keep)
		props.PlateauSizes, props.LeftEdges, props.RightEdges = pick(props.PlateauSizes, keep), pick(props.LeftEdges, keep), pick(props.RightEdges, keep)
		props.PeakHeights = pick(props.PeakHeights, keep)
		props.LeftThresholds, props.RightThresholds = pick(props.LeftThresholds, keep), pick(props.RightThresholds, keep)
		props.Prominences, props.LeftBases, props.RightBases = pick(props.Prominences, keep), pick(props.LeftBases, keep), pick(props.RightBases, keep)
		props.Widths, props.WidthHeights = pick(props.Widths, keep), pick(props.WidthHeights, keep)
		props.LeftIPs, props.RightIPs = pick(props.LeftIPs, keep), pick(props.RightIPs, keep)
	}
	if opt.PlateauSize != nil {
		keep := make([]bool, len(peaks))
		props.PlateauSizes = make([]int, len(peaks))
		for i := range peaks {
			props.PlateauSizes[i] = right[i] - left[i] + 1
			keep[i] = opt.PlateauSize.keep(float64(props.PlateauSizes[i]))
		}
		props.LeftEdges, props.RightEdges = left, right
		apply(keep)
	}
	if opt.Height != nil {
		props.PeakHeights = make([]float64, len(peaks))
		keep := make([]bool, len(peaks))
		for i, p := range peaks {
			props.PeakHeights[i] = x[p]
			keep[i] = opt.Height.keep(x[p])
		}
		apply(keep)
	}
	if opt.Threshold != nil {
		props.LeftThresholds, props.RightThresholds = make([]float64, len(peaks)), make([]float64, len(peaks))
		keep := make([]bool, len(peaks))
		for i, p := range peaks {
			l, r := x[p]-x[p-1], x[p]-x[p+1]
			props.LeftThresholds[i], props.RightThresholds[i] = l, r
			keep[i] = opt.Threshold.keep(l) && opt.Threshold.keep(r)
		}
		apply(keep)
	}
	if opt.Distance != 0 {
		pri := make([]float64, len(peaks))
		for i, p := range peaks {
			pri[i] = x[p]
		}
		apply(selectByDistance(peaks, pri, opt.Distance))
	}
	if opt.Prominence != nil || opt.Width != nil {
		props.Prominences, props.LeftBases, props.RightBases, err = PeakProminences(x, peaks, opt.Wlen)
		if err != nil {
			return nil, PeakProperties{}, err
		}
	}
	if opt.Prominence != nil {
		keep := make([]bool, len(peaks))
		for i, p := range props.Prominences {
			keep[i] = opt.Prominence.keep(p)
		}
		apply(keep)
	}
	if opt.Width != nil {
		props.Widths, props.WidthHeights, props.LeftIPs, props.RightIPs, err = PeakWidths(x, peaks, opt.RelHeight, props.Prominences, props.LeftBases, props.RightBases)
		if err != nil {
			return nil, PeakProperties{}, err
		}
		keep := make([]bool, len(peaks))
		for i, w := range props.Widths {
			keep[i] = opt.Width.keep(w)
		}
		apply(keep)
	}
	if peaks == nil {
		peaks = []int{}
	}
	return peaks, props, nil
}

package signal

import (
	"errors"
	"math"

	"github.com/go-fft/fft"
)

// ConvMode selects which part of the full convolution is returned.
type ConvMode int

const (
	// ConvFull returns the complete discrete convolution, len(a)+len(b)-1 samples.
	ConvFull ConvMode = iota
	// ConvSame returns the central part, the size of the first input.
	ConvSame
	// ConvValid returns only the samples computed without zero padding.
	ConvValid
)

// ConvMethod selects the algorithm of Convolve and Correlate.
type ConvMethod int

const (
	// ConvAuto picks the faster of direct and FFT (scipy's choose_conv_method).
	ConvAuto ConvMethod = iota
	// ConvDirect sums the products directly.
	ConvDirect
	// ConvFFT uses the FFT.
	ConvFFT
)

func checkMode(m ConvMode) error {
	if m < ConvFull || m > ConvValid {
		return errors.New("signal: acceptable mode flags are full, same or valid")
	}
	return nil
}

// centered returns the middle n samples of x.
func centered(x []float64, n int) []float64 {
	start := (len(x) - n) / 2
	return x[start : start+n]
}

// applyMode crops the full convolution of inputs of length n1 and n2.
func applyMode(full []float64, n1, n2 int, mode ConvMode) []float64 {
	switch mode {
	case ConvSame:
		full = centered(full, n1)
	case ConvValid:
		full = centered(full, max(n1, n2)-min(n1, n2)+1)
	}
	return append([]float64(nil), full...)
}

// convolveDirect is the full linear convolution by direct summation.
func convolveDirect(x, h []float64) []float64 {
	if len(x) < len(h) {
		x, h = h, x
	}
	out := make([]float64, len(x)+len(h)-1)
	if len(h) >= simdDotMin {
		// Over the zero-padded signal, out[n] = Σ_k hr[k] xp[n+k]: eight
		// outputs at a time, each a chain of multiply-adds in a register (a
		// per-sample dot-product call costs more than it computes at these
		// lengths).
		hr := reversed(h)
		xp := make([]float64, len(x)+2*(len(h)-1))
		copy(xp[len(h)-1:], x)
		correlate8(out, hr, xp)
		return out
	}
	for j, hv := range h {
		if hv == 0 {
			continue
		}
		axpy(out[j:j+len(x)], hv, x)
	}
	return out
}

// correlate8 sets out[n] = Σ_k hr[k]*xp[n+k] (k ascending) for every n, with
// len(xp) >= len(out)+len(hr)-1, eight outputs per pass in registers.
func correlate8(out, hr, xp []float64) {
	m := len(hr)
	n := 0
	for ; n+8 <= len(out); n += 8 {
		w := xp[n : n+m+7]
		var s0, s1, s2, s3, s4, s5, s6, s7 float64
		var t0, t1, t2, t3, t4, t5, t6, t7 float64
		// Two taps per step share one load of nine window samples, and the
		// even and odd taps go to separate partial sums: sixteen independent
		// multiply-add chains, so the FMA latency no longer bounds the loop.
		k := 0
		for ; k+2 <= m; k += 2 {
			ha, hb := hr[k], hr[k+1]
			v := w[k : k+9]
			s0 += ha * v[0]
			t0 += hb * v[1]
			s1 += ha * v[1]
			t1 += hb * v[2]
			s2 += ha * v[2]
			t2 += hb * v[3]
			s3 += ha * v[3]
			t3 += hb * v[4]
			s4 += ha * v[4]
			t4 += hb * v[5]
			s5 += ha * v[5]
			t5 += hb * v[6]
			s6 += ha * v[6]
			t6 += hb * v[7]
			s7 += ha * v[7]
			t7 += hb * v[8]
		}
		s0, s1, s2, s3 = s0+t0, s1+t1, s2+t2, s3+t3
		s4, s5, s6, s7 = s4+t4, s5+t5, s6+t6, s7+t7
		for ; k < m; k++ {
			hk := hr[k]
			v := w[k : k+8]
			s0 += hk * v[0]
			s1 += hk * v[1]
			s2 += hk * v[2]
			s3 += hk * v[3]
			s4 += hk * v[4]
			s5 += hk * v[5]
			s6 += hk * v[6]
			s7 += hk * v[7]
		}
		out[n], out[n+1], out[n+2], out[n+3] = s0, s1, s2, s3
		out[n+4], out[n+5], out[n+6], out[n+7] = s4, s5, s6, s7
	}
	for ; n < len(out); n++ {
		var s float64
		for k, hk := range hr {
			s += hk * xp[n+k]
		}
		out[n] = s
	}
}

// axpy adds a*x to y (len(y) == len(x)).
func axpy(y []float64, a float64, x []float64) {
	i := 0
	for ; i+4 <= len(x); i += 4 {
		y[i] += a * x[i]
		y[i+1] += a * x[i+1]
		y[i+2] += a * x[i+2]
		y[i+3] += a * x[i+3]
	}
	for ; i < len(x); i++ {
		y[i] += a * x[i]
	}
}

// convolveFFT is the full linear convolution by FFT.
func convolveFFT(x, h []float64) []float64 {
	n := len(x) + len(h) - 1
	nf := nextFastLen(n)
	px := make([]float64, nf)
	ph := make([]float64, nf)
	copy(px, x)
	copy(ph, h)
	sx, sh := fft.RFFT(px), fft.RFFT(ph)
	for i := range sx {
		sx[i] *= sh[i]
	}
	return fft.IRFFT(sx, nf)[:n]
}

// FFTConvolve convolves a and b using the FFT (scipy.signal.fftconvolve).
func FFTConvolve(a, b []float64, mode ConvMode) ([]float64, error) {
	if err := checkMode(mode); err != nil {
		return nil, err
	}
	if len(a) == 0 || len(b) == 0 {
		return []float64{}, nil
	}
	if len(a) == 1 && len(b) == 1 {
		return []float64{a[0] * b[0]}, nil
	}
	return applyMode(convolveFFT(a, b), len(a), len(b), mode), nil
}

// lambertWm1 is the lower real branch W_{-1}(x) for x in [-1/e, 0).
func lambertWm1(x float64) float64 {
	w := math.Log(-x) - math.Log(-math.Log(-x))
	for i := 0; ; i++ {
		ew := math.Exp(w)
		f := w*ew - x
		d := ew*(w+1) - (w+2)*f/(2*w+2)
		nw := w - f/d
		if math.Abs(nw-w) <= 1e-15*math.Abs(nw) || i == 50 {
			return nw
		}
		w = nw
	}
}

// calcOALens returns scipy's block layout for overlap-add: block size, overlap
// (zero meaning a single FFT), and the step sizes through each input.
func calcOALens(s1, s2 int) (block, overlap, step1, step2 int) {
	if s1 == s2 || s1 == 1 || s2 == 1 {
		return s1 + s2 - 1, 0, s1, s2
	}
	swapped := false
	if s2 > s1 {
		s1, s2 = s2, s1
		swapped = true
	}
	if float64(s2) >= float64(s1)/2 {
		if swapped {
			s1, s2 = s2, s1
		}
		return s1 + s2 - 1, 0, s1, s2
	}
	overlap = s2 - 1
	opt := -float64(overlap) * lambertWm1(-1/(2*math.E*float64(overlap)))
	block = nextFastLen(int(math.Ceil(opt)))
	if block >= s1 {
		if swapped {
			s1, s2 = s2, s1
		}
		return s1 + s2 - 1, 0, s1, s2
	}
	if !swapped {
		return block, overlap, block - s2 + 1, s2
	}
	return block, overlap, s2, block - s2 + 1
}

// OAConvolve convolves a and b by the overlap-add method, which is faster than
// FFTConvolve when one input is much longer than the other
// (scipy.signal.oaconvolve).
func OAConvolve(a, b []float64, mode ConvMode) ([]float64, error) {
	if err := checkMode(mode); err != nil {
		return nil, err
	}
	if len(a) == 0 || len(b) == 0 {
		return []float64{}, nil
	}
	s1, s2 := len(a), len(b)
	if s1 == s2 {
		return FFTConvolve(a, b, mode)
	}
	block, overlap, step1, step2 := calcOALens(s1, s2)
	if step1 == s1 && step2 == s2 {
		return FFTConvolve(a, b, mode)
	}
	final := s1 + s2 - 1
	nsteps := func(s, step int) int {
		if s <= step {
			return 1
		}
		n := (s + step) / step // ceil((s+1)/step)
		if (block-overlap)*n < final {
			n++
		}
		return n
	}
	n1, n2 := nsteps(s1, step1), nsteps(s2, step2)
	spectra := func(x []float64, n, step int) [][]complex128 {
		out := make([][]complex128, n)
		for i := range out {
			buf := make([]float64, block)
			lo := i * step
			if lo < len(x) {
				copy(buf[:step], x[lo:min(lo+step, len(x))])
			}
			out[i] = fft.RFFT(buf)
		}
		return out
	}
	sp1, sp2 := spectra(a, n1, step1), spectra(b, n2, step2)
	rows := max(n1, n2)
	ret := make([]float64, rows*(block-overlap)+overlap)
	for r := 0; r < rows; r++ {
		x1, x2 := sp1[min(r, n1-1)], sp2[min(r, n2-1)]
		prod := make([]complex128, len(x1))
		for i := range prod {
			prod[i] = x1[i] * x2[i]
		}
		seg := fft.IRFFT(prod, block)
		base := r * (block - overlap)
		for i, v := range seg {
			ret[base+i] += v
		}
	}
	return applyMode(ret[:final], s1, s2, mode), nil
}

// convOps estimates the operation counts of the direct and FFT methods for
// a 1-D convolution (scipy's _conv_ops).
func convOps(s1, s2 int, mode ConvMode) (fftOps, directOps float64) {
	switch mode {
	case ConvFull:
		directOps = float64(s1) * float64(s2)
	case ConvValid:
		if s2 >= s1 {
			directOps = float64(s2-s1+1) * float64(s1)
		} else {
			directOps = float64(s1-s2+1) * float64(s2)
		}
	default:
		if s1 < s2 {
			directOps = float64(s1) * float64(s2)
		} else {
			directOps = float64(s1)*float64(s2) - float64(s2/2)*float64((s2+1)/2)
		}
	}
	n := float64(s1 + s2 - 1)
	return 3 * n * math.Log(n), directOps
}

// ChooseConvMethod returns the faster method for convolving inputs of length
// n1 and n2, by SciPy's fitted cost model (scipy.signal.choose_conv_method
// without measuring).
func ChooseConvMethod(n1, n2 int, mode ConvMode) ConvMethod {
	fftOps, directOps := convOps(n1, n2, mode)
	offset := -1e-3
	var oFFT, oDirect, oOff float64
	switch mode {
	case ConvValid:
		oFFT, oDirect, oOff = 1.89095737e-9, 2.1364985e-10, offset
	case ConvFull:
		oFFT, oDirect, oOff = 1.7649070e-9, 2.1414831e-10, offset
	default:
		if n2 <= n1 {
			oFFT, oDirect, oOff = 3.2646654e-9, 2.8478277e-10, offset
		} else {
			oFFT, oDirect, oOff = 3.21635404e-9, 1.1773253e-8, -1e-5
		}
	}
	if oFFT*fftOps < oDirect*directOps+oOff {
		return ConvFFT
	}
	return ConvDirect
}

// Convolve convolves a and b (scipy.signal.convolve; MATLAB conv).
func Convolve(a, b []float64, mode ConvMode, method ConvMethod) ([]float64, error) {
	if err := checkMode(mode); err != nil {
		return nil, err
	}
	if method < ConvAuto || method > ConvFFT {
		return nil, errors.New("signal: acceptable method flags are auto, direct or fft")
	}
	if len(a) == 0 || len(b) == 0 {
		return []float64{}, nil
	}
	if method == ConvAuto {
		x, h := a, b
		if mode == ConvValid && len(x) < len(h) {
			x, h = h, x
		}
		method = ChooseConvMethod(len(x), len(h), mode)
	}
	if method == ConvFFT {
		return FFTConvolve(a, b, mode)
	}
	return applyMode(convolveDirect(a, b), len(a), len(b), mode), nil
}

// Correlate cross-correlates a and b: out[k] = sum_n a[n+k] b[n]
// (scipy.signal.correlate; MATLAB xcorr up to its lag convention).
func Correlate(a, b []float64, mode ConvMode, method ConvMethod) ([]float64, error) {
	return Convolve(a, reversed(b), mode, method)
}

// CorrelationLags returns the lag of each output of Correlate for inputs of
// the given lengths (scipy.signal.correlation_lags).
func CorrelationLags(n1, n2 int, mode ConvMode) ([]int, error) {
	if err := checkMode(mode); err != nil {
		return nil, err
	}
	full := make([]int, 0, n1+n2-1)
	for l := -n2 + 1; l < n1; l++ {
		full = append(full, l)
	}
	switch mode {
	case ConvSame:
		mid := len(full) / 2
		bound := n1 / 2
		if n1%2 == 0 {
			return full[mid-bound : mid+bound], nil
		}
		return full[mid-bound : mid+bound+1], nil
	case ConvValid:
		bound := n1 - n2
		var lags []int
		if bound >= 0 {
			for l := 0; l <= bound; l++ {
				lags = append(lags, l)
			}
		} else {
			for l := bound; l <= 0; l++ {
				lags = append(lags, l)
			}
		}
		return lags, nil
	}
	return full, nil
}

// Deconvolve divides the polynomial signal by divisor, returning the quotient
// and remainder (scipy.signal.deconvolve): signal = conv(divisor, quot) + rem.
func Deconvolve(signal, divisor []float64) (quot, rem []float64, err error) {
	if len(signal) == 0 || len(divisor) == 0 {
		return nil, nil, errors.New("signal: signal and divisor must be non-empty")
	}
	n, d := len(signal), len(divisor)
	if d > n {
		return []float64{}, append([]float64(nil), signal...), nil
	}
	imp := make([]float64, n-d+1)
	imp[0] = 1
	quot, _, err = LFilter(signal, divisor, imp, nil)
	if err != nil {
		return nil, nil, err
	}
	conv := convolveDirect(divisor, quot)
	rem = make([]float64, n)
	for i := range rem {
		rem[i] = signal[i] - conv[i]
	}
	return quot, rem, nil
}

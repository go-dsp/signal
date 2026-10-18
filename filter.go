package signal

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// LFilter filters x with the IIR or FIR filter b/a using the direct form II
// transposed structure (scipy.signal.lfilter; MATLAB filter). zi, if not nil,
// holds the initial state (length max(len(a), len(b))-1), and the final
// state zf is returned so that long signals can be filtered in pieces.
func LFilter(b, a, x, zi []float64) (y, zf []float64, err error) {
	if len(b) == 0 || len(a) == 0 {
		return nil, nil, errors.New("signal: b and a must be non-empty")
	}
	if a[0] == 0 {
		return nil, nil, errors.New("signal: a[0] must be non-zero")
	}
	k := max(len(a), len(b))
	if zi != nil && len(zi) != k-1 {
		return nil, nil, fmt.Errorf("signal: unexpected length of zi: expected %d, found %d", k-1, len(zi))
	}
	bb := make([]float64, k)
	aa := make([]float64, k)
	for i, v := range b {
		bb[i] = v / a[0]
	}
	for i, v := range a {
		aa[i] = v / a[0]
	}
	z := make([]float64, k) // z[k-1] stays 0 (the extra state of the shift)
	copy(z, zi)
	y = make([]float64, len(x))
	if len(a) == 1 {
		lfilterFIR(bb[:len(b)], x, z, y)
	} else {
		lfilterIIR(bb, aa, x, z, y)
	}
	return y, z[:k-1], nil
}

// lfilterFIR is the direct form II transposed filter with a == 1.
func lfilterFIR(b, x, z, y []float64) {
	nb := len(b)
	for n, xn := range x {
		y[n] = z[0] + b[0]*xn
		for i := 1; i < nb-1; i++ {
			z[i-1] = z[i] + b[i]*xn
		}
		if nb > 1 {
			z[nb-2] = b[nb-1] * xn
		}
	}
}

// lfilterIIR is the direct form II transposed filter.
func lfilterIIR(b, a, x, z, y []float64) {
	k := len(b)
	for n, xn := range x {
		yn := z[0] + b[0]*xn
		y[n] = yn
		for i := 0; i < k-2; i++ {
			z[i] = z[i+1] + b[i+1]*xn - a[i+1]*yn
		}
		z[k-2] = b[k-1]*xn - a[k-1]*yn
	}
}

// LFilterZI returns the initial state of the filter b/a for a step input of
// unit amplitude: the state in which a constant input of 1 gives a constant
// output (scipy.signal.lfilter_zi).
func LFilterZI(b, a []float64) ([]float64, error) {
	if len(b) == 0 || len(a) == 0 {
		return nil, errors.New("signal: b and a must be non-empty")
	}
	if a[0] == 0 {
		return nil, errors.New("signal: first coefficient of a must be non-zero")
	}
	bn := slices.Clone(b)
	an := slices.Clone(a)
	if a[0] != 1 {
		for i := range bn {
			bn[i] /= a[0]
		}
		for i := range an {
			an[i] /= a[0]
		}
	}
	var suma, sumb float64
	for _, v := range an {
		suma += v
	}
	for _, v := range bn {
		sumb += v
	}
	if suma == 0 {
		return nil, errors.New("signal: filter not stable due to sum(a) == 0, a pole at z = 1")
	}
	yinf := sumb / suma
	n := max(len(an), len(bn))
	d := make([]float64, n)
	for i := range d {
		if i < len(bn) {
			d[i] += bn[i]
		}
		if i < len(an) {
			d[i] -= yinf * an[i]
		}
	}
	// zi[k] = sum_{j > k} d[j]
	zi := make([]float64, n-1)
	var s float64
	for i := n - 1; i >= 1; i-- {
		s += d[i]
		zi[i-1] = s
	}
	return zi, nil
}

// LFilterIC constructs the initial state of lfilter for given past outputs y
// and inputs x (most recent first), scipy.signal.lfiltic. x may be nil
// (zero input history).
func LFilterIC(b, a, y, x []float64) ([]float64, error) {
	if len(a) == 0 {
		return nil, errors.New("signal: there must be at least one a coefficient")
	}
	if a[0] == 0 {
		return nil, errors.New("signal: first a filter coefficient must be non-zero")
	}
	n, m := len(a)-1, len(b)-1
	k := max(m, n)
	xx := make([]float64, max(m, len(x)))
	copy(xx, x)
	yy := make([]float64, max(n, len(y)))
	copy(yy, y)
	zi := make([]float64, k)
	for i := 0; i < m; i++ {
		for j := i + 1; j <= m; j++ {
			zi[i] += b[j] * xx[j-i-1]
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j <= n; j++ {
			zi[i] -= a[j] * yy[j-i-1]
		}
	}
	if a[0] != 1 {
		for i := range zi {
			zi[i] /= a[0]
		}
	}
	return zi, nil
}

// PadType is the signal extension used by FiltFilt.
type PadType int

const (
	// PadOdd extends by odd (point) reflection about the end points (default).
	PadOdd PadType = iota
	// PadEven extends by even reflection.
	PadEven
	// PadConstant extends with the end values.
	PadConstant
	// PadNone does not extend the signal.
	PadNone
)

// FiltMethod selects the way FiltFilt handles the ends of the signal.
type FiltMethod int

const (
	// FiltPad pads the signal (default).
	FiltPad FiltMethod = iota
	// FiltGust uses Gustafsson's method, choosing initial conditions that
	// make the forward-backward and backward-forward results agree.
	FiltGust
)

// FiltFiltOptions are the optional parameters of FiltFilt.
type FiltFiltOptions struct {
	PadType PadType
	// PadLen is the number of samples to extend at each end; zero means the
	// default of 3*max(len(a), len(b)). Use PadNone for no padding.
	PadLen int
	Method FiltMethod
	// IRLen is the impulse response length used by FiltGust (0: the whole
	// signal).
	IRLen int
}

func extend(x []float64, n int, t PadType) []float64 {
	out := make([]float64, 0, len(x)+2*n)
	l := len(x)
	switch t {
	case PadEven:
		for i := n; i >= 1; i-- {
			out = append(out, x[i])
		}
		out = append(out, x...)
		for i := 1; i <= n; i++ {
			out = append(out, x[l-1-i])
		}
	case PadOdd:
		for i := n; i >= 1; i-- {
			out = append(out, 2*x[0]-x[i])
		}
		out = append(out, x...)
		for i := 1; i <= n; i++ {
			out = append(out, 2*x[l-1]-x[l-1-i])
		}
	default:
		for i := 0; i < n; i++ {
			out = append(out, x[0])
		}
		out = append(out, x...)
		for i := 0; i < n; i++ {
			out = append(out, x[l-1])
		}
	}
	return out
}

func reversed(x []float64) []float64 {
	r := slices.Clone(x)
	slices.Reverse(r)
	return r
}

func validatePad(t PadType, padlen, ntaps, n int) (edge int, err error) {
	if t < PadOdd || t > PadNone {
		return 0, fmt.Errorf("signal: unknown pad type %d", t)
	}
	switch {
	case t == PadNone:
		edge = 0
	case padlen < 0:
		return 0, errors.New("signal: padlen must not be negative")
	case padlen == 0:
		edge = 3 * ntaps
	default:
		edge = padlen
	}
	if n <= edge {
		return 0, fmt.Errorf("signal: the length of the input vector x must be greater than padlen, which is %d", edge)
	}
	return edge, nil
}

// FiltFilt applies the filter b/a forward and backward, giving zero phase
// distortion and squared magnitude response (scipy.signal.filtfilt; MATLAB
// filtfilt). The signal is extended at both ends to reduce transients.
func FiltFilt(b, a, x []float64, o *FiltFiltOptions) ([]float64, error) {
	var opt FiltFiltOptions
	if o != nil {
		opt = *o
	}
	if len(b) == 0 || len(a) == 0 {
		return nil, errors.New("signal: b and a must be non-empty")
	}
	switch opt.Method {
	case FiltGust:
		return filtfiltGust(b, a, x, opt.IRLen)
	case FiltPad:
	default:
		return nil, errors.New("signal: method must be pad or gust")
	}
	edge, err := validatePad(opt.PadType, opt.PadLen, max(len(a), len(b)), len(x))
	if err != nil {
		return nil, err
	}
	zi, err := LFilterZI(b, a)
	if err != nil {
		return nil, err
	}
	ext := x
	if edge > 0 {
		ext = extend(x, edge, opt.PadType)
	}
	return filtfiltCore(ext, edge, func(sig []float64, x0 float64) []float64 {
		z := slices.Clone(zi)
		for i := range z {
			z[i] *= x0
		}
		y, _, _ := LFilter(b, a, sig, z)
		return y
	}), nil
}

// filtfiltCore runs the forward and backward passes of filtfilt around
// pass, which filters a signal starting from the steady state of its first
// sample.
func filtfiltCore(ext []float64, edge int, pass func(sig []float64, x0 float64) []float64) []float64 {
	y := pass(ext, ext[0])
	y = reversed(pass(reversed(y), y[len(y)-1]))
	if edge > 0 {
		y = y[edge : len(y)-edge]
	}
	return y
}

func validateSOS(sos SOS) error {
	if len(sos) == 0 {
		return errors.New("signal: sos must have at least one section")
	}
	for _, s := range sos {
		if s[3] != 1 {
			return errors.New("signal: sos[:, 3] should be all ones")
		}
	}
	return nil
}

// SOSFilt filters x with a cascade of second-order sections
// (scipy.signal.sosfilt). zi, if not nil, holds the initial state of each
// section (len(sos) rows of two values); the final state is returned.
func SOSFilt(sos SOS, x []float64, zi [][2]float64) (y []float64, zf [][2]float64, err error) {
	if err := validateSOS(sos); err != nil {
		return nil, nil, err
	}
	if zi != nil && len(zi) != len(sos) {
		return nil, nil, fmt.Errorf("signal: invalid zi: need %d sections, got %d", len(sos), len(zi))
	}
	zf = make([][2]float64, len(sos))
	copy(zf, zi)
	y = slices.Clone(x)
	for s, sec := range sos {
		b0, b1, b2, a1, a2 := sec[0], sec[1], sec[2], sec[4], sec[5]
		z0, z1 := zf[s][0], zf[s][1]
		for n, xc := range y {
			xn := b0*xc + z0
			z0 = b1*xc - a1*xn + z1
			z1 = b2*xc - a2*xn
			y[n] = xn
		}
		zf[s] = [2]float64{z0, z1}
	}
	return y, zf, nil
}

// SOSFiltZI returns the initial state of each section for a unit step input
// (scipy.signal.sosfilt_zi).
func SOSFiltZI(sos SOS) ([][2]float64, error) {
	if len(sos) == 0 {
		return nil, errors.New("signal: sos must have at least one section")
	}
	zi := make([][2]float64, len(sos))
	scale := 1.0
	for i, s := range sos {
		z, err := LFilterZI(s[:3], s[3:])
		if err != nil {
			return nil, err
		}
		zi[i] = [2]float64{scale * z[0], scale * z[1]}
		scale *= (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5])
	}
	return zi, nil
}

// SOSFiltFilt is the zero-phase forward-backward filter for second-order
// sections (scipy.signal.sosfiltfilt).
func SOSFiltFilt(sos SOS, x []float64, o *FiltFiltOptions) ([]float64, error) {
	var opt FiltFiltOptions
	if o != nil {
		opt = *o
	}
	if err := validateSOS(sos); err != nil {
		return nil, err
	}
	ntaps := 2*len(sos) + 1
	z2, z5 := 0, 0
	for _, s := range sos {
		if s[2] == 0 {
			z2++
		}
		if s[5] == 0 {
			z5++
		}
	}
	ntaps -= min(z2, z5)
	edge, err := validatePad(opt.PadType, opt.PadLen, ntaps, len(x))
	if err != nil {
		return nil, err
	}
	zi, err := SOSFiltZI(sos)
	if err != nil {
		return nil, err
	}
	ext := x
	if edge > 0 {
		ext = extend(x, edge, opt.PadType)
	}
	return filtfiltCore(ext, edge, func(sig []float64, x0 float64) []float64 {
		z := slices.Clone(zi)
		for i := range z {
			z[i][0] *= x0
			z[i][1] *= x0
		}
		y, _, _ := SOSFilt(sos, sig, z)
		return y
	}), nil
}

// filtfiltGust is Gustafsson's forward-backward filter (scipy's method="gust").
func filtfiltGust(b, a, x []float64, irlen int) ([]float64, error) {
	order := max(len(b), len(a)) - 1
	if order == 0 {
		s := b[0] / a[0]
		y := make([]float64, len(x))
		for i, v := range x {
			y[i] = s * s * v
		}
		return y, nil
	}
	n := len(x)
	if n == 0 {
		return nil, errors.New("signal: x must not be empty")
	}
	m := n
	if irlen > 0 && n > 2*irlen {
		m = irlen
	}
	// Obs: columns are the zero-input responses to unit initial states
	obs := make([][]float64, m)
	for i := range obs {
		obs[i] = make([]float64, order)
	}
	zi := make([]float64, order)
	zi[0] = 1
	col, _, err := LFilter(b, a, make([]float64, m), zi)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m; i++ {
		obs[i][0] = col[i]
	}
	for k := 1; k < order; k++ {
		for i := k; i < m; i++ {
			obs[i][k] = obs[i-k][0]
		}
	}
	obsr := reverseRows(obs)
	// S = lfilter(b, a, Obs[::-1], axis=0)
	s := make([][]float64, m)
	for i := range s {
		s[i] = make([]float64, order)
	}
	for k := 0; k < order; k++ {
		c := make([]float64, m)
		for i := range c {
			c[i] = obsr[i][k]
		}
		fc, _, _ := LFilter(b, a, c, nil)
		for i := range fc {
			s[i][k] = fc[i]
		}
	}
	sr := reverseRows(s)
	var mat [][]float64
	if m == n {
		for i := 0; i < n; i++ {
			row := make([]float64, 2*order)
			for k := 0; k < order; k++ {
				row[k] = sr[i][k] - obs[i][k]
				row[order+k] = obsr[i][k] - s[i][k]
			}
			mat = append(mat, row)
		}
	} else {
		for i := 0; i < 2*m; i++ {
			mat = append(mat, make([]float64, 2*order))
		}
		for i := 0; i < m; i++ {
			for k := 0; k < order; k++ {
				mat[i][k] = sr[i][k] - obs[i][k]
				mat[m+i][order+k] = obsr[i][k] - s[i][k]
			}
		}
	}
	yf, _, _ := LFilter(b, a, x, nil)
	yfb := reversed(mustFilter(b, a, reversed(yf)))
	yb := reversed(mustFilter(b, a, reversed(x)))
	ybf := mustFilter(b, a, yb)
	delta := make([]float64, 0, 2*m)
	if m == n {
		for i := range ybf {
			delta = append(delta, ybf[i]-yfb[i])
		}
	} else {
		for i := 0; i < m; i++ {
			delta = append(delta, ybf[i]-yfb[i])
		}
		for i := n - m; i < n; i++ {
			delta = append(delta, ybf[i]-yfb[i])
		}
	}
	ic, err := lstsq(mat, delta)
	if err != nil {
		return nil, err
	}
	yopt := slices.Clone(yfb)
	// wic = W @ ic, W = [Sr, Obsr] (block form when m < n)
	for i := 0; i < m; i++ {
		var w1, w2 float64
		for k := 0; k < order; k++ {
			w1 += sr[i][k] * ic[k]
			w2 += obsr[i][k] * ic[order+k]
		}
		if m == n {
			yopt[i] += w1 + w2
		} else {
			yopt[i] += w1
			yopt[n-m+i] += w2
		}
	}
	return yopt, nil
}

func mustFilter(b, a, x []float64) []float64 {
	y, _, _ := LFilter(b, a, x, nil)
	return y
}

func reverseRows(m [][]float64) [][]float64 {
	r := make([][]float64, len(m))
	for i := range m {
		r[len(m)-1-i] = m[i]
	}
	return r
}

// lstsq solves the least-squares problem min |A x - b| for a tall, full-rank
// A by Householder QR.
func lstsq(a [][]float64, b []float64) ([]float64, error) {
	m, n := len(a), len(a[0])
	r := make([][]float64, m)
	for i := range r {
		r[i] = slices.Clone(a[i])
	}
	y := slices.Clone(b)
	for k := 0; k < n; k++ {
		var nrm float64
		for i := k; i < m; i++ {
			nrm = math.Hypot(nrm, r[i][k])
		}
		if nrm == 0 {
			return nil, errors.New("signal: least-squares matrix is rank deficient")
		}
		if r[k][k] > 0 {
			nrm = -nrm
		}
		// v = x - nrm e_k, applied as a reflector
		v := make([]float64, m-k)
		for i := k; i < m; i++ {
			v[i-k] = r[i][k]
		}
		v[0] -= nrm
		vv := 0.0
		for _, e := range v {
			vv += e * e
		}
		if vv != 0 {
			for j := k; j < n; j++ {
				var d float64
				for i := k; i < m; i++ {
					d += v[i-k] * r[i][j]
				}
				d = 2 * d / vv
				for i := k; i < m; i++ {
					r[i][j] -= d * v[i-k]
				}
			}
			var d float64
			for i := k; i < m; i++ {
				d += v[i-k] * y[i]
			}
			d = 2 * d / vv
			for i := k; i < m; i++ {
				y[i] -= d * v[i-k]
			}
		}
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := y[i]
		for j := i + 1; j < n; j++ {
			s -= r[i][j] * x[j]
		}
		x[i] = s / r[i][i] // non-zero: each column's reflector left a non-zero diagonal
	}
	return x, nil
}

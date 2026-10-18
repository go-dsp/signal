package signal

import (
	"errors"
	"math"
)

// RemezType is the kind of filter Remez designs.
type RemezType int

const (
	// RemezBandpass is a symmetric multiband filter.
	RemezBandpass RemezType = iota
	// RemezDifferentiator has a gain proportional to frequency, with the
	// error weighted by 1/f.
	RemezDifferentiator
	// RemezHilbert is an antisymmetric filter.
	RemezHilbert
)

// RemezOptions are the optional parameters of Remez.
type RemezOptions struct {
	// Weight has one relative weight per band (default all 1).
	Weight []float64
	// Type defaults to RemezBandpass.
	Type RemezType
	// MaxIter defaults to 25.
	MaxIter int
	// GridDensity defaults to 16.
	GridDensity int
	// Fs is the sampling frequency (default 1).
	Fs float64
}

// ErrNotConverged is returned (with the best filter found) by Remez when the
// exchange algorithm does not converge within MaxIter iterations.
var ErrNotConverged = errors.New("signal: the Remez exchange did not converge; try a wider transition band")

// Remez designs an optimal equiripple FIR filter with the Parks–McClellan
// exchange algorithm (scipy.signal.remez; MATLAB firpm). bands holds pairs of
// band edges between 0 and Fs/2, desired the gain in each band.
func Remez(numtaps int, bands, desired []float64, o *RemezOptions) ([]float64, error) {
	var opt RemezOptions
	if o != nil {
		opt = *o
	}
	if opt.Fs == 0 {
		opt.Fs = 1
	}
	if err := validFs(opt.Fs); err != nil {
		return nil, err
	}
	if opt.MaxIter == 0 {
		opt.MaxIter = 25
	}
	if opt.GridDensity == 0 {
		opt.GridDensity = 16
	}
	if opt.Type < RemezBandpass || opt.Type > RemezHilbert {
		return nil, errors.New("signal: type must be bandpass, differentiator or hilbert")
	}
	nb := len(desired)
	if numtaps < 2 {
		return nil, errors.New("signal: numtaps must be at least 2")
	}
	if nb == 0 || len(bands) != 2*nb {
		return nil, errors.New("signal: bands must hold a pair of edges for each desired gain")
	}
	if opt.MaxIter < 1 || opt.GridDensity < 1 {
		return nil, errors.New("signal: MaxIter and GridDensity must be positive")
	}
	weight := opt.Weight
	if weight == nil {
		weight = ones(nb)
	}
	if len(weight) != nb {
		return nil, errors.New("signal: weight must have one entry per band")
	}
	edges := make([]float64, len(bands))
	for i, b := range bands {
		edges[i] = b / opt.Fs
		if edges[i] < 0 || edges[i] > 0.5 || (i > 0 && edges[i] < edges[i-1]) {
			return nil, errors.New("signal: band edges must be nondecreasing and between 0 and fs/2")
		}
	}
	for _, w := range weight {
		if !(w > 0) {
			return nil, errors.New("signal: weights must be positive")
		}
	}
	neg := opt.Type != RemezBandpass
	odd := numtaps%2 == 1
	nfcns := numtaps / 2
	if odd && !neg {
		nfcns++
	}
	delf := 0.5 / (float64(opt.GridDensity) * float64(nfcns))
	if neg && edges[0] < delf {
		edges[0] = delf
	}

	// the dense grid, the desired response and the weight on it
	var grid, des, wt []float64
	for b := 0; b < nb; b++ {
		lo, hi := edges[2*b], edges[2*b+1]
		k := int((hi-lo)/delf + 0.5)
		start := len(grid)
		f := lo
		for i := 0; i < k; i++ {
			grid = append(grid, f)
			f += delf
		}
		if k > 0 {
			grid[len(grid)-1] = hi
		}
		for _, f := range grid[start:] {
			d, w := desired[b], weight[b]
			if opt.Type == RemezDifferentiator {
				d = desired[b] * f
				if desired[b] >= 0.0001 {
					w = weight[b] / f
				}
			}
			des = append(des, d)
			wt = append(wt, w)
		}
	}
	if (!neg && !odd) || (neg && odd) { // Q has a zero at 0.5
		if n := len(grid); n > 0 && grid[n-1] > 0.5-delf {
			grid, des, wt = grid[:n-1], des[:n-1], wt[:n-1]
		}
	}
	ngrid := len(grid)
	if ngrid < nfcns+1 {
		return nil, errors.New("signal: the bands are too narrow for this many taps")
	}
	q := func(f float64) float64 {
		switch {
		case !neg && !odd:
			return math.Cos(math.Pi * f)
		case neg && odd:
			return math.Sin(2 * math.Pi * f)
		case neg:
			return math.Sin(math.Pi * f)
		}
		return 1
	}
	for j, f := range grid {
		if c := q(f); c != 1 {
			des[j] /= c
			wt[j] *= c
		}
	}

	iext := make([]int, nfcns+1)
	temp := float64(ngrid-1) / float64(nfcns)
	for j := range iext {
		iext[j] = int(float64(j) * temp)
	}
	iext[nfcns] = ngrid - 1
	x := make([]float64, nfcns+1)
	ad := make([]float64, nfcns+1)
	y := make([]float64, nfcns+1)
	errv := make([]float64, ngrid)

	// solve interpolates the extremal set: nodes x, weights ad, values y
	solve := func() {
		for i := range x {
			x[i] = math.Cos(2 * math.Pi * grid[iext[i]])
		}
		for j := range x {
			p := 1.0
			for k := range x {
				if k != j {
					p *= 2 * (x[j] - x[k])
				}
			}
			ad[j] = 1 / p
		}
		var num, den float64
		sgn := 1.0
		for i := range x {
			num += ad[i] * des[iext[i]]
			den += sgn * ad[i] / wt[iext[i]]
			sgn = -sgn
		}
		dev := num / den
		sgn = 1.0
		for i := range y {
			y[i] = des[iext[i]] - sgn*dev/wt[iext[i]]
			sgn = -sgn
		}
	}
	amp := func(f float64) float64 {
		xv := math.Cos(2 * math.Pi * f)
		var num, den float64
		for i := range x {
			d := xv - x[i]
			if d == 0 {
				return y[i]
			}
			t := ad[i] / d
			num += t * y[i]
			den += t
		}
		return num / den
	}

	converged := false
	for iter := 0; iter < opt.MaxIter; iter++ {
		solve()
		for j, f := range grid {
			errv[j] = wt[j] * (des[j] - amp(f))
		}
		next, ok := exchangeFn(errv, nfcns+1)
		if !ok {
			return nil, errors.New("signal: the Remez exchange failed to find enough extrema; try a wider transition band")
		}
		emax, emin := 0.0, math.Inf(1)
		for _, j := range next {
			emax = math.Max(emax, math.Abs(errv[j]))
			emin = math.Min(emin, math.Abs(errv[j]))
		}
		same := true
		for i := range next {
			if next[i] != iext[i] {
				same = false
			}
		}
		iext = next
		if same || (emax-emin)/emax < 1e-4 {
			converged = true
			break
		}
	}
	solve()

	// sample the amplitude response and transform to the impulse response
	h := make([]float64, numtaps)
	nf := float64(numtaps)
	m := (nf - 1) / 2
	k := (numtaps - 1) / 2
	a := make([]float64, k+1)
	for i := range a {
		f := float64(i) / nf
		a[i] = amp(f) * q(f)
	}
	for n := range h {
		t := float64(n) - m
		var s float64
		if !neg {
			s = a[0]
			for i := 1; i <= k; i++ {
				s += 2 * a[i] * math.Cos(2*math.Pi*float64(i)*t/nf)
			}
		} else {
			for i := 1; i <= k; i++ {
				s -= 2 * a[i] * math.Sin(2*math.Pi*float64(i)*t/nf)
			}
			if !odd { // the Nyquist sample of a type IV filter
				s -= q(0.5) * amp(0.5) * math.Sin(math.Pi*t)
			}
		}
		h[n] = s / nf
	}
	if !converged {
		return h, ErrNotConverged
	}
	return h, nil
}

// exchangeFn is exchange, replaceable by tests to reach the failure path.
var exchangeFn = exchange

// exchange picks the nfcns+1 alternating extrema of the error curve e
// (indices into the grid) with the exchange rules of Parks and McClellan.
func exchange(e []float64, want int) ([]int, bool) {
	var ext []int
	n := len(e)
	for j := 0; j < n; j++ {
		var prev, next float64
		hasPrev, hasNext := j > 0, j < n-1
		if hasPrev {
			prev = e[j-1]
		}
		if hasNext {
			next = e[j+1]
		}
		isExt := false
		switch {
		case e[j] > 0:
			isExt = (!hasPrev || e[j] >= prev) && (!hasNext || e[j] >= next)
		case e[j] < 0:
			isExt = (!hasPrev || e[j] <= prev) && (!hasNext || e[j] <= next)
		}
		if isExt {
			ext = append(ext, j)
		}
	}
	// keep the larger of consecutive extrema of the same sign
	var alt []int
	for _, j := range ext {
		if len(alt) > 0 && (e[alt[len(alt)-1]] > 0) == (e[j] > 0) {
			if math.Abs(e[j]) > math.Abs(e[alt[len(alt)-1]]) {
				alt[len(alt)-1] = j
			}
			continue
		}
		alt = append(alt, j)
	}
	for len(alt) > want {
		if math.Abs(e[alt[0]]) < math.Abs(e[alt[len(alt)-1]]) {
			alt = alt[1:]
		} else {
			alt = alt[:len(alt)-1]
		}
	}
	return alt, len(alt) == want
}

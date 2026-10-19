package signal

import (
	"errors"
	"fmt"
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
	if numtaps < 2 || numtaps > 1<<20 {
		return nil, errors.New("signal: numtaps must be between 2 and 2^20")
	}
	if nb == 0 || len(bands) != 2*nb {
		return nil, errors.New("signal: bands must hold a pair of edges for each desired gain")
	}
	if opt.MaxIter < 1 || opt.GridDensity < 1 || opt.GridDensity > 1<<12 {
		return nil, errors.New("signal: MaxIter must be positive and GridDensity between 1 and 4096")
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

	// the dense grid, the desired response and the weight on it. One
	// allocation holds them and the per-iteration buffers (error curve,
	// grid abscissae): repeated designs otherwise spend most of their time
	// in the garbage collector, not in the exchange.
	//
	// Each band holds lo, lo+delf, ... (accumulated) while at most hi, at
	// least one point, the last replaced by hi: SciPy's grid exactly, so
	// that both exchanges start from the same extremal set.
	total := 0
	for b := 0; b < nb; b++ {
		// (hi-lo)/delf bounds the count up to rounding of the sum.
		total += int((edges[2*b+1]-edges[2*b])/delf) + 2
	}
	if total > maxRemezGrid {
		return nil, fmt.Errorf("signal: the Remez grid would hold %d points (more than %d); lower numtaps or GridDensity", total, maxRemezGrid)
	}
	buf := make([]float64, 4*total)
	grid, des, wt := buf[:0:total], buf[total:total:2*total], buf[2*total:2*total:3*total]
	for b := 0; b < nb; b++ {
		lo, hi := edges[2*b], edges[2*b+1]
		start := len(grid)
		for f := lo; ; {
			grid = append(grid, f)
			if f += delf; !(f <= hi) {
				break
			}
		}
		grid[len(grid)-1] = hi
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

	// The grid does not move: its abscissae are computed once, not at every
	// exchange iteration.
	xg := buf[3*total : 3*total+ngrid]
	for j, f := range grid {
		xg[j] = math.Cos(2 * math.Pi * f)
	}
	x, ad, y, converged := remezExchange(xg, des, wt, nfcns, opt.MaxIter)

	amp := func(f float64) float64 { return baryEval(math.Cos(2*math.Pi*f), x, ad, y) }

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
	// The angle 2π·i·t/N, t = n-(N-1)/2, is π·j/N with the integer
	// j = i·(2n-N+1) mod 2N: one table of 2N sines or cosines replaces
	// numtaps·k calls (and reduces the argument exactly). The response is
	// symmetric (cosine) or antisymmetric (sine) about the centre, so only
	// the first half is summed.
	n2 := 2 * numtaps
	tab := make([]float64, n2)
	for j := range tab {
		if neg {
			tab[j] = math.Sin(math.Pi * float64(j) / nf)
		} else {
			tab[j] = math.Cos(math.Pi * float64(j) / nf)
		}
	}
	for n := 0; n < (numtaps+1)/2; n++ {
		t := float64(n) - m
		step := ((2*n-numtaps+1)%n2 + n2) % n2 // j for i = 1
		var s float64
		j := 0
		if !neg {
			s = a[0]
			for i := 1; i <= k; i++ {
				j += step
				if j >= n2 {
					j -= n2
				}
				s += 2 * a[i] * tab[j]
			}
		} else {
			for i := 1; i <= k; i++ {
				j += step
				if j >= n2 {
					j -= n2
				}
				s -= 2 * a[i] * tab[j]
			}
			if !odd { // the Nyquist sample of a type IV filter
				s -= q(0.5) * amp(0.5) * math.Sin(math.Pi*t)
			}
		}
		h[n] = s / nf
		if neg {
			h[numtaps-1-n] = -h[n]
		} else {
			h[numtaps-1-n] = h[n]
		}
	}
	if !converged {
		return h, ErrNotConverged
	}
	return h, nil
}

// maxRemezGrid bounds the dense grid (32 MiB for each of its four buffers;
// numtaps up to 2^19 at the default density), so that numtaps and
// GridDensity near their limits cannot request tens of gigabytes.
const maxRemezGrid = 1 << 22

// remezExchange runs the Remez exchange of McClellan, Parks and Rabiner on
// the grid abscissae xg = cos(2π·grid) with desired response des and weight
// wt, for nfcns cosines and at most itrmax iterations. It returns the final
// barycentric interpolant (nodes x, weights ad, values y) and whether the
// exchange converged.
//
// The control flow is the published FORTRAN program as SciPy's
// _sigtoolsmodule.cc carries it (Kvaleberg's C translation): the search for
// new extrema walks the error curve from each current extremum until it stops
// growing, instead of evaluating the whole grid at every iteration, which is
// what makes it fast. Arrays are 1-based and the labels keep the original
// numbers so that the two can be read side by side.
func remezExchange(xg, des, wt []float64, nfcns, itrmax int) (xo, ado, yo []float64, converged bool) {
	ngrid := len(xg)
	nz, nzz := nfcns+1, nfcns+2
	iext := make([]int, nzz+1)
	temp := float64(ngrid-1) / float64(nfcns)
	for j := 1; j <= nfcns; j++ {
		iext[j] = int(float64(j-1)*temp) + 1
	}
	iext[nz] = ngrid
	x := make([]float64, nz+1)
	ad := make([]float64, nz+1)
	y := make([]float64, nz+1)
	// errAt is the weighted error at grid point l (1-based).
	errAt := func(l int) float64 {
		return (baryEval(xg[l-1], x[1:], ad[1:], y[1:]) - des[l-1]) * wt[l-1]
	}

	var (
		k, k1, knz, klow, kup, nu, nut, jchnge, j, l int
		nut1, luck, niter, kn                        int
		dev, devl, comp, ynz, y1, e                  float64
	)
	devl = -1
	for {
	L100:
		iext[nzz] = ngrid + 1
		niter++
		if niter > itrmax {
			return x[1:], ad[1:], y[1:], false
		}
		for j = 1; j <= nz; j++ {
			x[j] = xg[iext[j]-1]
		}
		jet := (nfcns-1)/15 + 1
		for j = 1; j <= nz; j++ {
			ad[j] = lagrangeWeight(j, nz, jet, x)
		}
		var dnum, dden float64
		k = 1
		for j = 1; j <= nz; j++ {
			l = iext[j]
			dnum += ad[j] * des[l-1]
			dden += float64(k) * ad[j] / wt[l-1]
			k = -k
		}
		dev = dnum / dden
		nu = 1
		if dev > 0 {
			nu = -1
		}
		dev = -float64(nu) * dev
		k = nu
		for j = 1; j <= nz; j++ {
			l = iext[j]
			y[j] = des[l-1] + float64(k)*dev/wt[l-1]
			k = -k
		}
		if dev <= devl {
			// The deviation stopped growing before the extremal set
			// settled: SciPy raises "Failure to converge" here.
			return x[1:], ad[1:], y[1:], false
		}
		devl = dev
		jchnge = 0
		k1 = iext[1]
		knz = iext[nz]
		klow = 0
		nut = -nu
		j = 1

		// search for the extremal frequencies of the best approximation
	L200:
		if j == nzz {
			ynz = comp
		}
		if j >= nzz {
			goto L300
		}
		kup = iext[j+1]
		l = iext[j] + 1
		nut = -nut
		if j == 2 {
			y1 = comp
		}
		comp = dev
		if l >= kup {
			goto L220
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L220
		}
		comp = float64(nut) * e
	L210:
		l++
		if l >= kup {
			goto L215
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L215
		}
		comp = float64(nut) * e
		goto L210
	L215:
		iext[j] = l - 1
		j++
		klow = l - 1
		jchnge++
		goto L200
	L220:
		l--
	L225:
		l--
		if l <= klow {
			goto L250
		}
		e = errAt(l)
		if float64(nut)*e-comp > 0 {
			goto L230
		}
		if jchnge <= 0 {
			goto L225
		}
		goto L260
	L230:
		comp = float64(nut) * e
	L235:
		l--
		if l <= klow {
			goto L240
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L240
		}
		comp = float64(nut) * e
		goto L235
	L240:
		klow = iext[j]
		iext[j] = l + 1
		j++
		jchnge++
		goto L200
	L250:
		l = iext[j] + 1
		if jchnge > 0 {
			goto L215
		}
	L255:
		l++
		if l >= kup {
			goto L260
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L255
		}
		comp = float64(nut) * e
		goto L210
	L260:
		klow = iext[j]
		j++
		goto L200

	L300:
		if j > nzz {
			goto L320
		}
		if k1 > iext[1] {
			k1 = iext[1]
		}
		if knz < iext[nz] {
			knz = iext[nz]
		}
		nut1 = nut
		nut = -nu
		l = 0
		kup = k1
		comp = ynz * 1.00001
		luck = 1
	L310:
		l++
		if l >= kup {
			goto L315
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L310
		}
		comp = float64(nut) * e
		j = nzz
		goto L210
	L315:
		luck = 6
		goto L325
	L320:
		if luck > 9 {
			goto L350
		}
		if comp > y1 {
			y1 = comp
		}
		k1 = iext[nzz]
	L325:
		l = ngrid + 1
		klow = knz
		nut = -nut1
		comp = y1 * 1.00001
	L330:
		l--
		if l <= klow {
			goto L340
		}
		e = errAt(l)
		if float64(nut)*e-comp <= 0 {
			goto L330
		}
		j = nzz
		comp = float64(nut) * e
		luck += 10
		goto L235
	L340:
		if luck == 6 {
			goto L370
		}
		for j = 1; j <= nfcns; j++ {
			iext[nzz-j] = iext[nz-j]
		}
		iext[1] = k1
		goto L100
	L350:
		kn = iext[nzz]
		for j = 1; j <= nfcns; j++ {
			iext[j] = iext[j+1]
		}
		iext[nz] = kn
		goto L100
	L370:
		if jchnge <= 0 {
			return x[1:], ad[1:], y[1:], true
		}
	}
}

// lagrangeWeight is the barycentric weight of node k among x[1..n]
// (1-based), its product taken in m interleaved passes so that it does not
// overflow or underflow for many nodes (the FORTRAN function d).
func lagrangeWeight(k, n, m int, x []float64) float64 {
	r := 1.0
	q := x[k]
	for l := 1; l <= m; l++ {
		for j := l; j <= n; j += m {
			if j != k {
				r *= 2 * (q - x[j])
			}
		}
	}
	return 1 / r
}

// baryEval evaluates the barycentric interpolant with nodes x, weights ad
// and values y at xv; at a node it returns that node's value. Even and odd
// terms accumulate in separate sums, so two addition chains overlap instead
// of one waiting on its own latency; every quotient is still a true
// division. (Pairing terms to share one division was faster still but
// doubled the deviation from SciPy on one fixture; not worth it.)
func baryEval(xv float64, x, ad, y []float64) float64 {
	var n0, n1, d0, d1 float64
	ad, y = ad[:len(x)], y[:len(x)]
	i := 0
	for ; i+2 <= len(x); i += 2 {
		e0, e1 := xv-x[i], xv-x[i+1]
		if e0 == 0 || e1 == 0 {
			if e0 == 0 {
				return y[i]
			}
			return y[i+1]
		}
		t0, t1 := ad[i]/e0, ad[i+1]/e1
		d0 += t0
		d1 += t1
		n0 += t0 * y[i]
		n1 += t1 * y[i+1]
	}
	if i < len(x) {
		e := xv - x[i]
		if e == 0 {
			return y[i]
		}
		t := ad[i] / e
		d0 += t
		n0 += t * y[i]
	}
	return (n0 + n1) / (d0 + d1)
}

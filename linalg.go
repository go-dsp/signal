package signal

import (
	"errors"
	"math"
)

// sturmCount returns the number of eigenvalues of the symmetric tridiagonal
// matrix (diagonal d, off-diagonal e) that are smaller than x.
func sturmCount(d, e []float64, x float64) int {
	const tiny = 1e-300
	count := 0
	q := d[0] - x
	if q < 0 {
		count++
	}
	for i := 1; i < len(d); i++ {
		if q == 0 {
			q = tiny
		}
		q = d[i] - x - e[i-1]*e[i-1]/q
		if q < 0 {
			count++
		}
	}
	return count
}

// tridiagTopEigen returns the k largest eigenvalues (descending) of the
// symmetric tridiagonal matrix (d, e) and their unit-norm eigenvectors, by
// bisection on the Sturm sequence and inverse iteration.
func tridiagTopEigen(d, e []float64, k int) ([]float64, [][]float64) {
	n := len(d)
	lo, hi := math.Inf(1), math.Inf(-1)
	var norm1 float64
	for i := range d {
		r := 0.0
		if i > 0 {
			r += math.Abs(e[i-1])
		}
		if i < n-1 {
			r += math.Abs(e[i])
		}
		lo, hi = math.Min(lo, d[i]-r), math.Max(hi, d[i]+r)
		norm1 = math.Max(norm1, math.Abs(d[i])+r)
	}
	vals := make([]float64, k)
	vecs := make([][]float64, k)
	for t := 0; t < k; t++ {
		j := n - 1 - t // index of the eigenvalue among the ascending ones
		a, b := lo, hi
		for {
			mid := 0.5 * (a + b)
			if mid <= a || mid >= b {
				break
			}
			if sturmCount(d, e, mid) > j {
				b = mid
			} else {
				a = mid
			}
		}
		vals[t] = 0.5 * (a + b)
		vecs[t] = inverseIteration(d, e, vals[t], norm1, vecs[:t])
	}
	return vals, vecs
}

// inverseIteration returns the unit eigenvector of (d, e) for the eigenvalue
// lambda, kept orthogonal to the vectors in prev.
func inverseIteration(d, e []float64, lambda, norm1 float64, prev [][]float64) []float64 {
	n := len(d)
	x := make([]float64, n)
	seed := uint64(0x9e3779b97f4a7c15)
	for i := range x {
		seed = seed*6364136223846793005 + 1442695040888963407
		x[i] = float64(seed>>11)/float64(1<<53) - 0.5
	}
	for it := 0; it < 5; it++ {
		for _, p := range prev { // modified Gram–Schmidt
			var dot float64
			for i := range x {
				dot += x[i] * p[i]
			}
			for i := range x {
				x[i] -= dot * p[i]
			}
		}
		solveShiftedTridiag(d, e, lambda, norm1, x)
		var s float64
		for _, v := range x {
			s = math.Max(s, math.Abs(v))
		}
		for i := range x {
			x[i] /= s
		}
	}
	for _, p := range prev {
		var dot float64
		for i := range x {
			dot += x[i] * p[i]
		}
		for i := range x {
			x[i] -= dot * p[i]
		}
	}
	var nrm float64
	for _, v := range x {
		nrm += v * v
	}
	nrm = math.Sqrt(nrm)
	for i := range x {
		x[i] /= nrm
	}
	return x
}

// solveShiftedTridiag solves (T - shift I) x = b in place by Gaussian
// elimination with partial pivoting (LAPACK's dgtsv); exact zero pivots are
// replaced by a tiny multiple of the matrix norm so that the solve can be
// used for inverse iteration.
func solveShiftedTridiag(d0, e []float64, shift, norm1 float64, b []float64) {
	n := len(d0)
	d := make([]float64, n)
	for i := range d {
		d[i] = d0[i] - shift
	}
	if n == 1 {
		if d[0] == 0 {
			d[0] = 2.220446049250313e-16 * norm1
		}
		b[0] /= d[0]
		return
	}
	dl := make([]float64, n-1)
	du := make([]float64, n) // du[n-1] is unused; du[i+1] receives fill-in
	copy(dl, e)
	copy(du, e)
	tiny := 2.220446049250313e-16 * norm1
	for i := 0; i < n-1; i++ {
		if math.Abs(d[i]) >= math.Abs(dl[i]) {
			if d[i] == 0 {
				d[i] = tiny
			}
			f := dl[i] / d[i]
			d[i+1] -= f * du[i]
			b[i+1] -= f * b[i]
			dl[i] = 0
		} else {
			f := d[i] / dl[i]
			d[i] = dl[i]
			t := d[i+1]
			d[i+1] = du[i] - f*t
			if i < n-2 {
				dl[i] = du[i+1]
				du[i+1] = -f * dl[i]
			}
			du[i] = t
			b[i], b[i+1] = b[i+1], b[i]-f*b[i+1]
		}
	}
	if d[n-1] == 0 {
		d[n-1] = tiny
	}
	b[n-1] /= d[n-1]
	b[n-2] = (b[n-2] - du[n-2]*b[n-1]) / d[n-2]
	for i := n - 3; i >= 0; i-- {
		b[i] = (b[i] - du[i]*b[i+1] - dl[i]*b[i+2]) / d[i]
	}
}

// solveDense solves the n x n system a x = b by LU decomposition with partial
// pivoting. a and b are destroyed.
func solveDense(a [][]float64, b []float64) error {
	n := len(b)
	for c := 0; c < n; c++ {
		p := c
		for r := c + 1; r < n; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[p][c]) {
				p = r
			}
		}
		if a[p][c] == 0 {
			return errors.New("signal: singular matrix")
		}
		a[c], a[p] = a[p], a[c]
		b[c], b[p] = b[p], b[c]
		for r := c + 1; r < n; r++ {
			f := a[r][c] / a[c][c]
			for k := c; k < n; k++ {
				a[r][k] -= f * a[c][k]
			}
			b[r] -= f * b[c]
		}
	}
	for r := n - 1; r >= 0; r-- {
		s := b[r]
		for k := r + 1; k < n; k++ {
			s -= a[r][k] * b[k]
		}
		b[r] = s / a[r][r]
	}
	return nil
}

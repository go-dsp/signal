package signal

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
)

// hqrLimit is the QR iteration limit per eigenvalue (a variable so tests can
// reach the failure path).
var hqrLimit = 60

// Roots returns the roots of the polynomial p[0]*x^n + ... + p[n], like
// numpy.roots: the eigenvalues of the companion matrix (balanced, reduced by
// the shifted QR algorithm). Leading zeros are dropped and trailing zeros
// give exact zero roots. Roots of a real polynomial come in conjugate pairs.
//
// The order of the returned roots is unspecified.
func Roots(p []float64) ([]complex128, error) {
	for len(p) > 0 && p[0] == 0 {
		p = p[1:]
	}
	nz := 0
	for len(p) > 0 && p[len(p)-1] == 0 {
		p = p[:len(p)-1]
		nz++
	}
	var r []complex128
	if n := len(p) - 1; n > 0 {
		a := make([][]float64, n)
		for i := range a {
			a[i] = make([]float64, n)
		}
		for j := 0; j < n; j++ {
			a[0][j] = -p[j+1] / p[0]
		}
		for i := 1; i < n; i++ {
			a[i][i-1] = 1
		}
		balance(a)
		wr, wi, ok := hqr(a, hqrLimit)
		if !ok {
			return nil, errors.New("signal: root finding did not converge")
		}
		for i := range wr {
			r = append(r, complex(wr[i], wi[i]))
		}
	}
	return append(r, make([]complex128, nz)...), nil
}

// balance scales rows and columns of a by powers of two so that their norms
// are comparable (Parlett–Reinsch), improving eigenvalue accuracy.
func balance(a [][]float64) {
	n := len(a)
	for done := false; !done; {
		done = true
		for i := 0; i < n; i++ {
			var r, c float64
			for j := 0; j < n; j++ {
				if j != i {
					c += math.Abs(a[j][i])
					r += math.Abs(a[i][j])
				}
			}
			if c == 0 || r == 0 {
				continue
			}
			g := r / 2
			f := 1.0
			s := c + r
			for c < g {
				f *= 2
				c *= 4
			}
			g = r * 2
			for c > g {
				f /= 2
				c /= 4
			}
			if (c+r)/f < 0.95*s {
				done = false
				for j := 0; j < n; j++ {
					a[i][j] /= f
					a[j][i] *= f
				}
			}
		}
	}
}

func sign(a, b float64) float64 {
	if b >= 0 {
		return math.Abs(a)
	}
	return -math.Abs(a)
}

// hqr finds all eigenvalues of an upper Hessenberg matrix by the double-shift
// QR algorithm (EISPACK hqr). a is destroyed.
func hqr(a [][]float64, limit int) (wr, wi []float64, ok bool) {
	n := len(a)
	wr = make([]float64, n)
	wi = make([]float64, n)
	var anorm float64
	for i := 0; i < n; i++ {
		for j := max(i-1, 0); j < n; j++ {
			anorm += math.Abs(a[i][j])
		}
	}
	const eps = 2.220446049250313e-16
	nn := n - 1
	t := 0.0
	for nn >= 0 {
		its := 0
	iter:
		for {
			l := nn
			for ; l >= 1; l-- {
				s := math.Abs(a[l-1][l-1]) + math.Abs(a[l][l])
				if s == 0 {
					s = anorm
				}
				if math.Abs(a[l][l-1]) <= eps*s {
					a[l][l-1] = 0
					break
				}
			}
			x := a[nn][nn]
			if l == nn {
				wr[nn], wi[nn] = x+t, 0
				nn--
				break iter
			}
			y := a[nn-1][nn-1]
			w := a[nn][nn-1] * a[nn-1][nn]
			if l == nn-1 {
				p := 0.5 * (y - x)
				q := p*p + w
				z := math.Sqrt(math.Abs(q))
				x += t
				if q >= 0 {
					z = p + sign(z, p)
					wr[nn-1], wr[nn] = x+z, x+z
					if z != 0 {
						wr[nn] = x - w/z
					}
					wi[nn-1], wi[nn] = 0, 0
				} else {
					wr[nn-1], wr[nn] = x+p, x+p
					wi[nn-1], wi[nn] = z, -z
				}
				nn -= 2
				break iter
			}
			if its == limit {
				return nil, nil, false
			}
			if its == 10 || its == 20 {
				t += x
				for i := 0; i <= nn; i++ {
					a[i][i] -= x
				}
				s := math.Abs(a[nn][nn-1]) + math.Abs(a[nn-1][nn-2])
				x = 0.75 * s
				y = x
				w = -0.4375 * s * s
			}
			its++
			var p, q, r, z float64
			m := nn - 2
			for ; m >= l; m-- {
				z = a[m][m]
				r = x - z
				s := y - z
				p = (r*s-w)/a[m+1][m] + a[m][m+1]
				q = a[m+1][m+1] - z - r - s
				r = a[m+2][m+1]
				s = math.Abs(p) + math.Abs(q) + math.Abs(r)
				p /= s
				q /= s
				r /= s
				if m == l {
					break
				}
				u := math.Abs(a[m][m-1]) * (math.Abs(q) + math.Abs(r))
				v := math.Abs(p) * (math.Abs(a[m-1][m-1]) + math.Abs(z) + math.Abs(a[m+1][m+1]))
				if u <= eps*v {
					break
				}
			}
			for i := m + 2; i <= nn; i++ {
				a[i][i-2] = 0
				if i != m+2 {
					a[i][i-3] = 0
				}
			}
			for k := m; k < nn; k++ {
				if k != m {
					p = a[k][k-1]
					q = a[k+1][k-1]
					r = 0
					if k != nn-1 {
						r = a[k+2][k-1]
					}
					x = math.Abs(p) + math.Abs(q) + math.Abs(r)
					if x != 0 {
						p /= x
						q /= x
						r /= x
					}
				}
				s := sign(math.Sqrt(p*p+q*q+r*r), p)
				if s == 0 {
					continue
				}
				if k == m {
					if l != m {
						a[k][k-1] = -a[k][k-1]
					}
				} else {
					a[k][k-1] = -s * x
				}
				p += s
				x = p / s
				y = q / s
				z = r / s
				q /= p
				r /= p
				for j := k; j <= nn; j++ {
					p = a[k][j] + q*a[k+1][j]
					if k != nn-1 {
						p += r * a[k+2][j]
						a[k+2][j] -= p * z
					}
					a[k+1][j] -= p * y
					a[k][j] -= p * x
				}
				for i := l; i <= min(nn, k+3); i++ {
					p = x*a[i][k] + y*a[i][k+1]
					if k != nn-1 {
						p += z * a[i][k+2]
						a[i][k+2] -= p * r
					}
					a[i][k+1] -= p * q
					a[i][k] -= p
				}
			}
		}
	}
	return wr, wi, true
}

// polyFromRoots returns the monic real polynomial with the given roots,
// highest degree first (numpy.poly). The roots must be real or come in
// conjugate pairs; otherwise an error is returned.
func polyFromRoots(r []complex128) ([]float64, error) {
	c := make([]complex128, 1, len(r)+1)
	c[0] = 1
	for _, root := range r {
		c = append(c, 0)
		for i := len(c) - 1; i >= 1; i-- {
			c[i] -= root * c[i-1]
		}
	}
	out := make([]float64, len(c))
	for i, v := range c {
		out[i] = real(v)
	}
	if !conjugateClosed(r) {
		return nil, errors.New("signal: complex roots must come in conjugate pairs")
	}
	return out, nil
}

// conjugateClosed reports whether the multiset r is closed under
// conjugation (numpy's test in poly: sorted positive-imaginary roots equal
// the conjugates of the negative-imaginary ones).
func conjugateClosed(r []complex128) bool {
	var pos, neg []complex128
	for _, v := range r {
		switch {
		case imag(v) > 0:
			pos = append(pos, v)
		case imag(v) < 0:
			neg = append(neg, cmplx.Conj(v))
		}
	}
	if len(pos) != len(neg) {
		return false
	}
	cmp := func(a, b complex128) int {
		if real(a) != real(b) {
			if real(a) < real(b) {
				return -1
			}
			return 1
		}
		if imag(a) < imag(b) {
			return -1
		}
		if imag(a) > imag(b) {
			return 1
		}
		return 0
	}
	slices.SortFunc(pos, cmp)
	slices.SortFunc(neg, cmp)
	for i := range pos {
		if pos[i] != neg[i] {
			return false
		}
	}
	return true
}

// polyMul multiplies two real polynomials (full convolution).
func polyMul(a, b []float64) []float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	r := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			r[i+j] += x * y
		}
	}
	return r
}

// polyAdd adds two polynomials aligned at the constant term.
func polyAdd(a, b []float64) []float64 {
	if len(a) < len(b) {
		a, b = b, a
	}
	r := slices.Clone(a)
	off := len(a) - len(b)
	for i, v := range b {
		r[off+i] += v
	}
	return r
}

// polyval evaluates p (highest degree first) at the complex point x.
func polyval(p []float64, x complex128) complex128 {
	var s complex128
	for _, c := range p {
		s = s*x + complex(c, 0)
	}
	return s
}

// polyvalReal evaluates p at a real x.
func polyvalReal(p []float64, x float64) float64 {
	var s float64
	for _, c := range p {
		s = s*x + c
	}
	return s
}

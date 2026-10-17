package signal

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"

	"github.com/go-numerics/special"
)

const epsilon = 2e-16 // scipy.signal's EPSILON for filter design

func pow10m1(x float64) float64 { return math.Expm1(math.Ln10 * x) }

var errOrder = errors.New("signal: filter order must be nonnegative")

// ButtAP returns the zeros, poles and gain of the analog Butterworth lowpass
// prototype of order n, with cutoff 1 rad/s (scipy.signal.buttap).
func ButtAP(n int) (ZPK, error) {
	if n < 0 {
		return ZPK{}, errOrder
	}
	p := make([]complex128, n)
	for i := range p {
		m := float64(-n + 1 + 2*i)
		p[i] = -cmplx.Exp(complex(0, math.Pi*m/float64(2*n)))
	}
	return ZPK{P: p, K: 1}, nil
}

func prodNeg(c []complex128) complex128 {
	p := complex(1, 0)
	for _, v := range c {
		p *= -v
	}
	return p
}

// Cheb1AP returns the analog Chebyshev type I lowpass prototype of order n
// with rp dB of passband ripple, passband edge 1 rad/s (scipy.signal.cheb1ap).
func Cheb1AP(n int, rp float64) (ZPK, error) {
	if n < 0 {
		return ZPK{}, errOrder
	}
	if n == 0 {
		return ZPK{K: math.Pow(10, -rp/20)}, nil
	}
	eps := math.Sqrt(math.Pow(10, 0.1*rp) - 1)
	mu := 1 / float64(n) * math.Asinh(1/eps)
	p := make([]complex128, n)
	for i := range p {
		theta := math.Pi * float64(-n+1+2*i) / float64(2*n)
		p[i] = -cmplx.Sinh(complex(mu, theta))
	}
	k := real(prodNeg(p))
	if n%2 == 0 {
		k /= math.Sqrt(1 + eps*eps)
	}
	return ZPK{P: p, K: k}, nil
}

// Cheb2AP returns the analog Chebyshev type II lowpass prototype of order n
// with rs dB of stopband attenuation, stopband edge 1 rad/s
// (scipy.signal.cheb2ap).
func Cheb2AP(n int, rs float64) (ZPK, error) {
	if n < 0 {
		return ZPK{}, errOrder
	}
	if n == 0 {
		return ZPK{K: 1}, nil
	}
	de := 1 / math.Sqrt(math.Pow(10, 0.1*rs)-1)
	mu := math.Asinh(1/de) / float64(n)
	var z []complex128
	for m := -n + 1; m < n; m += 2 {
		if n%2 == 1 && m == 0 {
			continue // the zero at infinity
		}
		z = append(z, 1i/complex(math.Sin(float64(m)*math.Pi/float64(2*n)), 0))
	}
	p := make([]complex128, n)
	for i := range p {
		theta := math.Pi * float64(-n+1+2*i) / float64(2*n)
		p[i] = -1 / cmplx.Sinh(complex(mu, theta))
	}
	k := real(prodNeg(p) / prodNeg(z))
	return ZPK{Z: z, P: p, K: k}, nil
}

// Ellipj evaluates the Jacobi elliptic functions sn, cn and dn of argument u
// and parameter m = k^2 in [0, 1] (the algorithm of Cephes' ellpj, via the
// arithmetic-geometric mean; NaN outside the domain).
func Ellipj(u, m float64) (sn, cn, dn float64) {
	const machep = 1.11022302462515654042e-16
	switch {
	case math.IsNaN(u) || math.IsNaN(m) || m < 0 || m > 1:
		nan := math.NaN()
		return nan, nan, nan
	case m < 1e-9:
		t, b := math.Sin(u), math.Cos(u)
		ai := 0.25 * m * (u - t*b)
		return t - ai*b, b + ai*t, 1 - 0.5*m*t*t
	case m >= 0.9999999999:
		ai := 0.25 * (1 - m)
		b := math.Cosh(u)
		t := math.Tanh(u)
		phi := 1 / b
		twon := b * math.Sinh(u)
		sn = t + ai*(twon-u)/(b*b)
		ai *= t * phi
		return sn, phi - ai*(twon-u), phi + ai*(twon+u)
	}
	var a, c [9]float64
	a[0] = 1
	b := math.Sqrt(1 - m)
	c[0] = math.Sqrt(m)
	twon := 1.0
	i := 0
	for math.Abs(c[i]/a[i]) > machep && i <= 7 {
		ai := a[i]
		i++
		c[i] = (ai - b) / 2
		t := math.Sqrt(ai * b)
		a[i] = (ai + b) / 2
		b = t
		twon *= 2
	}
	phi := twon * a[i] * u
	for ; i > 0; i-- {
		t := c[i] * math.Sin(phi) / a[i]
		b = phi
		phi = (math.Asin(t) + phi) / 2
	}
	t := math.Cos(phi)
	return math.Sin(phi), t, t / math.Cos(phi-b)
}

// ellipDeg solves the degree equation of the elliptic filter for the
// parameter m given the nome-based modular relation (scipy's _ellipdeg).
func ellipDeg(n int, m1 float64) float64 {
	k1 := special.EllipK(m1)
	k1p := special.EllipKm1(m1)
	q1 := math.Exp(-math.Pi * k1p / k1)
	q := math.Pow(q1, 1/float64(n))
	var num, den float64
	for m := 0; m <= 7; m++ {
		num += math.Pow(q, float64(m*(m+1)))
	}
	for m := 1; m <= 8; m++ {
		den += math.Pow(q, float64(m*m))
	}
	return 16 * q * math.Pow(num/(1+2*den), 4)
}

// arcJacSN solves w = sn(z, m) for z by descending Landen transformations
// (scipy's _arc_jac_sn); w may be complex.
func arcJacSN(w complex128, m float64) (complex128, error) {
	complement := func(kx complex128) complex128 { return cmplx.Sqrt((1 - kx) * (1 + kx)) }
	k := math.Sqrt(m)
	if k > 1 {
		return cmplx.NaN(), nil
	}
	if k == 1 {
		return cmplx.Atanh(w), nil
	}
	ks := []float64{k}
	for niter := 0; ks[len(ks)-1] != 0; {
		kp := real(complement(complex(ks[len(ks)-1], 0)))
		ks = append(ks, (1-kp)/(1+kp))
		niter++
		if niter > 10 {
			return 0, errors.New("signal: Landen transformation not converging")
		}
	}
	kk := math.Pi / 2
	for _, v := range ks[1:] {
		kk *= 1 + v
	}
	wn := w
	for i := 0; i+1 < len(ks); i++ {
		wn = 2 * wn / (complex(1+ks[i+1], 0) * (1 + complement(complex(ks[i], 0)*wn)))
	}
	return complex(kk*2/math.Pi, 0) * cmplx.Asin(wn), nil
}

// arcJacSC1 is the inverse of sc(z, m) at real w.
func arcJacSC1(w, m float64) (float64, error) {
	z, err := arcJacSN(complex(0, w), m)
	if err != nil {
		return 0, err
	}
	if !(math.Abs(real(z)) <= 1e-14) {
		return 0, errors.New("signal: inverse Jacobi function has a real part")
	}
	return imag(z), nil
}

// EllipAP returns the analog elliptic (Cauer) lowpass prototype of order n
// with rp dB of passband ripple and rs dB of stopband attenuation, passband
// edge 1 rad/s (scipy.signal.ellipap).
func EllipAP(n int, rp, rs float64) (ZPK, error) {
	if n < 0 {
		return ZPK{}, errOrder
	}
	if n == 0 {
		return ZPK{K: math.Pow(10, -rp/20)}, nil
	}
	if n == 1 {
		p := -math.Sqrt(1 / pow10m1(0.1*rp))
		return ZPK{P: []complex128{complex(p, 0)}, K: -p}, nil
	}
	epsSq := pow10m1(0.1 * rp)
	eps := math.Sqrt(epsSq)
	ck1Sq := epsSq / pow10m1(0.1*rs)
	if ck1Sq == 0 {
		return ZPK{}, errors.New("signal: cannot design a filter with the given rp and rs")
	}
	val0 := special.EllipK(ck1Sq)
	m := ellipDeg(n, ck1Sq)
	capk := special.EllipK(m)
	nf := float64(n)

	type jac struct{ s, c, d float64 }
	var js []jac
	var z []complex128
	for j := 1 - n%2; j < n; j += 2 {
		s, c, d := Ellipj(float64(j)*capk/nf, m)
		js = append(js, jac{s, c, d})
		if math.Abs(s) > epsilon {
			z = append(z, complex(0, 1/(math.Sqrt(m)*s)))
		}
	}
	nz := len(z)
	for i := 0; i < nz; i++ {
		z = append(z, cmplx.Conj(z[i]))
	}
	r, err := arcJacSC1(1/eps, ck1Sq)
	if err != nil {
		return ZPK{}, err
	}
	v0 := capk * r / (nf * val0)
	sv, cv, dv := Ellipj(v0, 1-m)
	p := make([]complex128, len(js))
	for i, e := range js {
		ds := e.d * sv
		p[i] = -complex(e.c*e.d*sv*cv, e.s*dv) / complex(1-ds*ds, 0)
	}
	if n%2 == 1 {
		var norm float64
		for _, v := range p {
			norm += real(v * cmplx.Conj(v))
		}
		lim := epsilon * math.Sqrt(norm)
		np := len(p)
		for i := 0; i < np; i++ {
			if math.Abs(imag(p[i])) > lim {
				p = append(p, cmplx.Conj(p[i]))
			}
		}
	} else {
		np := len(p)
		for i := 0; i < np; i++ {
			p = append(p, cmplx.Conj(p[i]))
		}
	}
	k := real(prodNeg(p) / prodNeg(z))
	if n%2 == 0 {
		k /= math.Sqrt(1 + epsSq)
	}
	return ZPK{Z: z, P: p, K: k}, nil
}

// BesselNorm selects the frequency normalisation of the Bessel prototype.
type BesselNorm int

const (
	// BesselPhase (the default) matches the phase to half of the asymptotic
	// shift at 1 rad/s; the asymptotes are those of a Butterworth filter.
	BesselPhase BesselNorm = iota
	// BesselDelay normalises the low-frequency group delay to 1 s.
	BesselDelay
	// BesselMag puts the -3 dB magnitude point at 1 rad/s.
	BesselMag
)

// besselY evaluates (n >= 1) the ordinary Bessel polynomial y_n and its derivative at x
// by the three-term recurrence y_k = (2k-1) x y_{k-1} + y_{k-2}.
func besselY(n int, x complex128) (y, dy complex128) {
	y0, d0 := complex(1, 0), complex(0, 0) // n >= 1
	y1, d1 := 1+x, complex(1, 0)
	for k := 2; k <= n; k++ {
		c := complex(float64(2*k-1), 0)
		y0, y1, d0, d1 = y1, c*x*y1+y0, d1, c*(y1+x*d1)+d0
	}
	return y1, d1
}

// campos returns the starting points of Campos-Calderón for the zeros of the
// Bessel polynomial y_n (scipy's _campos_zeros).
func campos(n int) []complex128 {
	if n == 1 {
		return []complex128{-1}
	}
	nf := float64(n)
	n2, n3, n4, n5 := nf*nf, nf*nf*nf, nf*nf*nf*nf, nf*nf*nf*nf*nf
	s := 2*n2 - 3*n4 + n5
	b3 := (16 - 8*nf) / s
	b2 := (-24 - 12*nf + 12*n2) / s
	b1 := (8 + 24*nf - 12*n2 - 2*n3) / s
	b0 := (-6*nf + 5*n3 - n4) / s
	r := 2*n2 + n3
	a1 := (-6 - 6*nf) / r
	a2 := 6 / r
	x := make([]complex128, n)
	for i := range x {
		k := float64(i + 1)
		x[i] = complex(a1*k+a2*k*k, b0+b1*k+b2*k*k+b3*k*k*k)
	}
	return x
}

// aberthMaxIter is the iteration limit of the Aberth–Ehrlich root finder (a
// variable so tests can reach the failure path).
var aberthMaxIter = 50

// besselZeros returns the zeros of the Bessel polynomial y_n in the order of
// scipy's _bessel_zeros, refined to full precision and made exactly
// conjugate-symmetric.
func besselZeros(n int) ([]complex128, error) {
	x := campos(n)
	alpha := make([]complex128, n)
	for it := 0; it < aberthMaxIter; it++ {
		small := true
		for k := range x {
			y, dy := besselY(n, x[k])
			alpha[k] = -y / dy
			if cmplx.Abs(alpha[k]) > 1e-9 {
				small = false
			}
		}
		for k := range x {
			var beta complex128
			for j := range x {
				if j != k {
					beta += 1 / (x[k] - x[j])
				}
			}
			x[k] += alpha[k] / (1 + alpha[k]*beta)
		}
		if small {
			break
		}
	}
	// The polynomial's zeros are so ill-conditioned in double precision
	// (relative error ~ 10^(n/2) eps) that the final Newton steps are taken in
	// 320-bit arithmetic on the exact integer coefficients.
	coef := besselCoefs(n)
	out := make([]complex128, n)
	var sum complex128
	for k := range x {
		out[k] = newtonBig(coef, x[k], besselNewtonSteps)
	}
	for k := range out {
		out[k] = (out[k] + cmplx.Conj(out[n-1-k])) / 2
		sum += out[k]
	}
	// the zeros of y_n sum to -1
	if !(cmplx.Abs(sum+1) <= 1e-9) {
		return nil, errors.New("signal: Bessel zeros failed to converge")
	}
	return out, nil
}

// besselNewtonSteps is the number of high-precision Newton steps (a variable so
// tests can reach the failure path).
var besselNewtonSteps = 6

// besselCoefs returns the exact coefficients c[k] of x^k in y_n(x):
// (n+k)! / (2^k k! (n-k)!).
func besselCoefs(n int) []*big.Int {
	c := make([]*big.Int, n+1)
	for k := range c {
		num := new(big.Int).MulRange(int64(n-k+1), int64(n+k))
		den := new(big.Int).MulRange(1, int64(k))
		den.Lsh(den, uint(k))
		c[k] = num.Quo(num, den)
	}
	return c
}

// newtonBig refines a root x0 of the polynomial with exact coefficients c
// (lowest degree first) by Newton's method in 320-bit complex arithmetic.
func newtonBig(c []*big.Int, x0 complex128, steps int) complex128 {
	const prec = 320
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	xr, xi := nf().SetFloat64(real(x0)), nf().SetFloat64(imag(x0))
	for it := 0; it < steps; it++ {
		// Horner for y and y'
		yr, yi, dr, di := nf(), nf(), nf(), nf()
		for k := len(c) - 1; k >= 0; k-- {
			// d = d*x + y ; y = y*x + c[k]
			t1, t2 := nf().Mul(dr, xr), nf().Mul(di, xi)
			t3, t4 := nf().Mul(dr, xi), nf().Mul(di, xr)
			ndr, ndi := nf().Sub(t1, t2), nf().Add(t3, t4)
			ndr.Add(ndr, yr)
			ndi.Add(ndi, yi)
			t1, t2 = nf().Mul(yr, xr), nf().Mul(yi, xi)
			t3, t4 = nf().Mul(yr, xi), nf().Mul(yi, xr)
			yr, yi = nf().Sub(t1, t2), nf().Add(t3, t4)
			yr.Add(yr, nf().SetInt(c[k]))
			dr, di = ndr, ndi
		}
		// x -= y/d
		den := nf().Add(nf().Mul(dr, dr), nf().Mul(di, di))
		qr := nf().Add(nf().Mul(yr, dr), nf().Mul(yi, di))
		qi := nf().Sub(nf().Mul(yi, dr), nf().Mul(yr, di))
		xr.Sub(xr, qr.Quo(qr, den))
		xi.Sub(xi, qi.Quo(qi, den))
	}
	re, _ := xr.Float64()
	im, _ := xi.Float64()
	return complex(re, im)
}

// secant is scipy.optimize.newton without a derivative: the secant method
// from x0 with absolute tolerance tol.
func secant(f func(float64) float64, x0, tol float64) float64 {
	p0 := x0
	p1 := x0*(1+1e-4) + 1e-4
	q0, q1 := f(p0), f(p1)
	if math.Abs(q1) < math.Abs(q0) {
		p0, p1, q0, q1 = p1, p0, q1, q0
	}
	for it := 0; it < 50; it++ {
		if q1 == q0 {
			return (p1 + p0) / 2
		}
		var p float64
		if math.Abs(q1) > math.Abs(q0) {
			p = (-q0/q1*p1 + p0) / (1 - q0/q1)
		} else {
			p = (-q1/q0*p0 + p1) / (1 - q1/q0)
		}
		if math.Abs(p-p1) <= tol {
			return p
		}
		p0, q0 = p1, q1
		p1 = p
		q1 = f(p1)
	}
	return p1
}

// BesselAP returns the analog Bessel (Thomson) lowpass prototype of order n
// (scipy.signal.besselap).
func BesselAP(n int, norm BesselNorm) (ZPK, error) {
	if n < 0 {
		return ZPK{}, errOrder
	}
	if norm < BesselPhase || norm > BesselMag {
		return ZPK{}, errors.New("signal: unknown Bessel normalization")
	}
	if n == 0 {
		return ZPK{K: 1}, nil
	}
	x, err := besselZeros(n)
	if err != nil {
		return ZPK{}, err
	}
	p := make([]complex128, n)
	for i, v := range x {
		p[i] = 1 / v
	}
	// a_last = (2n-1)!! = (2n)!/(n! 2^n), kept exact until the float rounding
	aLast := new(big.Int).MulRange(int64(n+1), int64(2*n))
	aLast.Rsh(aLast, uint(n))
	af, _ := new(big.Float).SetInt(aLast).Float64()
	switch norm {
	case BesselPhase:
		// log10 of the big integer: exponent plus mantissa
		l10 := bigLog10(aLast)
		f := math.Pow(10, -l10/float64(n))
		for i := range p {
			p[i] *= complex(f, 0)
		}
		return ZPK{P: p, K: 1}, nil
	case BesselDelay:
		return ZPK{P: p, K: af}, nil
	}
	pc := p
	gain := func(w float64) float64 {
		d := complex(1, 0)
		for _, v := range pc {
			d *= complex(0, w) - v
		}
		return cmplx.Abs(complex(af, 0)/d) - 1/math.Sqrt2
	}
	nf := secant(gain, 1.5, 1.48e-8)
	for i := range p {
		p[i] /= complex(nf, 0)
	}
	return ZPK{P: p, K: math.Pow(nf, -float64(n)) * af}, nil
}

// bigLog10 returns log10 of a positive big integer (math.log10 of a Python int).
func bigLog10(x *big.Int) float64 {
	bl := x.BitLen()
	if bl <= 1000 {
		f, _ := new(big.Float).SetInt(x).Float64()
		return math.Log10(f)
	}
	shift := uint(bl - 100)
	top, _ := new(big.Float).SetInt(new(big.Int).Rsh(x, shift)).Float64()
	return math.Log10(top) + float64(shift)*math.Log10(2)
}

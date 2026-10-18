package signal

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
)

func relativeDegree(z, p []complex128) (int, error) {
	if d := len(p) - len(z); d >= 0 {
		return d, nil
	}
	return 0, errors.New("signal: improper transfer function: must have at least as many poles as zeros")
}

func scaleC(c []complex128, f complex128) []complex128 {
	r := make([]complex128, len(c))
	for i, v := range c {
		r[i] = f * v
	}
	return r
}

// LP2LPZPK transforms a lowpass prototype to a lowpass filter with cutoff wo
// (scipy.signal.lp2lp_zpk).
func LP2LPZPK(s ZPK, wo float64) (ZPK, error) {
	deg, err := relativeDegree(s.Z, s.P)
	if err != nil {
		return ZPK{}, err
	}
	return ZPK{scaleC(s.Z, complex(wo, 0)), scaleC(s.P, complex(wo, 0)), s.K * math.Pow(wo, float64(deg))}, nil
}

// LP2HPZPK transforms a lowpass prototype to a highpass filter with cutoff
// wo (scipy.signal.lp2hp_zpk).
func LP2HPZPK(s ZPK, wo float64) (ZPK, error) {
	deg, err := relativeDegree(s.Z, s.P)
	if err != nil {
		return ZPK{}, err
	}
	z := make([]complex128, len(s.Z), len(s.Z)+deg)
	for i, v := range s.Z {
		z[i] = complex(wo, 0) / v
	}
	p := make([]complex128, len(s.P))
	for i, v := range s.P {
		p[i] = complex(wo, 0) / v
	}
	z = append(z, make([]complex128, deg)...)
	k := s.K * real(prodNeg(s.Z)/prodNeg(s.P))
	return ZPK{z, p, k}, nil
}

// cmulE is the complex product with every operation rounded separately, as
// NumPy computes it: no fused multiply-add can change the sign of a zero.
func cmulE(a, b complex128) complex128 {
	ar, ai, br, bi := real(a), imag(a), real(b), imag(b)
	return complex(float64(ar*br)-float64(ai*bi), float64(ar*bi)+float64(ai*br))
}

// cdivE is NumPy's complex division (Smith's algorithm, scaling by a
// reciprocal), with the same signed zeros.
func cdivE(a, b complex128) complex128 {
	ar, ai, br, bi := real(a), imag(a), real(b), imag(b)
	abr, abi := math.Abs(br), math.Abs(bi)
	switch {
	case abr >= abi:
		if abr == 0 && abi == 0 {
			return complex(ar/abr, ai/abi)
		}
		rat := bi / br
		scl := 1.0 / (br + float64(bi*rat))
		return complex((ar+float64(ai*rat))*scl, (ai-float64(ar*rat))*scl)
	default:
		rat := br / bi
		scl := 1.0 / (bi + float64(br*rat))
		return complex((float64(ar*rat)+ai)*scl, (float64(ai*rat)-ar)*scl)
	}
}

// splitBand duplicates the points c of a lowpass (or highpass) prototype and
// shifts them to +wo and -wo: v +- sqrt(v^2 - wo^2).
func splitBand(c []complex128, wo float64) []complex128 {
	out := make([]complex128, 0, 2*len(c))
	root := func(v complex128) complex128 {
		sq := cmulE(v, v)
		return cmplx.Sqrt(complex(real(sq)-wo*wo, imag(sq)-0))
	}
	for _, v := range c {
		out = append(out, v+root(v))
	}
	for _, v := range c {
		out = append(out, v-root(v))
	}
	return out
}

// LP2BPZPK transforms a lowpass prototype to a bandpass filter centred on wo
// with bandwidth bw (scipy.signal.lp2bp_zpk).
func LP2BPZPK(s ZPK, wo, bw float64) (ZPK, error) {
	deg, err := relativeDegree(s.Z, s.P)
	if err != nil {
		return ZPK{}, err
	}
	shift := func(c []complex128) []complex128 {
		// (v*bw)/2 as two complex operations: NumPy's signed zeros decide which
		// of a conjugate pair the square root puts first
		lp := make([]complex128, len(c))
		for i, v := range c {
			lp[i] = cdivE(cmulE(v, complex(bw, 0)), 2)
		}
		return splitBand(lp, wo)
	}
	z := append(shift(s.Z), make([]complex128, deg)...)
	return ZPK{z, shift(s.P), s.K * math.Pow(bw, float64(deg))}, nil
}

// LP2BSZPK transforms a lowpass prototype to a bandstop filter centred on wo
// with bandwidth bw (scipy.signal.lp2bs_zpk).
func LP2BSZPK(s ZPK, wo, bw float64) (ZPK, error) {
	deg, err := relativeDegree(s.Z, s.P)
	if err != nil {
		return ZPK{}, err
	}
	shift := func(c []complex128) []complex128 {
		hp := make([]complex128, len(c))
		for i, v := range c {
			hp[i] = cdivE(complex(bw/2, 0), v)
		}
		return splitBand(hp, wo)
	}
	z := shift(s.Z)
	for i := 0; i < deg; i++ {
		z = append(z, complex(0, wo))
	}
	for i := 0; i < deg; i++ {
		z = append(z, complex(0, -wo))
	}
	k := s.K * real(prodNeg(s.Z)/prodNeg(s.P))
	return ZPK{z, shift(s.P), k}, nil
}

// BilinearZPK maps an analog filter to a digital one with the bilinear
// transform at sampling frequency fs (scipy.signal.bilinear_zpk). Zeros at
// infinity go to the Nyquist frequency.
func BilinearZPK(s ZPK, fs float64) (ZPK, error) {
	if !(fs > 0) || math.IsInf(fs, 0) {
		return ZPK{}, errors.New("signal: fs must be a positive finite number")
	}
	deg, err := relativeDegree(s.Z, s.P)
	if err != nil {
		return ZPK{}, err
	}
	fs2 := complex(2*fs, 0)
	z := make([]complex128, len(s.Z), len(s.Z)+deg)
	for i, v := range s.Z {
		z[i] = (fs2 + v) / (fs2 - v)
	}
	p := make([]complex128, len(s.P))
	for i, v := range s.P {
		p[i] = (fs2 + v) / (fs2 - v)
	}
	for i := 0; i < deg; i++ {
		z = append(z, -1)
	}
	num, den := complex(1, 0), complex(1, 0)
	for _, v := range s.Z {
		num *= fs2 - v
	}
	for _, v := range s.P {
		den *= fs2 - v
	}
	return ZPK{z, p, s.K * real(num/den)}, nil
}

func comb(n, k int) float64 {
	r := 1.0
	for j := 1; j <= k; j++ {
		r = r * float64(n-k+j) / float64(j)
	}
	return r
}

func dropLeadingZeros(a []float64) []float64 {
	for len(a) > 0 && a[0] == 0 {
		a = a[1:]
	}
	return a
}

func powInt(x float64, n int) float64 { return math.Pow(x, float64(n)) }

// LP2LP transforms a lowpass prototype b/a to a lowpass with cutoff wo
// (scipy.signal.lp2lp).
func LP2LP(b, a []float64, wo float64) ([]float64, []float64, error) {
	d, n := len(a), len(b)
	m := max(d, n)
	pwo := make([]float64, m)
	for i := range pwo {
		pwo[i] = math.Pow(wo, float64(m-1-i))
	}
	s1, s2 := max(n-d, 0), max(d-n, 0)
	nb := make([]float64, len(b))
	for i, v := range b {
		nb[i] = v * pwo[s1] / pwo[s2+i]
	}
	na := make([]float64, len(a))
	for i, v := range a {
		na[i] = v * pwo[s1] / pwo[s1+i]
	}
	return Normalize(nb, na)
}

// LP2HP transforms a lowpass prototype b/a to a highpass with cutoff wo
// (scipy.signal.lp2hp).
func LP2HP(b, a []float64, wo float64) ([]float64, []float64, error) {
	d, n := len(a), len(b)
	m := max(d, n)
	pwo := make([]float64, m)
	for i := range pwo {
		pwo[i] = 1
		if wo != 1 {
			pwo[i] = math.Pow(wo, float64(i))
		}
	}
	outa, outb := make([]float64, m), make([]float64, m)
	for i := 0; i < d; i++ {
		outa[i] = a[d-1-i] * pwo[i]
	}
	for i := 0; i < n; i++ {
		outb[i] = b[n-1-i] * pwo[i]
	}
	return Normalize(outb, outa)
}

// LP2BP transforms a lowpass prototype b/a to a bandpass centred on wo with
// bandwidth bw (scipy.signal.lp2bp).
func LP2BP(b, a []float64, wo, bw float64) ([]float64, []float64, error) {
	d, n := len(a)-1, len(b)-1
	ma := max(n, d)
	wosq := wo * wo
	build := func(c []float64, deg int) []float64 {
		out := make([]float64, deg+ma+1)
		for j := range out {
			val := 0.0
			for i := 0; i <= deg; i++ {
				for k := 0; k <= i; k++ {
					if ma-i+2*k == j {
						val += comb(i, k) * c[deg-i] * powInt(wosq, i-k) / powInt(bw, i)
					}
				}
			}
			out[len(out)-1-j] = val
		}
		return out
	}
	return Normalize(build(b, n), build(a, d))
}

// LP2BS transforms a lowpass prototype b/a to a bandstop centred on wo with
// bandwidth bw (scipy.signal.lp2bs).
func LP2BS(b, a []float64, wo, bw float64) ([]float64, []float64, error) {
	d, n := len(a)-1, len(b)-1
	m := max(n, d)
	wosq := wo * wo
	build := func(c []float64, deg int) []float64 {
		out := make([]float64, 2*m+1)
		for j := range out {
			val := 0.0
			for i := 0; i <= deg; i++ {
				for k := 0; k <= m-i; k++ {
					if i+2*k == j {
						val += comb(m-i, k) * c[deg-i] * powInt(wosq, m-i-k) * powInt(bw, i)
					}
				}
			}
			out[len(out)-1-j] = val
		}
		return out
	}
	return Normalize(build(b, n), build(a, d))
}

// Bilinear maps an analog transfer function b/a (in s) to a digital one (in
// z) by the bilinear transform at sampling frequency fs
// (scipy.signal.bilinear).
func Bilinear(b, a []float64, fs float64) ([]float64, []float64, error) {
	if !(fs > 0) || math.IsInf(fs, 0) {
		return nil, nil, errors.New("signal: fs must be a positive finite number")
	}
	b, a = dropLeadingZeros(b), dropLeadingZeros(a)
	fac := math.Sqrt(fs * 2)
	zp1 := []float64{1 / fac, 1 / fac} // (z + 1)/fac, lowest power first
	zm1 := []float64{-fac, fac}        // (z - 1)*fac
	pow := func(p []float64, n int) []float64 {
		r := []float64{1}
		for i := 0; i < n; i++ {
			r = polyMul(r, p)
		}
		return r
	}
	nn := max(len(a), len(b)) - 1
	sum := func(c []float64) []float64 {
		var acc []float64
		for q := 0; q < len(c); q++ {
			coef := c[len(c)-1-q]
			t := pow(zp1, nn-q)
			for i := range t {
				t[i] *= coef
			}
			t = polyMul(t, pow(zm1, q))
			if acc == nil {
				acc = t
			} else {
				for i := range t {
					acc[i] += t[i]
				}
			}
		}
		return acc
	}
	num, den := sum(b), sum(a)
	slices.Reverse(num)
	slices.Reverse(den)
	return Normalize(num, den)
}

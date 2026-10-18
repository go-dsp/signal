package signal

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
)

// ZPK is a filter as zeros, poles and gain. Roots of a real filter are real
// or come in conjugate pairs. In the digital case the transfer function is
// H(z) = K * prod(z - Z[i]) / prod(z - P[i]); in the analog case the same with s.
type ZPK struct {
	Z, P []complex128
	K    float64
}

// SOS is a cascade of second-order sections, each [b0 b1 b2 a0 a1 a2]; the
// digital form is b0 + b1 z^-1 + b2 z^-2 over a0 + a1 z^-1 + a2 z^-2 with
// a0 == 1. The gain is carried by the first section.
type SOS [][6]float64

const cplxTol = 100 * 2.220446049250313e-16

// Normalize scales the transfer function b/a so that a[0] == 1, trims the
// leading zeros of a and the leading numerator coefficients that are all
// below 1e-14 (keeping at least one), like scipy.signal.normalize.
func Normalize(b, a []float64) ([]float64, []float64, error) {
	for len(a) > 0 && a[0] == 0 {
		a = a[1:]
	}
	if len(a) == 0 {
		return nil, nil, errors.New("signal: denominator must have at least one nonzero element")
	}
	if len(b) == 0 {
		return nil, nil, errors.New("signal: empty numerator")
	}
	a0 := a[0]
	nb := make([]float64, len(b))
	for i, v := range b {
		nb[i] = v / a0
	}
	na := make([]float64, len(a))
	for i, v := range a {
		na[i] = v / a0
	}
	lead := 0
	for lead < len(nb) && math.Abs(nb[lead]) <= 1e-14 {
		lead++
	}
	if lead == len(nb) {
		lead--
	}
	return nb[lead:], na, nil
}

// TF2ZPK converts a transfer function b/a to zeros, poles and gain.
func TF2ZPK(b, a []float64) (ZPK, error) {
	b, a, err := Normalize(b, a)
	if err != nil {
		return ZPK{}, err
	}
	k := b[0]
	if k == 0 {
		return ZPK{}, errors.New("signal: numerator is identically zero")
	}
	nb := make([]float64, len(b))
	for i, v := range b {
		nb[i] = v / k
	}
	z, err := Roots(nb)
	if err != nil {
		return ZPK{}, err
	}
	p, err := Roots(a)
	if err != nil {
		return ZPK{}, err
	}
	return ZPK{z, p, k}, nil
}

// ZPK2TF converts zeros, poles and gain to a transfer function. Complex
// roots must come in conjugate pairs.
func ZPK2TF(s ZPK) (b, a []float64, err error) {
	b, err = polyFromRoots(s.Z)
	if err != nil {
		return nil, nil, err
	}
	a, err = polyFromRoots(s.P)
	if err != nil {
		return nil, nil, err
	}
	for i := range b {
		b[i] *= s.K
	}
	return b, a, nil
}

// SOS2TF converts second-order sections to a transfer function.
func SOS2TF(sos SOS) (b, a []float64) {
	b, a = []float64{1}, []float64{1}
	for _, s := range sos {
		b = polyMul(b, s[:3])
		a = polyMul(a, s[3:])
	}
	return b, a
}

// SOS2ZPK converts second-order sections to zeros, poles and gain.
func SOS2ZPK(sos SOS) (ZPK, error) {
	out := ZPK{K: 1}
	for _, s := range sos {
		z, err := TF2ZPK(s[:3], s[3:])
		if err != nil {
			return ZPK{}, err
		}
		out.Z = append(out.Z, z.Z...)
		out.P = append(out.P, z.P...)
		out.K *= z.K
	}
	return out, nil
}

// cplxReal splits z into the complex elements, one per conjugate pair (with
// positive imaginary part, sorted by real part then imaginary magnitude), and
// the real elements (sorted). It fails when a complex value has no conjugate.
func cplxReal(z []complex128, tol float64) (zc []complex128, zr []float64, err error) {
	if len(z) == 0 {
		return nil, nil, nil
	}
	z = slices.Clone(z)
	sort.SliceStable(z, func(i, j int) bool {
		if real(z[i]) != real(z[j]) {
			return real(z[i]) < real(z[j])
		}
		return math.Abs(imag(z[i])) < math.Abs(imag(z[j]))
	})
	var zp, zn []complex128
	for _, v := range z {
		switch {
		case math.Abs(imag(v)) <= tol*cmplx.Abs(v):
			zr = append(zr, real(v))
		case imag(v) > 0:
			zp = append(zp, v)
		default:
			zn = append(zn, v)
		}
	}
	if len(zp) != len(zn) {
		return nil, nil, errors.New("signal: array contains complex value with no matching conjugate")
	}
	byImag := func(c []complex128) {
		sort.SliceStable(c, func(i, j int) bool { return math.Abs(imag(c[i])) < math.Abs(imag(c[j])) })
	}
	for start := 0; start < len(zp); {
		stop := start
		for stop+1 < len(zp) && real(zp[stop+1])-real(zp[stop]) <= tol*cmplx.Abs(zp[stop]) {
			stop++
		}
		byImag(zp[start : stop+1])
		byImag(zn[start : stop+1])
		start = stop + 1
	}
	for i := range zp {
		if cmplx.Abs(zp[i]-cmplx.Conj(zn[i])) > tol*cmplx.Abs(zn[i]) {
			return nil, nil, errors.New("signal: array contains complex value with no matching conjugate")
		}
	}
	zc = make([]complex128, len(zp))
	for i := range zp {
		zc[i] = (zp[i] + cmplx.Conj(zn[i])) / 2
	}
	return zc, zr, nil
}

// cplxPair sorts z into conjugate pairs (negative imaginary part first,
// increasing real part) followed by the real numbers, like scipy's _cplxpair.
func cplxPair(z []complex128, tol float64) ([]complex128, error) {
	zc, zr, err := cplxReal(z, tol)
	if err != nil {
		return nil, err
	}
	out := make([]complex128, 0, len(z))
	for _, c := range zc {
		out = append(out, cmplx.Conj(c), c)
	}
	for _, r := range zr {
		out = append(out, complex(r, 0))
	}
	return out, nil
}

// Pairing selects how ZPK2SOS pairs poles with zeros.
type Pairing int

const (
	// PairNearest (the default for digital filters) pairs each pole nearest
	// the unit circle with the closest zero and pads to an even order.
	PairNearest Pairing = iota
	// PairKeepOdd keeps the odd section last, like PairNearest but without
	// padding the pole and zero counts to an even number.
	PairKeepOdd
	// PairMinimal gives sections of the lowest necessary order; it is the
	// only valid choice for analog filters.
	PairMinimal
)

func isReal(c complex128) bool { return imag(c) == 0 }

func countReal(c []complex128) (n int) {
	for _, v := range c {
		if isReal(v) {
			n++
		}
	}
	return n
}

func deleteAt(c []complex128, i int) []complex128 {
	return append(c[:i:i], c[i+1:]...)
}

// nearestIdx returns the index in fro of the element nearest to "to" among
// those that are real, complex or either.
func nearestIdx(fro []complex128, to complex128, which byte) int {
	order := make([]int, len(fro))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return cmplx.Abs(fro[order[i]]-to) < cmplx.Abs(fro[order[j]]-to)
	})
	for _, i := range order {
		switch {
		case which == 'a', which == 'r' && isReal(fro[i]), which == 'c' && !isReal(fro[i]):
			return i
		}
	}
	return -1
}

// singleSection builds one second-order section from at most two zeros and
// two poles, right-aligned like scipy's _single_zpksos.
func singleSection(z, p []complex128) [6]float64 {
	var s [6]float64
	b, _ := polyFromRoots(z)
	a, _ := polyFromRoots(p)
	copy(s[3-len(b):3], b)
	copy(s[6-len(a):6], a)
	return s
}

// ZPK2SOS converts zeros, poles and gain to second-order sections, ordering
// them so that the poles nearest the unit circle (the imaginary axis if
// analog) come last, as scipy.signal.zpk2sos does.
func ZPK2SOS(s ZPK, pairing Pairing, analog bool) (_ SOS, err error) {
	defer func() {
		// Running out of zeros of the required kind means the zeros and
		// poles cannot be paired (SciPy raises an IndexError there).
		if recover() != nil {
			err = errors.New("signal: zeros and poles cannot be paired into sections")
		}
	}()
	if analog && pairing != PairMinimal {
		return nil, errors.New("signal: for analog zpk2sos conversion, pairing must be PairMinimal")
	}
	if pairing < PairNearest || pairing > PairMinimal {
		return nil, fmt.Errorf("signal: unknown pairing %d", pairing)
	}
	z, p := slices.Clone(s.Z), slices.Clone(s.P)
	if len(z) == 0 && len(p) == 0 {
		if !analog {
			return SOS{{s.K, 0, 0, 1, 0, 0}}, nil
		}
		return SOS{{0, 0, s.K, 0, 0, 1}}, nil
	}
	var nSections int
	if pairing != PairMinimal {
		p = append(p, make([]complex128, max(len(z)-len(p), 0))...)
		z = append(z, make([]complex128, max(len(p)-len(z), 0))...)
		nSections = (max(len(p), len(z)) + 1) / 2
		if len(p)%2 == 1 && pairing == PairNearest {
			p = append(p, 0)
			z = append(z, 0)
		}
	} else {
		if len(p) < len(z) {
			return nil, errors.New("signal: for analog zpk2sos conversion, must have len(p)>=len(z)")
		}
		nSections = (len(p) + 1) / 2
	}
	zc, zr, err := cplxReal(z, cplxTol)
	if err != nil {
		return nil, err
	}
	z = joinCplxReal(zc, zr)
	pc, pr, err := cplxReal(p, cplxTol)
	if err != nil {
		return nil, err
	}
	p = joinCplxReal(pc, pr)

	idxWorst := func(p []complex128) int {
		best, bi := math.Inf(1), 0
		for i, v := range p {
			var d float64
			if analog {
				d = math.Abs(real(v))
			} else {
				d = math.Abs(1 - cmplx.Abs(v))
			}
			if d < best {
				best, bi = d, i
			}
		}
		return bi
	}

	sos := make(SOS, nSections)
	for si := nSections - 1; si >= 0; si-- {
		p1i := idxWorst(p)
		p1 := p[p1i]
		p = deleteAt(p, p1i)
		switch {
		case isReal(p1) && countReal(p) == 0:
			switch {
			case pairing != PairMinimal:
				zi := nearestIdx(z, p1, 'r')
				z1 := z[zi]
				z = deleteAt(z, zi)
				sos[si] = singleSection([]complex128{z1, 0}, []complex128{p1, 0})
			case len(z) > 0:
				zi := nearestIdx(z, p1, 'r')
				z1 := z[zi]
				z = deleteAt(z, zi)
				sos[si] = singleSection([]complex128{z1}, []complex128{p1})
			default:
				sos[si] = singleSection(nil, []complex128{p1})
			}
		case len(p)+1 == len(z) && !isReal(p1) && countReal(p) == 1 && countReal(z) == 1:
			zi := nearestIdx(z, p1, 'c')
			z1 := z[zi]
			z = deleteAt(z, zi)
			sos[si] = singleSection([]complex128{z1, cmplx.Conj(z1)}, []complex128{p1, cmplx.Conj(p1)})
		default:
			var p2 complex128
			if isReal(p1) {
				var realIdx []int
				var realP []complex128
				for i, v := range p {
					if isReal(v) {
						realIdx = append(realIdx, i)
						realP = append(realP, v)
					}
				}
				p2i := realIdx[idxWorst(realP)]
				p2 = p[p2i]
				p = deleteAt(p, p2i)
			} else {
				p2 = cmplx.Conj(p1)
			}
			if len(z) == 0 {
				sos[si] = singleSection(nil, []complex128{p1, p2})
				break
			}
			zi := nearestIdx(z, p1, 'a')
			z1 := z[zi]
			z = deleteAt(z, zi)
			switch {
			case !isReal(z1):
				sos[si] = singleSection([]complex128{z1, cmplx.Conj(z1)}, []complex128{p1, p2})
			case len(z) > 0:
				z2i := nearestIdx(z, p1, 'r')
				z2 := z[z2i]
				z = deleteAt(z, z2i)
				sos[si] = singleSection([]complex128{z1, z2}, []complex128{p1, p2})
			default:
				sos[si] = singleSection([]complex128{z1}, []complex128{p1, p2})
			}
		}
	}
	for i := 0; i < 3; i++ {
		sos[0][i] *= s.K
	}
	return sos, nil
}

// joinCplxReal concatenates the one-per-pair complex roots and the real roots
// (the two outputs of cplxReal), as scipy's np.concatenate(_cplxreal(z)).
func joinCplxReal(zc []complex128, zr []float64) []complex128 {
	out := slices.Clone(zc)
	for _, r := range zr {
		out = append(out, complex(r, 0))
	}
	return out
}

// TF2SOS converts a transfer function to second-order sections.
func TF2SOS(b, a []float64, pairing Pairing, analog bool) (SOS, error) {
	s, err := TF2ZPK(b, a)
	if err != nil {
		return nil, err
	}
	return ZPK2SOS(s, pairing, analog)
}

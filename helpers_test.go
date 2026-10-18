package signal

import (
	"cmp"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"
)

// sortedKeys gives a deterministic iteration order: oracle fixtures are keyed
// by the position of the call within the test.
func sortedKeys[K cmp.Ordered, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)) }

// randRoots returns n roots, real or in conjugate pairs, with radius in
// (rmin, rmax) and the given number of real roots (at least n%2).
func randRoots(r *rand.Rand, n, nreal int, rmin, rmax float64) []complex128 {
	if nreal > n {
		nreal = n
	}
	if (n-nreal)%2 == 1 {
		nreal++
	}
	var out []complex128
	for i := 0; i < nreal; i++ {
		v := rmin + (rmax-rmin)*r.Float64()
		if r.IntN(2) == 0 {
			v = -v
		}
		out = append(out, complex(v, 0))
	}
	for i := 0; i < (n-nreal)/2; i++ {
		rad := rmin + (rmax-rmin)*r.Float64()
		th := (0.02 + 0.96*r.Float64()) * math.Pi
		c := cmplx.Rect(rad, th)
		out = append(out, c, cmplx.Conj(c))
	}
	r.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

func randVec(r *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 2*r.Float64() - 1
	}
	return v
}

// matchRoots returns the largest distance between each root in want and the
// nearest unused root in got, relative to max(1, |want|).
func matchRoots(got, want []complex128) float64 {
	if len(got) != len(want) {
		return math.Inf(1)
	}
	used := make([]bool, len(got))
	worst := 0.0
	// match the roots with the fewest close neighbours first: simple greedy
	idx := make([]int, len(want))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return cmplx.Abs(want[idx[a]]) < cmplx.Abs(want[idx[b]]) })
	for _, i := range idx {
		best, bj := math.Inf(1), -1
		for j, g := range got {
			if used[j] {
				continue
			}
			if d := cmplx.Abs(g - want[i]); d < best {
				best, bj = d, j
			}
		}
		used[bj] = true
		worst = math.Max(worst, best/math.Max(1, cmplx.Abs(want[i])))
	}
	return worst
}

func toC(a Arr) []complex128 { return Out{a}.C() }

func sosFlat(s SOS) []float64 {
	var f []float64
	for _, r := range s {
		f = append(f, r[:]...)
	}
	return f
}

func sosFromFlat(f []float64) SOS {
	s := make(SOS, len(f)/6)
	for i := range s {
		copy(s[i][:], f[6*i:])
	}
	return s
}

func eqSlice(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func mustErr(t *testing.T, what string, err error) {
	t.Helper()
	if err == nil {
		t.Errorf("%s: expected an error", what)
	}
}

func mustOK(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

type bigInt = big.Int

func bigOne() *big.Int { return big.NewInt(1) }

func expf(x float64) float64 { return math.Exp(x) }

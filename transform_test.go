package signal

import (
	"math"
	"testing"
)

func TestZPKTransformsOracle(t *testing.T) {
	rng := newRNG(30)
	for n := 1; n <= 9; n++ {
		proto := ZPK{randRoots(rng, n-n/3, 1, 0.3, 3), randRoots(rng, n, 1, 0.3, 3), 0.5 + rng.Float64()}
		for i, v := range proto.P {
			proto.P[i] = complex(-math.Abs(real(v))-0.05, imag(v))
		}
		args := func(s ZPK, extra ...any) []any {
			return append([]any{C(s.Z), C(s.P), s.K}, extra...)
		}
		type tc struct {
			name string
			f    func() (ZPK, error)
			a    []any
		}
		for _, c := range []tc{
			{"lp2lp_zpk", func() (ZPK, error) { return LP2LPZPK(proto, 2.5) }, args(proto, 2.5)},
			{"lp2hp_zpk", func() (ZPK, error) { return LP2HPZPK(proto, 2.5) }, args(proto, 2.5)},
			{"lp2bp_zpk", func() (ZPK, error) { return LP2BPZPK(proto, 2.5, 0.7) }, args(proto, 2.5, 0.7)},
			{"lp2bs_zpk", func() (ZPK, error) { return LP2BSZPK(proto, 2.5, 0.7) }, args(proto, 2.5, 0.7)},
			{"bilinear_zpk", func() (ZPK, error) { return BilinearZPK(proto, 3) }, args(proto, 3.0)},
		} {
			got, err := c.f()
			mustOK(t, err)
			fn := "signal." + c.name
			var o Out
			if c.name == "bilinear_zpk" {
				o = MustOracle(t, fn, c.a, KW{})
			} else {
				o = MustOracle(t, fn, c.a, nil)
			}
			compareC(t, c.name+" z", got.Z, toC(o.Idx(0).Arr()), 1e-13)
			compareC(t, c.name+" p", got.P, toC(o.Idx(1).Arr()), 1e-13)
			compareVec(t, c.name+" k", []float64{got.K}, []float64{o.Idx(2).X()}, 1e-13)
		}
	}
	improper := ZPK{Z: []complex128{1, 2}, P: []complex128{-1}, K: 1}
	for _, f := range []func() error{
		func() error { _, e := LP2LPZPK(improper, 1); return e },
		func() error { _, e := LP2HPZPK(improper, 1); return e },
		func() error { _, e := LP2BPZPK(improper, 1, 1); return e },
		func() error { _, e := LP2BSZPK(improper, 1, 1); return e },
		func() error { _, e := BilinearZPK(improper, 1); return e },
		func() error { _, e := BilinearZPK(ZPK{}, 0); return e },
	} {
		mustErr(t, "improper", f())
	}
}

func TestTFTransformsOracle(t *testing.T) {
	rng := newRNG(31)
	for n := 1; n <= 6; n++ {
		for _, nb := range []int{n, n - 1, n + 1} {
			if nb < 1 {
				continue
			}
			b, a := randVec(rng, nb), randVec(rng, n+1)
			a[0] += 2
			a2 := make([]float64, len(a))
			copy(a2, a)
			type tc struct {
				name string
				f    func() ([]float64, []float64, error)
				args []any
			}
			for _, c := range []tc{
				{"lp2lp", func() ([]float64, []float64, error) { return LP2LP(b, a, 1.7) }, []any{F(b), F(a), 1.7}},
				{"lp2hp", func() ([]float64, []float64, error) { return LP2HP(b, a, 1.7) }, []any{F(b), F(a), 1.7}},
				{"lp2hp", func() ([]float64, []float64, error) { return LP2HP(b, a, 1) }, []any{F(b), F(a), 1.0}},
				{"lp2bp", func() ([]float64, []float64, error) { return LP2BP(b, a, 1.7, 0.4) }, []any{F(b), F(a), 1.7, 0.4}},
				{"lp2bs", func() ([]float64, []float64, error) { return LP2BS(b, a, 1.7, 0.4) }, []any{F(b), F(a), 1.7, 0.4}},
				{"bilinear", func() ([]float64, []float64, error) { return Bilinear(b, a, 4) }, []any{F(b), F(a), 4.0}},
			} {
				gb, ga, err := c.f()
				mustOK(t, err)
				o := MustOracle(t, "signal."+c.name, c.args, nil)
				compareVec(t, c.name+" b", gb, o.Idx(0).F(), 1e-12)
				compareVec(t, c.name+" a", ga, o.Idx(1).F(), 1e-12)
			}
		}
	}
	// leading zeros are dropped by bilinear
	gb, ga, err := Bilinear([]float64{0, 1}, []float64{0, 1, 2}, 1)
	mustOK(t, err)
	o := MustOracle(t, "signal.bilinear", []any{F([]float64{0, 1}), F([]float64{0, 1, 2}), 1.0}, nil)
	compareVec(t, "bilinear b", gb, o.Idx(0).F(), 1e-14)
	compareVec(t, "bilinear a", ga, o.Idx(1).F(), 1e-14)
	_, _, err = Bilinear([]float64{1}, []float64{1}, -1)
	mustErr(t, "fs", err)
	_, _, err = Bilinear([]float64{1}, nil, 1)
	mustErr(t, "empty", err)
	_, _, err = LP2LP([]float64{1}, []float64{0}, 1)
	mustErr(t, "zero denominator", err)
}

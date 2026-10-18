package signal

import (
	"math"
	"testing"
)

func winSpec(name string, params []float64) any {
	if len(params) == 0 {
		return name
	}
	items := []any{name}
	for _, p := range params {
		if math.IsNaN(p) {
			items = append(items, nil) // Python None
		} else {
			items = append(items, p)
		}
	}
	return Tup(items...)
}

func TestGetWindowOracle(t *testing.T) {
	type wc struct {
		name   string
		params []float64
		tol    float64
	}
	cases := []wc{
		{"boxcar", nil, 0}, {"triang", nil, 1e-15}, {"bartlett", nil, 1e-15}, {"hann", nil, 1e-14},
		{"hamming", nil, 1e-14}, {"blackman", nil, 1e-14}, {"blackmanharris", nil, 1e-14},
		{"nuttall", nil, 1e-14}, {"flattop", nil, 1e-14}, {"cosine", nil, 1e-15}, {"bohman", nil, 1e-14},
		{"parzen", nil, 1e-14}, {"barthann", nil, 1e-14},
		{"kaiser", []float64{8.6}, 1e-13}, {"kaiser", []float64{0}, 1e-13}, {"gaussian", []float64{7}, 1e-14},
		{"general_gaussian", []float64{1.5, 7}, 1e-14}, {"general_hamming", []float64{0.6}, 1e-14},
		{"tukey", []float64{0.5}, 1e-14}, {"tukey", []float64{0.2}, 1e-14}, {"tukey", []float64{0}, 0}, {"tukey", []float64{1}, 1e-14},
		{"exponential", []float64{math.NaN(), 3}, 1e-14}, {"chebwin", []float64{80}, 1e-12}, {"chebwin", []float64{60}, 1e-12},
		{"dpss", []float64{2.5}, 1e-9},
	}
	for _, c := range cases {
		for _, n := range []int{1, 2, 3, 4, 5, 8, 9, 16, 31, 64, 101, 128} {
			if c.name == "dpss" && n < 6 {
				continue
			}
			for _, fftbins := range []bool{true, false} {
				got, err := GetWindow(c.name, n, fftbins, c.params...)
				mustOK(t, err)
				o, oerr := Oracle(t, "signal.get_window", []any{winSpec(c.name, c.params), n}, KW{"fftbins": fftbins})
				if oerr != nil {
					t.Fatalf("%s(%v) n=%d: scipy: %v", c.name, c.params, n, oerr)
				}
				compareVec(t, "window "+c.name, got, o.F(), c.tol)
			}
		}
	}
	// general_cosine takes a list of coefficients
	for _, n := range []int{7, 8} {
		got, err := GetWindow("general_cosine", n, false, 0.3, 0.5, 0.2)
		mustOK(t, err)
		o := MustOracle(t, "signal.get_window", []any{Tup("general_cosine", F([]float64{0.3, 0.5, 0.2})), n}, KW{"fftbins": false})
		compareVec(t, "window general_cosine", got, o.F(), 1e-14)
	}
	// suffixes and aliases
	got, _ := GetWindow("hann_symmetric", 8, true)
	o := MustOracle(t, "signal.get_window", []any{"hann_symmetric", 8}, nil)
	compareVec(t, "window hann_symmetric", got, o.F(), 1e-14)
	got, _ = GetWindow("hann_periodic", 8, false)
	o = MustOracle(t, "signal.get_window", []any{"hann_periodic", 8}, nil)
	compareVec(t, "window hann_periodic", got, o.F(), 1e-14)
	for _, bad := range []struct {
		name string
		n    int
		p    []float64
	}{{"nosuch", 8, nil}, {"hann", 0, nil}, {"hann", 8, []float64{1}}, {"kaiser", 8, nil}, {"general_cosine", 8, nil},
		{"dpss", 8, nil}, {"dpss", 8, []float64{9}}, {"exponential_symmetric", 8, []float64{2, 3}}, {"general_gaussian", 8, []float64{1}}} {
		_, err := GetWindow(bad.name, bad.n, true, bad.p...)
		mustErr(t, bad.name, err)
	}
	if w := Hann(0, true); len(w) != 0 {
		t.Error("empty window")
	}
	if w := Tukey(0.5)(-2, true); len(w) != 0 {
		t.Error("tukey negative length")
	}
}

func TestDPSSOracle(t *testing.T) {
	type dc struct {
		m    int
		nw   float64
		k    int
		sym  bool
		norm DPSSNorm
		ns   any
	}
	cases := []dc{
		{64, 2.5, 4, true, DPSSNorm2, 2}, {65, 3, 5, true, DPSSNorm2, 2}, {64, 4, 7, false, DPSSNorm2, 2},
		{128, 4, 7, true, DPSSNorm2, 2}, {1000, 4, 7, true, DPSSNorm2, 2}, {512, 3.5, 1, true, DPSSNorm2, 2},
		{64, 2.5, 4, true, DPSSApproximate, "approximate"}, {63, 2.5, 3, true, DPSSApproximate, "approximate"},
		{64, 2.5, 4, true, DPSSSubsample, "subsample"}, {65, 2.5, 4, false, DPSSSubsample, "subsample"},
		{200, 6, 11, true, DPSSNorm2, 2}, {16, 1, 16, true, DPSSNorm2, 2}, {1024, 8, 4, false, DPSSApproximate, "approximate"},
	}
	for _, c := range cases {
		got, ratios, err := DPSS(c.m, c.nw, c.k, c.sym, c.norm)
		mustOK(t, err)
		o := MustOracle(t, "signal.windows.dpss", []any{c.m, c.nw}, KW{"Kmax": c.k, "sym": c.sym, "norm": c.ns, "return_ratios": true})
		want := o.Idx(0).Rows()
		if c.k == 1 {
			want = [][]float64{o.Idx(0).F()}
		}
		for i := range got {
			compareVec(t, "dpss window", got[i], want[i], 1e-9)
		}
		compareVec(t, "dpss ratios", ratios, o.Idx(1).F(), 1e-9)
	}
	_, _, err := DPSS(64, 2.5, 1, true, DPSSNorm(7))
	mustErr(t, "norm", err)
	_, _, err = DPSS(64, 2.5, 0, true, DPSSNorm2)
	mustErr(t, "kmax 0", err)
	_, _, err = DPSS(64, 2.5, 65, true, DPSSNorm2)
	mustErr(t, "kmax too large", err)
	_, _, err = DPSS(8, 4, 1, true, DPSSNorm2)
	mustErr(t, "NW too large", err)
	_, _, err = DPSS(64, 0, 1, true, DPSSNorm2)
	mustErr(t, "NW zero", err)
	w, r, err := DPSS(1, 2, 3, true, DPSSNorm2)
	if err != nil || len(w) != 3 || w[0][0] != 1 || r[0] != 1 {
		t.Error("length-one DPSS")
	}
	if w, _, _ := DPSS(0, 2, 0, true, DPSSNorm2); len(w) != 1 || len(w[0]) != 0 {
		t.Error("empty DPSS")
	}
	_ = math.Pi
}

func TestNextFastLen(t *testing.T) {
	cases := []int{1, 2, 3, 5, 6, 7, 11, 13, 17, 100, 1001, 4097, 99999, 1000003}
	for _, n := range cases {
		o := MustOracle(t, "fft.next_fast_len", []any{n}, KW{"real": true})
		if got := nextFastLen(n); float64(got) != o.X() {
			t.Errorf("nextFastLen(%d) = %d, scipy %v", n, got, o.X())
		}
	}
}

func TestTridiagEigen(t *testing.T) {
	rng := newRNG(50)
	for _, n := range []int{1, 2, 3, 10, 40} {
		d, e := randVec(rng, n), randVec(rng, max(n-1, 0))
		k := min(n, 3)
		vals, vecs := tridiagTopEigen(d, e, k)
		for i := range vals {
			// residual of T v = lambda v
			var res, nrm float64
			for r := 0; r < n; r++ {
				s := d[r] * vecs[i][r]
				if r > 0 {
					s += e[r-1] * vecs[i][r-1]
				}
				if r < n-1 {
					s += e[r] * vecs[i][r+1]
				}
				res = math.Max(res, math.Abs(s-vals[i]*vecs[i][r]))
				nrm += vecs[i][r] * vecs[i][r]
			}
			if res > 1e-12 || math.Abs(nrm-1) > 1e-12 {
				t.Errorf("n=%d eigenpair %d: residual %g norm %g", n, i, res, nrm)
			}
		}
	}
	// pivoting in the tridiagonal solve: zero diagonal with large couplings
	d := []float64{0, 0, 0, 0}
	e := []float64{1, 2, 3}
	b := []float64{1, 2, 3, 4}
	solveShiftedTridiag(d, e, 0.1, 5, b)
	if math.IsNaN(b[0]) {
		t.Error("NaN from pivoted solve")
	}
	// exactly singular
	b = []float64{1}
	solveShiftedTridiag([]float64{2}, nil, 2, 1, b)
	if math.IsInf(b[0], 0) || math.IsNaN(b[0]) {
		t.Error("singular 1x1")
	}
	b = []float64{1, 1}
	solveShiftedTridiag([]float64{1, 1}, []float64{0}, 1, 1, b)
	if math.IsNaN(b[0]) || math.IsNaN(b[1]) {
		t.Error("singular 2x2")
	}
}

func TestSolveDense(t *testing.T) {
	a := [][]float64{{0, 2, 1}, {1, 1, 1}, {4, 0, 1}}
	b := []float64{3, 3, 5}
	mustOK(t, solveDense(a, b))
	for i, want := range []float64{1, 1, 1} {
		if math.Abs(b[i]-want) > 1e-14 {
			t.Errorf("x[%d] = %g", i, b[i])
		}
	}
	mustErr(t, "singular", solveDense([][]float64{{1, 1}, {1, 1}}, []float64{1, 2}))
}

func TestWindowAliases(t *testing.T) {
	aliases := map[string][]string{
		"boxcar": {"box", "ones", "rect", "rectangular"}, "triang": {"triangle", "tri"},
		"bartlett": {"bart", "brt"}, "hann": {"han"}, "hamming": {"hamm", "ham"},
		"blackman": {"black", "blk"}, "blackmanharris": {"blackharr", "bkh"},
		"nuttall": {"nutl", "nut"}, "flattop": {"flat", "flt"}, "cosine": {"halfcosine"},
		"bohman": {"bman", "bmn"}, "parzen": {"parz", "par"}, "barthann": {"brthan", "bth"},
	}
	for canon, names := range aliases {
		want, err := GetWindow(canon, 9, false)
		mustOK(t, err)
		for _, a := range names {
			got, err := GetWindow(a, 9, false)
			mustOK(t, err)
			if !eqSlice(got, want) {
				t.Errorf("alias %s differs from %s", a, canon)
			}
		}
	}
	withParams := map[string][][]string{"kaiser": {{"ksr"}}, "gaussian": {{"gauss", "gss"}}, "tukey": {{"tuk"}}, "chebwin": {{"cheb"}}}
	for canon, names := range withParams {
		want, _ := GetWindow(canon, 9, false, 4)
		for _, a := range names[0] {
			got, err := GetWindow(a, 9, false, 4)
			mustOK(t, err)
			if !eqSlice(got, want) {
				t.Errorf("alias %s differs from %s", a, canon)
			}
		}
	}
	want, _ := GetWindow("exponential", 9, false, math.NaN(), 2)
	got, err := GetWindow("poisson", 9, false, math.NaN(), 2)
	mustOK(t, err)
	if !eqSlice(got, want) {
		t.Error("poisson alias")
	}
	if sturmCount([]float64{0, 0}, []float64{1}, 0) != 1 {
		t.Error("sturm count with a zero pivot")
	}
}

func TestExponentialCenterOracle(t *testing.T) {
	got := Exponential(2, 3)(9, false)
	o := MustOracle(t, "signal.windows.exponential", []any{9, 2.0, 3.0}, KW{"sym": false})
	compareVec(t, "window exponential center", got, o.F(), 1e-14)
}

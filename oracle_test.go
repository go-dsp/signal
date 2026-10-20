package signal

// Oracle harness. SciPy is the reference: a Go test drives one persistent
// Python process (scipy==1.18.1, numpy==2.5.4, started with `uv run`) through
// JSON lines over a pipe. Arrays travel as base64 little-endian float64 or
// complex128, so no decimal round trip loses a bit. The Python side is a
// generic dispatcher (no eval): it resolves a dotted name inside scipy and
// calls it.
//
// Every response is recorded in testdata/oracle/<Test>.json.gz keyed by the
// SHA-256 of the request line. Without -update the tests replay those
// fixtures (no Python needed, so CI on every architecture compares against the
// same SciPy output); with -update the oracle is called and the fixtures are
// rewritten:
//
//	GOWORK=off go test -update ./...

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
)

var update = flag.Bool("update", false, "call the SciPy oracle and rewrite testdata/oracle")

const oraclePy = `
import sys,json,base64,time
import numpy as np, scipy, scipy.signal, scipy.special, scipy.fft, scipy.linalg, scipy.optimize
ns={'signal':scipy.signal,'special':scipy.special,'fft':scipy.fft,'np':np,'windows':scipy.signal.windows,'linalg':scipy.linalg,'optimize':scipy.optimize}
def b64(a,dt): return base64.b64encode(np.ascontiguousarray(a,dtype=dt).tobytes()).decode()
def dec(x):
    if isinstance(x,dict):
        if 'f64' in x: return np.frombuffer(base64.b64decode(x['f64']),dtype='<f8').copy().reshape(x['shape'])
        if 'i64' in x: return np.frombuffer(base64.b64decode(x['i64']),dtype='<i8').copy().reshape(x['shape'])
        if 'c128' in x: return np.frombuffer(base64.b64decode(x['c128']),dtype='<c16').copy().reshape(x['shape'])
        if 'tuple' in x: return tuple(dec(i) for i in x['tuple'])
        return {k:dec(v) for k,v in x.items()}
    if isinstance(x,list): return [dec(i) for i in x]
    return x
def enc(x):
    if isinstance(x,(tuple,list)): return [enc(i) for i in x]
    if isinstance(x,dict): return {'dict':{k:enc(v) for k,v in x.items()}}
    if isinstance(x,(bool,np.bool_)): return bool(x)
    if isinstance(x,(int,np.integer,float,np.floating,complex,np.complexfloating,np.ndarray)):
        a=np.asarray(x)
        if a.dtype.kind=='c': return {'c128':b64(a,'<c16'),'shape':list(a.shape)}
        return {'f64':b64(a,'<f8'),'shape':list(a.shape)}
    if x is None or isinstance(x,str): return x
    return str(x)
for line in sys.stdin:
    r=json.loads(line)
    try:
        f=ns
        for p in r['fn'].split('.'): f=f[p] if isinstance(f,dict) else getattr(f,p)
        a=dec(r.get('args',[])); k=dec(r.get('kw',{}))
        if 'bench' in r:
            best=1e300
            for _ in range(r['rounds']):
                t0=time.perf_counter()
                for _ in range(r['bench']): f(*a,**k)
                best=min(best,(time.perf_counter()-t0)/r['bench'])
            out={'ok':enc(best)}
        else:
            out={'ok':enc(f(*a,**k))}
    except Exception as e:
        out={'err':type(e).__name__+': '+str(e)}
    sys.stdout.write(json.dumps(out)+'\n'); sys.stdout.flush()
`

// oracleProc is the single Python process shared by the whole test run.
type oracleProc struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

var (
	procMu   sync.Mutex
	proc     *oracleProc
	stores   = map[string]map[string]string{} // test -> "call number function" -> response line
	seqCount = map[string]int{}
	loaded   = map[string]map[string]string{}
)

func startOracle() (*oracleProc, error) {
	cmd := exec.Command("uv", "run", "--with", "scipy==1.18.1", "--with", "numpy==2.5.4", "python", "-c", oraclePy)
	cmd.Stderr = os.Stderr
	// one core for the oracle too (the performance table compares on one core)
	cmd.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1")
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &oracleProc{cmd: cmd, in: in, out: bufio.NewReaderSize(out, 1<<20)}, nil
}

func (p *oracleProc) ask(line []byte) (string, error) {
	if _, err := p.in.Write(append(line, '\n')); err != nil {
		return "", err
	}
	resp, err := p.out.ReadString('\n')
	return strings.TrimSpace(resp), err
}

func stopOracle() {
	if proc != nil {
		proc.in.Close()
		proc.cmd.Wait()
		proc = nil
	}
}

func fixturePath(test string) string {
	return filepath.Join("testdata", "oracle", test+".json.gz")
}

func loadStore(test string) map[string]string {
	if m, ok := loaded[test]; ok {
		return m
	}
	m := map[string]string{}
	loaded[test] = m
	f, err := os.Open(fixturePath(test))
	if err != nil {
		return m
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return m
	}
	json.NewDecoder(zr).Decode(&m)
	return m
}

func writeStores() error {
	if err := os.MkdirAll(filepath.Join("testdata", "oracle"), 0o755); err != nil {
		return err
	}
	for test, m := range stores {
		var buf bytes.Buffer
		zw, _ := gzip.NewWriterLevel(&buf, gzip.BestCompression)
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		ordered := make(map[string]string, len(m))
		for _, k := range keys {
			ordered[k] = m[k]
		}
		if err := json.NewEncoder(zw).Encode(ordered); err != nil {
			return err
		}
		zw.Close()
		if err := os.WriteFile(fixturePath(test), buf.Bytes(), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func TestMain(m *testing.M) {
	flag.Parse()
	code := m.Run()
	if *update {
		if err := writeStores(); err != nil {
			fmt.Fprintln(os.Stderr, "writing fixtures:", err)
			code = 1
		}
	}
	if err := writeAccuracy(); err != nil {
		fmt.Fprintln(os.Stderr, "writing accuracy table:", err)
		code = 1
	}
	stopOracle()
	os.Exit(code)
}

// ---- request encoding ------------------------------------------------------

type f64 struct {
	F64   string `json:"f64"`
	Shape []int  `json:"shape"`
}
type c128 struct {
	C128  string `json:"c128"`
	Shape []int  `json:"shape"`
}

// F encodes a float64 vector for the oracle.
func F(a []float64) any { return fShape(a, len(a)) }

// I encodes an int64 vector (Python integer array).
func I(a []int) any {
	b := make([]byte, 8*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(int64(v)))
	}
	return map[string]any{"i64": base64.StdEncoding.EncodeToString(b), "shape": []int{len(a)}}
}

// F2 encodes a rows x cols float64 matrix.
func F2(a [][]float64) any {
	var flat []float64
	for _, r := range a {
		flat = append(flat, r...)
	}
	return fShape(flat, len(a), len(a[0]))
}

func fShape(a []float64, shape ...int) any {
	b := make([]byte, 8*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(v))
	}
	return f64{base64.StdEncoding.EncodeToString(b), shape}
}

// C encodes a complex128 vector.
func C(a []complex128) any {
	b := make([]byte, 16*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint64(b[16*i:], math.Float64bits(real(v)))
		binary.LittleEndian.PutUint64(b[16*i+8:], math.Float64bits(imag(v)))
	}
	return c128{base64.StdEncoding.EncodeToString(b), []int{len(a)}}
}

// Tup encodes a Python tuple.
func Tup(items ...any) any { return map[string]any{"tuple": items} }

// KW is the keyword-argument map of an oracle call.
type KW map[string]any

// Arr is a decoded oracle array.
type Arr struct {
	Shape []int
	F     []float64
	C     []complex128
}

// Out is one decoded oracle response; its value is an Arr, a list, a string,
// a bool, nil or a JSON number.
type Out struct {
	v any
}

func decodeOut(x any) any {
	switch t := x.(type) {
	case []any:
		r := make([]any, len(t))
		for i, e := range t {
			r[i] = decodeOut(e)
		}
		return r
	case map[string]any:
		if d, ok := t["dict"]; ok {
			m := map[string]any{}
			for k, v := range d.(map[string]any) {
				m[k] = decodeOut(v)
			}
			return m
		}
		var shape []int
		for _, s := range t["shape"].([]any) {
			shape = append(shape, int(s.(float64)))
		}
		if s, ok := t["f64"]; ok {
			b, _ := base64.StdEncoding.DecodeString(s.(string))
			a := Arr{Shape: shape, F: make([]float64, len(b)/8)}
			for i := range a.F {
				a.F[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
			}
			return a
		}
		b, _ := base64.StdEncoding.DecodeString(t["c128"].(string))
		a := Arr{Shape: shape, C: make([]complex128, len(b)/16)}
		for i := range a.C {
			a.C[i] = complex(math.Float64frombits(binary.LittleEndian.Uint64(b[16*i:])),
				math.Float64frombits(binary.LittleEndian.Uint64(b[16*i+8:])))
		}
		return a
	}
	return x
}

// Idx returns the i-th element of a list response.
func (o Out) Idx(i int) Out { return Out{o.v.([]any)[i]} }

// Get returns a member of a dict response.
func (o Out) Get(key string) Out { return Out{o.v.(map[string]any)[key]} }

// Has reports whether a dict response has the member.
func (o Out) Has(key string) bool { _, ok := o.v.(map[string]any)[key]; return ok }

// Arr returns the response as an array.
func (o Out) Arr() Arr { return o.v.(Arr) }

// F returns the response as a float64 vector (flattened).
func (o Out) F() []float64 { return o.v.(Arr).F }

// C returns the response as a complex vector (a real array is promoted).
func (o Out) C() []complex128 {
	a := o.v.(Arr)
	if a.C != nil || len(a.F) == 0 {
		return a.C
	}
	c := make([]complex128, len(a.F))
	for i, v := range a.F {
		c[i] = complex(v, 0)
	}
	return c
}

// X returns the first element (scalar response).
func (o Out) X() float64 { return o.v.(Arr).F[0] }

// Len returns the length of a list response.
func (o Out) Len() int { return len(o.v.([]any)) }

// Bool returns a boolean response.
func (o Out) Bool() bool { return o.v.(bool) }

// Rows returns a 2-D float response as rows.
func (o Out) Rows() [][]float64 {
	a := o.v.(Arr)
	if len(a.Shape) != 2 {
		return nil
	}
	r := make([][]float64, a.Shape[0])
	for i := range r {
		r[i] = a.F[i*a.Shape[1] : (i+1)*a.Shape[1]]
	}
	return r
}

// Oracle calls (or replays) fn with args and keyword arguments.
// A Python exception is returned as an error.
func Oracle(t testing.TB, fn string, args []any, kw KW) (Out, error) {
	return oracleCall(t, fn, args, kw, 0, 0)
}

// MustOracle is Oracle that fails the test on error.
func MustOracle(t testing.TB, fn string, args []any, kw KW) Out {
	t.Helper()
	o, err := Oracle(t, fn, args, kw)
	if err != nil {
		t.Fatalf("oracle %s: %v", fn, err)
	}
	return o
}

func oracleCall(t testing.TB, fn string, args []any, kw KW, bench, rounds int) (Out, error) {
	t.Helper()
	req := map[string]any{"fn": fn, "args": args}
	if kw != nil {
		req["kw"] = kw
	}
	if bench > 0 {
		req["bench"], req["rounds"] = bench, rounds
	}
	line, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	test := strings.SplitN(t.Name(), "/", 2)[0]
	var resp string
	procMu.Lock()
	defer procMu.Unlock()
	// Fixtures are keyed by the position of the call within its test, not by
	// the bits of its arguments: test inputs are computed with math functions
	// and fused multiply-adds that round differently on different
	// architectures, but the sequence of oracle calls does not.
	if _, ok := seqCount[test]; !ok {
		t.Cleanup(func() { procMu.Lock(); delete(seqCount, test); procMu.Unlock() })
	}
	key := fmt.Sprintf("%05d %s", seqCount[test], fn)
	if bench == 0 {
		seqCount[test]++
	}
	if *update || bench > 0 {
		if proc == nil {
			if proc, err = startOracle(); err != nil {
				t.Fatal(err)
			}
		}
		if resp, err = proc.ask(line); err != nil {
			t.Fatalf("oracle %s: %v", fn, err)
		}
		if bench == 0 {
			if stores[test] == nil {
				stores[test] = map[string]string{}
			}
			stores[test][key] = resp
		}
	} else {
		var ok bool
		if resp, ok = loadStore(test)[key]; !ok {
			t.Fatalf("no fixture %q; run: GOWORK=off go test -update -run %s", key, test)
		}
	}
	var r map[string]any
	if err := json.Unmarshal([]byte(resp), &r); err != nil {
		t.Fatalf("oracle response: %v", err)
	}
	if e, ok := r["err"]; ok {
		return Out{}, fmt.Errorf("%v", e)
	}
	return Out{decodeOut(r["ok"])}, nil
}

// ---- accuracy bookkeeping ---------------------------------------------------

// ulpDist is the distance in units in the last place between a and b.
func ulpDist(a, b float64) float64 {
	if a == b {
		return 0
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.Inf(1)
	}
	ord := func(x float64) int64 {
		u := int64(math.Float64bits(x))
		if u < 0 {
			return math.MinInt64 - u
		}
		return u
	}
	d := ord(a) - ord(b)
	if d < 0 {
		d = -d
	}
	return float64(d)
}

type accStat struct {
	n, exact int
	maxULP   float64
	maxNorm  float64 // max over cases of max|g-w| / max|w|
}

var (
	accMu sync.Mutex
	acc   = map[string]*accStat{}
)

// compareVec compares got against want, records the statistics under group
// and fails if the normwise relative error exceeds tol. It returns the
// normwise error.
func compareVec(t testing.TB, group string, got, want []float64, tol float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", group, len(got), len(want))
	}
	var wmax, dmax, ulp float64
	exact := 0
	for i := range got {
		if a := math.Abs(want[i]); a > wmax {
			wmax = a
		}
		d := math.Abs(got[i] - want[i])
		if math.IsNaN(got[i]) != math.IsNaN(want[i]) {
			d = math.Inf(1)
		}
		if d > dmax {
			dmax = d
		}
		u := ulpDist(got[i], want[i])
		if u == 0 {
			exact++
		}
		if u > ulp && !math.IsInf(u, 1) {
			ulp = u
		}
	}
	norm := 0.0
	if wmax > 0 {
		norm = dmax / wmax
	} else {
		norm = dmax
	}
	accMu.Lock()
	s := acc[group]
	if s == nil {
		s = &accStat{}
		acc[group] = s
	}
	s.n += len(got)
	s.exact += exact
	s.maxULP = math.Max(s.maxULP, ulp)
	s.maxNorm = math.Max(s.maxNorm, norm)
	accMu.Unlock()
	if !(norm <= tol) {
		t.Errorf("%s: normwise error %.3g > %.3g", group, norm, tol)
	}
	return norm
}

func compareC(t testing.TB, group string, got, want []complex128, tol float64) float64 {
	t.Helper()
	g := make([]float64, 0, 2*len(got))
	w := make([]float64, 0, 2*len(want))
	for _, v := range got {
		g = append(g, real(v), imag(v))
	}
	for _, v := range want {
		w = append(w, real(v), imag(v))
	}
	return compareVec(t, group, g, w, tol)
}

func writeAccuracy() error {
	if !*update || len(acc) == 0 {
		return nil
	}
	// A filtered run (-run) sees only some groups: writing would replace
	// the full table with that subset.
	if f := flag.Lookup("test.run"); f != nil && f.Value.String() != "" {
		return nil
	}
	names := make([]string, 0, len(acc))
	for k := range acc {
		names = append(names, k)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("# Accuracy against SciPy 1.18.1 / NumPy 2.5.4\n\n")
	b.WriteString("Generated by `GOWORK=off go test -update ./...`; the same numbers are re-checked from the\n")
	b.WriteString("recorded fixtures (testdata/oracle) on every run. `exact` counts values bit-identical to the\n")
	b.WriteString("oracle; `max ULP` is the elementwise maximum over the non-identical values; `norm err` is\n")
	b.WriteString("max|got-want| / max|want| per case, maximised over cases (the right measure for polynomial\n")
	b.WriteString("coefficients and spectra, where elementwise ULPs near zero are noise).\n\n")
	b.WriteString("| group | values | exact | max ULP | norm err |\n|---|---:|---:|---:|---:|\n")
	for _, k := range names {
		s := acc[k]
		fmt.Fprintf(&b, "| %s | %d | %d (%.1f%%) | %.0f | %.2e |\n", k, s.n, s.exact, 100*float64(s.exact)/float64(s.n), s.maxULP, s.maxNorm)
	}
	if err := os.MkdirAll("docs", 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join("docs", "accuracy.md"), []byte(b.String()), 0o644)
}

// ---- self test of the harness ------------------------------------------------

func TestOracleHarness(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	o := MustOracle(t, "np.cumsum", []any{F(x)}, nil)
	compareVec(t, "harness/cumsum", cumsum(x), o.F(), 0)
	o = MustOracle(t, "signal.windows.hann", []any{5}, KW{"sym": false})
	if len(o.F()) != 5 {
		t.Fatal("hann length")
	}
	// tuple, complex, 2-D, list, bool, error, scalar paths
	o = MustOracle(t, "signal.get_window", []any{Tup("kaiser", 8.0), 7}, nil)
	if len(o.F()) != 7 {
		t.Fatal("kaiser length")
	}
	o = MustOracle(t, "np.conj", []any{C([]complex128{1 + 2i})}, nil)
	if o.C()[0] != 1-2i {
		t.Fatal("conj")
	}
	o = MustOracle(t, "signal.butter", []any{2, 0.3}, KW{"output": "sos"})
	if r := o.Rows(); len(r) != 1 || len(r[0]) != 6 {
		t.Fatal("sos rows")
	}
	o = MustOracle(t, "signal.butter", []any{2, 0.3}, nil)
	if o.Len() != 2 || o.Idx(0).Arr().Shape[0] != 3 {
		t.Fatal("ba list")
	}
	o = MustOracle(t, "signal.check_COLA", []any{"hann", 8, 4}, nil)
	if !o.Bool() {
		t.Fatal("cola")
	}
	o = MustOracle(t, "np.float64", []any{2.5}, nil)
	if o.X() != 2.5 {
		t.Fatal("scalar")
	}
	if _, err := Oracle(t, "signal.butter", []any{-1, 0.3}, nil); err == nil {
		t.Fatal("expected an oracle error")
	}
	o = MustOracle(t, "np.reshape", []any{F2([][]float64{{1, 2}, {3, 4}}), 4}, nil)
	if o.Rows() != nil || len(o.F()) != 4 {
		t.Fatal("reshape")
	}
	if ulpDist(1, math.Nextafter(1, 2)) != 1 || ulpDist(1, math.NaN()) != math.Inf(1) || ulpDist(-1, 1) < 1e18 {
		t.Fatal("ulpDist")
	}
}

func cumsum(x []float64) []float64 {
	r := make([]float64, len(x))
	s := 0.0
	for i, v := range x {
		s += v
		r[i] = s
	}
	return r
}

// C2 encodes a rows x cols complex128 matrix.
func C2(a [][]complex128) any {
	var flat []complex128
	for _, r := range a {
		flat = append(flat, r...)
	}
	c := C(flat).(c128)
	c.Shape = []int{len(a), len(a[0])}
	return c
}

package signal

// Performance against SciPy on one core. Run with
//
//	GOWORK=off go test -run TestPerf -perf -count=1 .
//
// It writes docs/perf.md. Go and SciPy are timed in interleaved rounds (a Go
// batch, then a SciPy batch of the same function, repeated), the best round
// of each is kept, and nothing is measured while the load average is above
// perfMaxLoad: the table records the load before and after. Only one
// measurement runs at a time.

import (
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

var perfFlag = flag.Bool("perf", false, "measure performance against SciPy and write docs/perf.md")

const perfMaxLoad = 8.0

func loadAvg() float64 {
	out, err := exec.Command("uptime").Output()
	if err != nil {
		return math.NaN()
	}
	m := regexp.MustCompile(`load averages?: ([0-9.,]+)`).FindStringSubmatch(string(out))
	if m == nil {
		return math.NaN()
	}
	v, _ := strconv.ParseFloat(strings.ReplaceAll(strings.Fields(m[1])[0], ",", "."), 64)
	return v
}

// waitForLoad blocks until the load average is below perfMaxLoad, or gives up.
func waitForLoad(max time.Duration) (float64, bool) {
	deadline := time.Now().Add(max)
	for {
		l := loadAvg()
		if l < perfMaxLoad {
			return l, true
		}
		if time.Now().After(deadline) {
			return l, false
		}
		time.Sleep(20 * time.Second)
	}
}

type perfCase struct {
	name string
	fn   string // SciPy function
	args []any
	kw   KW
	run  func() // the Go equivalent
}

// timeGo returns the best per-call time over a batch of calls totalling about d.
func timeGo(run func(), d time.Duration) float64 {
	run() // warm up
	start := time.Now()
	run()
	one := time.Since(start)
	n := int(d / (one + 1))
	n = max(n, 1)
	start = time.Now()
	for i := 0; i < n; i++ {
		run()
	}
	return time.Since(start).Seconds() / float64(n)
}

func TestPerf(t *testing.T) {
	if !*perfFlag {
		t.Skip("run with -perf to measure")
	}
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	rng := newRNG(100)
	x1m := testSignal(rng, 1_000_000)
	x100k := x1m[:100_000]
	x262k := x1m[:262_144]

	bq := butterSOSPerf(t, 8, 0.2)
	bb, ba := butterBAPerf(t, 4, 0.2)
	sosRowsArg := F2(sosRows(bq))
	h101 := randVec(rng, 101)
	h1k := randVec(rng, 1001)
	h10k := randVec(rng, 10001)
	one := func(v float64) []float64 { return []float64{v} }

	cases := []perfCase{
		{"sosfilt, 10^6 samples, order 8", "signal.sosfilt", []any{sosRowsArg, F(x1m)}, nil, func() { SOSFilt(bq, x1m, nil) }},
		{"lfilter, 10^6 samples, order 4", "signal.lfilter", []any{F(bb), F(ba), F(x1m)}, nil, func() { LFilter(bb, ba, x1m, nil) }},
		{"lfilter FIR 101 taps, 10^6 samples", "signal.lfilter", []any{F(h101), F([]float64{1}), F(x1m)}, nil, func() { LFilter(h101, []float64{1}, x1m, nil) }},
		{"filtfilt, 10^5 samples, order 4", "signal.filtfilt", []any{F(bb), F(ba), F(x100k)}, nil, func() { FiltFilt(bb, ba, x100k, nil) }},
		{"sosfiltfilt, 10^6 samples, order 8", "signal.sosfiltfilt", []any{sosRowsArg, F(x1m)}, nil, func() { SOSFiltFilt(bq, x1m, nil) }},
		{"welch, 10^6 samples, nperseg 1024", "signal.welch", []any{F(x1m)}, KW{"nperseg": 1024}, func() { Welch(x1m, &SpectralOptions{Nperseg: 1024}) }},
		{"welch, 10^6 samples, nperseg 256 (default)", "signal.welch", []any{F(x1m)}, nil, func() { Welch(x1m, nil) }},
		{"periodogram, 2^18 samples", "signal.periodogram", []any{F(x262k)}, nil, func() { Periodogram(x262k, nil) }},
		{"spectrogram, 10^5 samples", "signal.spectrogram", []any{F(x100k)}, nil, func() { Spectrogram(x100k, nil, SpecPSD) }},
		{"resample_poly 3/2, 2^18 samples", "signal.resample_poly", []any{F(x262k), 3, 2}, nil, func() { ResamplePoly(x262k, 3, 2, nil) }},
		{"resample_poly 1/4, 10^6 samples", "signal.resample_poly", []any{F(x1m), 1, 4}, nil, func() { ResamplePoly(x1m, 1, 4, nil) }},
		{"resample (FFT), 2^18 -> 3*2^16", "signal.resample", []any{F(x262k), 196608}, nil, func() { Resample(x262k, 196608, nil) }},
		{"decimate by 8, 10^6 samples", "signal.decimate", []any{F(x1m), 8}, nil, func() { Decimate(x1m, 8, nil) }},
		{"upfirdn 3/2, 101 taps, 10^5 samples", "signal.upfirdn", []any{F(h101), F(x100k), 3, 2}, nil, func() { Upfirdn(h101, x100k, 3, 2, "", 0) }},
		{"convolve direct, 10^5 x 101", "signal.convolve", []any{F(x100k), F(h101)}, KW{"method": "direct"}, func() { Convolve(x100k, h101, ConvFull, ConvDirect) }},
		{"convolve direct, 10^5 x 1001", "signal.convolve", []any{F(x100k), F(h1k)}, KW{"method": "direct"}, func() { Convolve(x100k, h1k, ConvFull, ConvDirect) }},
		{"fftconvolve, 10^6 x 10^4", "signal.fftconvolve", []any{F(x1m), F(h10k)}, nil, func() { FFTConvolve(x1m, h10k, ConvFull) }},
		{"oaconvolve, 10^6 x 10^3", "signal.oaconvolve", []any{F(x1m), F(h1k)}, nil, func() { OAConvolve(x1m, h1k, ConvFull) }},
		{"hilbert, 2^18 samples", "signal.hilbert", []any{F(x262k)}, nil, func() { Hilbert(x262k, 0) }},
		{"find_peaks with prominence, 10^6 samples", "signal.find_peaks", []any{F(x1m)}, KW{"prominence": 0.5}, func() { FindPeaks(x1m, &FindPeaksOptions{Prominence: AtLeast(0.5)}) }},
		{"firwin, 4001 taps", "signal.firwin", []any{4001, 0.3}, nil, func() { Firwin(4001, one(0.3), nil) }},
		{"firwin, 100001 taps (Kaiser)", "signal.firwin", []any{100001, 0.3}, KW{"window": Tup("kaiser", 8.0)}, func() { Firwin(100001, one(0.3), &FirwinOptions{Window: Kaiser(8)}) }},
		{"remez, 101 taps", "signal.remez", []any{101, F([]float64{0, 0.1, 0.15, 0.5}), F([]float64{1, 0})}, nil, func() { Remez(101, []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}, nil) }},
		{"remez, 501 taps", "signal.remez", []any{501, F([]float64{0, 0.1, 0.12, 0.5}), F([]float64{1, 0})}, nil, func() { Remez(501, []float64{0, 0.1, 0.12, 0.5}, []float64{1, 0}, nil) }},
		{"firls, 301 taps", "signal.firls", []any{301, F([]float64{0, 0.3, 0.4, 1}), F([]float64{1, 1, 0, 0})}, nil, func() { Firls(301, []float64{0, 0.3, 0.4, 1}, []float64{1, 1, 0, 0}, nil, 0) }},
		{"butter order 8 (zpk to sos)", "signal.butter", []any{8, 0.2}, KW{"output": "sos"}, func() { z, _ := Butter(8, one(0.2), Lowpass, nil); z.SOS(false) }},
		{"ellip order 10 (zpk to sos)", "signal.ellip", []any{10, 1.0, 60.0, 0.2}, KW{"output": "sos"}, func() { z, _ := Ellip(10, 1, 60, one(0.2), Lowpass, nil); z.SOS(false) }},
		{"dpss 1024, NW 4, K 7", "signal.windows.dpss", []any{1024, 4.0}, KW{"Kmax": 7}, func() { DPSS(1024, 4, 7, true, DPSSNorm2) }},
	}

	loadStart, ok := waitForLoad(2 * time.Minute)
	if !ok {
		t.Skipf("load average %.1f stays above %.0f; not measuring", loadStart, perfMaxLoad)
	}
	var rows []string
	const rounds = 5
	for _, c := range cases {
		if l, ok := waitForLoad(5 * time.Minute); !ok {
			rows = append(rows, fmt.Sprintf("| %s | not measured (load %.1f) | | |", c.name, l))
			continue
		}
		bestGo, bestPy := math.Inf(1), math.Inf(1)
		var py float64
		for r := 0; r < rounds; r++ {
			g := timeGo(c.run, 150*time.Millisecond)
			bestGo = math.Min(bestGo, g)
			number := max(1, int(0.15/g))
			number = min(number, 50)
			o, err := oracleCall(t, c.fn, c.args, c.kw, number, 1)
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			py = o.X()
			bestPy = math.Min(bestPy, py)
		}
		rows = append(rows, fmt.Sprintf("| %s | %s | %s | %.2f |", c.name, fmtDur(bestGo), fmtDur(bestPy), bestGo/bestPy))
		t.Logf("%-45s go %s  scipy %s  ratio %.2f", c.name, fmtDur(bestGo), fmtDur(bestPy), bestGo/bestPy)
	}
	loadEnd := loadAvg()
	var b strings.Builder
	b.WriteString("# Performance against SciPy\n\n")
	b.WriteString("One core (GOMAXPROCS=1; the SciPy process runs with OMP/OpenBLAS threads set to 1), best of 5 interleaved rounds\n")
	b.WriteString("(a Go batch, then a SciPy batch of the same call, repeated), `uptime` checked before each case (load below 8).\n")
	b.WriteString("A ratio below 1 means this package is faster. Regenerate with\n`GOWORK=off go test -run TestPerf -perf -count=1 .`\n\n")
	fmt.Fprintf(&b, "Host: %s/%s, %d logical CPUs, %s; SciPy 1.18.1, NumPy 2.5.4. Load average at start %.1f, at end %.1f. Measured %s.\n\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version(), loadStart, loadEnd, time.Now().Format("2006-01-02"))
	b.WriteString("| case | Go | SciPy | Go / SciPy |\n|---|---:|---:|---:|\n")
	for _, r := range rows {
		b.WriteString(r + "\n")
	}
	if err := os.MkdirAll("docs", 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join("docs", "perf.md"), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fmtDur(s float64) string {
	switch {
	case s < 1e-6:
		return fmt.Sprintf("%.0f ns", s*1e9)
	case s < 1e-3:
		return fmt.Sprintf("%.1f µs", s*1e6)
	case s < 1:
		return fmt.Sprintf("%.2f ms", s*1e3)
	}
	return fmt.Sprintf("%.2f s", s)
}

func butterSOSPerf(t *testing.T, n int, wn float64) SOS {
	z, err := Butter(n, []float64{wn}, Lowpass, nil)
	mustOK(t, err)
	s, err := z.SOS(false)
	mustOK(t, err)
	return s
}

func butterBAPerf(t *testing.T, n int, wn float64) (b, a []float64) {
	z, err := Butter(n, []float64{wn}, Lowpass, nil)
	mustOK(t, err)
	b, a, err = z.TF()
	mustOK(t, err)
	return b, a
}

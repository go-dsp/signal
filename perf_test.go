package signal

// Performance against SciPy on one core, measured by refbench
// (github.com/go-scikit/refbench, performance campaign wave W1).
//
//	go test -run TestPerf .                          CI mode: allocations, table, denominator (no Python, no timing)
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//
// Go runs under GOMAXPROCS=1; SciPy runs in the oracle process (oracle_test.go:
// OMP/OpenBLAS/MKL/VecLib threads at 1) and is timed inside that process, so
// the pipe is not in the interval. Rounds alternate Go and SciPy, and the
// verdict is on the median. docs/perf.json is the record, docs/perf-table.md is
// generated from it, docs/perf-exempt.md names the public functions with no case.

import (
	"runtime"
	"testing"
	"time"

	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

type perfCase struct {
	name string
	fn   string // SciPy function
	args []any
	kw   KW
	goFn string // the public function measured
	run  func() // the Go equivalent
}

// pyTimer times iters SciPy calls inside the oracle process: the oracle returns
// the per-call time of a loop of `iters` calls.
func pyTimer(t *testing.T, c perfCase) refbench.Timer {
	return func(iters int) (time.Duration, error) {
		o, err := oracleCall(t, c.fn, c.args, c.kw, iters, 1)
		if err != nil {
			return 0, err
		}
		return time.Duration(o.X() * float64(iters) * float64(time.Second)), nil
	}
}

func perfCases(t *testing.T) []refbench.Case {
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

	cs := []perfCase{
		{"sosfilt, 10^6 samples, order 8", "signal.sosfilt", []any{sosRowsArg, F(x1m)}, nil, "SOSFilt", func() { SOSFilt(bq, x1m, nil) }},
		{"lfilter, 10^6 samples, order 4", "signal.lfilter", []any{F(bb), F(ba), F(x1m)}, nil, "LFilter", func() { LFilter(bb, ba, x1m, nil) }},
		{"lfilter FIR 101 taps, 10^6 samples", "signal.lfilter", []any{F(h101), F([]float64{1}), F(x1m)}, nil, "LFilter", func() { LFilter(h101, []float64{1}, x1m, nil) }},
		{"filtfilt, 10^5 samples, order 4", "signal.filtfilt", []any{F(bb), F(ba), F(x100k)}, nil, "FiltFilt", func() { FiltFilt(bb, ba, x100k, nil) }},
		{"sosfiltfilt, 10^6 samples, order 8", "signal.sosfiltfilt", []any{sosRowsArg, F(x1m)}, nil, "SOSFiltFilt", func() { SOSFiltFilt(bq, x1m, nil) }},
		{"welch, 10^6 samples, nperseg 1024", "signal.welch", []any{F(x1m)}, KW{"nperseg": 1024}, "Welch", func() { Welch(x1m, &SpectralOptions{Nperseg: 1024}) }},
		{"welch, 10^6 samples, nperseg 256 (default)", "signal.welch", []any{F(x1m)}, nil, "Welch", func() { Welch(x1m, nil) }},
		{"periodogram, 2^18 samples", "signal.periodogram", []any{F(x262k)}, nil, "Periodogram", func() { Periodogram(x262k, nil) }},
		{"spectrogram, 10^5 samples", "signal.spectrogram", []any{F(x100k)}, nil, "Spectrogram", func() { Spectrogram(x100k, nil, SpecPSD) }},
		{"resample_poly 3/2, 2^18 samples", "signal.resample_poly", []any{F(x262k), 3, 2}, nil, "ResamplePoly", func() { ResamplePoly(x262k, 3, 2, nil) }},
		{"resample_poly 1/4, 10^6 samples", "signal.resample_poly", []any{F(x1m), 1, 4}, nil, "ResamplePoly", func() { ResamplePoly(x1m, 1, 4, nil) }},
		{"resample (FFT), 2^18 -> 3*2^16", "signal.resample", []any{F(x262k), 196608}, nil, "Resample", func() { Resample(x262k, 196608, nil) }},
		{"decimate by 8, 10^6 samples", "signal.decimate", []any{F(x1m), 8}, nil, "Decimate", func() { Decimate(x1m, 8, nil) }},
		{"upfirdn 3/2, 101 taps, 10^5 samples", "signal.upfirdn", []any{F(h101), F(x100k), 3, 2}, nil, "Upfirdn", func() { Upfirdn(h101, x100k, 3, 2, "", 0) }},
		{"convolve direct, 10^5 x 101", "signal.convolve", []any{F(x100k), F(h101)}, KW{"method": "direct"}, "Convolve", func() { Convolve(x100k, h101, ConvFull, ConvDirect) }},
		{"convolve direct, 10^5 x 1001", "signal.convolve", []any{F(x100k), F(h1k)}, KW{"method": "direct"}, "Convolve", func() { Convolve(x100k, h1k, ConvFull, ConvDirect) }},
		{"fftconvolve, 10^6 x 10^4", "signal.fftconvolve", []any{F(x1m), F(h10k)}, nil, "FFTConvolve", func() { FFTConvolve(x1m, h10k, ConvFull) }},
		{"oaconvolve, 10^6 x 10^3", "signal.oaconvolve", []any{F(x1m), F(h1k)}, nil, "OAConvolve", func() { OAConvolve(x1m, h1k, ConvFull) }},
		{"hilbert, 2^18 samples", "signal.hilbert", []any{F(x262k)}, nil, "Hilbert", func() { Hilbert(x262k, 0) }},
		{"find_peaks with prominence, 10^6 samples", "signal.find_peaks", []any{F(x1m)}, KW{"prominence": 0.5}, "FindPeaks", func() { FindPeaks(x1m, &FindPeaksOptions{Prominence: AtLeast(0.5)}) }},
		{"firwin, 4001 taps", "signal.firwin", []any{4001, 0.3}, nil, "Firwin", func() { Firwin(4001, one(0.3), nil) }},
		{"firwin, 100001 taps (Kaiser)", "signal.firwin", []any{100001, 0.3}, KW{"window": Tup("kaiser", 8.0)}, "Firwin", func() { Firwin(100001, one(0.3), &FirwinOptions{Window: Kaiser(8)}) }},
		{"remez, 101 taps", "signal.remez", []any{101, F([]float64{0, 0.1, 0.15, 0.5}), F([]float64{1, 0})}, nil, "Remez", func() { Remez(101, []float64{0, 0.1, 0.15, 0.5}, []float64{1, 0}, nil) }},
		{"remez, 501 taps", "signal.remez", []any{501, F([]float64{0, 0.1, 0.12, 0.5}), F([]float64{1, 0})}, nil, "Remez", func() { Remez(501, []float64{0, 0.1, 0.12, 0.5}, []float64{1, 0}, nil) }},
		{"firls, 301 taps", "signal.firls", []any{301, F([]float64{0, 0.3, 0.4, 1}), F([]float64{1, 1, 0, 0})}, nil, "Firls", func() { Firls(301, []float64{0, 0.3, 0.4, 1}, []float64{1, 1, 0, 0}, nil, 0) }},
		{"butter order 8 (zpk to sos)", "signal.butter", []any{8, 0.2}, KW{"output": "sos"}, "Butter", func() { z, _ := Butter(8, one(0.2), Lowpass, nil); z.SOS(false) }},
		{"ellip order 10 (zpk to sos)", "signal.ellip", []any{10, 1.0, 60.0, 0.2}, KW{"output": "sos"}, "Ellip", func() { z, _ := Ellip(10, 1, 60, one(0.2), Lowpass, nil); z.SOS(false) }},
		{"dpss 1024, NW 4, K 7", "signal.windows.dpss", []any{1024, 4.0}, KW{"Kmax": 7}, "DPSS", func() { DPSS(1024, 4, 7, true, DPSSNorm2) }},
	}
	out := make([]refbench.Case, len(cs))
	for i, c := range cs {
		out[i] = refbench.Case{
			Name:      c.name,
			Func:      c.goFn,
			Reference: "scipy.signal (" + c.fn + ")",
			Parity:    refbench.ParityNone, // the equality tests against SciPy live in oracle_test.go and the per-function tests
			Go: func(iters int) {
				for range iters {
					c.run()
				}
			},
			Ref: pyTimer(t, c),
		}
	}
	return out
}

func TestPerf(t *testing.T) {
	// Allocation counts are those of a plain amd64/arm64 build: the race
	// detector, 32-bit and emulated architectures change them.
	if perfRace || (runtime.GOARCH != "amd64" && runtime.GOARCH != "arm64") {
		t.Skip("refbench CI mode runs on plain amd64/arm64 builds (the `perf` job)")
	}
	rbtest.Perf(t, refbench.CIConfig{Root: ".", Cases: perfCases(t)}, refbench.Options{
		Repo:        "go-dsp/signal",
		RefVersions: map[string]string{"scipy": "1.18.1", "numpy": "2.5.4"},
	})
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

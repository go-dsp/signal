# Performance against SciPy

## Method (refbench)

Measured by [refbench](https://github.com/go-scikit/refbench) (performance
campaign wave W1, [go-scikit/roadmap#27](https://github.com/go-scikit/roadmap/issues/27)):
one core on both sides (`GOMAXPROCS=1`; the SciPy process runs with OMP/OpenBLAS/MKL/VecLib threads at 1), each
side a calibrated loop timed inside its own process, rounds alternating Go-then-SciPy and SciPy-then-Go, the
1-minute load read before and after every round (gate: below 8), the verdict on the median with the
quartiles. `ratio = Go / SciPy`; below 1 means this package is faster.

```sh
go test -run TestPerf .                        # CI mode: allocations, table, every public function has a case or an exemption
go test -run TestPerf -cpu 1 -perf.run .       # measure; refuses at a load of 8 or more
go test -run TestPerf -perf.record .           # allocation counts only (a deferred report)
```

[perf.json](perf.json) is the record, [perf-table.md](perf-table.md) is generated from it, and
[perf-exempt.md](perf-exempt.md) names the public functions that have no case yet, with the reason.

**Status 2026-11-07: timing deferred.** The host's 1-minute load was about 33 when this migration was made,
so [perf-table.md](perf-table.md) holds allocation counts only (verdict "deferred"). The table below is the
previous harness's, measured 2026-10-19 with a different statistic (best of 5 rather than the median) and
kept until the first refbench timing run replaces it.

## Previous measurement (best of 5, 2026-10-19)

Host: darwin/arm64, 16 logical CPUs, go1.27.2; SciPy 1.18.1, NumPy 2.5.4. Load average at start 4.7, at end 4.1. Measured 2026-10-19.

| case | Go | SciPy | Go / SciPy |
|---|---:|---:|---:|
| sosfilt, 10^6 samples, order 8 | 4.63 ms | 7.20 ms | 0.64 |
| lfilter, 10^6 samples, order 4 | 7.27 ms | 8.87 ms | 0.82 |
| lfilter FIR 101 taps, 10^6 samples | 14.53 ms | 14.78 ms | 0.98 |
| filtfilt, 10^5 samples, order 4 | 1.56 ms | 1.85 ms | 0.85 |
| sosfiltfilt, 10^6 samples, order 8 | 10.40 ms | 14.94 ms | 0.70 |
| welch, 10^6 samples, nperseg 1024 | 7.83 ms | 30.74 ms | 0.25 |
| welch, 10^6 samples, nperseg 256 (default) | 8.42 ms | 86.92 ms | 0.10 |
| periodogram, 2^18 samples | 3.26 ms | 15.32 ms | 0.21 |
| spectrogram, 10^5 samples | 374.9 µs | 497.1 µs | 0.75 |
| resample_poly 3/2, 2^18 samples | 4.05 ms | 4.19 ms | 0.97 |
| resample_poly 1/4, 10^6 samples | 6.59 ms | 15.35 ms | 0.43 |
| resample (FFT), 2^18 -> 3*2^16 | 1.80 ms | 3.08 ms | 0.58 |
| decimate by 8, 10^6 samples | 10.52 ms | 15.36 ms | 0.68 |
| upfirdn 3/2, 101 taps, 10^5 samples | 2.71 ms | 2.73 ms | 0.99 |
| convolve direct, 10^5 x 101 | 1.01 ms | 1.48 ms | 0.68 |
| convolve direct, 10^5 x 1001 | 9.69 ms | 12.80 ms | 0.76 |
| fftconvolve, 10^6 x 10^4 | 10.47 ms | 19.78 ms | 0.53 |
| oaconvolve, 10^6 x 10^3 | 7.03 ms | 8.63 ms | 0.81 |
| hilbert, 2^18 samples | 3.79 ms | 4.07 ms | 0.93 |
| find_peaks with prominence, 10^6 samples | 17.12 ms | 18.96 s | 0.00 |
| firwin, 4001 taps | 84.0 µs | 130.5 µs | 0.64 |
| firwin, 100001 taps (Kaiser) | 3.15 ms | 3.44 ms | 0.91 |
| remez, 101 taps | 146.9 µs | 168.9 µs | 0.87 |
| remez, 501 taps | 1.17 ms | 6.54 ms | 0.18 |
| firls, 301 taps | 393.1 µs | 774.5 µs | 0.51 |
| butter order 8 (zpk to sos) | 2.3 µs | 282.7 µs | 0.01 |
| ellip order 10 (zpk to sos) | 5.6 µs | 397.7 µs | 0.01 |
| dpss 1024, NW 4, K 7 | 2.99 ms | 3.46 ms | 0.86 |

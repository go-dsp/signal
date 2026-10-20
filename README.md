# go-dsp/signal

Signal processing in pure Go (`CGO_ENABLED=0`): filter design and filtering,
spectral estimation, resampling and peak finding, with the semantics of SciPy's
`scipy.signal` and, where it differs, a note on MATLAB's.

```go
import "github.com/go-dsp/signal"

z, _ := signal.Butter(8, []float64{0.2}, signal.Lowpass, nil) // zeros, poles, gain
sos, _ := z.SOS(false)                                        // second-order sections
y, _ := signal.SOSFiltFilt(sos, x, nil)                       // zero-phase filtering
f, p, _ := signal.Welch(x, &signal.SpectralOptions{Fs: 1000, Nperseg: 1024})
```

The module is `github.com/go-dsp/signal`, BSD-3-Clause. It depends on
[go-fft/fft](https://github.com/go-fft/fft) (transforms and the classic
windows), [go-numerics/special](https://github.com/go-numerics/special)
(complete elliptic integrals, `I0`) and
[go-simd/floats](https://github.com/go-simd/floats) (the assembly dot
product used by the FIR kernels).

`remez.go`'s exchange follows the Parks–McClellan program of J. H. McClellan,
T. W. Parks and L. R. Rabiner (IEEE Trans. Audio Electroacoust., 1973) as
SciPy's `_sigtoolsmodule.cc` carries it in E. Kvaleberg's C translation
(SciPy License, BSD-3-Clause; Copyright 2005 Travis Oliphant and the SciPy
developers); its control flow and label numbers are kept so the two read side
by side.

## Conventions

* Signals and coefficients are `[]float64`; filters have real coefficients and
  signals are real and one-dimensional (no `axis` argument: loop over the
  channels of a matrix).
* A filter is a `ZPK` (zeros, poles, gain), a transfer function `(b, a)` or an
  `SOS` (second-order sections). **Use SOS above order 8**: the transfer
  function form loses accuracy fast (see [docs/verification.md](docs/verification.md)).
* Functions that can fail return an `error`; nothing panics on bad arguments
  (the fuzz targets enforce it), and sizes are bounded (`2^28` elements) so
  absurd arguments fail instead of exhausting memory.
* Option structs have useful zero values: `nil` means SciPy's defaults.
  Where SciPy's default differs between functions (detrend, scaling) the zero
  value is "the default of this function".

## What is provided

| area | functions | SciPy | MATLAB |
|---|---|---|---|
| representations | `ZPK`, `SOS`, `Normalize`, `TF2ZPK`, `ZPK2TF`, `ZPK2SOS`, `TF2SOS`, `SOS2TF`, `SOS2ZPK`, `Roots` | `normalize`, `tf2zpk`, `zpk2tf`, `zpk2sos`, `tf2sos`, `sos2tf`, `sos2zpk`, `numpy.roots` | `tf2zp`, `zp2tf`, `zp2sos`, `tf2sos`, `sos2tf`, `sos2zp`, `roots` |
| responses | `Freqz`, `SOSFreqz`, `FreqzZPK`, `GroupDelay`, `Freqs`, `FreqsZPK` | `freqz`, `sosfreqz`, `freqz_zpk`, `group_delay`, `freqs`, `freqs_zpk` | `freqz`, `grpdelay`, `freqs` |
| analog prototypes | `ButtAP`, `Cheb1AP`, `Cheb2AP`, `EllipAP` (own Jacobi `Ellipj`), `BesselAP` | `buttap`, `cheb1ap`, `cheb2ap`, `ellipap`, `besselap` | `buttap`, `cheb1ap`, `cheb2ap`, `ellipap`, `besselap` |
| band transforms | `LP2LP`, `LP2HP`, `LP2BP`, `LP2BS` and the `...ZPK` forms, `Bilinear`, `BilinearZPK` | `lp2lp` ... `lp2bs`, `bilinear` | `lp2lp`, `lp2hp`, `lp2bp`, `lp2bs`, `bilinear` |
| IIR design | `IIRFilter`, `Butter`, `Cheby1`, `Cheby2`, `Ellip`, `BesselFilter`, `IIRDesign`, `IIRNotch`, `IIRPeak` | `iirfilter`, `butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirdesign`, `iirnotch`, `iirpeak` | `butter`, `cheby1`, `cheby2`, `ellip`, `besself`, `designfilt`, `iirnotch`, `iirpeak` |
| order selection | `ButtOrd`, `Cheb1Ord`, `Cheb2Ord`, `EllipOrd` | `buttord`, `cheb1ord`, `cheb2ord`, `ellipord` | `buttord`, `cheb1ord`, `cheb2ord`, `ellipord` |
| FIR design | `Firwin`, `Firwin2`, `Remez` (Parks–McClellan), `Firls`, `KaiserOrd`, `KaiserBeta`, `KaiserAtten` | `firwin`, `firwin2`, `remez`, `firls`, `kaiserord` | `fir1`, `fir2`, `firpm`, `firls`, `kaiserord` |
| windows | `Hann`, `Hamming`, `Blackman`, `BlackmanHarris`, `Bartlett`, `Boxcar`, `Triang`, `Parzen`, `Bohman`, `Nuttall`, `Flattop`, `Cosine`, `Barthann`, `Kaiser`, `Tukey`, `Gaussian`, `GeneralGaussian`, `GeneralHamming`, `GeneralCosine`, `Exponential`, `Chebwin`, `DPSS` (Slepian, with concentration ratios), `GetWindow` | `windows.*`, `get_window` | `hann`, `hamming`, `blackman`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, `dpss`, ... |
| filtering | `LFilter`, `LFilterZI`, `LFilterIC`, `FiltFilt` (odd/even/constant padding and Gustafsson), `SOSFilt`, `SOSFiltZI`, `SOSFiltFilt` | `lfilter`, `lfilter_zi`, `lfiltic`, `filtfilt`, `sosfilt`, `sosfilt_zi`, `sosfiltfilt` | `filter`, `filtfilt` |
| convolution | `Convolve`, `Correlate`, `CorrelationLags`, `FFTConvolve`, `OAConvolve`, `ChooseConvMethod`, `Deconvolve` | `convolve`, `correlate`, `correlation_lags`, `fftconvolve`, `oaconvolve`, `choose_conv_method`, `deconvolve` | `conv`, `xcorr`, `fftfilt`, `deconv` |
| spectral | `Periodogram`, `Welch`, `CSD`, `Coherence`, `Spectrogram`, `STFT`, `ISTFT`, `CheckCOLA`, `CheckNOLA`, `Lombscargle` | `periodogram`, `welch`, `csd`, `coherence`, `spectrogram`, `stft`, `istft`, `check_COLA`, `check_NOLA`, `lombscargle` | `periodogram`, `pwelch`, `cpsd`, `mscohere`, `spectrogram`, `stft`, `istft`, `plomb` |
| resampling | `Resample` (FFT), `ResamplePoly` (polyphase), `Decimate`, `Upfirdn` | `resample`, `resample_poly`, `decimate`, `upfirdn` | `resample`, `decimate`, `upfirdn` |
| analytic signal, trend | `Hilbert`, `Detrend` | `hilbert`, `detrend` | `hilbert`, `detrend` |
| peaks | `FindPeaks` (height, threshold, distance, prominence, width, plateau), `PeakProminences`, `PeakWidths` | `find_peaks`, `peak_prominences`, `peak_widths` | `findpeaks`, `peakwidths`, `peakprom` |

Differences worth knowing:

* `STFT`/`ISTFT` follow SciPy's legacy `stft`/`istft` (the `ShortTimeFFT` class
  is not provided).
* SciPy emits warnings where Go returns values: a singular group delay is set to
  zero silently; `Remez` returns the best filter together with `ErrNotConverged`
  (SciPy raises; a degenerate design whose taps would not be finite returns
  nil).
* `EllipAP` returns an error (SciPy returns NaN poles) when the stopband
  attenuation is below the passband ripple.
* Bessel prototypes use exact integer polynomial coefficients refined in
  320-bit arithmetic (double precision loses the zeros above order 8) and are
  limited to order 150, where the delay gain overflows `float64`.
* The `PadModeReflect`, `PadModeAntireflect`, `PadModeSmooth` and `PadModeLine`
  extensions of a one-sample signal are undefined in SciPy; here they repeat
  the sample.

Not provided: N-dimensional and `axis` variants, complex-coefficient filters,
`dlti`/`lti` classes, `ShortTimeFFT`, `savgol_filter`, `czt`, continuous
wavelets and `find_peaks_cwt` (wavelets are planned in this organisation).

## Verification

SciPy 1.18.1 with NumPy 2.5.4 is the oracle, driven from Go
(`oracle_test.go`: one persistent Python process, base64 float64/complex128 over
JSON lines, so no decimal round trip loses a bit). Every comparison is recorded
in `testdata/oracle` and replayed from there, so `go test` needs no Python and
the same recorded SciPy output is checked on every architecture CI runs.
`GOWORK=off go test -update ./...` re-records it and rewrites
[docs/accuracy.md](docs/accuracy.md), the table of maximum ULP errors,
normwise relative errors and exact-equality counts for 225 groups of values;
[docs/verification.md](docs/verification.md) explains the measures and the few
places where SciPy itself is the less accurate side (ill-conditioned elliptic
and Bessel designs, high-order transfer functions).

* 100% statement coverage with `-race`, including the error paths.
* `gofmt`, `go vet`, `govulncheck`; amd64, arm64, 386 (and under qemu riscv64,
  loong64, ppc64le, s390x, arm) and every `GOOS/GOARCH` pair compiling the tests.
* Fuzz targets for the design constructors, the spectral estimators, the
  convolutions and the peak finder.

## Performance

[docs/perf.md](docs/perf.md): one core against SciPy on the same host, in
interleaved rounds, with the load average checked
(`-perf -perfcase <regexp>` measures a subset without rewriting the table).
The FIR kernels call the SIMD dot product of go-simd/floats; the recursions
are serial by nature, so `SOSFilt` runs four (or two) sections sample-major
to overlap their dependency chains and the FIR form of `LFilter` is a direct
chain eight outputs wide, both bit-identical to the one-section,
one-output form. `Remez` runs the McClellan–Parks–Rabiner exchange (its
search walks the error curve from each extremum instead of evaluating the
whole grid every iteration), `PeakProminences` finds every base in two
monotonic-stack passes (O(n), where the per-peak scan SciPy uses is
O(n·peaks)), and `Spectrogram` computes its PSD through one reused spectrum
buffer.

## Layout

| file | contents |
|---|---|
| `convert.go`, `poly.go` | representations, conversions, polynomial roots |
| `freq.go` | frequency responses, group delay |
| `analog.go`, `transform.go`, `iir.go` | prototypes, band transforms, IIR design and order selection |
| `fir.go`, `remez.go`, `windows.go`, `linalg.go` | FIR design, Parks–McClellan, windows and DPSS, small solvers |
| `filter.go`, `convolve.go` | filtering, convolution |
| `spectral.go`, `resample.go` | spectral estimation, resampling, peaks |
| `*_test.go`, `testdata/oracle` | tests, SciPy oracle harness and recorded responses |

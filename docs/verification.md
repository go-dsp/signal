# Verification

How the package is checked against the reference tools of the domain, what the
numbers in [accuracy.md](accuracy.md) mean, and where the reference itself is
the less accurate side.

## The oracle

SciPy 1.18.1 with NumPy 2.5.4 (`uv run --with scipy==1.18.1 --with numpy==2.5.4
python -c ...`). The Python side is a 40-line generic dispatcher (no `eval`): it
resolves a dotted name inside `scipy`/`numpy`, decodes the arguments, calls the
function and encodes the result. It runs as one persistent process, speaks JSON
lines over a pipe and exits on end of input. Arrays travel as base64
little-endian float64, int64 or complex128, so no decimal round trip loses a
bit. The Go side is `oracle_test.go`; every other tool (comparisons, accuracy
tables, the performance harness) is Go.

Every request/response pair is recorded in `testdata/oracle/<Test>.json.gz`,
keyed by the position of the call within its test. `go test` replays them, so
CI needs no Python and the same recorded SciPy output is checked on amd64,
arm64, 386, arm, riscv64, loong64, ppc64le and s390x. (Keying by position, not
by the bits of the arguments, matters: test signals are built with `math.Sin`
and similar, which round differently on architectures that fuse multiply-adds.)
`GOWORK=off go test -update ./...` re-records everything and rewrites
`accuracy.md`.

## Measures

For each group of values `accuracy.md` gives

* **values**: how many numbers were compared;
* **exact**: how many are bit-identical to SciPy's (measured on the recording
  host, arm64, where Go fuses multiply-adds and NumPy's compiled loops do not);
* **max ULP**: the largest elementwise distance in units in the last place over
  the values that differ. It is meaningless near zero (a coefficient of 1e-17
  that differs in its last digit is millions of ULPs) and is shown for
  completeness;
* **norm err**: `max|got - want| / max|want|` per case, maximised over the cases
  of the group. This is the right measure for polynomial coefficients, spectra
  and responses, and the one the tests assert (tolerances between 0 and 1e-5,
  stated next to each comparison).

Roots and poles are compared as sets (greedy nearest matching) where SciPy's
order comes from LAPACK's eigenvalue solver, and elementwise where the order is
fixed by the algorithm (prototypes, band transforms, `zpk2sos`, `iirfilter`).
Where SciPy's own order depends on signed zeros (the square root of a negative
real number with an imaginary part of `-0.0` is `-i*s`, not `+i*s`) the Go code
reproduces the sign of the zero so that the conjugate pairs come out in the
same order.

## Where the numbers are not 1e-16

| group | norm err | why |
|---|---:|---|
| elliptic prototypes (`ellipap k`) | 1.7e-8 | Pole positions are ill-conditioned in the stopband attenuation and the order. The tests compare up to order 16. At order 18-30 SciPy itself misses the requested passband ripple by 1e-7 to 6e-3 (checked by evaluating `|H(j)|^2` of both implementations), and so does this package, differently. |
| order selection wn (`buttord`, `cheb1ord`, `cheb2ord`, `ellipord`) and `iirdesign` | 1e-7 | Band-stop edges come from a bounded scalar search with `xtol = 1e-5` (`fminbound`); a one-ULP difference in the objective moves the optimum by far more than 1 ULP. |
| `besselap` norm `mag` | 1e-13 | SciPy finds the -3 dB point with a secant search that stops at 1.5e-8; this package uses the same search. |
| `iirfilter` transfer functions at order 16-40 (`high ba num`) | 1.2e-8 | The monomial form of a high-order filter is ill-conditioned: coefficient errors of 1e-8 are in the problem, not in either implementation. Compared in `zpk` (4e-16) and `sos` (4e-16) form the two agree to rounding. This is why the package recommends SOS. |
| `tf2sos`, `zpk2tf` | 1.6e-14 | Same cause, at moderate order. |
| `dpss` windows | 1.5e-13 | Eigenvectors of the tridiagonal problem by bisection and inverse iteration; LAPACK's `stebz`/`stein` differ in rounding. |
| `chebwin` | 2.6e-13 | The window is the inverse FFT of a Chebyshev polynomial sampled near its zeros. |
| `remez` | 4.8e-14 | Parks-McClellan converges to the optimum of the discrete problem, and the grid is built as SciPy builds it. |
| `filtfilt` with `method="gust"` | 4.7e-15 | Least squares by Householder QR instead of LAPACK's SVD driver. |

## Where SciPy is the less accurate side

The zeros of Bessel polynomials are notoriously ill-conditioned. SciPy computes
them with `kve` and polishes in double precision; this package refines them with
Newton's method in 320-bit arithmetic on the exact integer coefficients
(`besselap`). Against a 60-digit `mpmath` solution SciPy's poles are accurate to
1e-15; in double precision alone the same recurrence loses 1e-8 by order 17,
which is why the refinement exists. The two agree to 1e-12 up to order 25.

## Corner cases

The comparisons include the degenerate inputs, not only random ones: filters
with `a[0] != 1`, with fewer zeros than poles, FIR filters through the IIR path,
unit-circle poles (singular group delay), one-sample and one-segment signals,
odd and even lengths everywhere, every window and every pad mode, filters longer
than the signal, `up` larger than the filter length, plateaus in `find_peaks`,
empty inputs, and every error path (checked to be an error in Go, and, where
SciPy raises too, to be an error there).

## Fuzzing and limits

`limits_test.go` holds the fuzz targets (`FuzzFirwin`, `FuzzIIRFilter`,
`FuzzWelchOptions`, `FuzzFindPeaks`, `FuzzConvolve`) and a table of absurd
arguments (orders of 2^30, billions of taps, huge FFT lengths and resampling
factors) that must produce an error, not a panic or an allocation. Output sizes
are bounded at 2^28 elements, analog prototype orders at 65536 (Bessel at 150),
`firls` at 16384 taps (dense normal equations).

## Gates

`gofmt`, `go vet`, `go test -race` with 100% statement coverage
(`-coverpkg=./...`), GOARCH=386 (run in a 32-bit container locally, natively in
CI), `govulncheck`, and the `go test -c` of every `GOOS/GOARCH` pair in CI.

# Public functions without a performance case

Read by `refbench` in the CI mode (`go test -run TestPerf .`): every public
function of the module has a case in `perf_test.go` or is named here with its
reason. The method and the measured table are in [perf.md](perf.md) and
[perf-table.md](perf-table.md). The 27 cases register the heavy paths (filters,
spectral estimates, resampling, convolution, peak finding, FIR/IIR design,
DPSS); the rows below are the functions not yet registered. "Not yet" is an
open item of the performance campaign
([go-scikit/roadmap#27](https://github.com/go-scikit/roadmap/issues/27)), not a
claim that the function is as fast as SciPy's.

## Option constructors, no SciPy counterpart

- `AtLeast`: option constructor, O(1), no reference function.
- `AtMost`: as `AtLeast`.
- `Between`: as `AtLeast`.
- `Overlap`: parameter helper, O(1), no SciPy function.

## Timed inside a registered case

- `ZPK.SOS`: timed inside the `butter` and `ellip` cases ("zpk to sos").
- `ZPK.TF`: same conversion family as `ZPK.SOS`; SciPy's `zpk2tf`, O(n^2) on filter orders below 20; not yet registered.

## Window generators (scipy.signal.windows), O(n) closed forms, not yet registered

- `Barthann`: `scipy.signal.windows.barthann`.
- `Bartlett`: `scipy.signal.windows.bartlett`.
- `Blackman`: `scipy.signal.windows.blackman`.
- `BlackmanHarris`: `scipy.signal.windows.blackmanharris`.
- `Bohman`: `scipy.signal.windows.bohman`.
- `Boxcar`: `scipy.signal.windows.boxcar`.
- `Chebwin`: `scipy.signal.windows.chebwin`, an FFT of a Chebyshev polynomial; not yet registered.
- `Cosine`: `scipy.signal.windows.cosine`.
- `Exponential`: `scipy.signal.windows.exponential`.
- `Flattop`: `scipy.signal.windows.flattop`.
- `Gaussian`: `scipy.signal.windows.gaussian`.
- `GeneralCosine`: `scipy.signal.windows.general_cosine`.
- `GeneralGaussian`: `scipy.signal.windows.general_gaussian`.
- `GeneralHamming`: `scipy.signal.windows.general_hamming`.
- `GetWindow`: `scipy.signal.get_window`, a dispatcher over the generators above.
- `Hamming`: `scipy.signal.windows.hamming`.
- `Hann`: `scipy.signal.windows.hann`.
- `Kaiser`: `scipy.signal.windows.kaiser`.
- `Nuttall`: `scipy.signal.windows.nuttall`.
- `Parzen`: `scipy.signal.windows.parzen`.
- `Triang`: `scipy.signal.windows.triang`.
- `Tukey`: `scipy.signal.windows.tukey`.

## Filter design, order selection and transforms (scipy.signal), small sizes, not yet registered

- `BesselAP`: `scipy.signal.besselap`.
- `BesselFilter`: `scipy.signal.bessel`.
- `Bilinear`: `scipy.signal.bilinear`.
- `BilinearZPK`: `scipy.signal.bilinear_zpk`.
- `ButtAP`: `scipy.signal.buttap`.
- `ButtOrd`: `scipy.signal.buttord`.
- `Cheb1AP`: `scipy.signal.cheb1ap`.
- `Cheb1Ord`: `scipy.signal.cheb1ord`.
- `Cheb2AP`: `scipy.signal.cheb2ap`.
- `Cheb2Ord`: `scipy.signal.cheb2ord`.
- `Cheby1`: `scipy.signal.cheby1`.
- `Cheby2`: `scipy.signal.cheby2`.
- `EllipAP`: `scipy.signal.ellipap`.
- `EllipOrd`: `scipy.signal.ellipord`.
- `Ellipj`: `scipy.special.ellipj`, evaluated on a handful of points inside the elliptic designs.
- `Firwin2`: `scipy.signal.firwin2`.
- `IIRDesign`: `scipy.signal.iirdesign`.
- `IIRFilter`: `scipy.signal.iirfilter`.
- `IIRNotch`: `scipy.signal.iirnotch`.
- `IIRPeak`: `scipy.signal.iirpeak`.
- `KaiserAtten`: `scipy.signal.kaiser_atten`, O(1).
- `KaiserBeta`: `scipy.signal.kaiser_beta`, O(1).
- `KaiserOrd`: `scipy.signal.kaiserord`, O(1).
- `LP2BP`: `scipy.signal.lp2bp`.
- `LP2BPZPK`: `scipy.signal.lp2bp_zpk`.
- `LP2BS`: `scipy.signal.lp2bs`.
- `LP2BSZPK`: `scipy.signal.lp2bs_zpk`.
- `LP2HP`: `scipy.signal.lp2hp`.
- `LP2HPZPK`: `scipy.signal.lp2hp_zpk`.
- `LP2LP`: `scipy.signal.lp2lp`.
- `LP2LPZPK`: `scipy.signal.lp2lp_zpk`.
- `Normalize`: `scipy.signal.normalize`.
- `Roots`: `numpy.roots` (an eigenvalue problem), used by the designs above.
- `SOS2TF`: `scipy.signal.sos2tf`.
- `SOS2ZPK`: `scipy.signal.sos2zpk`.
- `TF2SOS`: `scipy.signal.tf2sos`.
- `TF2ZPK`: `scipy.signal.tf2zpk`.
- `ZPK2SOS`: `scipy.signal.zpk2sos`, timed inside the `butter` and `ellip` cases through `ZPK.SOS`.
- `ZPK2TF`: `scipy.signal.zpk2tf`.

## Frequency responses and filter state, not yet registered

- `Freqs`: `scipy.signal.freqs`.
- `FreqsZPK`: `scipy.signal.freqs_zpk`.
- `Freqz`: `scipy.signal.freqz`.
- `FreqzZPK`: `scipy.signal.freqz_zpk`.
- `GroupDelay`: `scipy.signal.group_delay`.
- `SOSFreqz`: `scipy.signal.sosfreqz`.
- `LFilterIC`: `scipy.signal.lfiltic`.
- `LFilterZI`: `scipy.signal.lfilter_zi`.
- `SOSFiltZI`: `scipy.signal.sosfilt_zi`.

## Spectral estimates, convolution family and peak helpers, not yet registered

- `CSD`: `scipy.signal.csd`, the same segment loop as the `welch` cases.
- `Coherence`: `scipy.signal.coherence`, the same segment loop as the `welch` cases.
- `STFT`: `scipy.signal.stft`.
- `ISTFT`: `scipy.signal.istft`.
- `CheckCOLA`: `scipy.signal.check_COLA`.
- `CheckNOLA`: `scipy.signal.check_NOLA`.
- `Lombscargle`: `scipy.signal.lombscargle`, O(n*m) with trigonometric kernels.
- `LombscargleAmplitude`: as `Lombscargle`.
- `Correlate`: `scipy.signal.correlate`, the same engine as the `convolve` cases.
- `CorrelationLags`: `scipy.signal.correlation_lags`, O(n) index vector.
- `ChooseConvMethod`: `scipy.signal.choose_conv_method`, a timing heuristic by design.
- `Deconvolve`: `scipy.signal.deconvolve`, an IIR filter run.
- `Detrend`: `scipy.signal.detrend`.
- `PeakProminences`: `scipy.signal.peak_prominences`, the stage timed inside `find_peaks`.
- `PeakWidths`: `scipy.signal.peak_widths`, the stage timed inside `find_peaks`.

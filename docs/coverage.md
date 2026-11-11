# Coverage against SciPy and MATLAB

Inventory of what `go-dsp/signal` (main, v0.1.0) implements against
`scipy.signal` v1.18.1, the MATLAB Signal Processing Toolbox and the wavelet
basics (MATLAB Wavelet Toolbox, PyWavelets as the open reference). Status:
**done** (same semantics and options that matter), **partial** (what is
missing is named), **missing**. "Dep" is what a missing item needs beyond this
package.

Sources. SciPy: `scipy/signal/__init__.py` at tag v1.18.1 (the rows follow its
autosummary sections in order) and the `scipy.signal.windows` namespace (window
names from knowledge, not re-read at the tag). MATLAB: the MathWorks function
lists could not be read by the tool used (client-rendered pages), so the MATLAB
column is built from the toolbox's documented function families and must be
re-checked against the MathWorks reference lists before a slice is started.
PyWavelets: names from knowledge, to be re-checked. Go side: README "Not
provided", `go doc`, and the sources on main.

Conventions that apply to many rows (not repeated in every cell): the Go
functions take real, one-dimensional `[]float64` signals and real filter
coefficients, with no `axis` argument ("1-D real only"); N-D, `axis` and complex
variants are listed once, as slice 10.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| SciPy v1.18.1 | 65 | 18 | 49 | 132 |
| MATLAB and wavelets | 54 | 18 | 68 | 140 |

A row counts under SciPy when its SciPy cell is filled and under MATLAB when its
MATLAB cell is filled.

## Convolution

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `convolve` | `conv` | `Convolve` | partial | N-D |
| 2 | `correlate` | `xcorr` | `Correlate` | partial | N-D; MATLAB `xcorr` normalisation options (`biased`, `unbiased`, `coeff`), `maxlag`, matrix input |
| 3 | `fftconvolve` | `fftfilt` | `FFTConvolve` | partial | N-D, `axes` |
| 4 | `oaconvolve` | `fftfilt` | `OAConvolve` | partial | N-D, `axes` |
| 5 | `convolve2d` | `conv2` | | missing | none (direct 2-D; FFT form via go-fft) |
| 6 | `correlate2d` | `xcorr2` | | missing | row 5 |
| 7 | `sepfir2d` | `conv2(h,v,A)` | | missing | none |
| 8 | `choose_conv_method` | | `ChooseConvMethod` | partial | `measure=True` |
| 9 | `correlation_lags` | `xcorr` lags | `CorrelationLags` | done | |

## B-splines

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 10 | `gauss_spline` | | | missing | none |
| 11 | `cspline1d`, `cspline1d_eval` | `csapi` (Curve Fitting) | | missing | recursive symmetric IIR filter (row 21 `symiirorder1`); or `go-approx/interp` |
| 12 | `qspline1d`, `qspline1d_eval` | | | missing | same |
| 13 | `cspline2d`, `qspline2d`, `spline_filter` | | | missing | same, 2-D |
| 14 | `whittaker_henderson` | `smoothdata`-like | | missing | banded solver (own) |

## Filtering

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 15 | `order_filter` | `ordfilt2` (Image Processing) | | missing | none |
| 16 | `medfilt` | `medfilt1` | | missing | none (sliding median; O(n log k) or histogram) |
| 17 | `medfilt2d` | `medfilt2` (Image Processing) | | missing | none |
| 18 | `wiener` | `wiener2` (Image Processing) | | missing | none |
| 19 | | `hampel`, `filloutliers`, `movmedian`, `movmean`, `smoothdata` | | missing | none (moving-window statistics) |
| 20 | | `sgolayfilt`, `sgolay` | | missing | row 22 |
| 21 | `symiirorder1`, `symiirorder2` | | | missing | none (cascade of causal and anticausal first and second order sections) |
| 22 | `savgol_filter` | `sgolayfilt` | | missing | row 23 (`savgol_coeffs`), edge modes `mirror`, `nearest`, `constant`, `wrap`, `interp` |
| 23 | `savgol_coeffs` | `sgolay` | | missing | small least-squares solve (own) |
| 24 | `lfilter` | `filter` | `LFilter` | partial | N-D and `axis`; MATLAB `filter` dimension argument |
| 25 | `lfiltic` | `filtic` | `LFilterIC` | done | |
| 26 | `lfilter_zi` | `filtic` (steady state) | `LFilterZI` | done | |
| 27 | `filtfilt` | `filtfilt` | `FiltFilt` | partial | N-D and `axis` |
| 28 | `deconvolve` | `deconv` | `Deconvolve` | done | |
| 29 | `sosfilt` | `sosfilt`(via `dsp.BiquadFilter`), `filter(sos)` | `SOSFilt` | partial | N-D and `axis` |
| 30 | `sosfilt_zi` | | `SOSFiltZI` | done | |
| 31 | `sosfiltfilt` | `filtfilt(sos,g,x)` | `SOSFiltFilt` | partial | N-D and `axis` |
| 32 | `hilbert` | `hilbert` | `Hilbert` | partial | N-D, `axis` |
| 33 | `hilbert2` | | | missing | go-fft 2-D |
| 34 | `envelope` | `envelope` | | missing | `Hilbert` (have), a low-pass for the RMS/peak modes |
| 35 | `decimate` | `decimate` | `Decimate` | partial | N-D; `ftype='iir'` and `'fir'` done |
| 36 | `detrend` | `detrend` | `Detrend` | partial | N-D; MATLAB polynomial order above 1 |
| 37 | `resample` | `resample` | `Resample` | partial | complex input, `t` (time vector), `domain='freq'`, N-D |
| 38 | `resample_poly` | `resample` | `ResamplePoly` | partial | N-D; MATLAB `resample` Kaiser design parameters |
| 39 | `upfirdn` | `upfirdn` | `Upfirdn` | partial | N-D, complex |
| 40 | | `downsample`, `upsample`, `interp` | | missing | none (trivial) |

## Filter design

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 41 | `bilinear` | `bilinear` | `Bilinear` | done | |
| 42 | `bilinear_zpk` | `bilinear` | `BilinearZPK` | done | |
| 43 | `findfreqs` | | | missing | none (trivial) |
| 44 | `firls` | `firls` | `Firls` | done | |
| 45 | `firwin` | `fir1` | `Firwin` | done | |
| 46 | `firwin2` | `fir2` | `Firwin2` | done | |
| 47 | `firwin_2d` | `ftrans2`, `fwind1` (Image Processing) | | missing | row 5 and `Firwin` |
| 48 | `freqs` | `freqs` | `Freqs` | done | |
| 49 | `freqs_zpk` | `freqs` | `FreqsZPK` | done | |
| 50 | `freqz` | `freqz` | `Freqz` | done | |
| 51 | `sosfreqz`, `freqz_sos` | `freqz(sos)` | `SOSFreqz` | done | |
| 52 | `freqz_zpk` | `zp2tf` + `freqz` | `FreqzZPK` | done | |
| 53 | `gammatone` | `gammatone` (Audio Toolbox) | | missing | none (FIR and IIR gammatone) |
| 54 | `group_delay` | `grpdelay` | `GroupDelay` | done | |
| 55 | `iirdesign` | `designfilt` | `IIRDesign` | partial | MATLAB `designfilt` front end (specification-driven, many families); constraint sets |
| 56 | `iirfilter` | | `IIRFilter` | done | |
| 57 | `kaiser_atten` | | `KaiserAtten` | done | |
| 58 | `kaiser_beta` | | `KaiserBeta` | done | |
| 59 | `kaiserord` | `kaiserord` | `KaiserOrd` | done | |
| 60 | `minimum_phase` | `rceps` (minimum-phase via cepstrum) | | missing | go-fft; homomorphic method and `'hilbert'` method |
| 61 | `remez` | `firpm` | `Remez` | done | |
| 62 | `unique_roots` | | | missing | none |
| 63 | `residue`, `invres` | `residue` | | missing | row 62 |
| 64 | `residuez`, `invresz` | `residuez` | | missing | row 62 |
| 65 | `BadCoefficients` | | | missing | not applicable (a warning) |
| 66 | `abcd_normalize` | | | missing | with state space (row 91) |
| 67 | `band_stop_obj` | | | missing | internal helper of `buttord`; trivial to export |
| 68 | `besselap`, `buttap`, `cheb1ap`, `cheb2ap`, `ellipap` | `besselap`, `buttap`, `cheb1ap`, `cheb2ap`, `ellipap` | `BesselAP`, `ButtAP`, `Cheb1AP`, `Cheb2AP`, `EllipAP` | done | |
| 69 | `lp2bp`, `lp2bs`, `lp2hp`, `lp2lp` | same | `LP2BP`, `LP2BS`, `LP2HP`, `LP2LP` | done | |
| 70 | `lp2bp_zpk`, `lp2bs_zpk`, `lp2hp_zpk`, `lp2lp_zpk` | | `LP2BPZPK`, `LP2BSZPK`, `LP2HPZPK`, `LP2LPZPK` | done | |
| 71 | `normalize` | | `Normalize` | done | |
| 72 | `butter`, `buttord` | `butter`, `buttord` | `Butter`, `ButtOrd` | done | |
| 73 | `cheby1`, `cheb1ord` | `cheby1`, `cheb1ord` | `Cheby1`, `Cheb1Ord` | done | |
| 74 | `cheby2`, `cheb2ord` | `cheby2`, `cheb2ord` | `Cheby2`, `Cheb2Ord` | done | |
| 75 | `ellip`, `ellipord` | `ellip`, `ellipord` | `Ellip`, `EllipOrd` | done | |
| 76 | `bessel` | `besself` | `BesselFilter` | done | |
| 77 | `iirnotch`, `iirpeak` | `iirnotch`, `iirpeak` | `IIRNotch`, `IIRPeak` | done | |
| 78 | `iircomb` | `iircomb` | | missing | none |
| 79 | | `firpmord`, `firpmord` order estimate for `firpm` | | missing | none (Herrmann-Rabiner-Chan formula) |
| 80 | | `firceqrip`, `firgr`, `cfirpm`, `firhalfband`, `firnyquist`, `maxflat` | | missing | `Remez` with constraints (`firceqrip`), complex Remez (`cfirpm`); `maxflat` is closed form |
| 81 | | `rcosdesign`, `firrcos`, `gaussdesign` | | missing | none (raised-cosine and Gaussian pulse shaping) |
| 82 | | `yulewalk`, `prony`, `stmcb`, `invfreqz`, `invfreqs` | | missing | dense least squares |
| 83 | | `lpc`, `levinson`, `rc2ac`, `ac2rc`, `ac2poly`, `poly2ac`, `rc2poly`, `poly2rc`, `lsf2poly`, `poly2lsf` | | missing | none (Levinson-Durbin) |
| 84 | | `ellipke` | `Ellipj` (own) | partial | `ellipke` lives in `go-numerics/special` (dependency already) |
| 85 | | `isstable`, `isminphase`, `ismaxphase`, `isallpass`, `islinphase` | | missing | `Roots` (have) |
| 86 | | `phasez`, `zerophase`, `impz`, `stepz`, `phasedelay`, `zplane` | | missing | `Freqz` (have); `zplane` is plotting |
| 87 | | `fdesign`, `filterBuilder`, `designfilt` objects | | missing | out of scope (object model); row 55 |
| 88 | | `dsp.FIRFilter`, `dsp.BiquadFilter`, `dsp.CICDecimator`, `dsp.FarrowRateConverter` (streaming System objects) | | missing | a streaming filter type holding state across calls (`LFilter` already takes and returns `zi`) |

## Linear systems

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 89 | `lti`, `StateSpace`, `TransferFunction`, `ZerosPolesGain` | `ss`, `tf`, `zpk` (Control System Toolbox) | `ZPK`, `(b, a)` | partial | no state-space type, no `lti` object |
| 90 | `dlti`, `dlsim`, `dimpulse`, `dstep`, `dfreqresp`, `dbode` | `lsim`, `impulse`, `step`, `bode` (discrete) | | missing | state space (row 91); `Freqz` covers `dfreqresp` for TF |
| 91 | `tf2ss`, `zpk2ss`, `ss2tf`, `ss2zpk` | `tf2ss`, `zp2ss`, `ss2tf`, `ss2zp` | | missing | dense eigenvalues (for `ss2zpk`), `Roots` (have) |
| 92 | `lsim`, `impulse`, `step`, `freqresp`, `bode` | same | | missing | matrix exponential (dense `expm`); `freqresp` for TF is `Freqs` (have) |
| 93 | `tf2zpk`, `tf2sos`, `zpk2tf`, `zpk2sos`, `sos2zpk`, `sos2tf` | `tf2zp`, `tf2sos`, `zp2tf`, `zp2sos`, `sos2zp`, `sos2tf` | `TF2ZPK`, `TF2SOS`, `ZPK2TF`, `ZPK2SOS`, `SOS2ZPK`, `SOS2TF` | done | |
| 94 | `cont2discrete` | `c2d` (Control) | | missing | zoh, foh, bilinear (have `Bilinear`), impulse, matched; zoh needs dense `expm` |
| 95 | `place_poles` | `place` (Control) | | missing | dense linear algebra (KNV0, YT methods) |
| 96 | | `sos2ss`, `ss2sos`, `latc2tf`, `tf2latc` | | missing | row 91 |

## Waveforms

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 97 | `chirp` | `chirp` | | missing | none (linear, quadratic, logarithmic, hyperbolic) |
| 98 | `gausspulse` | `gauspuls` | | missing | none |
| 99 | `max_len_seq` | `mlseq`-like (Communications) | | missing | none (LFSR) |
| 100 | `sawtooth`, `square` | `sawtooth`, `square` | | missing | none |
| 101 | `sweep_poly` | | | missing | `go-approx/poly` `Polyval` |
| 102 | `unit_impulse` | `impz`-like | | missing | none |
| 103 | | `pulstran`, `rectpuls`, `tripuls`, `sinc`, `diric` | | missing | none |

## Window functions (`scipy.signal.windows`, `get_window`)

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 104 | `get_window` | | `GetWindow` | partial | names for `taylor`, `lanczos`, `kaiser_bessel_derived` (rows 125, 128, 129) |
| 105 | `barthann` | `barthannwin` | `Barthann` | done | |
| 106 | `bartlett` | `bartlett` | `Bartlett` | done | |
| 107 | `blackman` | `blackman` | `Blackman` | done | |
| 108 | `blackmanharris` | `blackmanharris` | `BlackmanHarris` | done | |
| 109 | `bohman` | `bohmanwin` | `Bohman` | done | |
| 110 | `boxcar` | `rectwin` | `Boxcar` | done | |
| 111 | `chebwin` | `chebwin` | `Chebwin` | done | |
| 112 | `cosine` | | `Cosine` | done | |
| 113 | `dpss` | `dpss` | `DPSS` | done | |
| 114 | `exponential` | | `Exponential` | done | |
| 115 | `flattop` | `flattopwin` | `Flattop` | done | |
| 116 | `gaussian` | `gausswin` | `Gaussian` | done | |
| 117 | `general_cosine` | | `GeneralCosine` | done | |
| 118 | `general_gaussian` | | `GeneralGaussian` | done | |
| 119 | `general_hamming` | | `GeneralHamming` | done | |
| 120 | `hamming` | `hamming` | `Hamming` | done | |
| 121 | `hann` | `hann`, `hanning` | `Hann` | done | |
| 122 | `kaiser` | `kaiser` | `Kaiser` | done | |
| 123 | `nuttall` | `nuttallwin` | `Nuttall` | done | |
| 124 | `parzen` | `parzenwin` | `Parzen` | done | |
| 125 | `taylor` | `taylorwin` | | missing | none |
| 126 | `triang` | `triang` | `Triang` | done | |
| 127 | `tukey` | `tukeywin` | `Tukey` | done | |
| 128 | `lanczos` | | | missing | none |
| 129 | `kaiser_bessel_derived` | `kbdwin` | `KaiserBesselDerived` | done | even length only (nil otherwise); no `GetWindow` name yet |
| 130 | | `window`, `wvtool`, `wintool` | `GetWindow` | partial | `window` front end; the tools are interactive |

## Peak finding

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 131 | `argrelmin`, `argrelmax`, `argrelextrema` | `islocalmin`, `islocalmax` | | missing | none |
| 132 | `find_peaks` | `findpeaks` | `FindPeaks` | done | |
| 133 | `find_peaks_cwt` | `findpeaks` + `cwt` | | missing | CWT (wavelet slice) and `ricker` wavelet |
| 134 | `peak_prominences` | `findpeaks` prominence | `PeakProminences` | done | |
| 135 | `peak_widths` | `findpeaks` widths | `PeakWidths` | done | |
| 136 | | `findchangepts`, `dtw`, `findsignal` | | missing | none (change-point search, dynamic time warping) |

## Spectral analysis

| # | SciPy | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 137 | `periodogram` | `periodogram` | `Periodogram` | done | |
| 138 | `welch` | `pwelch` | `Welch` | done | |
| 139 | `csd` | `cpsd` | `CSD` | done | |
| 140 | `coherence` | `mscohere` | `Coherence` | done | |
| 141 | `spectrogram` | `spectrogram` | `Spectrogram` | done | |
| 142 | `lombscargle` | `plomb` | `Lombscargle` | done | |
| 143 | `vectorstrength` | | | missing | none |
| 144 | `ShortTimeFFT` | `stft`, `istft` | | missing | go-fft; the class API (mfft, dual window, `p_range`, `extent`, lazy `stft_detrend`) |
| 145 | `closest_STFT_dual_window` | | | missing | `ShortTimeFFT` |
| 146 | `stft`, `istft` (legacy) | `stft`, `istft` | `STFT`, `ISTFT` | done | |
| 147 | `check_COLA`, `check_NOLA` | `iscola` | `CheckCOLA`, `CheckNOLA` | done | |
| 148 | `czt`, `CZT`, `czt_points` | `czt` | | missing | go-fft (Bluestein) |
| 149 | `zoom_fft`, `ZoomFFT` | `zoomfft`-like | | missing | row 148 |
| 150 | | `pspectrum` | | missing | `Welch`, `Spectrogram` (have); persistence spectrum |
| 151 | | `pburg`, `pcov`, `pmcov`, `pyulear`, `arburg`, `arcov`, `armcov`, `aryule` | | missing | Levinson-Durbin (row 83), covariance-method least squares |
| 152 | | `pmtm` (multitaper) | | missing | `DPSS` (have), go-fft |
| 153 | | `pmusic`, `peig`, `rootmusic`, `rooteig`, `esprit` | | missing | dense eigendecomposition |
| 154 | | `cwt`, `icwt`, `fsst`, `ifsst`, `tfridge`, `wvd`, `ambgfun`, `cqt`, `icqt` | | missing | wavelet slice and go-fft |
| 155 | | `instfreq`, `instbw`, `meanfreq`, `medfreq`, `powerbw`, `obw`, `bandpower`, `rms`, `peak2peak`, `peak2rms`, `rssq` | | missing | none |
| 156 | | `snr`, `thd`, `sinad`, `sfdr`, `toi`, `enbw` | | missing | `Periodogram` (have) |
| 157 | | `xcov`, `xcorr`, `cceps`, `rceps`, `icceps`, `dct`, `idct`, `dst` | | missing | `xcov`: none; cepstra and DCT/DST: go-fft |
| 158 | | `emd`, `vmd`, `hht`, `ssqueeze` | | missing | spline envelopes (`go-approx/interp`); research-grade methods |
| 159 | | `unwrap`, `fftshift`, `buffer`, `shiftdata`, `bitrevorder`, `mag2db`, `db2mag`, `pow2db`, `db2pow` | | missing | none (utilities) |

## Wavelets (basics)

PyWavelets (`pywt`) is the open reference. From knowledge, `scipy.signal.cwt`,
`morlet`, `morlet2`, `ricker`, `daub`, `qmf` and `cascade` were deprecated in
SciPy 1.12 and removed in 1.15 (not in the 1.18.1 `__init__.py` list above;
`find_peaks_cwt` remains); this needs a re-check. The README of this module says
wavelets are planned in this organisation.

| # | SciPy | MATLAB / PyWavelets | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 160 | | `cwt` (`pywt.cwt`) | | missing | CWT via FFT (go-fft) and wavelet functions |
| 161 | | `dwt`, `idwt`, `dwt2`, `idwt2`, `wavedec`, `waverec`, `wavedec2`, `waverec2` | | missing | filter banks (below) and `Upfirdn` (have); boundary modes (`zero`, `symmetric`, `periodization`, `reflect`, `periodic`, `smooth`, `constant`) |
| 162 | | `swt`, `iswt`, `modwt`, `imodwt`, `modwtmra` | | missing | row 161 |
| 163 | | `wpdec`, `wprec`, `wpt`, `iwpt` (packets), `wpbest`, `wentropy` | | missing | row 161 |
| 164 | | wavelet families: `haar`, `db1..45`, `sym2..20`, `coif1..17`, `bior`, `rbio`, `dmey`, `meyer`, `morl`, `mexh`, `gaus`, `cgau`, `cmor`, `shan`, `fbsp` | | missing | coefficient tables (public numerical constants); `db` and `sym` via spectral factorisation (`Roots`, have) |
| 165 | | `wavefun`, `waveinfo`, `wfilters`, `orthfilt`, `biorfilt`, `qmf` | | missing | row 164 |
| 166 | | `wden`, `wthresh`, `thselect`, `wdenoise` | | missing | row 161 |
| 167 | | `cwtfilterbank`, `cwtft`, `dtt` (dual-tree) | | missing | row 160 |

## Proposed slices

Ordered by value per cost; "dep" names what has to exist first.

1. **Waveforms and utilities** (rows 40, 43, 67, 97 to 103, 131, 143, 155, 159): `Chirp`, `GaussPulse`, `MaxLenSeq`, `Sawtooth`, `Square`, `SweepPoly`, `UnitImpulse`, `ArgRelExtrema`, `FindFreqs`, `Upsample`/`Downsample`, band power and RMS. Dep: none (`SweepPoly` uses `go-approx/poly`).
2. **Windows** (rows 104, 125, 128, 129): `Taylor`, `Lanczos`, `KaiserBesselDerived`, `GetWindow` names. Dep: none.
3. **Smoothing and rank filters** (rows 14 to 23, 36): `SavgolCoeffs`, `SavgolFilter`, `MedFilt`, `OrderFilter`, `Wiener`, `Whittaker-Henderson`, `SymIIROrder1/2`, `GaussSpline`, `CSpline1D`, `QSpline1D`. Dep: none.
4. **Partial fractions and design leftovers** (rows 62 to 64, 78 to 81, 83, 85, 86): `UniqueRoots`, `Residue`, `Residuez`, `Invres`, `IIRComb`, `Firpmord`, `Maxflat`, `Rcosdesign`, `Gaussdesign`, `Levinson`, `Lpc`, stability tests, `Impz`. Dep: none.
5. **Spectral estimation extensions** (rows 144, 145, 148 to 152, 156, 157): `ShortTimeFFT`, `CZT`, `ZoomFFT`, `Pmtm`, AR methods (`Pburg`, `Pyulear`, `Pcov`), `Pspectrum`, SNR family, cepstra. Dep: go-fft (Bluestein, DCT/DST); Levinson from slice 4.
6. **Linear systems** (rows 89 to 92, 94 to 96, 66): state-space type, `TF2SS`, `SS2TF`, `Lsim`, `Impulse`, `Step`, `Bode`, `Cont2Discrete`, `PlacePoles`. Dep: dense linear algebra (matrix exponential, eigenvalues, Schur).
7. **2-D** (rows 5 to 7, 17, 33, 47): `Convolve2D`, `Correlate2D`, `SepFir2D`, `MedFilt2D`, `Hilbert2`, `Firwin2D`. Dep: go-fft 2-D.
8. **Wavelets** (rows 133, 154, 160 to 167): coefficient tables and `DWT`/`IDWT`/`WaveDec`/`WaveRec` first (they need only `Upfirdn`), then `SWT`, `MODWT`, packets, `CWT`, `FindPeaksCWT`. Dep: go-fft for `CWT`; coefficient tables.
9. **Streaming and object APIs** (rows 87, 88, 29): stateful filter types for `LFilter`/`SOSFilt`, CIC, Farrow; `designfilt`-style builder. Dep: none; a design decision on the API.
10. **N-D, `axis` and complex** (rows 1 to 4, 24, 27, 31, 32, 35 to 39): `axis` and N-D arrays through go-ndarray, complex signals and coefficients. Dep: go-ndarray; decision on the scalar constraint.
11. **Advanced MATLAB-only** (rows 82, 153, 158): `Prony`, `Invfreqz`, `Yulewalk`, MUSIC/ESPRIT, `EMD`/`VMD`. Dep: dense eigendecomposition; spline envelopes from `go-approx/interp`.

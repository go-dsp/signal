// Package signal is signal processing in pure Go: filter design and
// filtering, spectral estimation and resampling, with SciPy's semantics
// (scipy.signal) and, where they differ, a note on MATLAB's.
//
// All vectors are []float64 and filters have real coefficients. Filters travel
// as transfer-function coefficients (TF), zeros-poles-gain (ZPK) or
// second-order sections (SOS); prefer SOS above order 8, where the transfer
// function form loses accuracy.
package signal

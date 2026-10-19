package signal

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestPSDColumnsBitIdentical checks the fused PSD spectrogram path against
// the per-segment cross-spectra it replaces, bit for bit.
func TestPSDColumnsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	x := make([]float64, 3001)
	for i := range x {
		x[i] = r.NormFloat64() + 0.01*float64(i)
	}
	for _, o := range []SpectralOptions{
		{Fs: 1},
		{Fs: 2, Nperseg: 100, Nfft: 128, Detrend: DetrendLinear},
		{Fs: 1, Nperseg: 101, Nfft: 101, Detrend: DetrendNone, Scaling: ScaleSpectrum},
		{Fs: 1, Nperseg: 64, TwoSided: true},
		{Fs: 1, Nperseg: 63, TwoSided: true, Noverlap: Overlap(10)},
	} {
		opt := o
		seg, err := opt.resolve(len(x), specDefaults)
		if err != nil {
			t.Fatal(err)
		}
		want := transposeReal(crossSpectra(x, nil, seg), func(c complex128) float64 { return real(c) })
		got := psdColumns(x, seg)
		if len(got) != len(want) {
			t.Fatalf("%+v: %d rows, want %d", o, len(got), len(want))
		}
		for i := range want {
			for k := range want[i] {
				if math.Float64bits(got[i][k]) != math.Float64bits(want[i][k]) {
					t.Fatalf("%+v: [%d][%d] = %v, want %v", o, i, k, got[i][k], want[i][k])
				}
			}
		}
	}
}

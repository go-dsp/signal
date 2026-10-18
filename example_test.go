package signal_test

import (
	"fmt"
	"math"

	"github.com/go-dsp/signal"
)

func ExampleButter() {
	z, _ := signal.Butter(2, []float64{0.2}, signal.Lowpass, nil)
	b, a, _ := z.TF()
	fmt.Printf("b = %.6f\na = %.6f\n", b, a)
	// Output:
	// b = [0.067455 0.134911 0.067455]
	// a = [1.000000 -1.142981 0.412802]
}

func ExampleFirwin() {
	h, _ := signal.Firwin(21, []float64{0.3}, nil)
	var sum float64
	for _, v := range h {
		sum += v
	}
	fmt.Printf("%d taps, DC gain %.6f\n", len(h), sum)
	// Output: 21 taps, DC gain 1.000000
}

func ExampleConvolve() {
	y, _ := signal.Convolve([]float64{1, 2, 3}, []float64{0, 1, 0.5}, signal.ConvFull, signal.ConvAuto)
	fmt.Println(y)
	// Output: [0 1 2.5 4 1.5]
}

func ExampleFindPeaks() {
	peaks, props, _ := signal.FindPeaks([]float64{0, 1, 0, 2, 0, 1.5, 0}, &signal.FindPeaksOptions{Height: signal.AtLeast(1.2)})
	fmt.Println(peaks, props.PeakHeights)
	// Output: [3 5] [2 1.5]
}

func ExampleWelch() {
	x := make([]float64, 4096)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * 100 * float64(i) / 1000) // 100 Hz at fs = 1000 Hz
	}
	f, p, _ := signal.Welch(x, &signal.SpectralOptions{Fs: 1000, Nperseg: 500})
	best := 0
	for i := range p {
		if p[i] > p[best] {
			best = i
		}
	}
	fmt.Printf("peak at %.0f Hz\n", f[best])
	// Output: peak at 100 Hz
}

func ExampleResamplePoly() {
	x := make([]float64, 100)
	for i := range x {
		x[i] = float64(i)
	}
	y, _ := signal.ResamplePoly(x, 3, 2, nil)
	fmt.Println(len(y))
	// Output: 150
}

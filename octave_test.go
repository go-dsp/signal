package signal

// MATLAB semantics against Octave 9.2 with its signal package (GPL, so used
// strictly as an oracle: run, never read or ported). Each case is an Octave
// expression assigning a vector to r and the Go computation of the same
// vector; Octave prints its result with 17 significant digits (a lossless
// round trip; its printf has no %a). The responses are recorded in testdata/octave and
// replayed; regenerate with
//
//	GOWORK=off go test -run TestOctave -octave -count=1 .
//
// which needs Docker and the image built as described in docs/verification.md.

import (
	"bytes"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var octaveFlag = flag.Bool("octave", false, "run the Octave oracle and record testdata/octave")

const octaveImage = "go-dsp-octave-signal:9.2"

type octCase struct {
	name string
	expr string // Octave statements leaving the result vector in r
	tol  float64
	got  func(t *testing.T) []float64
}

func fl(v ...[]float64) []float64 {
	var out []float64
	for _, x := range v {
		out = append(out, x...)
	}
	return out
}

func octInput(name string, x []float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = [", name)
	for i, v := range x {
		if i > 0 {
			b.WriteString(" ")
		}
		b.WriteString(strconv.FormatFloat(v, 'g', 17, 64))
	}
	b.WriteString("];\n")
	return b.String()
}

func octaveCases(t *testing.T) []octCase {
	rng := newRNG(200)
	x := testSignal(rng, 64)
	baOf := func(z ZPK, err error) func(*testing.T) []float64 {
		return func(t *testing.T) []float64 {
			mustOK(t, err)
			b, a, err := z.TF()
			mustOK(t, err)
			return fl(b, a)
		}
	}
	one := func(v float64) []float64 { return []float64{v} }
	butter4, e1 := Butter(4, one(0.3), Lowpass, nil)
	butterHP, e2 := Butter(3, one(0.4), Highpass, nil)
	butterBP, e3 := Butter(3, []float64{0.2, 0.5}, Bandpass, nil)
	butterBS, e4 := Butter(2, []float64{0.2, 0.5}, Bandstop, nil)
	cheb1, e5 := Cheby1(4, 1, one(0.3), Lowpass, nil)
	cheb2, e6 := Cheby2(4, 40, one(0.3), Lowpass, nil)
	ell, e7 := Ellip(4, 1, 40, one(0.3), Lowpass, nil)
	return []octCase{
		{"butter_low4", "[b,a]=butter(4,0.3); r=[b a];", 1e-12, baOf(butter4, e1)},
		{"butter_high3", "[b,a]=butter(3,0.4,'high'); r=[b a];", 1e-12, baOf(butterHP, e2)},
		{"butter_band3", "[b,a]=butter(3,[0.2 0.5]); r=[b a];", 1e-11, baOf(butterBP, e3)},
		{"butter_stop2", "[b,a]=butter(2,[0.2 0.5],'stop'); r=[b a];", 1e-11, baOf(butterBS, e4)},
		{"cheby1_4", "[b,a]=cheby1(4,1,0.3); r=[b a];", 1e-11, baOf(cheb1, e5)},
		{"cheby2_4", "[b,a]=cheby2(4,40,0.3); r=[b a];", 1e-11, baOf(cheb2, e6)},
		// Octave's ellip is a different, less exact design (1e-5 apart)
		{"ellip_4", "[b,a]=ellip(4,1,40,0.3); r=[b a];", 1e-4, baOf(ell, e7)},
		{"buttord", "[n,w]=buttord(0.2,0.3,1,40); r=[n w];", 1e-9, func(t *testing.T) []float64 {
			n, w, err := ButtOrd(one(0.2), one(0.3), 1, 40, false, 0)
			mustOK(t, err)
			return fl([]float64{float64(n)}, w)
		}},
		{"cheb1ord", "[n,w]=cheb1ord(0.2,0.3,1,40); r=[n w];", 1e-9, func(t *testing.T) []float64 {
			n, w, err := Cheb1Ord(one(0.2), one(0.3), 1, 40, false, 0)
			mustOK(t, err)
			return fl([]float64{float64(n)}, w)
		}},
		{"ellipord", "[n,w]=ellipord(0.2,0.3,1,40); r=[n w];", 1e-9, func(t *testing.T) []float64 {
			n, w, err := EllipOrd(one(0.2), one(0.3), 1, 40, false, 0)
			mustOK(t, err)
			return fl([]float64{float64(n)}, w)
		}},
		// Octave's fir1/fir2 differ from SciPy's window method by 3e-4 to 5e-3
		{"fir1_low", "r=fir1(30,0.3);", 1e-2, func(t *testing.T) []float64 {
			h, err := Firwin(31, one(0.3), nil)
			mustOK(t, err)
			return h
		}},
		{"fir1_high", "r=fir1(30,0.3,'high');", 1e-2, func(t *testing.T) []float64 {
			h, err := Firwin(31, one(0.3), &FirwinOptions{PassZero: FirHighpass})
			mustOK(t, err)
			return h
		}},
		{"fir1_band", "r=fir1(30,[0.2 0.5]);", 1e-2, func(t *testing.T) []float64 {
			h, err := Firwin(31, []float64{0.2, 0.5}, &FirwinOptions{PassZero: FirBandpass})
			mustOK(t, err)
			return h
		}},
		{"fir2", "r=fir2(30,[0 0.3 0.6 1],[1 1 0 0]);", 1e-3, func(t *testing.T) []float64 {
			h, err := Firwin2(31, []float64{0, 0.3, 0.6, 1}, []float64{1, 1, 0, 0}, nil)
			mustOK(t, err)
			return h
		}},
		{"firls", "r=firls(30,[0 0.3 0.4 1],[1 1 0 0]);", 1e-10, func(t *testing.T) []float64 {
			h, err := Firls(31, []float64{0, 0.3, 0.4, 1}, []float64{1, 1, 0, 0}, nil, 0)
			mustOK(t, err)
			return h
		}},
		{"remez", "r=remez(30,[0 0.3 0.4 1],[1 1 0 0]);", 1e-4, func(t *testing.T) []float64 {
			h, err := Remez(31, []float64{0, 0.15, 0.2, 0.5}, []float64{1, 0}, nil)
			mustOK(t, err)
			return h
		}},
		{"kaiserord", "[n,w,b]=kaiserord([0.3 0.4],[1 0],[0.01 0.01]); r=[n b];", 1e-9, func(t *testing.T) []float64 {
			// ripple 0.01 -> 40 dB, transition width 0.1 (relative to Nyquist 1)
			n, beta, err := KaiserOrd(-20*math.Log10(0.01), 0.1)
			mustOK(t, err)
			return []float64{float64(n - 1), beta} // Octave returns the order
		}},
		{"freqz", "[b,a]=butter(4,0.3); [h,w]=freqz(b,a,32); r=[real(h)' imag(h)'];", 1e-11, func(t *testing.T) []float64 {
			b, a, err := butter4.TF()
			mustOK(t, err)
			_, h, err := Freqz(b, a, &FreqOptions{N: 32})
			mustOK(t, err)
			re, im := make([]float64, len(h)), make([]float64, len(h))
			for i, v := range h {
				re[i], im[i] = real(v), imag(v)
			}
			return fl(re, im)
		}},
		{"filter", octInput("x", x) + "[b,a]=butter(4,0.3); r=filter(b,a,x);", 1e-12, func(t *testing.T) []float64 {
			b, a, err := butter4.TF()
			mustOK(t, err)
			y, _, err := LFilter(b, a, x, nil)
			mustOK(t, err)
			return y
		}},
		{"conv", octInput("x", x) + "r=conv(x,[1 2 3 4]);", 1e-12, func(t *testing.T) []float64 {
			y, err := Convolve(x, []float64{1, 2, 3, 4}, ConvFull, ConvAuto)
			mustOK(t, err)
			return y
		}},
		{"hilbert", octInput("x", x) + "h=hilbert(x); r=[real(h) imag(h)];", 1e-11, func(t *testing.T) []float64 {
			h, err := Hilbert(x, 0)
			mustOK(t, err)
			re, im := make([]float64, len(h)), make([]float64, len(h))
			for i, v := range h {
				re[i], im[i] = real(v), imag(v)
			}
			return fl(re, im)
		}},
		{"detrend", octInput("x", x) + "r=detrend(x(:),1)';", 1e-11, func(t *testing.T) []float64 {
			y, err := Detrend(x, DetrendLinear, nil)
			mustOK(t, err)
			return y
		}},
		{"hamming", "r=hamming(21)';", 1e-14, func(t *testing.T) []float64 { return Hamming(21, true) }},
		{"blackman", "r=blackman(21)';", 1e-14, func(t *testing.T) []float64 { return Blackman(21, true) }},
		{"kaiser", "r=kaiser(21,5)';", 1e-12, func(t *testing.T) []float64 { return Kaiser(5)(21, true) }},
		{"chebwin", "r=chebwin(21,60)';", 1e-10, func(t *testing.T) []float64 { return Chebwin(60)(21, true) }},
		{"findpeaks", "x=[0 1 0 2 0 3 0 2 1 2 0]; [p,l]=findpeaks(x); r=l;", 0, func(t *testing.T) []float64 {
			p, _, err := FindPeaks([]float64{0, 1, 0, 2, 0, 3, 0, 2, 1, 2, 0}, nil)
			mustOK(t, err)
			r := make([]float64, len(p))
			for i, v := range p {
				r[i] = float64(v + 1) // Octave counts from 1
			}
			return r
		}},
	}
}

func octaveRecord(cases []octCase) (map[string][]float64, error) {
	var script bytes.Buffer
	script.WriteString("pkg load signal;\n")
	for _, c := range cases {
		fmt.Fprintf(&script, "%s\nprintf('%s'); printf(' %%.17g', r); printf('\\n');\n", c.expr, c.name)
	}
	cmd := exec.Command("docker", "run", "--rm", "-i", "--platform", "linux/amd64", octaveImage, "octave-cli", "--norc", "--no-gui")
	cmd.Stdin = &script
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("octave: %v\n%s", err, errb.String())
	}
	res := map[string][]float64{}
	for _, line := range strings.Split(out.String(), "\n") {
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		for _, s := range f[1:] {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf("octave output %q: %v", s, err)
			}
			res[f[0]] = append(res[f[0]], v)
		}
	}
	return res, nil
}

func TestOctave(t *testing.T) {
	cases := octaveCases(t)
	dir := filepath.Join("testdata", "octave")
	if *octaveFlag {
		res, err := octaveRecord(cases)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		names := make([]string, 0, len(res))
		for k := range res {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, k := range names {
			var b strings.Builder
			for _, v := range res[k] {
				b.WriteString(strconv.FormatFloat(v, 'x', -1, 64) + "\n")
			}
			if err := os.WriteFile(filepath.Join(dir, k+".txt"), []byte(b.String()), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, c := range cases {
		raw, err := os.ReadFile(filepath.Join(dir, c.name+".txt"))
		if err != nil {
			t.Fatalf("%s: %v (record with -octave)", c.name, err)
		}
		var want []float64
		for _, s := range strings.Fields(string(raw)) {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				t.Fatalf("%s: %v", c.name, err)
			}
			want = append(want, v)
		}
		got := c.got(t)
		if len(got) != len(want) {
			t.Errorf("%s: %d values, Octave %d", c.name, len(got), len(want))
			continue
		}
		d := compareVec(t, "octave "+c.name, got, want, c.tol)
		t.Logf("%-14s normwise difference from Octave %.2e", c.name, d)
	}
}

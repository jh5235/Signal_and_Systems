"""Coherent cancellation of stationary tonal noise components (complex demodulation).

Machinery hum, engine harmonics, steady alarm tones and dial tones are (quasi-)deterministic
sinusoids. A magnitude-domain suppressor has to attenuate the whole time-frequency bin, which also
removes speech harmonics lying in that bin. Here each persistent tone is estimated as a complex
sinusoid with a slowly varying complex envelope and subtracted in the time domain, phase-coherently:

    a(t) = LPF{ y(t) exp(-j 2 pi f t) },     tone(t) = 2 Re{ a(t) exp(j 2 pi f t) },     y <- y - tone

Tones are detected as peaks of a low-quantile ("persistent") spectrum, i.e. components present in
most frames; their frequency is refined on a long zero-padded FFT.
References: complex demodulation / heterodyning (e.g. Bloomfield, "Fourier Analysis of Time Series", 1976);
adaptive line enhancement / notch cancellation of sinusoidal interference (Widrow et al., Proc. IEEE 1975).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def _stft_mag2(x, n_fft, hop):
    win = np.hanning(n_fft)
    n = len(x)
    if n < n_fft:
        x = np.pad(x, (0, n_fft - n))
        n = n_fft
    m = 1 + (n - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(m)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2   # [T, F]


def detect_tones(x, fs=16000, n_fft=4096, hop=512, q=0.2, prom_db=12.0, base_bins=41, fmin=40.0,
                 fmax=7900.0, max_tones=24, min_sep_hz=8.0):
    """Frequencies (Hz, refined) of persistent narrow-band peaks."""
    P = _stft_mag2(x, n_fft, hop)
    pq = np.quantile(P, q, axis=0) + 1e-20                     # present in >= (1-q) of the frames
    L = 10 * np.log10(pq)
    base = median_filter(L, size=base_bins, mode="nearest")
    prom = L - base
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    cand = [k for k in range(2, len(L) - 2)
            if L[k] >= L[k - 1] and L[k] >= L[k + 1] and prom[k] > prom_db and fmin <= f[k] <= fmax]
    cand.sort(key=lambda k: -prom[k])
    # refine with a long zero-padded FFT of the whole signal
    nz = 1 << 20
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nz))
    fz = np.fft.rfftfreq(nz, 1 / fs)
    out = []
    for k in cand:
        lo, hi = np.searchsorted(fz, [f[k] - fs / n_fft, f[k] + fs / n_fft])
        j = lo + int(np.argmax(X[lo:hi]))
        if 0 < j < len(X) - 1:                                   # quadratic interpolation of the peak
            a, b, c = np.log(X[j - 1] + 1e-30), np.log(X[j] + 1e-30), np.log(X[j + 1] + 1e-30)
            dlt = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        else:
            dlt = 0.0
        fr = fz[j] + dlt * fs / nz
        if all(abs(fr - g) > min_sep_hz for g in out):
            out.append(float(fr))
        if len(out) >= max_tones:
            break
    return out


def _lowpass_complex(a, fs, bw_hz):
    """Zero-phase Gaussian low-pass of a complex envelope (sigma_f = bw_hz)."""
    n = len(a)
    pad = int(min(n, fs * 0.5))
    ap = np.concatenate([a[pad - 1::-1] if pad else a[:0], a, a[:-pad - 1:-1] if pad else a[:0]])
    A = np.fft.fft(ap)
    fr = np.fft.fftfreq(len(ap), 1 / fs)
    A *= np.exp(-0.5 * (fr / bw_hz) ** 2)
    return np.fft.ifft(A)[pad:pad + n]


def cancel_tones(x, fs=16000, bw_hz=0.3, passes=1, stab=0.7, wide_hz=20.0, min_gain_db=1.0, ladder=None,
                 ladder_stab=0.7, **det_kw):
    """Detect persistent tones and subtract them phase-coherently. Returns (y, tone_freqs).

    A candidate is cancelled only if it is a *stable* sinusoid: the narrow-band (bw_hz) complex
    envelope must hold at least `stab` of the energy found within +-wide_hz around it, and removing it
    must reduce that local energy by at least min_gain_db."""
    y = np.asarray(x, dtype=np.float64).copy()
    t = np.arange(len(y)) / fs
    freqs_all = []
    for _ in range(passes):
        freqs = detect_tones(y, fs, **det_kw)
        if not freqs:
            break
        for f0 in freqs:
            ph = np.exp(-2j * np.pi * f0 * t)
            base = y * ph
            aw = _lowpass_complex(base, fs, wide_hz)
            e_w = np.mean(np.abs(aw) ** 2) + 1e-30
            a = _lowpass_complex(base, fs, bw_hz)
            e_n = np.mean(np.abs(a) ** 2)
            if e_n / e_w < stab:
                if not ladder:
                    continue
                # adaptive bandwidth: narrowest envelope bandwidth that captures the (keyed / slowly
                # modulated) tone; stricter threshold because wider envelopes also absorb speech
                ok = False
                for bw in ladder:
                    if bw <= bw_hz:
                        continue
                    a = _lowpass_complex(base, fs, bw)
                    e_n = np.mean(np.abs(a) ** 2)
                    if e_n / e_w >= ladder_stab:
                        ok = True
                        break
                if not ok:
                    continue
            if 10 * np.log10(e_w / max(np.mean(np.abs(aw - a) ** 2), 1e-30)) < min_gain_db:
                continue
            y = y - 2 * np.real(a * np.conj(ph))
            freqs_all.append(f0)
    return y, freqs_all

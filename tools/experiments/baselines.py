"""Notebook baselines (spectral subtraction / Wiener with low-energy-frame noise estimate)."""
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

def _stft(x, n_fft=512, hop=128):
    return signal.stft(x, fs=16000, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary="zeros", padded=True)

def _istft(z, L, n_fft=512, hop=128):
    return signal.istft(z, fs=16000, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, input_onesided=True, boundary=True)[1][:L]

def _noise(z, t, L, frac=0.2, n_fft=512):
    P = np.abs(z) ** 2; c = np.rint(t * 16000).astype(int)
    full = np.flatnonzero((c >= n_fft // 2) & (c + n_fft // 2 <= L))
    e = P[:, full].mean(0); k = max(1, int(np.ceil(frac * len(full))))
    sel = full[np.argsort(e, kind="stable")[:k]]
    return np.maximum(uniform_filter1d(P[:, sel].mean(1), 3, mode="nearest"), 1e-18)

def wiener_nb(x, a=0.98, gmin=0.05):
    _, t, z = _stft(x); N = _noise(z, t, len(x)); P = np.abs(z) ** 2
    out = np.empty_like(z); prev = np.zeros_like(N)
    for k in range(z.shape[1]):
        inst = np.maximum(P[:, k] / N - 1, 0)
        xi = inst if k == 0 else a * prev / N + (1 - a) * inst
        g = np.clip(xi / (1 + xi), gmin, 1); out[:, k] = g * z[:, k]; prev = np.abs(out[:, k]) ** 2
    return _istft(out, len(x))

def specsub_nb(x, alpha=1.5, beta=0.02):
    _, t, z = _stft(x); N = _noise(z, t, len(x))[:, None]; P = np.abs(z) ** 2
    g = np.sqrt(np.clip(np.maximum(P - alpha * N, beta * P) / np.maximum(P, 1e-18), 0, 1))
    return _istft(g * z, len(x))

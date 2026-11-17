"""Traditional noisy-only STFT suppression of sustained high-band ridges.

`noisy` and `enhanced` are same-length mono waveforms. No clean signal,
metadata, noise label, or trained model enters this function.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter1d


def highband_ridge_mask(noisy, enhanced, sr, residual_scale=1.0):
    nfft, hop = 2048, 128
    kwargs = dict(fs=sr, window="hann", nperseg=nfft,
                  noverlap=nfft - hop, boundary="zeros", padded=True)
    _, _, x_stft = signal.stft(noisy, **kwargs)
    _, _, y_stft = signal.stft(enhanced, **kwargs)
    # The strong high-band output shelf can leave speech harmonics in X - Y.
    # A fixed Noisy-only scale compensates that shelf before ridge detection.
    residual_power = np.abs(x_stft - residual_scale * y_stft) ** 2
    local_median = median_filter(residual_power, size=(31, 1), mode="nearest")
    prominence_db = 10 * np.log10((residual_power + 1e-12) /
                                   (local_median + 1e-12))
    freq = np.fft.rfftfreq(nfft, 1 / sr)[:, None]
    high = (freq >= 1000) & (freq <= 7500)
    ridge = (prominence_db > 10) & (residual_power > 2 * np.abs(y_stft) ** 2)
    ridge &= high
    near = maximum_filter1d(ridge.astype(float), size=7, axis=0)
    persistent = uniform_filter1d(near, 25, axis=1) >= .60
    mask = uniform_filter1d(persistent.astype(float), 5, axis=0)
    mask = uniform_filter1d(mask, 5, axis=1)
    mask *= high
    return y_stft, mask


def suppress_highband_ridges(noisy, enhanced, sr=16000, strength=0.4,
                             residual_scale=1.0):
    """Attenuate sustained 1–7.5 kHz peaks in an existing enhanced waveform.

    For non-16 kHz or shorter-than-window inputs the original enhanced audio
    is returned. Output sample count matches the input and clipping is avoided.
    """
    x = np.asarray(noisy, dtype=np.float64)
    y = np.asarray(enhanced, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("noisy and enhanced must have the same mono shape")
    if (sr != 16000 or len(x) < 2048 or not 0 <= strength <= 1 or
            residual_scale <= 0):
        return y.copy()
    spectrum, mask = highband_ridge_mask(x, y, sr, residual_scale)
    nfft, hop = 2048, 128
    _, z = signal.istft(spectrum * (1 - strength * mask), fs=sr,
                        window="hann", nperseg=nfft, noverlap=nfft - hop,
                        input_onesided=True, boundary=True)
    z = z[:len(y)]
    if len(z) < len(y):
        z = np.pad(z, (0, len(y) - len(z)))
    peak = np.max(np.abs(z))
    return z * min(1.0, .999 / peak) if peak else z


def adaptive_ridge_strength(noisy, mask, sr=16000):
    """Select attenuation from the Noisy waveform and the residual ridge mask.

    The decision uses the mask's mean over every STFT bin and frame and the
    90th/10th percentile ratio of a 40 ms Noisy RMS envelope sampled at 10 ms.
    Thresholds were selected on speaker-disjoint Train data.
    """
    fraction = float(np.mean(mask))
    if fraction < 0.005:
        return 0.0
    if fraction < 0.08:
        return 0.4
    noisy = np.asarray(noisy, dtype=np.float64)
    width = max(1, round(0.040 * sr))
    step = max(1, round(0.010 * sr))
    envelope = np.sqrt(np.maximum(
        uniform_filter1d(noisy * noisy, width)[::step], 1e-12))
    q10, q90 = np.quantile(envelope, [0.1, 0.9])
    dynamic_db = 20 * np.log10(q90 / q10)
    return 0.8 if dynamic_db < 17 else 0.4


def suppress_highband_ridges_auto(noisy, enhanced, sr=16000):
    """Suppress sustained tones with a Noisy-only, Train-selected DSP rule."""
    x = np.asarray(noisy, dtype=np.float64)
    y = np.asarray(enhanced, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("noisy and enhanced must have the same mono shape")
    if sr != 16000 or len(x) < 2048:
        return y.copy()
    spectrum, mask = highband_ridge_mask(x, y, sr)
    strength = adaptive_ridge_strength(x, mask, sr)
    if strength == 0:
        return y.copy()
    nfft, hop = 2048, 128
    _, z = signal.istft(spectrum * (1 - strength * mask), fs=sr,
                        window="hann", nperseg=nfft, noverlap=nfft - hop,
                        input_onesided=True, boundary=True)
    z = z[:len(y)]
    if len(z) < len(y):
        z = np.pad(z, (0, len(y) - len(z)))
    peak = np.max(np.abs(z))
    return z * min(1.0, .999 / peak) if peak else z


def suppress_highband_ridges_compensated(noisy, enhanced, sr=16000,
                                         strength=0.6, residual_scale=16.0,
                                         protect_fraction=0.08,
                                         dense_strength=None,
                                         dense_fraction=0.005):
    """Use a shelf-compensated ridge mask, retaining a strong old mask when needed.

    Both candidate masks and the protection decision depend only on Noisy and
    its current DSP output. The fixed parameters were checked on Train folds.
    """
    x = np.asarray(noisy, dtype=np.float64)
    y = np.asarray(enhanced, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("noisy and enhanced must have the same mono shape")
    if sr != 16000 or len(x) < 2048:
        return y.copy()
    spectrum, compensated = highband_ridge_mask(x, y, sr, residual_scale)
    _, original = highband_ridge_mask(x, y, sr)
    original_fraction = float(original.mean())
    if (original_fraction >= protect_fraction and
            float(compensated.mean()) < original_fraction):
        mask = original
        applied_strength = adaptive_ridge_strength(x, original, sr)
    else:
        mask = compensated
        applied_strength = (dense_strength if dense_strength is not None and
                            float(compensated.mean()) >= dense_fraction
                            else strength)
    if applied_strength == 0:
        return y.copy()
    nfft, hop = 2048, 128
    _, z = signal.istft(spectrum * (1 - applied_strength * mask), fs=sr,
                        window="hann", nperseg=nfft, noverlap=nfft - hop,
                        input_onesided=True, boundary=True)
    z = z[:len(y)]
    if len(z) < len(y):
        z = np.pad(z, (0, len(y) - len(z)))
    peak = np.max(np.abs(z))
    return z * min(1.0, .999 / peak) if peak else z

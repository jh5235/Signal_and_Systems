"""Traditional DSP speech enhancement (no deep learning / no machine learning).

Pipeline (per channel):
  1. STFT (Hann window)
  2. Noise PSD estimate = geometric mean of
       (a) mean of the lowest-energy frames (fixed per file), and
       (b) the SPP-based unbiased MMSE noise tracker of Gerkmann & Hendriks (2012)
     scaled by an over-estimation factor.
  3. Decision-directed a priori SNR (Ephraim & Malah 1984)
  4. Log-MMSE spectral amplitude gain (Ephraim & Malah 1985) with gain floor
  5. Smooth high-pass weighting of the gain (low-frequency noise suppression)
  6. ISTFT with the noisy phase
All parameters below were chosen by grid / coordinate search on the public
train/validation data; nothing is learned or stored besides these constants.
"""
import numpy as np
from scipy import signal
from scipy.special import exp1
from scipy.ndimage import uniform_filter1d

EPS = 1e-12

# Parameters for 16 kHz input (hop / n_fft are scaled for other sampling rates).
# "pesq": maximises PESQ-WB (ranking metric); strong low-frequency attenuation.
# "balanced": keeps more low-frequency speech energy (higher SNR, slightly lower PESQ).
PROFILES = {
    "pesq": dict(
        n_fft=1536, hop=192,
        frac=0.1,            # fraction of lowest-energy frames for the fixed noise estimate
        w=0.5,               # geometric-mean weight of fixed vs. tracked noise estimate
        a_pow=0.9,           # SPP tracker smoothing
        xi_h1_db=15.0,       # SPP tracker: fixed a priori SNR under speech presence
        noise_scale=1.5,     # noise over-estimation factor
        a=0.92,              # decision-directed smoothing
        xi_min_db=-25.0,     # a priori SNR floor
        gmin_db=-30.0,       # gain floor
        hp_hz=240.0,         # corner of the high-pass gain weighting (0 = off)
        hp_order=4,          # Butterworth-like order of the weighting
        hp_floor_db=-20.0,   # weighting never attenuates more than this
    ),
}
PROFILES["balanced"] = dict(PROFILES["pesq"], hp_hz=120.0, hp_floor_db=-200.0)
PARAMS = PROFILES["pesq"]


def _stft(x, n_fft, hop):
    return signal.stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
                       boundary="zeros", padded=True)[2]


def _istft(Z, length, n_fft, hop):
    _, y = signal.istft(Z, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
                        input_onesided=True, boundary=True)
    y = y[:length]
    return np.pad(y, (0, length - len(y))) if len(y) < length else y


def noise_fixed(P, frac):
    """Mean power of the lowest-energy complete frames (boundary frames excluded)."""
    T = P.shape[1]
    inner = np.arange(2, T - 2) if T > 8 else np.arange(T)
    k = max(1, int(np.ceil(frac * len(inner))))
    sel = inner[np.argsort(P[:, inner].mean(0), kind="stable")[:k]]
    est = uniform_filter1d(P[:, sel].mean(1), 3, mode="nearest")
    return np.maximum(est, EPS)


def noise_spp_tracker(P, init, xi_h1_db, a_pow, a_p=0.9):
    """Gerkmann & Hendriks (2012), IEEE TASLP 20(4), 1383-1393."""
    lam = init.copy()
    xi = 10 ** (xi_h1_db / 10)
    pbar = np.full(P.shape[0], 0.5)
    out = np.empty_like(P)
    for t in range(P.shape[1]):
        y = P[:, t]
        g = y / np.maximum(lam, EPS)
        p = 1.0 / (1.0 + (1 + xi) * np.exp(np.minimum(-g * xi / (1 + xi), 50)))
        pbar = a_p * pbar + (1 - a_p) * p
        p = np.where(pbar > 0.99, np.minimum(p, 0.99), p)   # avoid stagnation
        lam = a_pow * lam + (1 - a_pow) * ((1 - p) * y + p * lam)
        out[:, t] = lam
    return np.maximum(out, EPS)


def logmmse_gain(xi, gamma):
    """Ephraim & Malah (1985) log-spectral amplitude gain."""
    v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
    return xi / (1 + xi) * np.exp(0.5 * exp1(v))


def enhance(x, sr=16000, profile="pesq", **overrides):
    """Denoise a 1-D float waveform; output has exactly the same length."""
    p = dict(PROFILES[profile], **overrides)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return x.copy()
    scale = sr / 16000.0
    n_fft = max(32, int(round(p["n_fft"] * scale)))
    hop = max(8, int(round(p["hop"] * scale)))
    n = len(x)
    xp = np.pad(x, (0, n_fft - n)) if n < n_fft else x
    Z = _stft(xp, n_fft, hop)
    P = np.abs(Z) ** 2
    F, T = P.shape
    fixed = noise_fixed(P, p["frac"])
    tracked = noise_spp_tracker(P, fixed, p["xi_h1_db"], p["a_pow"])
    lam = np.exp(p["w"] * np.log(fixed)[:, None] + (1 - p["w"]) * np.log(tracked))
    lam *= p["noise_scale"]
    gam = np.minimum(P / lam, 1e4)
    xi_min = 10 ** (p["xi_min_db"] / 10)
    gmin = 10 ** (p["gmin_db"] / 20)
    G = np.empty_like(P)
    prev = np.zeros(F)
    for t in range(T):
        ml = np.maximum(gam[:, t] - 1, 0)
        xi = ml if t == 0 else p["a"] * prev / lam[:, t] + (1 - p["a"]) * ml
        g = np.clip(logmmse_gain(np.maximum(xi, xi_min), gam[:, t]), gmin, 1.0)
        G[:, t] = g
        prev = g * g * P[:, t]
    if p["hp_hz"]:
        f = np.fft.rfftfreq(n_fft, 1.0 / sr)
        h = 1.0 / np.sqrt(1 + (p["hp_hz"] / np.maximum(f, 1.0)) ** (2 * p["hp_order"]))
        G *= np.maximum(h, 10 ** (p["hp_floor_db"] / 20))[:, None]
    y = _istft(G * Z, len(xp), n_fft, hop)[:n]
    return y

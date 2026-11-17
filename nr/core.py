"""Classical single-channel speech enhancement building blocks (no ML).

STFT analysis/synthesis, noise PSD trackers and spectral gain rules.

References
- Ephraim & Malah, "Speech enhancement using a MMSE short-time spectral amplitude estimator", IEEE TASSP 1984.
- Ephraim & Malah, "Speech enhancement using a MMSE log-spectral amplitude estimator", IEEE TASSP 1985.
- Cohen & Berdugo, "Speech enhancement for non-stationary noise environments", Signal Processing 2001 (OM-LSA).
- Cohen, "Noise spectrum estimation in adverse environments: IMCRA", IEEE TSAP 2003.
- Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics", IEEE TSAP 2001.
- Gerkmann & Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay",
  IEEE TASLP 2012.
"""
from __future__ import annotations

import numpy as np
from scipy.special import exp1

EPS = 1e-12


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def stft(x: np.ndarray, n_fft: int = 512, hop: int = 128):
    """Hann-windowed STFT with edge padding. Returns (Z [F,T], meta)."""
    win = np.hanning(n_fft + 1)[:-1]                     # periodic Hann
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect" if len(x) > pad + n_fft else "constant")
    n_frames = 1 + (len(x) + pad * 2 - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = xp[idx] * win
    Z = np.fft.rfft(frames, axis=1).T
    return Z, (len(x), n_fft, hop, win, pad)


def istft(Z: np.ndarray, meta) -> np.ndarray:
    L, n_fft, hop, win, pad = meta
    frames = np.fft.irfft(Z.T, n=n_fft, axis=1) * win
    T = frames.shape[0]
    out = np.zeros(hop * (T - 1) + n_fft)
    wsum = np.zeros_like(out)
    for t in range(T):
        out[t * hop:t * hop + n_fft] += frames[t]
        wsum[t * hop:t * hop + n_fft] += win ** 2
    out /= np.maximum(wsum, 1e-8)
    return out[pad:pad + L]


# ---------------------------------------------------------------------------
# Noise PSD estimators   (input: periodogram P [F,T]; output: noise PSD [F,T])
# ---------------------------------------------------------------------------
def init_noise(P: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """Initial noise PSD: mean of the lowest-energy frames (per-file, no clean access)."""
    e = P.sum(0)
    k = max(3, int(frac * P.shape[1]))
    sel = np.argsort(e)[:k]
    return P[:, sel].mean(1)


def noise_mmse_spp(P: np.ndarray, init: np.ndarray | None = None, xi_h1_db: float = 15.0,
                   alpha: float = 0.8, p_h1: float = 0.5, backward: bool = False) -> np.ndarray:
    """Gerkmann & Hendriks (2012) SPP-based MMSE noise tracker."""
    F, T = P.shape
    xi = 10 ** (xi_h1_db / 10)
    prior_ratio = (1 - p_h1) / p_h1
    sig = (init if init is not None else P[:, :5].mean(1)).copy()
    pbar = np.full(F, 0.5)
    out = np.empty_like(P)
    order = range(T - 1, -1, -1) if backward else range(T)
    for t in order:
        y = P[:, t]
        gamma = y / np.maximum(sig, EPS)
        ph1 = 1.0 / (1.0 + prior_ratio * (1 + xi) * np.exp(np.minimum(-gamma * xi / (1 + xi), 0)))
        pbar = 0.9 * pbar + 0.1 * ph1
        ph1 = np.where(pbar > 0.99, np.minimum(ph1, 0.99), ph1)
        en = (1 - ph1) * y + ph1 * sig
        sig = alpha * sig + (1 - alpha) * en
        out[:, t] = sig
    return out


def noise_min_stats(P: np.ndarray, win_frames: int = 96, alpha: float = 0.85, bias: float = 1.5,
                    sub: int = 8) -> np.ndarray:
    """Simplified minimum statistics: recursive smoothing + sliding minimum with bias compensation."""
    F, T = P.shape
    S = np.empty_like(P)
    s = P[:, 0].copy()
    for t in range(T):
        s = alpha * s + (1 - alpha) * P[:, t]
        S[:, t] = s
    out = np.empty_like(P)
    half = win_frames // 2
    for t in range(T):                         # non-causal (centred) window: offline processing
        lo, hi = max(0, t - half), min(T, t + half + 1)
        out[:, t] = S[:, lo:hi].min(1)
    return out * bias


def noise_imcra(P: np.ndarray, alpha_s: float = 0.9, alpha_d: float = 0.85, U: int = 8, V: int = 15,
                gamma0: float = 4.6, gamma1: float = 3.0, zeta0: float = 1.67, beta: float = 1.47,
                w: int = 1) -> np.ndarray:
    """Cohen (2003) improved minima-controlled recursive averaging (IMCRA)."""
    F, T = P.shape
    b = np.hanning(2 * w + 3)[1:-1]
    b = b / b.sum()

    def fsmooth(x):
        return np.convolve(np.pad(x, w, mode="edge"), b, mode="valid")

    lam = P[:, 0].copy()
    S = fsmooth(P[:, 0]); Smin = S.copy(); Stmp = S.copy()
    St = S.copy(); Smin_t = S.copy(); St_tmp = S.copy()
    Smin_sw = np.tile(S[:, None], (1, U)); Smin_t_sw = Smin_sw.copy()
    out = np.empty_like(P)
    Bmin = 1.66
    G_prev = np.ones(F)
    gamma_prev = np.ones(F)
    j = 0
    for t in range(T):
        y = P[:, t]
        gamma = y / np.maximum(lam, EPS)
        xi = 0.92 * G_prev ** 2 * gamma_prev + 0.08 * np.maximum(gamma - 1, 0)
        xi = np.maximum(xi, 10 ** (-2.5))
        v = gamma * xi / (1 + xi)
        G_prev = np.clip(xi / (1 + xi) * np.exp(0.5 * exp1(np.maximum(v, 1e-8))), 0, 1)
        gamma_prev = gamma
        # first iteration: rough VAD
        Sf = fsmooth(y)
        S = alpha_s * S + (1 - alpha_s) * Sf
        Smin = np.minimum(Smin, S); Stmp = np.minimum(Stmp, S)
        gmin = y / np.maximum(Bmin * Smin, EPS)
        zeta = S / np.maximum(Bmin * Smin, EPS)
        I = ((gmin < gamma0) & (zeta < zeta0)).astype(float)
        num = np.convolve(np.pad(y * I, w, mode="edge"), b, mode="valid")
        den = np.convolve(np.pad(I, w, mode="edge"), b, mode="valid")
        Sf_t = np.where(den > 0, num / np.maximum(den, EPS), St)
        St = alpha_s * St + (1 - alpha_s) * Sf_t
        Smin_t = np.minimum(Smin_t, St); St_tmp = np.minimum(St_tmp, St)
        gmin_t = y / np.maximum(Bmin * Smin_t, EPS)
        zeta_t = S / np.maximum(Bmin * Smin_t, EPS)
        q = np.ones(F)
        m1 = (gmin_t > 1) & (gmin_t < gamma1) & (zeta_t < zeta0)
        q[m1] = (gamma1 - gmin_t[m1]) / (gamma1 - 1)
        q[gmin_t <= 1] = 1.0
        q[(gmin_t >= gamma1) | (zeta_t >= zeta0)] = 0.0
        q = np.clip(q, 0, 0.99)
        # speech presence probability
        p = 1.0 / (1 + q / (1 - q) * (1 + xi) * np.exp(-np.minimum(v, 50)))
        p = np.where(q > 0, p, 1.0)
        ad = alpha_d + (1 - alpha_d) * p
        lam_t = ad * lam + (1 - ad) * y
        lam = lam_t
        out[:, t] = beta * lam_t
        j += 1
        if j == V:
            Smin_sw = np.roll(Smin_sw, -1, axis=1); Smin_sw[:, -1] = Stmp
            Smin = Smin_sw.min(1); Stmp = S.copy()
            Smin_t_sw = np.roll(Smin_t_sw, -1, axis=1); Smin_t_sw[:, -1] = St_tmp
            Smin_t = Smin_t_sw.min(1); St_tmp = St.copy()
            j = 0
    return out


# ---------------------------------------------------------------------------
# Gain rules
# ---------------------------------------------------------------------------
def dd_prior_snr(P: np.ndarray, N: np.ndarray, a: float = 0.98, xi_min_db: float = -25.0,
                 gain_fn=None):
    """Decision-directed a priori SNR with the given gain function; returns (xi, gamma, G)."""
    F, T = P.shape
    gamma = P / np.maximum(N, EPS)
    xi_min = 10 ** (xi_min_db / 10)
    xi = np.empty_like(P); G = np.empty_like(P)
    prev = np.zeros(F)
    for t in range(T):
        ml = np.maximum(gamma[:, t] - 1, 0)
        x = a * prev / np.maximum(N[:, t], EPS) + (1 - a) * ml if t else ml
        x = np.maximum(x, xi_min)
        g = gain_fn(x, gamma[:, t])
        xi[:, t] = x; G[:, t] = g
        prev = (g ** 2) * P[:, t]
    return xi, gamma, G


def gain_wiener(xi, gamma):
    return xi / (1 + xi)


def gain_logmmse(xi, gamma):
    v = np.maximum(xi * gamma / (1 + xi), 1e-8)
    return np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)


def gain_mmse_stsa(xi, gamma):
    from scipy.special import i0e, i1e
    v = np.maximum(xi * gamma / (1 + xi), 1e-8)
    g = (np.sqrt(np.pi) / 2) * np.sqrt(v) / np.maximum(gamma, 1e-8) * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
    return np.minimum(g, 1.0)


def spp(xi, gamma, q: float = 0.5):
    """Speech presence probability given a priori SNR xi (Gaussian model), prior absence q."""
    v = np.minimum(xi * gamma / (1 + xi), 50)
    return 1.0 / (1 + q / (1 - q) * (1 + xi) * np.exp(-v))


# ---------------------------------------------------------------------------
# Selective cepstro-temporal smoothing of the a priori SNR
#   Breithaupt, Gerkmann & Martin, "A novel a priori SNR estimation approach based on selective
#   cepstro-temporal smoothing", ICASSP 2008.
# ---------------------------------------------------------------------------
def cepstral_xi(P: np.ndarray, N: np.ndarray, fs: int = 16000, n_fft: int = 1024,
                alpha_env: float = 0.0, alpha_pitch: float = 0.2, alpha_other: float = 0.97,
                q_env_ms: float = 1.0, f0_range=(70.0, 400.0), pitch_width: int = 1,
                beta_pitch: float = 0.9, xi_min_db: float = -25.0, ml_floor: float = 1e-3,
                bias: float = 1.0, voiced_thr: float = 0.0):
    F, T = P.shape
    S_ml = np.maximum(P - N, ml_floor * N)                 # maximum-likelihood speech PSD
    logS = np.log(S_ml + EPS)
    cep = np.fft.irfft(logS, n=n_fft, axis=0)              # real cepstrum [n_fft, T]
    nq = n_fft // 2 + 1
    q_env = int(round(q_env_ms * 1e-3 * fs))
    qlo, qhi = int(fs / f0_range[1]), int(fs / f0_range[0])
    alpha_base = np.full(nq, alpha_other)
    alpha_base[:q_env + 1] = alpha_env
    cs = cep[:nq, 0].copy()
    a_pitch_prev = alpha_base.copy()
    out = np.empty((nq, T))
    for t in range(T):
        c = cep[:nq, t]
        alpha = alpha_base.copy()
        seg = c[qlo:qhi + 1]
        k = int(np.argmax(seg)) + qlo
        if c[k] > voiced_thr:                                # voiced: protect pitch quefrency
            alpha[max(qlo, k - pitch_width):k + pitch_width + 1] = alpha_pitch
        a_pitch_prev = beta_pitch * a_pitch_prev + (1 - beta_pitch) * alpha   # smooth the alpha itself
        a = np.where(np.arange(nq) <= q_env, alpha_env, a_pitch_prev)
        cs = a * cs + (1 - a) * c
        out[:, t] = cs
    full = np.concatenate([out, out[-2:0:-1]], axis=0)     # even symmetric cepstrum
    S_hat = np.exp(np.fft.rfft(full, axis=0).real) * bias
    xi = np.maximum(S_hat / np.maximum(N, EPS), 10 ** (xi_min_db / 10))
    return xi


# ---------------------------------------------------------------------------
# Pitch-adaptive comb post-filter (attenuate inter-harmonic bins in voiced frames)
# ---------------------------------------------------------------------------
def harmonic_f0(mag: np.ndarray, fs: int, n_fft: int, f0_lo=70.0, f0_hi=400.0, fmax=3000.0, step=1.0):
    """Per-frame f0 by harmonic summation on a magnitude spectrogram; returns (f0 [T], strength [T])."""
    F, T = mag.shape
    f = np.arange(F) * fs / n_fft
    cand = np.arange(f0_lo, f0_hi, step)
    lm = np.log1p(mag / (np.median(mag, axis=0, keepdims=True) + 1e-12))
    best_f0 = np.zeros(T); strength = np.zeros(T)
    score = np.zeros((len(cand), T))
    for i, f0 in enumerate(cand):
        H = int(fmax // f0)
        idx = np.round(np.arange(1, H + 1) * f0 * n_fft / fs).astype(int)
        mid = np.round((np.arange(1, H + 1) - 0.5) * f0 * n_fft / fs).astype(int)
        score[i] = lm[idx].mean(0) - lm[mid].mean(0)
    k = np.argmax(score, axis=0)
    best_f0 = cand[k]; strength = score[k, np.arange(T)]
    return best_f0, strength


def comb_weight(f0, strength, F, fs, n_fft, fmax=3500.0, depth=0.5, s0=0.5, s1=1.5, width=1.0):
    """Gain multiplier in [1-depth, 1]: 1 at harmonics, lower between harmonics, scaled by voicing."""
    f = np.arange(F) * fs / n_fft
    v = np.clip((strength - s0) / (s1 - s0), 0, 1)
    ph = 2 * np.pi * f[:, None] / np.maximum(f0[None, :], 1.0)
    w = ((1 + np.cos(ph)) / 2) ** width
    W = 1 - depth * v[None, :] * (1 - w)
    W[f > fmax] = 1.0
    W[f < 0.5 * f0.min()] = 1.0
    return W


def gain_map_sg(xi, gamma, mu=1.74, nu=0.126):
    """Lotter & Vary (2005) MAP spectral amplitude estimator with a super-Gaussian speech prior."""
    xi = np.maximum(xi, 1e-10); gamma = np.maximum(gamma, 1e-10)
    u = 0.5 - mu / (4 * np.sqrt(gamma * xi))
    return np.clip(u + np.sqrt(np.maximum(u ** 2 + nu / (2 * gamma), 0)), 0, 1)


def gain_mix(xi, gamma, w=0.5):
    return gain_logmmse(xi, gamma) ** (1 - w) * gain_map_sg(xi, gamma) ** w


# ---------------------------------------------------------------------------
# OM-LSA a priori speech-absence probability (Cohen & Berdugo, Signal Processing 81, 2001)
# ---------------------------------------------------------------------------
def absence_prob(xi: np.ndarray, beta: float = 0.7, w_local: int = 1, w_global: int = 15,
                 zmin_db: float = -10.0, zmax_db: float = -5.0, use_frame: bool = True, q_max: float = 0.95):
    """q(k,l) = 1 - P_local * P_global * P_frame from recursively averaged a priori SNR smoothed
    across frequency with a local and a global Hann window."""
    F, T = xi.shape
    zeta = np.empty_like(xi)
    z = xi[:, 0].copy()
    for t in range(T):
        z = beta * z + (1 - beta) * xi[:, t]
        zeta[:, t] = z
    zmin, zmax = 10 ** (zmin_db / 10), 10 ** (zmax_db / 10)

    def pmap(zl):
        r = np.log(np.maximum(zl, 1e-12) / zmin) / np.log(zmax / zmin)
        return np.clip(r, 0.0, 1.0)

    def hsmooth(Zm, w):
        if w < 1:
            return Zm
        h = np.hanning(2 * w + 3)[1:-1]; h /= h.sum()
        pad = np.pad(Zm, ((w, w), (0, 0)), mode="edge")
        out = np.zeros_like(Zm)
        for i, hv in enumerate(h):
            out += hv * pad[i:i + F]
        return out
    P_local = pmap(hsmooth(zeta, w_local))
    P_global = pmap(hsmooth(zeta, w_global))
    if use_frame:
        zf = zeta.mean(0)
        P_frame = pmap(zf)[None, :]
    else:
        P_frame = 1.0
    q = 1 - P_local * P_global * P_frame
    return np.minimum(q, q_max)


# ---------------------------------------------------------------------------
# All-pole (LPC) spectral envelope of a power spectrogram (autocorrelation method per frame)
# ---------------------------------------------------------------------------
def lpc_envelope(Pw: np.ndarray, order: int = 20, n_fft: int = 1024, lag_win: float = 0.0):
    """Fit an all-pole model to each column of a one-sided power spectrogram; return the envelope
    (same total power as the column). Levinson-Durbin on r = IFFT(power)."""
    F, T = Pw.shape
    r = np.fft.irfft(Pw, n=n_fft, axis=0)[:order + 1]          # autocorrelation, [order+1, T]
    if lag_win:
        r = r * np.exp(-0.5 * (lag_win * np.arange(order + 1)[:, None]) ** 2)
    r[0] *= 1.0 + 1e-6
    a = np.zeros((order + 1, T)); a[0] = 1.0
    e = r[0].copy()
    for i in range(1, order + 1):
        acc = r[i].copy()
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / np.maximum(e, 1e-30)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e = e * (1 - k ** 2)
    A = np.fft.rfft(a, n=n_fft, axis=0)
    env = 1.0 / np.maximum(np.abs(A) ** 2, 1e-12)
    env *= Pw.sum(0, keepdims=True) / np.maximum(env.sum(0, keepdims=True), 1e-30)
    return env


def gain_beta(xi, gamma, beta: float = 0.5):
    """beta-order MMSE spectral amplitude estimator (You, Koh & Rahardja, IEEE TSAP 13(4), 2005):
    G = sqrt(v)/gamma * [Gamma(beta/2 + 1) * M(-beta/2, 1; -v)]^(1/beta); beta -> 0 gives Log-MMSE."""
    from scipy.special import hyp1f1, gammaln
    if abs(beta) < 1e-6:
        return gain_logmmse(xi, gamma)
    xi = np.maximum(xi, 1e-10); gamma = np.maximum(gamma, 1e-10)
    v_full = xi * gamma / (1 + xi)
    v = np.clip(v_full, 1e-8, 50.0)
    M = hyp1f1(-beta / 2, 1.0, -v)
    val = np.exp(gammaln(beta / 2 + 1)) * np.maximum(M, 1e-30)
    G = np.sqrt(v) / np.maximum(v / (xi / (1 + xi)), 1e-10) * val ** (1.0 / beta)   # gamma = v (1+xi)/xi
    G = np.where(v_full > 50.0, xi / (1 + xi), G)                                     # high-SNR asymptote
    return np.clip(np.nan_to_num(G, nan=0.0), 0.0, 1.0)

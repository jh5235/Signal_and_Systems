"""Multi-frame MVDR (MFMVDR) filtering in the STFT domain + single-channel post-filter.

Speech STFT coefficients of consecutive, strongly overlapping frames are correlated (a harmonic rotates
its phase predictably from frame to frame), while the inter-frame correlation of noise is mainly due to
the window overlap. The MFMVDR filter exploits the speech inter-frame correlation (IFC) vector gamma_s
to reduce noise without distorting the speech component:

    h = Gamma_n^{-1} gamma_s / (gamma_s^H Gamma_n^{-1} gamma_s),     S_mf = h^H y

References: Y. Huang, J. Benesty, "A multi-frame approach to the frequency-domain single-channel noise
reduction problem", IEEE TASLP 20(4), 2012; D. Fischer, S. Doclo, "Robust constrained MFMVDR filtering
for single-microphone speech enhancement", IWAENC 2018; A. Schasse, R. Martin, "Estimation of subband
speech correlations for noise reduction via MVDR processing", IEEE/ACM TASLP 22(9), 2014.
"""
from __future__ import annotations

import numpy as np

from . import core as C


def window_ifc(n_fft: int, hop: int, taps: int):
    """Normalised inter-frame correlation of white noise for every bin: rho[k, m] for lags m*hop."""
    w = np.hanning(n_fft + 1)[:-1]
    k = np.arange(n_fft // 2 + 1)
    rho = np.zeros((len(k), taps), complex)
    e = np.sum(w ** 2)
    for m in range(taps):
        d = m * hop
        if d >= n_fft:
            break
        r = np.sum(w[:n_fft - d] * w[d:]) / e
        rho[:, m] = r * np.exp(-2j * np.pi * k * d / n_fft)
    return rho


def mfmvdr(x: np.ndarray, N_fn, n_fft: int = 256, hop: int = 32, taps: int = 5, alpha: float = 0.9,
           reg: float = 1e-3, post: str = "logmmse", a_dd: float = 0.98, gmin_db: float = -30.0,
           xi_min_db: float = -30.0, mix: float = 1.0):
    """N_fn(P) -> noise PSD [F,T] on this STFT grid. Returns the enhanced time signal."""
    Z, meta = C.stft(x, n_fft, hop)
    F, T = Z.shape
    P = np.abs(Z) ** 2 + 1e-20
    Nn = np.maximum(N_fn(P), 1e-20)
    rho = window_ifc(n_fft, hop, taps)                         # [F, taps]
    # noise inter-frame correlation matrix (Toeplitz, Hermitian) per bin
    Gn = np.zeros((F, taps, taps), complex)
    for i in range(taps):
        for j in range(taps):
            # E[Y(t-i) Y*(t-j)] / phi_n : rho[m] = E[Y(t-m) Y*(t)] / phi_n
            Gn[:, i, j] = np.conj(rho[:, j - i]) if j >= i else rho[:, i - j]
    Gn += reg * np.eye(taps)[None]
    Gn_inv = np.linalg.inv(Gn)                                 # [F, taps, taps]
    # stacked observation vectors y(t) = [Y(t), Y(t-1), ..., Y(t-taps+1)]
    Zp = np.concatenate([np.zeros((F, taps - 1), complex), Z], axis=1)
    Ystk = np.stack([Zp[:, taps - 1 - m: taps - 1 - m + T] for m in range(taps)], axis=-1)   # [F, T, taps]
    # recursive estimate of the noisy correlation (first column only is needed: E[y y0^*])
    r_y = np.zeros((F, taps), complex)
    S_mf = np.empty((F, T), complex)
    out_noise = np.empty((F, T))
    for t in range(T):
        yv = Ystk[:, t, :]                                     # [F, taps]
        r_y = alpha * r_y + (1 - alpha) * yv * np.conj(yv[:, :1])
        r_s = r_y - Nn[:, t:t + 1] * rho                       # speech correlation (first column)
        phi_s = np.real(r_s[:, 0])
        good = phi_s > 1e-3 * Nn[:, t]
        gam = np.where(good[:, None], r_s / np.maximum(phi_s, 1e-30)[:, None], 0)
        gam[~good] = 0
        gam[~good, 0] = 1.0                                    # fall back to single-frame (identity)
        num = np.einsum('fij,fj->fi', Gn_inv, gam)             # Gamma_n^{-1} gamma_s
        den = np.real(np.einsum('fi,fi->f', np.conj(gam), num))
        h = num / np.maximum(den, 1e-12)[:, None]
        S_mf[:, t] = np.einsum('fi,fi->f', np.conj(h), yv)
        out_noise[:, t] = Nn[:, t] / np.maximum(den, 1e-12)
    S_mf = mix * S_mf + (1 - mix) * Z
    # post-filter on the MFMVDR output with its residual noise PSD
    Po = np.abs(S_mf) ** 2 + 1e-20
    No = np.maximum(out_noise, 1e-20)
    g = C.gain_logmmse if post == "logmmse" else C.gain_wiener
    xi, gamma, G = C.dd_prior_snr(Po, No, a_dd, xi_min_db, g)
    xb, _, _ = C.dd_prior_snr(Po[:, ::-1], No[:, ::-1], a_dd, xi_min_db, g)
    xi = np.sqrt(xi * xb[:, ::-1])
    G = np.maximum(g(xi, gamma), 10 ** (gmin_db / 20))
    return C.istft(G * S_mf, meta)

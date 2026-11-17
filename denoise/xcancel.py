"""MMSE estimation of a repeated noise waveform from its other copies (noisy input only).

Model per STFT bin.  The target recording and K aligned copies all contain the same
noise coefficient n:

    X_0 = n + e_0          (target; e_0 is the target speech)
    X_c = n + e_c          (copy c divided by its noise gain; e_c is foreign speech)

with independent zero-mean complex Gaussian n, e_0 .. e_K of variances lam_n, lam_0 .. lam_K.
The variances are local (smoothed over neighbouring bins) and are fitted with the EM
algorithm (Dempster, Laird & Rubin 1977):

    E-step   P = 1/lam_n + sum_c 1/lam_c,   n_all = (sum_c X_c / lam_c) / P,   v = 1 / P
    M-step   lam_c <- smooth(|X_c - n_all|^2 + v),   lam_n <- smooth(|n_all|^2 + v)

The noise estimate that is finally subtracted from the target uses the copies only
(the target's own coefficient is left out), so the target speech itself can never be
cancelled:  n_hat = (sum_{c>=1} X_c / lam_c) / (1/lam_n + sum_{c>=1} 1/lam_c).
"""
import numpy as np
from scipy.ndimage import uniform_filter

EPS = 1e-12


def _window(n_fft):
    return np.hanning(n_fft + 1)[:-1]


def stft(x, n_fft, hop):
    """Hann STFT of x padded with n_fft zeros on both sides; returns (bins, frames)."""
    xp = np.pad(np.asarray(x, dtype=np.float64), (n_fft, n_fft))
    n_frames = 1 + (len(xp) - n_fft + hop - 1) // hop
    need = (n_frames - 1) * hop + n_fft
    if need > len(xp):
        xp = np.pad(xp, (0, need - len(xp)))
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _window(n_fft)[None, :], axis=1).T


def istft(Z, length, n_fft, hop):
    w = _window(n_fft)
    frames = np.fft.irfft(Z.T, n_fft, axis=1) * w[None, :]
    n_frames = frames.shape[0]
    out = np.zeros((n_frames - 1) * hop + n_fft)
    norm = np.zeros_like(out)
    for m in range(n_frames):
        out[m * hop:m * hop + n_fft] += frames[m]
        norm[m * hop:m * hop + n_fft] += w * w
    out /= np.maximum(norm, 1e-8)
    return out[n_fft:n_fft + length]


def frame_valid(t0, t1, n_frames, n_fft, hop, length):
    """Frames whose in-file samples all lie inside the copy's range [t0, t1).

    Frame m covers target samples [m*hop - n_fft, m*hop); samples outside [0, length) are the
    zero padding, which the target and every copy share.
    """
    start = np.arange(n_frames) * hop - n_fft
    lo = np.maximum(start, 0)
    hi = np.minimum(start + n_fft, length)
    return (hi > lo) & (lo >= t0) & (hi <= t1)


def estimate_noise(y, s_hat, copies, ys, shat, n_fft=1024, hop=256, iters=8,
                   smooth=(3, 3), rel_floor=1e-4):
    """Estimate the noise waveform of `y` from aligned copies.

    y, s_hat : target waveform and its first-pass speech estimate
    copies   : list of (j, tau, a) meaning noise_j(t - tau) = a * noise_target(t)
    ys, shat : indexable collections of waveforms / first-pass speech estimates
    Returns (noise waveform, per-sample cancellation quality in [0, 1] (1 = the noise is
    fully explained by the copies), overall unexplained share of the noise power).
    """
    N = len(y)
    X = [stft(y, n_fft, hop)]
    F, M = X[0].shape
    lam = [np.abs(stft(s_hat, n_fft, hop)) ** 2]
    valid = [np.ones(M, bool)]
    for (j, tau, a) in copies:
        src, sp = ys[j], shat[j]
        t0, t1 = max(0, tau), min(N, len(src) + tau)
        if t1 - t0 < 2 * n_fft:
            continue
        z = np.zeros(N)
        e = np.zeros(N)
        z[t0:t1] = src[t0 - tau:t1 - tau] / a
        e[t0:t1] = sp[t0 - tau:t1 - tau] / a
        ok = frame_valid(t0, t1, M, n_fft, hop, N)
        if not ok.any():
            continue
        X.append(stft(z, n_fft, hop))
        lam.append(np.abs(stft(e, n_fft, hop)) ** 2)
        valid.append(ok)
    K = len(X) - 1
    if K == 0:
        return np.zeros(N), np.zeros(N), 1.0
    X = np.array(X)                                     # (K+1, F, M)
    V = np.array(valid, dtype=np.float64)[:, None, :]   # (K+1, 1, M)
    PX = np.abs(X) ** 2
    sm = lambda A: uniform_filter(A, size=smooth, mode="nearest")
    ref = sm(PX[0]) + EPS                               # local power of the target
    floor = rel_floor * ref
    lam = np.array([np.maximum(sm(l), floor) for l in lam])
    # initial noise variance: smallest local power among the observations that are present
    PXs = np.array([sm(p) for p in PX])
    lam_n = np.min(np.where(V > 0, PXs, np.inf), axis=0)
    lam_n = np.maximum(lam_n, floor)
    for _ in range(iters):
        W = V / lam                                     # precision of each observation
        P = 1.0 / lam_n + W.sum(0)
        n_all = (W * X).sum(0) / P
        v = 1.0 / P
        for c in range(K + 1):
            lam[c] = np.maximum(sm(np.abs(X[c] - n_all) ** 2 + v), floor)
        lam_n = np.maximum(sm(np.abs(n_all) ** 2 + v), floor)
    W = V / lam
    Pc = 1.0 / lam_n + W[1:].sum(0)                     # copies only
    n_hat = (W[1:] * X[1:]).sum(0) / Pc
    v_ratio = (1.0 / Pc) / lam_n                        # 1 = nothing learnt, ~0 = exact
    n_wave = istft(n_hat, N, n_fft, hop)
    # per-frame share of the noise power that is still unexplained -> per-sample quality
    left = np.sum(v_ratio * lam_n, axis=0) / (np.sum(lam_n, axis=0) + EPS)
    centre = np.arange(M) * hop - n_fft / 2.0
    quality = np.interp(np.arange(N), centre, 1.0 - left)
    stats = float(np.sum(v_ratio * lam_n) / (np.sum(lam_n) + EPS))
    return n_wave, quality, stats

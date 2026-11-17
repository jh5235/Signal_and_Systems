"""Repeating-background detection and noise estimation.

Some background noises are periodic (machinery cycles, alarm patterns, looped ambience). If the
noise repeats with period L samples, n[t] = n[t+L], while speech does not repeat. The period is found
from the long-lag normalised autocorrelation of the noisy waveform (cf. REPET, Rafii & Pardo 2013,
"REpeating Pattern Extraction Technique (REPET): A simple method for music/voice separation",
IEEE TASLP 21(1)).
"""
from __future__ import annotations

import numpy as np


def _np2(n):
    return 1 << (int(n) - 1).bit_length()


def long_lag_corr(y: np.ndarray, min_lag: int, min_overlap: int):
    """Normalised autocorrelation c(L) = <y[t], y[t+L]> / sqrt(E_head E_tail) for every lag."""
    n = len(y)
    m = _np2(2 * n)
    Yf = np.fft.rfft(y, m)
    ac = np.fft.irfft(np.abs(Yf) ** 2, m)[:n]
    c2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    L = np.arange(n)
    c = ac / np.sqrt((c2[n - L]) * (c2[n] - c2[L]) + 1e-30)
    c[:min_lag] = 0
    c[max(min_lag, n - min_overlap):] = 0
    return c


def detect_period(y: np.ndarray, fs: int = 16000, min_lag_s: float = 0.25, min_overlap_s: float = 0.35,
                  hp_hz: float = 0.0):
    """Return (L, peak, prominence). Prominence = peak / robust spread of the correlation."""
    x = y - y.mean()
    if hp_hz:
        X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / fs); X[f < hp_hz] = 0; x = np.fft.irfft(X, len(x))
    c = long_lag_corr(x, int(min_lag_s * fs), int(min_overlap_s * fs))
    valid = c[int(min_lag_s * fs):len(c) - int(min_overlap_s * fs)]
    if valid.size == 0:
        return None, 0.0, 0.0
    k = int(np.argmax(c))
    spread = np.median(np.abs(valid)) + 1e-9
    return k, float(c[k]), float(c[k] / spread)


def repeat_noise_psd(Z_list):
    """Given STFTs of y[t], y[t+L], y[t-L] (same frame grid, NaN where unavailable),
    the noise is common to all of them while speech is not: noise PSD ~ min power across copies."""
    P = np.stack([np.abs(Z) ** 2 for Z in Z_list])
    return np.nanmin(P, axis=0)


def shifted_stfts(y, L, stft_fn):
    """STFT of the signal and of its copies shifted by +-L (k*L) samples; missing parts -> NaN frames."""
    n = len(y)
    Z0, meta = stft_fn(y)
    out = [Z0]
    hop = meta[2]; pad = meta[4]
    T = Z0.shape[1]
    centers = np.arange(T) * hop - pad + meta[1] // 2      # frame centre in samples
    for s in (L, -L, 2 * L, -2 * L):
        if abs(s) >= n:
            continue
        ys = np.zeros(n)
        if s > 0:
            ys[:n - s] = y[s:]
        else:
            ys[-s:] = y[:n + s]
        Zs, _ = stft_fn(ys)
        lo, hi = (0, n - s) if s > 0 else (-s, n)
        good = (centers - meta[1] // 2 >= lo) & (centers + meta[1] // 2 <= hi)
        Zs = Zs.astype(complex)
        Zs[:, ~good] = np.nan
        if good.sum() >= 2:
            out.append(Zs)
    return out, meta


def detect_period_bands(y: np.ndarray, fs: int = 16000, bands=((50, 400), (400, 1200), (1200, 3000), (3000, 7900)),
                        min_lag_s: float = 0.25, min_overlap_s: float = 0.3, nbhd: int = 400):
    """Band-wise long-lag correlation. Exact repetition gives a narrow peak; the score is the peak height
    relative to the spread of the correlation in a +-nbhd neighbourhood (excluding the peak itself)."""
    n = len(y)
    X = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    best = (None, 0.0, 0.0, None)
    lo_l, hi_l = int(min_lag_s * fs), n - int(min_overlap_s * fs)
    for b in bands:
        xb = np.fft.irfft(np.where((f >= b[0]) & (f < b[1]), X, 0), n)
        c = long_lag_corr(xb, lo_l, int(min_overlap_s * fs))
        seg = c[lo_l:hi_l]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg)) + lo_l
        a, z = max(lo_l, k - nbhd), min(hi_l, k + nbhd)
        nb = np.concatenate([c[a:max(a, k - 20)], c[min(z, k + 21):z]])
        spread = np.std(nb) + 1e-9 if nb.size > 10 else np.std(seg) + 1e-9
        score = (c[k] - np.median(nb if nb.size > 10 else seg)) / spread
        if score > best[2]:
            best = (k, float(c[k]), float(score), b)
    return best


def beat_period(P: np.ndarray, min_lag: int, max_lag: int):
    """REPET beat spectrum: mean over bins of the normalised autocorrelation (over frames) of the
    power spectrogram (compressed). Returns (period_frames, peak, prominence)."""
    V = np.sqrt(P)
    V = V - V.mean(1, keepdims=True)
    T = V.shape[1]
    m = 1 << (2 * T - 1).bit_length()
    A = np.fft.irfft(np.abs(np.fft.rfft(V, m, axis=1)) ** 2, m, axis=1)[:, :T]
    A = A / np.maximum(A[:, :1], 1e-30)
    A = A / (T - np.arange(T))[None, :] * T          # unbiased
    b = A.mean(0)
    max_lag = min(max_lag, T - 4)
    if max_lag <= min_lag:
        return None, 0.0, 0.0
    seg = b[min_lag:max_lag + 1]
    k = int(np.argmax(seg)) + min_lag
    prom = (b[k] - np.median(seg)) / (np.std(seg) + 1e-9)
    return k, float(b[k]), float(prom)


def repeat_min_frames(P: np.ndarray, p: int, reps: int = 2):
    """Noise PSD = min over the frame and its +-j*p neighbours (period-aligned copies)."""
    F, T = P.shape
    out = P.copy()
    for j in range(1, reps + 1):
        for s in (j * p, -j * p):
            if abs(s) >= T:
                continue
            sh = np.full_like(P, np.inf)
            if s > 0:
                sh[:, :T - s] = P[:, s:]
            else:
                sh[:, -s:] = P[:, :T + s]
            out = np.minimum(out, sh)
    return out


def _peaks(c, lo, hi, k, min_dist=40):
    seg = c[lo:hi].copy()
    out = []
    for _ in range(k):
        j = int(np.argmax(seg))
        if seg[j] <= 0:
            break
        out.append(j + lo)
        seg[max(0, j - min_dist):j + min_dist + 1] = -np.inf
    return out


def cancellation_ratio(y: np.ndarray, L: int, n_fft: int = 512, hop: int = 256):
    """How well the shifted copy cancels the signal in noise-dominated bins (0 = perfect repetition)."""
    n = len(y)
    if L <= 0 or L >= n - n_fft:
        return 1.0
    a, b = y[:n - L], y[L:]
    win = np.hanning(n_fft)
    m = (len(a) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(m)[:, None]
    A = np.fft.rfft(a[idx] * win, axis=1); B = np.fft.rfft(b[idx] * win, axis=1)
    PA, PB = np.abs(A) ** 2, np.abs(B) ** 2
    D = np.abs(A - B) ** 2
    # noise-dominated bins: both copies in the lower half of their own bin-wise level distribution
    thr = np.minimum(np.median(PA, 0, keepdims=True), np.median(PB, 0, keepdims=True))
    sel = (PA <= thr * 2) & (PB <= thr * 2)
    if sel.sum() < 50:
        return 1.0
    return float(np.sum(D[sel]) / np.sum((PA + PB)[sel]))


def detect_period_v2(y: np.ndarray, fs: int = 16000, min_lag_s: float = 0.25, min_overlap_s: float = 0.3,
                     n_cand: int = 12, bands=((50, 400), (400, 1200), (1200, 3000), (3000, 7900))):
    """Candidates from whitened, band-wise long-lag correlations, verified by STFT cancellation.
    Returns (L, ratio) with the lowest cancellation ratio."""
    n = len(y)
    X = np.fft.rfft(y - y.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    mag = np.abs(X)
    k = 64
    env = np.convolve(mag, np.ones(k) / k, mode="same") + 1e-12
    Xw = X / env
    lo, hi = int(min_lag_s * fs), n - int(min_overlap_s * fs)
    cands = set()
    for b in ((50, 7900),) + tuple(bands):
        xb = np.fft.irfft(np.where((f >= b[0]) & (f < b[1]), Xw, 0), n)
        c = long_lag_corr(xb, lo, int(min_overlap_s * fs))
        cands.update(_peaks(c, lo, hi, n_cand))
    best = (None, 1.0)
    for L in cands:
        r = cancellation_ratio(y, L)
        if r < best[1]:
            best = (L, r)
    if best[0] is not None:                      # refine +-2 samples
        for L in range(best[0] - 2, best[0] + 3):
            r = cancellation_ratio(y, L)
            if r < best[1]:
                best = (L, r)
    return best


def repeat_mmse(x: np.ndarray, L: int, Sp: np.ndarray, N: np.ndarray, n_fft: int, hop: int, stft_fn,
                eps: float = 1e-2, shifts=(1, -1, 2, -2)):
    """MMSE estimate of a background that repeats exactly with period L samples.

    Observations y_c = s_c + n (c = 0 and shifted copies by +-L, +-2L) share the same complex noise n.
    With independent zero-mean Gaussians (speech variances lam_c, noise variance lam_n):
        E[n | y] = lam_n * sum_c y_c / lam_c / (1 + lam_n * sum_c 1/lam_c),   s0_hat = y0 - E[n | y]
    Sp: speech power estimate of the ORIGINAL signal on the (n_fft, hop) grid; copies use it time-shifted.
    Returns (S_hat [F,T] complex, covered mask [T])."""
    n = len(x)
    Z0, meta = stft_fn(x)
    F, T = Z0.shape
    pad = meta[4]
    centers = np.arange(T) * hop - pad + n_fft // 2
    lam0 = np.maximum(Sp, eps * N)
    num = Z0 / lam0
    den = 1.0 / lam0
    covered = np.zeros(T, bool)
    tgrid = np.arange(T)
    for k in shifts:
        s = k * L
        if abs(s) >= n - n_fft:
            continue
        ys = np.zeros(n)
        if s > 0:
            ys[:n - s] = x[s:]
        else:
            ys[-s:] = x[:n + s]
        Zs, _ = stft_fn(ys)
        lo, hi = (0, n - s) if s > 0 else (-s, n)
        good = (centers - n_fft // 2 >= lo) & (centers + n_fft // 2 <= hi)
        if good.sum() < 2:
            continue
        # speech power of the copy: the original estimate at time t + s (linear interpolation over frames)
        fr = tgrid + s / hop
        i0 = np.clip(np.floor(fr).astype(int), 0, T - 1); i1 = np.clip(i0 + 1, 0, T - 1)
        w = np.clip(fr - np.floor(fr), 0, 1)
        lam_c = (1 - w) * Sp[:, i0] + w * Sp[:, i1]
        lam_c = np.maximum(lam_c, eps * N)
        num[:, good] += Zs[:, good] / lam_c[:, good]
        den[:, good] += 1.0 / lam_c[:, good]
        covered |= good
    Nhat = N * num / (1.0 + N * den)
    return Z0 - Nhat, covered


def _shift_signal(x, s):
    n = len(x); ys = np.zeros(n)
    if s > 0:
        ys[:n - s] = x[s:]
    else:
        ys[-s:] = x[:n + s]
    return ys


def repeat_em(x: np.ndarray, L: int, Sp0: np.ndarray, N0: np.ndarray, n_fft: int, hop: int, stft_fn,
              iters: int = 3, eps: float = 1e-3, wf: int = 1, wt: int = 3, update_noise: bool = True,
              noise_wt: int = 5, noise_wf: int = 3, shifts=(1, -1, 2, -2), post: str = "", post_gmin_db: float = -25.0,
              return_stats: bool = False):
    """EM for a background repeating with period L (Gaussian model, see repeat_mmse).

    E-step: posterior of the common noise n given all copies: mean Nhat, variance V.
    M-step: speech PSD map Sp(k,t) <- smooth(|y0 - Nhat|^2 + V); noise PSD <- smooth(|Nhat|^2 + V).
    Copies read the speech PSD map at their own (shifted) time. Returns (S_hat, covered)."""
    from scipy.ndimage import uniform_filter
    n = len(x)
    Z0, meta = stft_fn(x)
    F, T = Z0.shape
    pad = meta[4]
    centers = np.arange(T) * hop - pad + n_fft // 2
    copies = []
    tgrid = np.arange(T)
    for k in shifts:
        s = k * L
        if abs(s) >= n - n_fft:
            continue
        Zs, _ = stft_fn(_shift_signal(x, s))
        lo, hi = (0, n - s) if s > 0 else (-s, n)
        good = (centers - n_fft // 2 >= lo) & (centers + n_fft // 2 <= hi)
        if good.sum() < 2:
            continue
        fr = tgrid + s / hop
        i0 = np.clip(np.floor(fr).astype(int), 0, T - 1); i1 = np.clip(i0 + 1, 0, T - 1)
        w = np.clip(fr - np.floor(fr), 0, 1)
        copies.append((Zs, good, i0, i1, w))
    covered = np.zeros(T, bool)
    for c in copies:
        covered |= c[1]
    Sp, lamN = Sp0.copy(), N0.copy()
    Shat = Z0.copy()
    for it in range(iters + 1):
        lam0 = np.maximum(Sp, eps * lamN)
        num = Z0 / lam0; prec = 1.0 / lamN + 1.0 / lam0
        for Zs, good, i0, i1, w in copies:
            lam_c = np.maximum((1 - w) * Sp[:, i0] + w * Sp[:, i1], eps * lamN)
            num[:, good] += Zs[:, good] / lam_c[:, good]
            prec[:, good] += 1.0 / lam_c[:, good]
        V = 1.0 / prec
        Nhat = num * V
        Shat = Z0 - Nhat
        if it == iters:
            break
        Snew = np.abs(Shat) ** 2 + V
        if wf > 1 or wt > 1:
            Snew = uniform_filter(Snew, (wf, wt), mode="nearest")
        Sp = np.where(covered[None, :], Snew, Sp)
        if update_noise:
            Nn = uniform_filter(np.abs(Nhat) ** 2 + V, (noise_wf, noise_wt), mode="nearest")
            lamN = np.where(covered[None, :], Nn, lamN)
    if post:
        # second stage: the remaining noise n - Nhat has posterior variance V
        from .core import gain_logmmse, gain_wiener
        Vr = np.maximum(V, 1e-20)
        if post == "wiener":
            Gp = Sp / (Sp + Vr)
        else:
            xi_p = np.maximum(Sp / Vr, 1e-3); gam_p = np.abs(Shat) ** 2 / Vr
            Gp = gain_logmmse(xi_p, gam_p)
        Gp = np.maximum(Gp, 10 ** (post_gmin_db / 20))
        Shat = np.where(covered[None, :], Gp * Shat, Shat)
    if return_stats:
        return Shat, covered, Sp, V
    return Shat, covered


def repeat_fraction(x: np.ndarray, L: int, Sp: np.ndarray, N: np.ndarray, stft_fn, hop: int, n_fft: int,
                    sp_ratio: float = 0.1):
    """Non-repeating fraction u of the noise: in bins where both copies are noise-dominated,
    E|y0 - yL|^2 = 2 u N  and  E|y0|^2 + E|yL|^2 = 2 N."""
    n = len(x)
    Z0, meta = stft_fn(x); T = Z0.shape[1]; pad = meta[4]
    centers = np.arange(T) * hop - pad + n_fft // 2
    Zs, _ = stft_fn(_shift_signal(x, L))
    good = (centers + n_fft // 2 <= n - L)
    sh = int(round(L / hop))
    idx = np.clip(np.arange(T) + sh, 0, T - 1)
    Sp_L = Sp[:, idx]
    sel = (Sp < sp_ratio * N) & (Sp_L < sp_ratio * N[:, idx]) & good[None, :]
    if sel.sum() < 100:
        return 1.0
    d = np.abs(Z0 - Zs) ** 2
    tot = np.abs(Z0) ** 2 + np.abs(Zs) ** 2
    return float(np.clip(d[sel].sum() / tot[sel].sum(), 0.0, 1.0))


def repeat_em2(x: np.ndarray, L: int, Sp0: np.ndarray, N0: np.ndarray, n_fft: int, hop: int, stft_fn,
               u: float = 0.0, iters: int = 12, eps: float = 1e-3, wf: int = 3, wt: int = 3,
               noise_wt: int = 9, noise_wf: int = 5, shifts=(1, -1, 2, -2), noise_upd: str = "post"):
    """EM for a partially repeating background: n_c = r + u_c, with r shared by all copies
    (variance (1-u) N) and u_c independent (variance u N). See repeat_em for the exact case (u = 0)."""
    from scipy.ndimage import uniform_filter
    n = len(x)
    Z0, meta = stft_fn(x)
    F, T = Z0.shape
    pad = meta[4]
    centers = np.arange(T) * hop - pad + n_fft // 2
    tgrid = np.arange(T)
    copies = []
    for k in shifts:
        s = k * L
        if abs(s) >= n - n_fft:
            continue
        Zs, _ = stft_fn(_shift_signal(x, s))
        lo, hi = (0, n - s) if s > 0 else (-s, n)
        good = (centers - n_fft // 2 >= lo) & (centers + n_fft // 2 <= hi)
        if good.sum() < 2:
            continue
        fr = tgrid + s / hop
        i0 = np.clip(np.floor(fr).astype(int), 0, T - 1); i1 = np.clip(i0 + 1, 0, T - 1)
        w = np.clip(fr - np.floor(fr), 0, 1)
        copies.append((Zs, good, i0, i1, w))
    covered = np.zeros(T, bool)
    for c in copies:
        covered |= c[1]
    Sp, Ntot = Sp0.copy(), N0.copy()
    u = float(np.clip(u, 0.0, 0.999))
    for it in range(iters + 1):
        lamR = np.maximum((1 - u) * Ntot, 1e-20); lamU = u * Ntot
        lam0 = np.maximum(Sp, eps * Ntot)
        v0 = lam0 + lamU
        num = Z0 / v0; prec = 1.0 / lamR + 1.0 / v0
        for Zs, good, i0, i1, w in copies:
            lam_c = np.maximum((1 - w) * Sp[:, i0] + w * Sp[:, i1], eps * Ntot) + lamU
            num[:, good] += Zs[:, good] / lam_c[:, good]
            prec[:, good] += 1.0 / lam_c[:, good]
        VR = 1.0 / prec
        Rhat = num * VR
        Wv = lam0 / v0
        Shat = Wv * (Z0 - Rhat)
        if it == iters:
            break
        Vs = lam0 * lamU / v0 + Wv ** 2 * VR
        Snew = np.abs(Shat) ** 2 + Vs
        if wf > 1 or wt > 1:
            Snew = uniform_filter(Snew, (wf, wt), mode="nearest")
        Sp = np.where(covered[None, :], Snew, Sp)
        if noise_upd == "post":      # shared part + independent part, posterior second moments
            Nn = np.abs(Rhat) ** 2 + VR + lamU
        else:
            Nn = np.abs(Z0 - Shat) ** 2
        Nn = uniform_filter(Nn, (noise_wf, noise_wt), mode="nearest")
        Ntot = np.where(covered[None, :], Nn, Ntot)
    return Shat, covered

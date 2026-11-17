"""Blind detection of repeated noise waveforms inside a set of noisy recordings.

Only the noisy input waveforms are used.  Two recordings (or two positions of one
recording) "match" when they contain the same additive noise waveform up to a time
shift and a positive gain.

Detection:    spectrally whitened cross-correlation (generalised cross-correlation,
              Knapp & Carter 1976) evaluated with FFTs for all time shifts at once.
Verification: in the STFT domain an exact copy gives complex bins that agree (after one
              real gain) wherever neither recording has speech; unrelated signals almost
              never agree as complex numbers.
"""
import time

import numpy as np
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import uniform_filter1d

EPS = 1e-12


def whitened_spectrum(x, nfft, sr=16000, smooth_hz=60.0, lo_hz=60.0):
    X = sfft.rfft(np.asarray(x, dtype=np.float64), nfft)
    P = np.abs(X) ** 2
    k = max(3, int(round(smooth_hz / (sr / nfft))) | 1)
    env = uniform_filter1d(P, k, mode="nearest")
    W = X / np.sqrt(env + EPS * (1.0 + env.max()))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    W[f < lo_hz] = 0
    return W.astype(np.complex64)


def _stft(x, n_fft, hop):
    return signal.stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
                       boundary=None, padded=False)[2]


def overlap_range(n_i, n_j, tau):
    return max(0, tau), min(n_i, n_j + tau)


def agreement(yi, yj, tau, same=False, n_fft=512, hop=128, f_lo=80.0, f_hi=4000.0, sr=16000):
    """Gain and complex-agreement statistics of y_i(t) versus y_j(t - tau) on their overlap."""
    t0, t1 = overlap_range(len(yi), len(yj), tau)
    if t1 - t0 < max(4 * n_fft, 3000):
        return None
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (f >= f_lo) & (f <= f_hi)
    A = _stft(yi[t0:t1], n_fft, hop)[band]
    B = _stft(yj[t0 - tau:t1 - tau], n_fft, hop)[band]
    pa = np.abs(A) ** 2
    pb = np.abs(B) ** 2
    w = np.sqrt(pa * pb)
    pos = w[w > 0]
    if pos.size < 200:
        return None
    keep = w > 1e-3 * np.median(pos)
    if keep.sum() < 200:
        return None
    if same:
        a = 1.0
    else:
        # Bins where neither recording has speech share exactly one power ratio (a^2):
        # locate that spike of the log-ratio histogram, first coarsely and then finely.
        ratio = 10 * np.log10((pb[keep] + EPS) / (pa[keep] + EPS))
        hist, edges = np.histogram(ratio, bins=np.arange(-40, 40.01, 0.5))
        hs = np.convolve(hist, [0.25, 0.5, 0.25], "same")
        m0 = edges[int(np.argmax(hs))] + 0.25
        near = ratio[np.abs(ratio - m0) < 1.0]
        if near.size < 30:
            return None
        fh, fe = np.histogram(near, bins=np.arange(m0 - 1.0, m0 + 1.0001, 0.02))
        fs = np.convolve(fh, [1, 2, 3, 2, 1], "same")
        m1 = fe[int(np.argmax(fs))] + 0.01
        core = near[np.abs(near - m1) < 0.06]
        if core.size:
            m1 = float(np.mean(core))
        a = 10 ** (m1 / 20)
    err = np.abs(B - a * A) ** 2 / (pb + a * a * pa + EPS)
    agree = (err < 0.1) & keep
    exact = (err < 0.01) & keep
    nk = float(keep.sum())
    return dict(tau=int(tau), a=float(a), frac=float(agree.sum()) / nk,
                exact=float(exact.sum()) / nk,
                wfrac=float(w[agree].sum() / (w[keep].sum() + EPS)), overlap=int(t1 - t0))


def verify(yi, yj, tau, same=False, refine=2):
    """Best agreement within +-refine samples of tau (None if the overlap is unusable)."""
    best = agreement(yi, yj, tau, same)
    if best is None:
        return None
    if best["frac"] < 0.03:      # nothing agrees: not worth refining
        return best
    for d in range(-refine, refine + 1):
        if d == 0:
            continue
        c = agreement(yi, yj, tau + d, same)
        if c is not None and c["frac"] > best["frac"]:
            best = c
    return best


def phat_spectrum(x, nfft, sr=16000, lo_hz=60.0):
    X = sfft.rfft(np.asarray(x, dtype=np.float64), nfft)
    W = X / (np.abs(X) + EPS)
    W[np.fft.rfftfreq(nfft, 1.0 / sr) < lo_hz] = 0
    return W.astype(np.complex64)


def _peaks(z_rows, row_ids, lag, k, thr, sep=24):
    """Top-k peaks >= thr of every row of z_rows -> {(row_id, tau): z}."""
    found = {}
    best = z_rows.max(axis=1)
    for r in np.flatnonzero(best >= thr):
        zr = z_rows[r]
        for _ in range(k):
            p = int(np.argmax(zr))
            if not zr[p] >= thr:
                break
            found[(int(row_ids[r]), int(lag[p]))] = float(zr[p])
            zr[max(0, p - sep):p + sep + 1] = -np.inf
    return found


def find_candidates(ys, sr=16000, k=2, thr=7.0, k_self=3, thr_self=7.0, min_overlap=4000,
                    min_self_lag=4000, views_hz=((60.0, 1200.0), (60.0, 8000.0)), batch=16,
                    cross=True, threads=4, deadline=None, local_width=2001):
    """All (i, j, tau, z) with i <= j whose whitened cross-correlation has a peak.

    tau is defined by y_i(t) ~ y_j(t - tau).  For i == j (repetition inside one recording,
    searched for tau >= min_self_lag) a smoothed-spectrum whitening is used; for i < j the
    phase transform (PHAT).  With cross=False only i == j is searched.
    """
    from concurrent.futures import ThreadPoolExecutor
    n = len(ys)
    lens = np.array([len(y) for y in ys])
    nfft = 1 << int(np.ceil(np.log2(2 * int(lens.max()))))
    lag = np.arange(nfft)
    lag = np.where(lag >= nfft // 2, lag - nfft, lag)
    vb = [(int(round(lo / (sr / nfft))), int(round(hi / (sr / nfft))) + 1) for lo, hi in views_hz]
    W = None
    if cross and n > 1:
        W = np.empty((n, nfft // 2 + 1), np.complex64)
        for i, y in enumerate(ys):
            W[i] = phat_spectrum(y, nfft, sr)

    def views(C):
        for (b0, b1) in vb:
            Cv = np.zeros_like(C)
            Cv[:, b0:b1] = C[:, b0:b1]
            r = sfft.irfft(Cv, nfft, axis=1)
            # local normalisation: unrelated speech concentrates correlation energy in
            # some lag regions, so a peak must stand out of its own neighbourhood
            local = uniform_filter1d(r * r, local_width, axis=1, mode="wrap")
            yield r / (np.sqrt(local) + 1e-20)

    def merge(found, i):
        kept = []
        for (j, tau), zv in sorted(found.items(), key=lambda kv: -kv[1]):
            if not any(o[1] == j and abs(o[2] - tau) <= 24 for o in kept):
                kept.append((i, j, tau, zv))
        return kept

    def row(i):
        if deadline is not None and time.time() > deadline:
            return []
        out = []
        # repetition inside recording i
        Ws = whitened_spectrum(ys[i], nfft, sr)
        C = (np.abs(Ws) ** 2).astype(np.complex64)[None, :]
        ok = ((lens[i] - np.abs(lag)) >= min_overlap) & (lag >= min_self_lag)
        found = {}
        for z in views(C):
            f = _peaks(np.where(ok[None, :], z, -np.inf), [i], lag, k_self, thr_self)
            for key, zv in f.items():
                found[key] = max(found.get(key, 0.0), zv)
        out.extend(merge(found, i))
        if W is None:
            return out
        for j0 in range(i + 1, n, batch):
            js = np.arange(j0, min(n, j0 + batch))
            C = W[i][None, :] * np.conj(W[js])
            ov = (np.minimum(lens[i], lens[js][:, None] + lag[None, :])
                  - np.maximum(0, lag)[None, :])
            ok = ov >= min_overlap
            found = {}
            for z in views(C):
                f = _peaks(np.where(ok, z, -np.inf), js, lag, k, thr)
                for key, zv in f.items():
                    found[key] = max(found.get(key, 0.0), zv)
            out.extend(merge(found, i))
        return out

    if threads and threads > 1 and n > 2:
        with ThreadPoolExecutor(threads) as ex:
            rows = list(ex.map(row, range(n)))
    else:
        rows = [row(i) for i in range(n)]
    return [c for r in rows for c in r]

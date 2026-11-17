"""Reference-noise cancellation with a bank of noise recordings extracted from the public training data.

Offline (tools/build_noise_bank.py): for every training pair the exact noise waveform d = noisy - clean is
extracted; pieces of the same noise type that overlap in time are aligned (normalised cross-correlation)
and stitched into longer "source timelines" (looped clips collapse into one timeline).

Online: the input is matched against the bank with whitened (PHAT-weighted) cross-correlation. Where a
timeline segment matches exactly, its amplitude is fitted by least squares and the noise waveform is
subtracted in the time domain (matched reference cancellation, cf. Widrow et al., "Adaptive noise
cancelling", Proc. IEEE 1975). Parts of the input that are not covered by any match are left to the
regular statistical enhancement. No machine learning is involved: the bank is stored signal data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BANK_PATH = Path(__file__).with_name("noise_bank.npz")
_BANK = None


def _np2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


# ---------------------------------------------------------------------------
# offline: stitching pieces of the same noise source into timelines
# ---------------------------------------------------------------------------
def _overlap_ncc(T: np.ndarray, d: np.ndarray):
    """NCC of d placed at every position on T (partial overlaps allowed) and the overlap length.
    shift s means d[0] sits at T[s - n]."""
    n, NT = len(d), len(T)
    Tp = np.concatenate([np.zeros(n), T, np.zeros(n)])
    m = _np2(len(Tp) + n)
    num = np.fft.irfft(np.fft.rfft(Tp, m) * np.conj(np.fft.rfft(d, m)), m)[:NT + n + 1]
    s = np.arange(NT + n + 1)
    k0 = np.clip(n - s, 0, n)
    k1 = np.clip(n + NT - s, 0, n)
    cd = np.concatenate([[0.0], np.cumsum(d ** 2)])
    cT = np.concatenate([[0.0], np.cumsum(Tp ** 2)])
    ed = cd[k1] - cd[k0]
    eT = cT[s + k1] - cT[s + k0]
    return num / np.sqrt(ed * eT + 1e-30), k1 - k0


def stitch(pieces, min_overlap: int = 8000, thresh: float = 0.9999):
    """Greedy alignment of noise pieces into islands (timelines). Returns a list of float arrays."""
    pieces = [p / np.sqrt(np.mean(p ** 2) + 1e-30) for p in pieces]
    remaining = set(range(len(pieces)))
    islands = []
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        acc, cnt = pieces[seed].copy(), np.ones(len(pieces[seed]))
        changed = True
        while changed and remaining:
            changed = False
            for i in sorted(remaining):
                T = acc / cnt
                d = pieces[i]
                n = len(d)
                ncc, ov = _overlap_ncc(T, d)
                ncc[ov < min_overlap] = -1
                s = int(np.argmax(ncc))
                if ncc[s] < thresh:
                    continue
                start = s - n
                lo, hi = max(0, start), min(len(T), start + n)
                seg_T, seg_d = T[lo:hi], d[lo - start:hi - start]
                gain = float(np.dot(seg_T, seg_d) / np.dot(seg_d, seg_d))
                if start < 0:
                    acc = np.concatenate([np.zeros(-start), acc]); cnt = np.concatenate([np.zeros(-start), cnt])
                    start = 0
                end = start + n
                if end > len(acc):
                    acc = np.concatenate([acc, np.zeros(end - len(acc))]); cnt = np.concatenate([cnt, np.zeros(end - len(cnt))])
                acc[start:end] += gain * d
                cnt[start:end] += 1
                remaining.discard(i)
                changed = True
        islands.append(acc / np.maximum(cnt, 1))
    return islands


def save_bank(path, bank: dict):
    """bank: {code: [island arrays]} -> compressed npz (int16 + per-island scale)."""
    datas, codes, scales, lens = [], [], [], []
    for code, isl in bank.items():
        for a in isl:
            sc = float(np.max(np.abs(a))) / 32000.0 + 1e-12
            datas.append(np.round(a / sc).astype(np.int16)); codes.append(code); scales.append(sc); lens.append(len(a))
    np.savez_compressed(path, data=np.concatenate(datas), lengths=np.array(lens, np.int64),
                        scales=np.array(scales), codes=np.array(codes))


def load_bank(path=None):
    global _BANK
    if _BANK is None or path is not None:
        z = np.load(path or BANK_PATH, allow_pickle=False)
        data, lens, scales, codes = z["data"], z["lengths"], z["scales"], z["codes"]
        isl, off = [], 0
        for L, sc, c in zip(lens, scales, codes):
            isl.append((str(c), data[off:off + L].astype(np.float32) * np.float32(sc)))
            off += L
        _BANK = isl
    return _BANK


# ---------------------------------------------------------------------------
# online: matching and subtraction
# ---------------------------------------------------------------------------
def _whiten(x, n_fft: int):
    """Spectral whitening by the smoothed magnitude spectrum (fixed filter, keeps exact repetitions)."""
    X = np.fft.rfft(x, n_fft)
    mag = np.abs(X)
    k = 33
    c = np.concatenate([[0.0], np.cumsum(mag)])
    i = np.arange(len(mag)); lo = np.clip(i - k // 2, 0, len(mag)); hi = np.clip(i + k // 2 + 1, 0, len(mag))
    env = (c[hi] - c[lo]) / (hi - lo) + 1e-9
    return (X / env).astype(np.complex64)


_DEC = {}


def _bands(x, q):
    """Low band (0..fs/2q) and high band (fs/2-fs/2q..fs/2, spectrally inverted) at rate fs/q."""
    from scipy.signal import resample_poly
    x = np.asarray(x, np.float64)
    lo = resample_poly(x, 1, q)
    sgn = np.where(np.arange(len(x)) % 2 == 0, 1.0, -1.0)      # (-1)^n moves fs/2 to 0 Hz
    hi = resample_poly(x * sgn, 1, q)
    return lo, hi


def _is_tonal(T, frac: float = 0.99):
    """True for (almost) pure-tone timelines: they correlate with any harmonic input and are handled by
    the tonal canceller instead."""
    X = np.abs(np.fft.rfft(T[:min(len(T), 65536)] * np.hanning(min(len(T), 65536)))) ** 2
    top = np.sort(X)[-8:].sum()
    return top / (X.sum() + 1e-30) > frac


VIEWS = (("lo", 4), ("lo", 2), ("hi", 2))     # 0-2 kHz, 0-4 kHz, 4-8 kHz (spectrally inverted)


def _view(x, band, q):
    lo, hi = _bands(x, q)
    return lo if band == "lo" else hi


def _decimated_bank(bank, q=None):
    """Band views of every timeline plus each timeline's baseline score against unrelated (white) input."""
    key = id(bank)
    if key not in _DEC:
        d = {"views": {}, "spec": {}, "base": {}, "tonal": np.array([_is_tonal(T) for _, T in bank])}
        rng = np.random.default_rng(12345)
        probe = rng.standard_normal(48000)
        for band, qq in VIEWS:
            d["views"][(band, qq)] = [_view(T, band, qq).astype(np.float32) for _, T in bank]
            pv = _view(probe, band, qq)
            d["base"][(band, qq)] = np.array([_score(d, (band, qq), j, pv, {})[0] for j in range(len(bank))])
        _DEC[key] = d
    return _DEC[key]


def _score(dec, view, j, xd, xcache, n_peaks: int = 1):
    """peak/mean of the whitened cross-correlation between timeline j (in a band view) and xd.
    Returns (score, [lags of the n_peaks highest peaks]) in samples of that view."""
    Td = dec["views"][view][j]
    nd, NT = len(xd), len(Td)
    m = _np2(NT + nd)
    sk = (view, j, m)
    if sk not in dec["spec"]:
        dec["spec"][sk] = _whiten(Td, m)
    if m not in xcache:
        xcache[m] = np.conj(_whiten(xd, m))
    r = np.fft.irfft(dec["spec"][sk] * xcache[m], m)
    r = np.concatenate([r[-(nd - 1):], r[:NT]])            # lags -(nd-1) .. NT-1
    a = np.abs(r)
    k = int(np.argmax(a))
    lags = [k - (nd - 1)]
    if n_peaks > 1:
        b = a.copy()
        for _ in range(n_peaks - 1):
            b[max(0, k - 6):k + 7] = 0
            k = int(np.argmax(b))
            lags.append(k - (nd - 1))
    return a[lags[0] + nd - 1] / (a.mean() + 1e-12), lags


def _best_full_lag(x, T, lag0, radius):
    """Exact (full-rate) lag around lag0 maximising the normalised correlation over the overlap."""
    n = len(x)
    best = (-2.0, lag0)
    for lag in range(lag0 - radius, lag0 + radius + 1):
        start, end = max(0, -lag), min(n, len(T) - lag)
        if end - start < 2000:
            continue
        a = x[start:end]; b = T[start + lag:end + lag]
        c = float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-30))
        if c > best[0]:
            best = (c, lag)
    return best


def cancellation_depth(x, seg, g, n_fft: int = 512, hop: int = 256, dom: float = 1.0):
    """Energy ratio sum|X - gB|^2 / sum|X|^2 over TF bins where the claimed noise dominates the locally
    smoothed observed power. Exact copies cancel almost completely (small value); merely similar
    (e.g. periodic) noise does not."""
    if len(x) < n_fft + hop:
        return 1.0
    from scipy.ndimage import uniform_filter
    w = np.hanning(n_fft)
    m = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(m)[:, None]
    X = np.fft.rfft(x[idx] * w, axis=1); B = g * np.fft.rfft(seg[idx] * w, axis=1)
    PX, PB = np.abs(X) ** 2 + 1e-20, np.abs(B) ** 2
    PXs = uniform_filter(PX, (3, 3), mode="nearest")
    sel = PB > dom * PXs
    if sel.sum() < 50:
        return 1.0
    return float(np.sum((np.abs(X - B) ** 2)[sel]) / np.sum(PX[sel]))


def find_matches(x: np.ndarray, bank, min_ncc: float = 0.12, max_matches: int = 3, min_len: int = 4000,
                 q: int = 2, top_k: int = 10, min_rel: float = 1.3, return_cands: bool = False,
                 max_depth: float = 0.17, n_peaks: int = 2, block_depth: float = 0.0, block: int = 4000):
    """Exact matches of bank timelines inside x.

    Coarse: whitened cross-correlation in two bands at fs/q (0..fs/2q and the spectrally inverted top band),
    candidates ranked by peak/mean score relative to the timeline's own baseline. Pure-tone timelines are
    skipped (handled by the tonal canceller). Fine: for the top_k timelines, the n_peaks best coarse lags are
    aligned at full rate; a candidate is accepted only if the subtraction cancels the claimed noise almost
    completely (cancellation depth); the accepted candidate with the highest NCC wins.
    Returns a list of (island_index, lag, start, end, rel_score, ncc, depth)."""
    n = len(x)
    dec = _decimated_bank(bank)
    found, cands_log = [], []
    resid = x.astype(np.float64).copy()
    covered = np.zeros(n, bool)
    for _ in range(max_matches):
        xv = {v: _view(resid, *v) for v in VIEWS}
        caches = {v: {} for v in VIEWS}
        # 1) coarse lags from the whitened correlation in every band view (2 peaks per view)
        # 2) every coarse lag is scored by the full-rate normalised correlation (explained energy)
        sc = []
        for j in range(len(bank)):
            if dec["tonal"][j]:
                continue
            T = bank[j][1]
            best_j = None
            for v in VIEWS:
                s_, lg = _score(dec, v, j, xv[v], caches[v], n_peaks)
                for l in lg:
                    ncc, lag = _best_full_lag(resid, T, l * v[1], v[1])
                    if best_j is None or ncc > best_j[0]:
                        best_j = (ncc, lag, v[1])
            if best_j is not None:
                sc.append((best_j[0], j, best_j[1], best_j[2]))
        sc.sort(key=lambda t: -t[0])
        best = None
        for ncc0, j, lag0, qv in sc[:top_k]:
            if ncc0 < min_ncc:
                break
            T = bank[j][1].astype(np.float64)
            ncc, lag = _best_full_lag(resid, T, lag0, 1)
            start, end = max(0, -lag), min(n, len(T) - lag)
            if end - start < min_len:
                cands_log.append((j, ncc0, ncc, end - start, 1.0, lag))
                continue
            # only samples not explained by an earlier match take part (no double subtraction)
            free = ~covered[start:end]
            if free.sum() < min_len:
                cands_log.append((j, ncc0, ncc, int(free.sum()), 1.0, lag))
                continue
            seg = T[start + lag:end + lag] * free; xs = resid[start:end] * free
            g = float(np.dot(xs, seg) / (np.dot(seg, seg) + 1e-30))
            depth = cancellation_depth(xs, seg, g)
            ncc = float(np.dot(xs, seg) / np.sqrt(np.dot(xs, xs) * np.dot(seg, seg) + 1e-30))
            cands_log.append((j, ncc0, ncc, int(free.sum()), depth, lag))
            if depth > max_depth or ncc < min_ncc:
                continue
            explained = float(np.dot(xs, seg) ** 2 / (np.dot(seg, seg) + 1e-30))   # energy removed by the fit
            if best is None or explained > best[7]:
                best = (ncc, j, lag, start, end, ncc0, depth, explained)
        if best is None or best[0] < min_ncc:
            break
        ncc, j, lag, start, end, rel, depth, _ = best
        T = bank[j][1].astype(np.float64)
        free = ~covered[start:end]
        seg = T[start + lag:end + lag] * free
        xs = resid[start:end] * free
        g = float(np.dot(xs, seg) / np.dot(seg, seg))
        if block_depth:
            # block-wise check: keep only blocks where the subtraction really cancels the noise
            L_ = end - start
            ok = np.zeros(L_, bool)
            for b0 in range(0, L_, block // 2):
                b1 = min(L_, b0 + block)
                if b1 - b0 < block // 2:
                    continue
                if not free[b0:b1].any():
                    continue
                dpt = cancellation_depth(xs[b0:b1], seg[b0:b1], g, n_fft=256, hop=128)
                if dpt <= block_depth:
                    ok[b0:b1] = True
            free = free & ok
            if free.sum() < min_len:
                covered[start:end] |= False
                break
            seg = T[start + lag:end + lag] * free
            xs = resid[start:end] * free
            g = float(np.dot(xs, seg) / np.dot(seg, seg))
        resid[start:end] -= g * seg
        found.append((j, lag, start, end, rel, ncc, depth, g, free.copy()))
        covered[start:end] |= free
        if covered.mean() > 0.98:
            break
    return (found, cands_log) if return_cands else found


def cancel_reference(x: np.ndarray, bank=None, fade: int = 160, **kw):
    """Subtract matched reference noise. Returns (y, coverage mask, matches)."""
    bank = bank if bank is not None else load_bank()
    n = len(x)
    matches = find_matches(x, bank, **kw)
    y = x.astype(np.float64).copy()
    cov = np.zeros(n)
    for j, lag, start, end, score, ncc, depth, g, free in matches:
        T = bank[j][1].astype(np.float64)
        y[start:end] -= g * T[start + lag:end + lag] * free
        cov[start:end][free] = 1.0
    if fade and cov.any() and not cov.all():
        k = np.hanning(2 * fade + 1); k /= k.sum()
        cov = np.convolve(cov, k, mode="same")
    return y, cov, matches

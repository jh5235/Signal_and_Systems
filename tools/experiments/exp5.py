import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from scipy.ndimage import uniform_filter1d
from nr import core as C

def band_edges(n_bins, n_bands, fs=16000, fmin=50):
    f = np.linspace(0, fs / 2, n_bins)
    e = np.geomspace(fmin, fs / 2, n_bands)
    idx = np.unique(np.concatenate([[0], np.searchsorted(f, e), [n_bins]]))
    return idx

def band_noise(P, n_bands=24, delta_db=4.0, pct=5, tsmooth=3, min_frames=4):
    F, T = P.shape
    edges = band_edges(F, n_bands)
    N0 = np.empty(F)
    for lo, hi in zip(edges[:-1], edges[1:]):
        E = P[lo:hi].sum(0)
        Es = uniform_filter1d(E, tsmooth, mode="nearest") if tsmooth > 1 else E
        thr = np.percentile(Es, pct) * 10 ** (delta_db / 10)
        sel = np.flatnonzero(Es <= thr)
        if len(sel) < min_frames:
            sel = np.argsort(Es)[:min_frames]
        N0[lo:hi] = P[lo:hi, sel].mean(1)
    return N0

def enh5(x, n_fft=1024, hop=256, a=0.98, gmin_db=-25, xi_min_db=-25, n_bands=24, delta_db=4.0, pct=5,
         tsmooth=3, min_frames=4, fsm=1, mode="band", sfrac=0.1, nscale=1.0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    if mode == "band":
        N0 = band_noise(P, n_bands, delta_db, pct, tsmooth, min_frames)
    else:
        N0 = C.init_noise(P, sfrac)
    if fsm > 1:
        N0 = uniform_filter1d(N0, fsm, mode="nearest")
    N = np.tile(N0[:, None] * nscale, (1, P.shape[1]))
    xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, C.gain_logmmse)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh5, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

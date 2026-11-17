"""Band-wise SPP-weighted tracking of a modulated noise level on top of the long-term shape."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from scipy.ndimage import gaussian_filter1d
from nr import core as C, pipeline as PL
from xi_analysis import our_xi, BP
def mod2(x, q=3.0, q0=0.5, n_bands=8, sig_t=2.0, lam=0.3, mix=1.0, lt=True, gmax_db=12, gmin_db=-12):
    Z, meta, P, N, F, gamma, xi, pp = our_xi(x, BP)
    w = (1 - C.spp(xi, gamma, q0)) ** q
    Nm = N.mean(1, keepdims=True)
    Nlt = (w * P).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1e-9) if lt else Nm
    Fb, T = P.shape
    edges = np.unique(np.round(np.geomspace(3, Fb, n_bands + 1)).astype(int)); edges[0] = 0
    R = P / Nlt
    Gt = np.ones_like(P)
    for lo, hi in zip(edges[:-1], edges[1:]):
        num = (w[lo:hi] * R[lo:hi]).sum(0); den = w[lo:hi].sum(0)
        num = gaussian_filter1d(num, sig_t, mode='nearest'); den = gaussian_filter1d(den, sig_t, mode='nearest')
        g = (num + lam * (hi - lo)) / (den + lam * (hi - lo))
        Gt[lo:hi] = np.clip(g, 10 ** (gmin_db / 10), 10 ** (gmax_db / 10))[None, :]
    Nnew = Nlt * Gt
    corr = (Nnew / N) ** mix
    Z, meta, P, N, F, gamma, xi, pp = our_xi(x, BP, lambda N_, a, b: corr)
    G = np.maximum(C.gain_logmmse(xi, gamma), 10 ** (pp["gmin_db"] / 20))
    G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / P)))
    return C.istft(G * Z, meta)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(mod2, split, **cfg)
        print(ek.report(r, str(cfg), by=("snr",), base=base), f"{time.time()-t:.0f}s", flush=True)

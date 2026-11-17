"""Self-estimated per-bin long-term noise spectrum correction (no oracle)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from scipy.ndimage import uniform_filter1d, uniform_filter
from nr import core as C, pipeline as PL
from xi_analysis import our_xi, BP
def _bias_fixed(q, xih1_db, pri):
    """E[P w(P)] / (E[w] E[P]) for exponential P (unit mean) with w = (1 - p_H1(P))^q."""
    g = np.linspace(0, 30, 30001)[1:]; pdf = np.exp(-g)
    xi = 10 ** (xih1_db / 10)
    ph1 = 1 / (1 + pri * (1 + xi) * np.exp(-g * xi / (1 + xi)))
    w = (1 - ph1) ** q
    return float(np.sum(g * w * pdf) / np.sum(w * pdf))
def lt_fix(x, wmode="gain", q=2.0, iters=1, fsm=1, tsm=1, clip_db=20.0, src="final", q0=0.5, xih1_db=15.0, debias=False):
    p = dict(PL.DEFAULTS, **BP)
    corr = None
    for it in range(iters):
        nf = (lambda N, a, b, c=corr: c) if corr is not None else None
        Z, meta, P, N, F, gamma, xi, pp = our_xi(x, BP, nf)
        G = C.gain_logmmse(xi, gamma)
        if wmode == "gain":      w = np.clip(1 - G ** 2, 0, 1) ** q
        elif wmode == "xi":      w = (1.0 / (1.0 + xi)) ** q
        elif wmode == "spp":     w = (1 - C.spp(xi, gamma, q0)) ** q
        elif wmode == "sppfix":    # fixed-xi_H1 SPP (Gerkmann & Hendriks), independent of our xi
            xh = 10 ** (xih1_db / 10); pri = q0 / (1 - q0)
            ph1 = 1 / (1 + pri * (1 + xh) * np.exp(-np.minimum(gamma * xh / (1 + xh), 50)))
            w = (1 - ph1) ** q
        if fsm > 1 or tsm > 1:
            w = uniform_filter(w, (fsm, tsm), mode="nearest")
        Nlt = (w * P).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        if debias and wmode == "sppfix":
            Nlt = Nlt / _bias_fixed(q, xih1_db, q0 / (1 - q0))
        Nm = N.mean(1, keepdims=True)
        c = np.clip(Nlt / Nm, 10 ** (-clip_db / 10), 10 ** (clip_db / 10))
        corr = c if corr is None else corr * c
    Z, meta, P, N, F, gamma, xi, pp = our_xi(x, BP, lambda N, a, b: corr)
    G = np.maximum(C.gain_logmmse(xi, gamma), 10 ** (pp["gmin_db"] / 20))
    G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / P)))
    return C.istft(G * Z, meta)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(lt_fix, split, **cfg)
        print(ek.report(r, str(cfg), by=("snr",), base=base), f"{time.time()-t:.0f}s", flush=True)

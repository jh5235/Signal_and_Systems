"""Post-processing of the a priori SNR map (single resolution, no repetition) — compare to 'none'."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from scipy.ndimage import grey_opening, gaussian_filter, uniform_filter, minimum_filter, maximum_filter
from nr import core as C, pipeline as PL
from xi_analysis import our_xi, BP
def post(xi, mode, **k):
    L = np.log(xi)
    if mode == "none": return xi
    if mode == "open":        # union of time- and frequency-elongated openings (removes isolated peaks)
        a = grey_opening(L, size=(1, k.get("t", 3))); b = grey_opening(L, size=(k.get("f", 3), 1))
        return np.exp(np.maximum(a, b))
    if mode == "open_mix":
        a = grey_opening(L, size=(1, k.get("t", 3))); b = grey_opening(L, size=(k.get("f", 3), 1))
        w = k.get("w", 0.5); return np.exp(w * np.maximum(a, b) + (1 - w) * L)
    if mode == "logsmooth":
        return np.exp(gaussian_filter(L, (k.get("sf", 1.0), k.get("st", 1.0)), mode="nearest"))
    if mode == "clamp":       # xi <= kappa * local log-mean
        Ls = gaussian_filter(L, (k.get("sf", 2.0), k.get("st", 1.0)), mode="nearest")
        return np.exp(np.minimum(L, Ls + np.log(k.get("kappa", 3.0))))
    if mode == "open_t":
        return np.exp(grey_opening(L, size=(1, k.get("t", 3))))
    if mode == "open_2d":
        return np.exp(grey_opening(L, size=(k.get("f", 2), k.get("t", 2))))
def env_xi(P, N, p, w=9, a=0.9, tw=1):
    """Envelope-level a priori SNR: smooth power across frequency (arithmetic), then DD."""
    from scipy.ndimage import uniform_filter1d
    Ps = uniform_filter1d(P, w, axis=0, mode="nearest"); Ns = uniform_filter1d(N, w, axis=0, mode="nearest")
    if tw > 1:
        Ps = uniform_filter1d(Ps, tw, axis=1, mode="nearest")
    xs, _, _ = C.dd_prior_snr(Ps, Ns, a, p["xi_min_db"], C.gain_logmmse)
    xb, _, _ = C.dd_prior_snr(Ps[:, ::-1], Ns[:, ::-1], a, p["xi_min_db"], C.gain_logmmse)
    return np.sqrt(xs * xb[:, ::-1])
def fn(x, mode="none", **k):
    Z, meta, P, N, F, gamma, xi, p = our_xi(x, BP)
    if mode.startswith("lpc"):
        Ps = np.maximum(xi * N, 1e-20)              # speech power implied by our xi
        E = C.lpc_envelope(Ps, k.get("order", 20), p["n_fft"], k.get("lag_win", 0.0))
        if k.get("tsm", 0):
            from scipy.ndimage import uniform_filter1d
            E = uniform_filter1d(E, k["tsm"], axis=1, mode="nearest")
        xe = E / N
        if mode == "lpc_geo": w = k.get("mix", 0.5); xi = xi ** (1 - w) * xe ** w
        elif mode == "lpc_min": xi = np.minimum(xi, k.get("kappa", 1.0) * xe)
        elif mode == "lpc_only": xi = xe
    elif mode.startswith("env"):
        xe = env_xi(P, N, p, k.get("w", 9), k.get("a", 0.9), k.get("tw", 1))
        if mode == "env_only": xi = xe
        elif mode == "env_geo": w = k.get("mix", 0.5); xi = xi ** (1 - w) * xe ** w
        elif mode == "env_min": xi = np.minimum(xi, k.get("kappa", 1.0) * xe)
    else:
        xi = post(xi, mode, **k)
    xi = np.maximum(xi, 10 ** (p["xi_min_db"] / 10))
    G = np.maximum(C.gain_logmmse(xi, gamma), 10 ** (p["gmin_db"] / 20))
    G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / P)))
    return C.istft(G * Z, meta)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, **cfg)
        print(ek.report(r, str(cfg), by=("snr",), base=base), f"{time.time()-t:.0f}s", flush=True)

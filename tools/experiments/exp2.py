import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from nr import core as C

def static_noise(P, sfrac):
    return C.init_noise(P, sfrac)

def tv_noise(P, w, N0, sig_t, sig_f, lam):
    """SPP-weighted non-causal averaging of noisy power; falls back to N0 where little noise evidence."""
    num = gaussian_filter1d(w * P, sig_t, axis=1, mode="nearest")
    den = gaussian_filter1d(w, sig_t, axis=1, mode="nearest")
    if sig_f:
        num = gaussian_filter1d(num, sig_f, axis=0, mode="nearest"); den = gaussian_filter1d(den, sig_f, axis=0, mode="nearest")
    return (num + lam * N0[:, None]) / (den + lam)

def enh2(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25, xi_min_db=-25, passes=1,
         q=0.5, sig_t=4.0, sig_f=1.0, lam=0.05, wpow=1.0, nscale=1.0, wmode="spp"):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N0 = static_noise(P, sfrac)
    N = np.tile(N0[:, None], (1, P.shape[1]))
    for it in range(passes):
        xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, C.gain_logmmse)
        if it == passes - 1 and passes > 0 and it > 0:
            break
        if passes == 1:
            break
        if wmode == "spp":
            w = (1 - C.spp(xi, gamma, q)) ** wpow
        else:  # noise fraction by gain
            w = (1 - G ** 2) ** wpow
        N = tv_noise(P, w, N0, sig_t, sig_f, lam) * nscale
    if passes > 1:
        xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, C.gain_logmmse)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh2, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

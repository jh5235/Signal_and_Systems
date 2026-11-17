import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time, functools
from scipy.ndimage import uniform_filter1d
from nr import core as C

def smooth_tf(P, fw, alpha):
    S = uniform_filter1d(P, fw, axis=0, mode="nearest") if fw > 1 else P.copy()
    if alpha:
        for t in range(1, S.shape[1]):
            S[:, t] = alpha * S[:, t - 1] + (1 - alpha) * S[:, t]
    return S

@functools.lru_cache(maxsize=None)
def q_comp(qt, fw, alpha, n_fft, hop):
    """Bias factor mean/quantile for pure white Gaussian noise under the same STFT+smoothing."""
    rng = np.random.default_rng(0); vals = []
    for _ in range(8):
        Z, _ = C.stft(rng.standard_normal(48000), n_fft, hop); P = np.abs(Z) ** 2
        S = smooth_tf(P, fw, alpha); vals.append(P.mean() / np.quantile(S[:, 5:-5], qt, axis=1).mean())
    return float(np.mean(vals))

def quantile_noise(P, qt=0.3, fw=3, alpha=0.5, n_fft=1024, hop=256):
    S = smooth_tf(P, fw, alpha)
    return np.quantile(S, qt, axis=1) * q_comp(qt, fw, alpha, n_fft, hop)

def enh3(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25, xi_min_db=-25, mode="min",
         qt=0.3, fw=3, alpha=0.5, nscale=1.0, qscale=1.0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    Nf = C.init_noise(P, sfrac)
    Nq = quantile_noise(P, qt, fw, alpha, n_fft, hop) * qscale
    N0 = {"frames": Nf, "q": Nq, "min": np.minimum(Nf, Nq), "geo": np.sqrt(Nf * Nq)}[mode] * nscale
    N = np.tile(N0[:, None], (1, P.shape[1]))
    xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, C.gain_logmmse)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh3, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

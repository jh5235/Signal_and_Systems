import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from nr import core as C
c_, n_, m_ = ek.load('validation')
def find(x):
    return int(np.flatnonzero((n_ == x).all(1))[0])
def oracle(x, mode="psd_t", n_fft=1024, hop=256, a=0.98, gmin_db=-25, smooth=0.0):
    i = find(x); d = n_[i] - c_[i]
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z)**2
    D = np.abs(C.stft(d, n_fft, hop)[0])**2
    if mode == "psd_static": N = np.tile(D.mean(1, keepdims=True), (1, P.shape[1]))
    else:
        N = D.copy()
        if smooth:   # recursive smoothing over time
            for t in range(1, N.shape[1]): N[:, t] = smooth * N[:, t-1] + (1-smooth) * D[:, t]
    xi, g, G = C.dd_prior_snr(P, N, a, -25, C.gain_logmmse)
    G = np.maximum(G, 10**(gmin_db/20))
    if mode == "irm":
        S = np.abs(C.stft(c_[i], n_fft, hop)[0])**2
        G = np.maximum(np.sqrt(S/(S+D+1e-12)), 10**(gmin_db/20))
    return C.istft(G*Z, meta)
base = ek.noisy_scores('validation')
for kw in [dict(mode="psd_static"), dict(mode="psd_t", smooth=0.7), dict(mode="psd_t", smooth=0.0), dict(mode="irm")]:
    r = ek.evaluate(oracle, 'validation', **kw); print(ek.report(r, str(kw), by=("diff",), base=base))

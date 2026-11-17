import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from nr import core as C
c_, n_, m_ = ek.load('validation')
_lookup = {}
def find(x):
    key = x[:64].tobytes()
    if key not in _lookup:
        _lookup[key] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lookup[key]
def orc(x, n_fft=1024, hop=256, a=0.98, gmin_db=-25, xi_min_db=-25, gain="logmmse", nscale=1.0, gexp=1.0, static=True):
    i = find(x); d = n_[i] - c_[i]
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z)**2
    D = np.abs(C.stft(d, n_fft, hop)[0])**2
    N = np.tile(D.mean(1, keepdims=True), (1, P.shape[1])) if static else D
    N = N * nscale
    gf = {"wiener": C.gain_wiener, "logmmse": C.gain_logmmse, "stsa": C.gain_mmse_stsa}[gain]
    xi, g, G = C.dd_prior_snr(P, N, a, xi_min_db, gf)
    G = np.maximum(G ** gexp, 10**(gmin_db/20))
    return C.istft(G*Z, meta)
if __name__ == "__main__":
    base = ek.noisy_scores('validation')
    for kw in eval(sys.argv[1]):
        r = ek.evaluate(orc, 'validation', **kw); print(ek.report(r, str(kw), by=("diff",), base=base), flush=True)

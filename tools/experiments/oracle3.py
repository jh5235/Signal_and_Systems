"""Oracle diagnostics: which knowledge would help most?"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections
from nr import core as C, pipeline as PL
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'validation'
c_, n_, m_ = ek.load(SPLIT)
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]
def orc(x, mode, n_fft=1280, hop=256, gmin_db=-36.5):
    i = find(x); s = c_[i]; d = n_[i] - s
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2 + 1e-20
    S = np.abs(C.stft(s, n_fft, hop)[0]) ** 2; D = np.abs(C.stft(d, n_fft, hop)[0]) ** 2
    Ds = D.mean(1, keepdims=True) * np.ones_like(D)
    Dt = D.copy()
    for t in range(1, D.shape[1]): Dt[:, t] = 0.7 * Dt[:, t-1] + 0.3 * D[:, t]
    if mode == "xi_true_Nstatic":     # perfect a priori SNR, static true noise PSD
        G = C.gain_logmmse(S / Ds, P / Ds)
    elif mode == "xi_true_Nest":      # perfect speech power, our estimated noise (static)
        N = np.tile(C.init_noise(P, 0.12)[:, None], (1, P.shape[1])) * 1.19
        G = C.gain_logmmse(S / N, P / N)
    elif mode == "Ntrue_smooth":      # DD with smoothed true instantaneous noise
        xi, g, G = C.dd_prior_snr(P, Dt, 0.982, -30, C.gain_logmmse)
    elif mode == "wiener_true":       # S/(S+D) with true periodograms (upper bound of gain-only)
        G = np.sqrt(S / (S + D))
    G = np.maximum(G, 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)
if __name__ == "__main__":
    modes = sys.argv[1].split(',')
    base = ek.noisy_scores(SPLIT)
    for mo in modes:
        r = ek.evaluate(orc, SPLIT, mode=mo)
        print(ek.report(r, mo, by=('diff', 'snr'), base=base), flush=True)

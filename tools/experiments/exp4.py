import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import core as C
FLOOR = np.load('/tmp/claude-0/-mnt-c-Users-jwoo4-Yonsei--------FIND/061e7bb7-ac30-4c94-b5de-8171e19e6951/scratchpad/clean_floor_geo.npy')
c_, n_, m_ = ek.load('validation')
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]

def enh4(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25, xi_min_db=-25, floor_db=None,
         floor_mode="outfloor", subtract=0.0, oracle_floor=False, nscale=1.0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N0 = C.init_noise(P, sfrac)
    F = None
    if oracle_floor:
        i = find(x); F = C.init_noise(np.abs(C.stft(c_[i], n_fft, hop)[0]) ** 2, sfrac)
    elif floor_db is not None:
        F = FLOOR * 10 ** (floor_db / 10)
    if F is not None and subtract:
        N0 = np.maximum(N0 - subtract * F, 0.05 * N0)
    N = np.tile(N0[:, None] * nscale, (1, P.shape[1]))
    xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, C.gain_logmmse)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    if F is not None and floor_mode == "outfloor":
        G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / np.maximum(P, 1e-20))))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh4, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

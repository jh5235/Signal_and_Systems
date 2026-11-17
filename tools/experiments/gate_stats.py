import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, json
from multiprocessing import Pool
from nr import core as C, pipeline as PL
from nr.repet import repeat_fraction, detect_period_bands, long_lag_corr
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
c_, n_, m_ = ek.load(SPLIT)
LOOPS = {int(k): v for k, v in json.load(open(f'{ek.CACHE}/loops_{SPLIT}.json')).items()}
def f(i):
    x = n_[i]; p = dict(PL.DEFAULTS, **PL.BEST)
    L = PL.detect_repetition(x, 'both', 0.0, submult=0.8, **PL.BEST)     # threshold 0: always return best
    a = detect_period_bands(x); r = x - PL.enhance(x, **PL.BEST); b = detect_period_bands(r)
    score = max(a[2], b[2])
    if L is None: return i, None, score, 1.0, 0.0
    n_fft, hop = p['n_fft'], p['hop']
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z)**2 + 1e-20
    N, N0 = PL.estimate_noise(P, p); N = N * p['nscale']
    xi, _, G = C.dd_prior_snr(P, N, p['a'], p['xi_min_db'], C.gain_logmmse)
    Sp = (G * np.abs(Z))**2
    u = repeat_fraction(x, L, Sp, N, lambda v: C.stft(v, n_fft, hop), hop, n_fft)
    cr = long_lag_corr(r - r.mean(), 1000, 3000)[L]
    return i, L, score, u, cr
if __name__ == "__main__":
    with Pool(15) as p: R = p.map(f, range(len(c_)))
    out = []
    for i, L, sc, u, cr in R:
        t = LOOPS.get(i); ok = t is not None and L is not None and any(abs(L - k * t) <= 2 for k in (1, 2, 0.5))
        out.append((i, -1 if L is None else L, sc, u, cr, int(ok), int(t is not None)))
    A = np.array(out, float); np.save(f'{ek.CACHE}/gate_{SPLIT}.npy', A)
    for thr in (5, 6, 8):
        m = A[:, 2] > thr
        for ug in (0.3, 0.5, 0.7, 1.01):
            s = m & (A[:, 3] < ug)
            print(f'score>{thr} u<{ug}: selected {int(s.sum())}, correct {int((s & (A[:,5]==1)).sum())}, false {int((s & (A[:,5]==0)).sum())} (true loops {int(A[:,6].sum())})')
    print('u for correct:', np.percentile(A[A[:,5]==1,3], [10,50,90]).round(3), ' u for false (score>6):', np.percentile(A[(A[:,5]==0)&(A[:,2]>6),3], [10,50,90]).round(3) if ((A[:,5]==0)&(A[:,2]>6)).any() else None)
    print('resid corr for correct:', np.percentile(A[A[:,5]==1,4], [10,50,90]).round(3), ' false(score>6):', np.percentile(A[(A[:,5]==0)&(A[:,2]>6),4], [10,50,90]).round(3) if ((A[:,5]==0)&(A[:,2]>6)).any() else None)

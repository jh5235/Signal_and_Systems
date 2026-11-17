import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from multiprocessing import Pool
from nr import core as C, pipeline as PL
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'validation'
c_, n_, m_ = ek.load(SPLIT)
p = dict(PL.DEFAULTS, **PL.BEST); n_fft, hop = p['n_fft'], p['hop']
f = np.fft.rfftfreq(n_fft, 1 / 16000)
edges_hz = np.geomspace(60, 8000, 22)
def stats(i):
    x = n_[i]
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2 + 1e-20
    N, N0 = PL.estimate_noise(P, p); N = N * p['nscale']
    xi, gamma, G = PL._prior_snr(Z, meta, P, N, p, C.gain_logmmse, C.gain_logmmse)
    w = (1 - C.spp(xi, gamma, 0.5)) ** 3
    Nlt = (w * P).sum(1) / np.maximum(w.sum(1), 1e-9)
    ms = PL.min_stat_cap(P, n_fft, hop, 3, 5)
    S_hat = (G ** 2) * P
    D = np.abs(C.stft(x - c_[i], n_fft, hop)[0]) ** 2
    out = []
    for lo, hi in zip(edges_hz[:-1], edges_hz[1:]):
        b = (f >= lo) & (f < hi)
        if b.sum() < 1: continue
        Eb = np.log(P[b].sum(0)); ref = np.log(S_hat[~b].sum(0) + 1e-20)
        rho = np.corrcoef(Eb[2:-2], ref[2:-2])[0, 1]
        ratio = 10 * np.log10(Nlt[b].mean() / ms[b].mean())
        truth = 10 * np.log10(D[b].mean(1).mean() / Nlt[b].mean())     # < -10: gross over-estimate
        out.append((lo, rho, ratio, truth, m_[i][1]))
    return out
if __name__ == "__main__":
    with Pool(15) as pool: R = pool.map(stats, range(len(c_)))
    A = [r for rr in R for r in rr]
    rho = np.array([a[1] for a in A]); ratio = np.array([a[2] for a in A]); truth = np.array([a[3] for a in A])
    gross = truth < -10
    print('bands', len(A), 'gross over-estimated', gross.mean().round(3))
    for rt in (3, 6, 10):
        for pt in (0.5, 0.6, 0.7, 0.8):
            sel = (ratio > rt) & (rho > pt)
            print(f'ratio>{rt} rho>{pt}: selected {sel.mean():.3f}  precision {gross[sel].mean() if sel.any() else 0:.2f}  recall {(sel & gross).sum() / max(gross.sum(), 1):.2f}  median truth {np.median(truth[sel]) if sel.any() else 0:.1f}')
    np.save(f'{ek.CACHE}/band_stats_{SPLIT}.npy', np.array([(a[0], a[1], a[2], a[3]) for a in A]))

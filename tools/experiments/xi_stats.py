import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections
from multiprocessing import Pool
from nr import core as C
from xi_analysis import our_xi, BP
c_, n_, m_ = ek.load('validation')
def stats(i):
    Z, meta, P, N, F, gamma, xi, p = our_xi(n_[i], BP)
    S = np.abs(C.stft(c_[i], p["n_fft"], p["hop"])[0]) ** 2
    D = np.abs(C.stft(n_[i] - c_[i], p["n_fft"], p["hop"])[0]) ** 2
    xt = S / N
    nd = xt <= 1                                   # noise-dominant bins (w.r.t. our noise estimate)
    G = C.gain_logmmse(xi, gamma)
    # contribution of residual noise power from noise-dominant bins
    res = (G ** 2 * D)[nd].sum() / (G ** 2 * D).sum()
    nbias = np.median(10 * np.log10((D[nd] + 1e-20) / N[nd]))     # true noise vs estimate in those bins
    over = np.mean(xi[nd] > 1)                     # fraction estimated speech-dominant although noise-dominant
    lerr = np.median(10 * np.log10(xi[nd] / np.maximum(xt[nd], 1e-3)))
    # where do the over-estimates come from: gamma>1 (noise above estimate) or smoothing?
    ov = nd & (xi > 1)
    g_ov = np.median(gamma[ov]) if ov.any() else np.nan
    Gnd = np.mean(G[nd])
    return m_[i][1], res, nbias, over, lerr, g_ov, Gnd, nd.mean()
with Pool(15) as pool: R = pool.map(stats, range(len(c_)))
A = np.array([r[1:] for r in R], float)
print('ALL: resid-noise share from noise-dominant bins %.2f | true-noise/est in those bins (median dB) %.1f | frac xi>1 %.3f | median log err %.1f dB | median gamma at over-est %.2f | mean G %.3f | frac nd bins %.2f' % tuple(np.nanmean(A, 0)))
g = collections.defaultdict(list)
for r in R: g[r[0]].append(r[1:])
for k, v in sorted(g.items(), key=lambda x: -len(x[1]))[:10]:
    v = np.array(v, float); print(f'{k:20s} n={len(v):3d} ' + ' '.join(f'{x:6.2f}' for x in np.nanmean(v, 0)))

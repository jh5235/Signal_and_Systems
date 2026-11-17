import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections
from nr import core as C
from exp8 import enh8
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
c_, n_, m_ = ek.load(SPLIT)
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]
def orc(x, static=True, smooth=0.7):
    i = find(x); d = n_[i] - c_[i]
    Z, meta = C.stft(x, 1024, 256); P = np.abs(Z)**2; D = np.abs(C.stft(d, 1024, 256)[0])**2
    if static: N = np.tile(D.mean(1, keepdims=True), (1, P.shape[1]))
    else:
        N = D.copy()
        for t in range(1, N.shape[1]): N[:, t] = smooth * N[:, t-1] + (1-smooth) * D[:, t]
    xi, g, G = C.dd_prior_snr(P, N, 0.98, -25, C.gain_logmmse)
    return C.istft(np.maximum(G, 10**(-25/20)) * Z, meta)
if __name__ == "__main__":
    r0 = ek.noisy_scores(SPLIT)[0]
    r1 = ek.evaluate(enh8, SPLIT, step='hrnr', tsnr_mix=0.5, gmin_db=-30)['pesq']
    r2 = ek.evaluate(orc, SPLIT)['pesq']; r3 = ek.evaluate(orc, SPLIT, static=False)['pesq']
    g = collections.defaultdict(list)
    for k in range(len(m_)): g[m_[k][1]].append((r0[k], r1[k], r2[k], r3[k]))
    print(f'ALL n={len(m_)} noisy %.3f ours %.3f orac_s %.3f orac_t %.3f' % tuple(np.mean([r0, r1, r2, r3], 1)))
    print('cat n noisy ours orac_s orac_t | weighted gap(orac_t-ours)*n/N')
    for c, v in sorted(g.items(), key=lambda x: -len(x[1])):
        v = np.array(v); m = v.mean(0); print(f'{c:22s}{len(v):4d} ' + ' '.join(f'{x:.3f}' for x in m) + f' | {(m[3]-m[1])*len(v)/len(m_):.4f}')
    np.savez(f'{ek.CACHE}/{SPLIT}_ref_scores.npz', ours=r1, orac_s=r2, orac_t=r3)

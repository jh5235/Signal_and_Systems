import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from multiprocessing import Pool
from nr import core as C
from nr.pipeline import enhance
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
c_, n_, m_ = ek.load(SPLIT)
def feats(i):
    x = n_[i]; Z, _ = C.stft(x, 1024, 256); P = np.abs(Z)**2 + 1e-20
    N0 = C.init_noise(P, 0.1)
    a = C.noise_mmse_spp(P, N0); b = C.noise_mmse_spp(P, N0, backward=True); T = np.sqrt(a*b)
    L = np.log10(T / N0[:, None]) * 10
    w = N0 / N0.sum()                      # weight bins by noise power
    f1 = float(np.sum(w * np.median(np.abs(L), 1)))          # typical deviation tracker vs static
    f2 = float(np.sum(w * np.std(L, 1)))                    # temporal variability of tracker
    f3 = float(np.sum(w * np.mean(L, 1)))                   # mean offset
    # band energy temporal variability within low-energy half of frames
    e = 10*np.log10(P.sum(0)); f4 = float(np.std(np.sort(e)[:len(e)//2]))
    return f1, f2, f3, f4
if __name__ == "__main__":
    with Pool(15) as p: F = np.array(p.map(feats, range(len(c_))))
    base = dict(floor=True, floor_cap_db=-45)
    S = {m: ek.evaluate(enhance, SPLIT, nmode=m, **base)['pesq'] for m in ('static', 'geo', 'tracker')}
    np.savez(f'{ek.CACHE}/selfeat_{SPLIT}.npz', F=F, **S)
    for m, v in S.items(): print(m, v.mean())
    print('oracle select', np.max(np.stack(list(S.values())), 0).mean())

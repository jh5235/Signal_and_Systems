import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, json
from multiprocessing import Pool
from nr import pipeline as PL
SPLIT = sys.argv[1]; cfgs = eval(sys.argv[2])
c_, n_, m_ = ek.load(SPLIT)
LOOPS = {int(k): v for k, v in json.load(open(f'{ek.CACHE}/loops_{SPLIT}.json')).items()}
def f(args):
    i, cfg = args
    return PL.detect_repetition(n_[i], 'both', **cfg, **PL.BEST)
for cfg in cfgs:
    with Pool(15) as p: R = p.map(f, [(i, cfg) for i in range(len(c_))])
    sel = [i for i, L in enumerate(R) if L]
    def exact(i, L):
        t = LOOPS.get(i)
        return t is not None and any(abs(L - k * t) <= 2 for k in (1, 2, 3))
    def tonal_ok(i, L):          # pure tone N099: any multiple of 64 samples is valid
        return m_[i][2] == 'N099' and abs(L / 64 - round(L / 64)) * 64 <= 2
    good = [i for i in sel if exact(i, R[i]) or tonal_ok(i, R[i])]
    print(cfg, f'selected {len(sel)}, exact/valid {len(good)}, other {len(sel)-len(good)} (true loops {len(LOOPS)})', flush=True)

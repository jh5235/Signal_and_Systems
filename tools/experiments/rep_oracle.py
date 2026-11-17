import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0,'/mnt/c/Users/jwoo4/Yonsei/신호및시스템/FIND/noise_reduction_challenge/noisy_code')
import numpy as np, evalkit as ek, json
from noise_bank import detect_loop
from nr import pipeline as PL
from run2 import BP
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'validation'
c_, n_, m_ = ek.load(SPLIT)
LOOPS = {}
path = f'{ek.CACHE}/loops_{SPLIT}.json'
try: LOOPS = {int(k): v for k, v in json.load(open(path)).items()}
except FileNotFoundError:
    from multiprocessing import Pool
    def f(i): return detect_loop(n_[i]-c_[i], min_lag=4000, min_overlap=5600)[0]
    with Pool(15) as p: Ls = p.map(f, range(len(c_)))
    LOOPS = {i: L for i, L in enumerate(Ls) if L}; json.dump(LOOPS, open(path, 'w'))
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]
def fn(x, use=True, **kw):
    i = find(x); L = LOOPS.get(i) if use else None
    return PL.enhance(x, period=L, **dict(BP, **kw))
if __name__ == "__main__":
    idx = sorted(LOOPS)
    print('loop files', len(idx))
    for cfg in eval(sys.argv[1]):
        r = ek.evaluate(fn, SPLIT, idx=idx, **cfg); print(cfg, 'PESQ on loop files %.4f' % np.nanmean(r['pesq']), flush=True)

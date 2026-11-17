import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections, itertools, json
from nr import pipeline as PL
from run2 import BP
def fn(x, **kw): return PL.enhance_auto(x, rep_thr=6, **dict(BP, **kw))
res = {}
for ns, gm, rho in itertools.product((0.9, 1.19, 1.5), (-30, -36.5, -43), (0.5,)):
    r = ek.evaluate(fn, 'train', nscale=ns, gmin_db=gm)
    g = collections.defaultdict(list)
    for k, p in enumerate(r['pesq']): g[r['meta'][k][4]].append(p)
    res[(ns, gm)] = {k: float(np.mean(v)) for k, v in g.items()}
    print(ns, gm, ' '.join(f'{k}:{v:.4f}' for k, v in sorted(res[(ns, gm)].items())), flush=True)
for snr in sorted(next(iter(res.values()))):
    best = max(res, key=lambda c: res[c][snr]); print('SNR', snr, 'best', best, round(res[best][snr], 4), 'default', round(res[(1.19, -36.5)][snr], 4))

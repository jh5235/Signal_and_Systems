"""Random search of the full single-resolution pipeline (repetition EM + long-term correction) on a train subset."""
import sys, os, json, time; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from nr import pipeline as PL
log, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(seed)
IDX = list(range(0, 1296, 3))
def fn(x, **kw):
    lt = dict(q=kw.pop('lt_q'))
    return PL.enhance_auto(x, rep_thr=6.0, rep_mix_auto=0.6, det='both', submult=0.95, rep_em=PL.REP_EM, lt_corr=lt, **kw)
def space():
    b = dict(PL.BEST)
    b.update(a=float(rng.uniform(0.97, 0.99)), gmin_db=float(rng.uniform(-42, -28)), xi_min_db=float(rng.uniform(-36, -24)),
             rho=float(rng.uniform(0.3, 0.7)), tsnr_mix=float(rng.uniform(0.4, 0.85)), sfrac=float(rng.uniform(0.08, 0.18)),
             spp_alpha=float(rng.uniform(0.78, 0.9)), xi_h1=float(rng.uniform(9, 16)), nscale=float(rng.uniform(0.95, 1.4)),
             floor_cap_db=float(rng.uniform(-52, -43)), lt_q=float(rng.uniform(2, 5)))
    return b
if __name__ == "__main__":
    # reference = current BEST first
    cfgs = [dict(PL.BEST, lt_q=3.0)] + [space() for _ in range(n)]
    for k, cfg in enumerate(cfgs):
        t = time.time(); r = ek.evaluate(fn, 'train', idx=IDX, **cfg)
        rec = dict(cfg=cfg, pesq=float(np.nanmean(r['pesq'])), snr=float(np.mean(r['snr'])), sec=time.time() - t)
        with open(log, 'a') as f: f.write(json.dumps(rec) + '\n')
        print(k, round(rec['pesq'], 4), round(rec['sec']), flush=True)

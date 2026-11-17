"""Evaluate the full submission pipeline (enhance_signal) with BEST overrides."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import pipeline as PL
def fn(x, lt_q=None, ens=None, tonal=None, rep_em=None, w=None, **kw):
    if lt_q is not None: PL.LT_CORR = dict(q=lt_q)
    if ens is not None: PL.ENSEMBLE = ens
    if tonal is not None: PL.TONAL = tonal
    if rep_em is not None: PL.REP_EM = rep_em
    if w is not None: PL.ENS_WEIGHTS = w
    return PL.enhance_signal(x, 16000, **kw)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, **cfg)
        print(ek.report(r, str(cfg)[:150], by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

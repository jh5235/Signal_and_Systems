"""Evaluate pipeline variants (on top of PL.BEST) on a split, with per-category output."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time, collections
from nr import pipeline as PL
CATS = ["machine", "alarm_siren", "wind", "traffic_vehicle", "water", "animal", "aircraft_cockpit", "broadband_noise", "phone_dialing", "vehicle_interior"]
def by_cat(r):
    g = collections.defaultdict(list)
    for k, p in enumerate(r['pesq']): g[r['meta'][k][1]].append(p)
    return {k: np.mean(v) for k, v in g.items()}
def fn(x, fname="enhance_auto", **kw):
    return getattr(PL, fname)(x, **dict(PL.BEST, **kw))
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, **cfg); bc = by_cat(r)
        print(ek.report(r, str(cfg)[:160], by=("diff",), base=base), f"{time.time()-t:.0f}s")
        print("   " + " ".join(f"{c[:8]}:{bc.get(c, np.nan):.3f}" for c in CATS), flush=True)

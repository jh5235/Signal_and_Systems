import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import pipeline as PL
from run import by_cat, CATS
BP = dict(a=0.982, gmin_db=-36.5, xi_min_db=-30.1, rho=0.5, tsnr_mix=0.62, sfrac=0.122, spp_alpha=0.835, xi_h1=12.1,
          nscale=1.19, floor_cap_db=-47.2, hop=256, n_fft=1280, floor=True, nmode="geo")
def fn(x, fname="enhance", **kw):
    return getattr(PL, fname)(x, **dict(BP, **kw))
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, **cfg); bc = by_cat(r)
        print(ek.report(r, str(cfg)[:150], by=("diff",), base=base), f"{time.time()-t:.0f}s")
        print("   " + " ".join(f"{c[:8]}:{bc.get(c, np.nan):.3f}" for c in CATS), flush=True)

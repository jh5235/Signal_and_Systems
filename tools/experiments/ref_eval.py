"""Evaluate the reference-noise version on a split with a given bank (bank preloaded before forking)."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections
from nr import pipeline as PL, refnoise as R
BANK = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'bank_train.npz')
bank = R.load_bank(BANK)
R.cancel_reference(np.random.default_rng(0).standard_normal(48000) * 0.01, bank)    # warm caches before fork
def fn(x, max_depth=None, ref=None, block_depth=None, views=None, **kw):
    if views is not None and R.VIEWS != views:
        R.VIEWS = views; R._DEC.clear()
    if block_depth is not None:
        import inspect
        d = list(R.find_matches.__defaults__); names = list(inspect.signature(R.find_matches).parameters)[2:]
        d[names.index('block_depth')] = block_depth; R.find_matches.__defaults__ = tuple(d)
    if ref is not None: PL.REF.update(ref)
    if max_depth is not None:
        import functools
        R.find_matches.__defaults__ = tuple(list(R.find_matches.__defaults__[:-1]) + [max_depth])
    return PL.enhance_signal_ref(x, 16000, bank=bank, **kw)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, procs=12, **cfg)
        print(ek.report(r, str(cfg), by=("diff", "snr", "cat"), base=base), f"{time.time()-t:.0f}s", flush=True)

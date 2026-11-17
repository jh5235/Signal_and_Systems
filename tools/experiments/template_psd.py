"""Upper bound / realistic test: per-noise-type long-term spectral templates from TRAIN noise (profiles.json)."""
import sys, os, json; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import core as C, pipeline as PL
PROF = json.load(open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'noise_reduction_challenge', 'noisy_code', 'profiles.json'), encoding='utf-8'))
CODES = sorted(PROF)
TPL = np.array([10 ** (np.array(PROF[c]['psd_db']) / 10) for c in CODES])          # [88, 257] (512-pt grid)
c_, n_, m_ = ek.load(sys.argv[2] if len(sys.argv) > 2 else 'validation')
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]
def tpl_on_grid(t, nb):
    return np.interp(np.linspace(0, 1, nb), np.linspace(0, 1, len(t)), t)
def classify(x):
    """nearest template (log-spectral distance after level alignment) to the low-energy noisy spectrum"""
    Z, _ = C.stft(x, 512, 128); P = np.abs(Z) ** 2 + 1e-20
    N0 = C.init_noise(P, 0.1)
    L = np.log(N0[3:250]); best, bi = 1e9, 0
    for j in range(len(CODES)):
        T = np.log(TPL[j, 3:250]); d = L - T; d = d - np.median(d)
        e = np.mean(np.abs(d))
        if e < best: best, bi = e, j
    return bi
def fn(x, mode='none', mix=1.0, use_oracle=True):
    p = dict(PL.DEFAULTS, **PL.BEST)
    if mode == 'none':
        return PL.enhance_tonal(x, tonal=PL.TONAL, lt_corr=PL.LT_CORR, lt_prior=PL.LT_PRIOR, **PL.BEST)
    j = CODES.index(m_[find(x)][2]) if use_oracle else classify(x)
    T = TPL[j]
    def nfix(Nlt_mean, nb):
        Tg = tpl_on_grid(T, nb)
        r = np.log(Nlt_mean / Tg); a = np.exp(np.median(r))          # level of the template from our estimate
        return (a * Tg / Nlt_mean) ** mix
    return PL.enhance_tonal(x, tonal=PL.TONAL, lt_corr=dict(PL.LT_CORR, tpl_fix=nfix), lt_prior=PL.LT_PRIOR, **PL.BEST)
if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else 'validation'
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(fn, split, **cfg)
        print(ek.report(r, str(cfg), by=('diff',), base=base), f'{time.time()-t:.0f}s', flush=True)

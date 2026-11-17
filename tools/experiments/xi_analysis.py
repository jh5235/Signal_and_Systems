"""Where does our a priori SNR estimate go wrong? Replace parts with the oracle and measure PESQ."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from nr import core as C, pipeline as PL
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'validation'
c_, n_, m_ = ek.load(SPLIT)
_lk = {}
def find(x):
    k = x[:64].tobytes()
    if k not in _lk: _lk[k] = int(np.flatnonzero((n_[:, :64] == x[:64]).all(1))[0])
    return _lk[k]
BP = dict(PL.BEST)
def our_xi(x, p, nfix=None):
    """Replicates PL.enhance up to the final a priori SNR (single resolution, no repetition)."""
    p = dict(PL.DEFAULTS, **p)
    n_fft, hop = p["n_fft"], p["hop"]
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2 + 1e-20
    N, N0 = PL.estimate_noise(P, p); N = N * p["nscale"]
    if nfix is not None:
        N = N * nfix(N, n_fft, hop)
    F = PL.estimate_floor(P, N0, p, n_fft)
    N = np.maximum(N - F[:, None], 0.1 * N)
    gamma = P / N; g = C.gain_logmmse; xmin = 10 ** (p["xi_min_db"] / 10)
    xi, _, _ = C.dd_prior_snr(P, N, p["a"], p["xi_min_db"], g)
    xb, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], p["a"], p["xi_min_db"], g); xi = np.sqrt(xi * xb[:, ::-1])
    G = g(xi, gamma)
    xi2 = np.maximum(G ** 2 * gamma, xmin) ** p["tsnr_mix"] * xi ** (1 - p["tsnr_mix"]); G = g(xi2, gamma)
    s = C.istft(G * Z, meta); H = C.stft(np.maximum(s, 0), n_fft, hop)[0]
    xi3 = np.maximum((p["rho"] * G ** 2 * P + (1 - p["rho"]) * np.abs(H) ** 2) / N, xmin)
    return Z, meta, P, N, F, gamma, xi3, p
def fn(x, mode="ours", lvl=None):
    i = find(x)
    nfix = None
    if lvl is not None:
        D = np.abs(C.stft(n_[i] - c_[i], BP["n_fft"], BP["hop"])[0]) ** 2
        if lvl == "global":
            nfix = lambda N, a, b: D.sum() / N.sum()
        elif lvl == "perbin":
            nfix = lambda N, a, b: (D.mean(1, keepdims=True) / N.mean(1, keepdims=True))
        elif lvl == "perbin_down":   # only reduce where we over-estimate
            nfix = lambda N, a, b: np.minimum(1.0, D.mean(1, keepdims=True) / N.mean(1, keepdims=True))
        elif lvl == "perbin_up":     # only raise where we under-estimate
            nfix = lambda N, a, b: np.maximum(1.0, D.mean(1, keepdims=True) / N.mean(1, keepdims=True))
        elif lvl == "perbin_gross":   # only bins where we over-estimate by > 10 dB (noise-free bands)
            nfix = lambda N, a, b: np.where(D.mean(1, keepdims=True) < 0.1 * N.mean(1, keepdims=True),
                                            D.mean(1, keepdims=True) / N.mean(1, keepdims=True), 1.0)
        elif lvl == "perbin_gross6":
            nfix = lambda N, a, b: np.where(D.mean(1, keepdims=True) < 0.25 * N.mean(1, keepdims=True),
                                            D.mean(1, keepdims=True) / N.mean(1, keepdims=True), 1.0)
        elif lvl == "perbin_smooth":  # smooth (1/3-octave-like) correction only
            from scipy.ndimage import uniform_filter1d
            nfix = lambda N, a, b: np.exp(uniform_filter1d(np.log(D.mean(1, keepdims=True) / N.mean(1, keepdims=True)), 25, axis=0, mode='nearest'))
    Z, meta, P, N, F, gamma, xi, p = our_xi(x, BP, nfix)
    S = np.abs(C.stft(c_[i], p["n_fft"], p["hop"])[0]) ** 2
    xt = np.maximum(S / N, 10 ** (p["xi_min_db"] / 10))
    if mode == "oracle": xi = xt
    elif mode == "oracle_where_speech": xi = np.where(xt > 1, xt, xi)        # fix speech-dominant bins
    elif mode == "oracle_where_noise": xi = np.where(xt <= 1, xt, xi)        # fix noise-dominant bins
    elif mode == "oracle_lowfreq": f = np.fft.rfftfreq(p["n_fft"], 1/16000); xi = np.where(f[:, None] < 2000, xt, xi)
    elif mode == "oracle_highfreq": f = np.fft.rfftfreq(p["n_fft"], 1/16000); xi = np.where(f[:, None] >= 2000, xt, xi)
    elif mode == "oracle_smooth":   # oracle speech PSD smoothed in time like a realistic estimator would
        St = S.copy()
        for t in range(1, S.shape[1]): St[:, t] = 0.5 * St[:, t-1] + 0.5 * S[:, t]
        xi = np.maximum(St / N, 10 ** (p["xi_min_db"] / 10))
    elif mode == "oracle_env":      # oracle smooth spectral envelope only (no harmonic fine structure)
        from scipy.ndimage import uniform_filter1d
        xi = np.maximum(uniform_filter1d(S, 9, axis=0) / N, 10 ** (p["xi_min_db"] / 10))
    G = np.maximum(C.gain_logmmse(xi, gamma), 10 ** (p["gmin_db"] / 20))
    G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / P)))
    return C.istft(G * Z, meta)
if __name__ == "__main__":
    base = ek.noisy_scores(SPLIT)
    for mo in sys.argv[1].split(','):
        r = ek.evaluate(fn, SPLIT, mode=mo)
        print(ek.report(r, mo, by=('snr',), base=base), flush=True)

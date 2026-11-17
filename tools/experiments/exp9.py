import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time, collections
from scipy.ndimage import uniform_filter1d, minimum_filter1d, gaussian_filter1d
from nr import core as C
from exp8 import gfun

def noise_est(P, mode, sfrac=0.1, win=24, alpha=0.7, fw=3, bias=None, spp_alpha=0.8, xi_h1=15.0):
    N0 = C.init_noise(P, sfrac)
    if mode == "static":
        return np.tile(N0[:, None], (1, P.shape[1]))
    if mode == "minwin":   # smoothed periodogram + centred sliding minimum + bias, capped by nothing
        S = uniform_filter1d(P, fw, axis=0, mode="nearest")
        for t in range(1, S.shape[1]): S[:, t] = alpha * S[:, t-1] + (1-alpha) * S[:, t]
        M = minimum_filter1d(S, win, axis=1, mode="nearest")
        M = uniform_filter1d(M, win // 2, axis=1, mode="nearest")
        return M * (bias or 1.0)
    if mode == "spp":
        return C.noise_mmse_spp(P, N0, xi_h1, spp_alpha)
    if mode in ("geo", "min", "max", "wgeo"):
        a = C.noise_mmse_spp(P, N0, xi_h1, spp_alpha); b = C.noise_mmse_spp(P, N0, xi_h1, spp_alpha, backward=True)
        T = np.sqrt(a * b); S0 = np.tile(N0[:, None], (1, P.shape[1]))
        if mode == "geo": return np.sqrt(T * S0)
        if mode == "min": return np.minimum(T, S0)
        if mode == "max": return np.maximum(T, S0)
    if mode == "spp_bi":
        a = C.noise_mmse_spp(P, N0, xi_h1, spp_alpha); b = C.noise_mmse_spp(P, N0, xi_h1, spp_alpha, backward=True)
        return np.sqrt(a * b)

def enh9(x, n_fft=1024, hop=256, a=0.98, gmin_db=-30, xi_min_db=-25, rho=0.5, tsnr_mix=0.5, nmode="static",
         nscale=1.0, **nk):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N = noise_est(P, nmode, **nk) * nscale
    gamma = P / N; f1 = C.gain_logmmse; xmin = 10 ** (xi_min_db / 10)
    xi, _, _ = C.dd_prior_snr(P, N, a, xi_min_db, f1)
    xb, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], a, xi_min_db, f1); xi = np.sqrt(xi * xb[:, ::-1])
    G = f1(xi, gamma)
    xi2 = np.maximum(G ** 2 * gamma, xmin) ** tsnr_mix * xi ** (1 - tsnr_mix)
    G = f1(xi2, gamma)
    s = C.istft(G * Z, meta); H = C.stft(np.maximum(s, 0), n_fft, hop)[0]
    xi3 = np.maximum((rho * G ** 2 * P + (1 - rho) * np.abs(H) ** 2) / N, xmin)
    G = np.maximum(f1(xi3, gamma), 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

def by_cat(r):
    g = collections.defaultdict(list)
    for k, p in enumerate(r['pesq']): g[r['meta'][k][1]].append(p)
    return {k: np.mean(v) for k, v in g.items()}

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    cats = ["machine", "alarm_siren", "wind", "traffic_vehicle", "water", "animal", "aircraft_cockpit", "broadband_noise"]
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh9, split, **cfg); bc = by_cat(r)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s")
        print("   " + " ".join(f"{c[:8]}:{bc[c]:.3f}" for c in cats), flush=True)

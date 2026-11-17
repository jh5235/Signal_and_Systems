import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import core as C

def gfun(name):
    return {"wiener": C.gain_wiener, "logmmse": C.gain_logmmse}[name]

def enh8(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25, xi_min_db=-25, ncausal=True,
         step="tsnr", g1="logmmse", g2="logmmse", rho=0.5, nl="hwr", hscale=1.0, tsnr_mix=1.0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N0 = C.init_noise(P, sfrac); N = np.tile(N0[:, None], (1, P.shape[1]))
    gamma = P / N
    f1 = gfun(g1)
    xi, _, G = C.dd_prior_snr(P, N, a, xi_min_db, f1)
    if ncausal:
        xb, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], a, xi_min_db, f1); xi = np.sqrt(xi * xb[:, ::-1])
    G = f1(xi, gamma)
    xmin = 10 ** (xi_min_db / 10)
    if step in ("tsnr", "hrnr"):
        xi2 = np.maximum(G ** 2 * gamma, xmin)
        xi2 = xi2 ** tsnr_mix * xi ** (1 - tsnr_mix)
        G = gfun(g2)(xi2, gamma)
    if step == "hrnr":
        s = C.istft(G * Z, meta)
        h = np.maximum(s, 0) if nl == "hwr" else np.abs(s)
        H = C.stft(h * hscale, n_fft, hop)[0]
        xi3 = np.maximum((rho * G ** 2 * P + (1 - rho) * np.abs(H) ** 2) / N, xmin)
        G = gfun(g2)(xi3, gamma)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh8, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

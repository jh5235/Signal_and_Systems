import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time, json, os
from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter
from nr import core as C

def enh6(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25, xi_min_db=-25, nscale=1.0, gain="logmmse",
         gexp=1.0, med=(1, 1), vad_db=None, vad_thr=0.5, q=0.5, ncausal=False, gsmooth=0.0, hpf=0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N0 = C.init_noise(P, sfrac) * nscale
    N = np.tile(N0[:, None], (1, P.shape[1]))
    gf = {"wiener": C.gain_wiener, "logmmse": C.gain_logmmse, "stsa": C.gain_mmse_stsa}[gain]
    xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, gf)
    if ncausal:   # average forward and backward decision-directed passes (offline)
        xib, gb, Gb = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], a, xi_min_db, gf)
        xi = np.sqrt(xi * xib[:, ::-1]); G = gf(xi, gamma)
    if gexp != 1.0:
        G = G ** gexp
    if med != (1, 1):
        G = median_filter(G, size=med, mode="nearest")
    if gsmooth:
        G = gaussian_filter(G, gsmooth, mode="nearest")
    gmin = 10 ** (gmin_db / 20)
    G = np.maximum(G, gmin)
    if vad_db is not None:   # extra attenuation in frames with low speech presence
        p = C.spp(xi, gamma, q)
        fp = uniform_filter1d(p.mean(0), 3, mode="nearest")
        att = np.where(fp < vad_thr, 10 ** (vad_db / 20), 1.0)
        G = np.maximum(G * att[None, :], gmin * 10 ** (vad_db / 20))
    if hpf:
        f = np.fft.rfftfreq(n_fft, 1 / 16000); G[f < hpf] *= 0.1
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh6, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

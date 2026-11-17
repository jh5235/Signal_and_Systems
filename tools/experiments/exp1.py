import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import core as C

def enh(x, noise="spp", gain="logmmse", a=0.98, gmin_db=-20, xi_min_db=-25, omlsa=False, q=0.5,
        n_fft=512, hop=128, bidir=False, init_frac=0.1, spp_alpha=0.8, xi_h1=15.0, ms_win=96, ms_bias=1.5,
        nscale=1.0, sfrac=0.2, smode="frames", hybrid=False, hpf=0):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    if noise == "spp":
        init = C.init_noise(P, init_frac)
        N = C.noise_mmse_spp(P, init, xi_h1, spp_alpha)
        if bidir:
            Nb = C.noise_mmse_spp(P, init, xi_h1, spp_alpha, backward=True)
            N = np.sqrt(N * Nb)
    elif noise == "imcra":
        N = C.noise_imcra(P)
    elif noise == "ms":
        N = C.noise_min_stats(P, ms_win, bias=ms_bias)
    elif noise == "static":
        if smode == "frames":
            n0 = C.init_noise(P, sfrac)
        else:   # per-bin mean of lowest sfrac fraction
            k = max(2, int(sfrac * P.shape[1])); n0 = np.sort(P, 1)[:, :k].mean(1)
        N = np.tile(n0[:, None], (1, P.shape[1]))
        if hybrid:
            Nt = C.noise_mmse_spp(P, n0, xi_h1, spp_alpha)
            N = np.maximum(N, Nt)
    N = N * nscale
    if hpf:
        f = np.fft.rfftfreq(n_fft, 1/16000)
        N[f < hpf] = np.maximum(N[f < hpf], 1e3 * P[f < hpf])
    gf = {"wiener": C.gain_wiener, "logmmse": C.gain_logmmse, "stsa": C.gain_mmse_stsa}[gain]
    xi, gamma, G = C.dd_prior_snr(P, N, a, xi_min_db, gf)
    gmin = 10 ** (gmin_db / 20)
    if omlsa:
        p = C.spp(xi, gamma, q)
        G = G ** p * gmin ** (1 - p)
    else:
        G = np.maximum(G, gmin)
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    base = ek.noisy_scores("validation")
    configs = eval(sys.argv[1]) if len(sys.argv) > 1 else [{}]
    for cfg in configs:
        t = time.time(); r = ek.evaluate(enh, "validation", **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, time
from nr import core as C

def enh7(x, n_fft=1024, hop=256, sfrac=0.1, gmin_db=-25, xi_min_db=-25, gain="logmmse", mix=0.0, a=0.98, ncausal=True, **ck):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2
    N0 = C.init_noise(P, sfrac); N = np.tile(N0[:, None], (1, P.shape[1]))
    gamma = P / N
    xi = C.cepstral_xi(P, N, n_fft=n_fft, xi_min_db=xi_min_db, **ck)
    if mix:
        gf = C.gain_logmmse
        xd, _, _ = C.dd_prior_snr(P, N, a, xi_min_db, gf)
        if ncausal:
            xb, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], a, xi_min_db, gf); xd = np.sqrt(xd * xb[:, ::-1])
        xi = xi ** (1 - mix) * xd ** mix
    gf = {"wiener": C.gain_wiener, "logmmse": C.gain_logmmse}[gain]
    G = np.maximum(gf(xi, gamma), 10 ** (gmin_db / 20))
    return C.istft(G * Z, meta)

if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "validation"
    base = ek.noisy_scores(split)
    for cfg in eval(sys.argv[1]):
        t = time.time(); r = ek.evaluate(enh7, split, **cfg)
        print(ek.report(r, str(cfg), by=("diff",), base=base), f"{time.time()-t:.0f}s", flush=True)

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from nr import core as C
c_, n_, m_ = ek.load('validation')
def gains(x, n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-25):
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z)**2
    N = np.tile(C.init_noise(P, sfrac)[:, None], (1, P.shape[1]))
    xi, g, G = C.dd_prior_snr(P, N, a, -25, C.gain_logmmse)
    xib, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], a, -25, C.gain_logmmse)
    G = C.gain_logmmse(np.sqrt(xi*xib[:, ::-1]), g)
    return np.maximum(G, 10**(gmin_db/20)), meta
def part(i, which):
    G, meta = gains(n_[i])
    S = C.stft(c_[i], 1024, 256)[0]; D = C.stft(n_[i]-c_[i], 1024, 256)[0]
    s_out = C.istft(G*S, meta); d_out = C.istft(G*D, meta)
    y = {"speech_only": s_out, "noise_only": c_[i] + d_out, "both": s_out + d_out}[which]
    return ek.pesq_wb(c_[i], y)
from multiprocessing import Pool
if len(sys.argv) == 1:
  for w in ["speech_only", "noise_only", "both"]:
      with Pool(15) as p: r = p.starmap(part, [(i, w) for i in range(len(c_))])
      print(w, np.mean(r))

def part2(i, mode):
    G, meta = gains(n_[i])
    S = C.stft(c_[i], 1024, 256)[0]; D = C.stft(n_[i]-c_[i], 1024, 256)[0]
    e = 10*np.log10((np.abs(S)**2).sum(0)+1e-12); act = e > e.max() - 35
    Gd = G.copy()
    if mode == "zero_absent": Gd[:, ~act] = 0
    if mode == "zero_present": Gd[:, act] = 0
    if mode == "absent_m10": Gd[:, ~act] *= 10**(-10/20)
    d_out = C.istft(Gd*D, meta); s_out = C.istft(G*S, meta)
    return ek.pesq_wb(c_[i], c_[i] + d_out), ek.pesq_wb(c_[i], s_out + d_out), act.mean()
if __name__ == "__main__" and len(sys.argv) > 1:
    for w in ["zero_absent", "zero_present", "absent_m10"]:
        with Pool(15) as p: r = np.array(p.starmap(part2, [(i, w) for i in range(len(c_))]))
        print(w, 'noise_only %.3f  both %.3f  active frac %.2f' % tuple(r.mean(0)))

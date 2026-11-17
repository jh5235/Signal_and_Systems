"""Discriminability of phase-coherence features for speech- vs noise-dominated TF bins (oracle labels)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek, collections
from multiprocessing import Pool
from nr import core as C, pipeline as PL
c_, n_, m_ = ek.load('validation')
p = dict(PL.DEFAULTS, **PL.BEST); n_fft, hop = p['n_fft'], p['hop']
def princ(a): return np.angle(np.exp(1j * a))
def feats(i):
    x = n_[i]
    Z, meta = C.stft(x, n_fft, hop); P = np.abs(Z) ** 2 + 1e-20
    N, N0 = PL.estimate_noise(P, p); N = N * p['nscale']
    xi, gamma, G = PL._prior_snr(Z, meta, P, N, p, C.gain_logmmse, C.gain_logmmse)
    S = np.abs(C.stft(c_[i], n_fft, hop)[0]) ** 2; D = np.abs(C.stft(x - c_[i], n_fft, hop)[0]) ** 2
    F, T = Z.shape
    k = np.arange(F)[:, None]
    ph = np.angle(Z)
    d = np.zeros_like(ph); d[:, 1:] = princ(ph[:, 1:] - ph[:, :-1] - 2 * np.pi * k * hop / n_fft)
    e = np.exp(1j * d)
    coh3 = np.abs(e + np.roll(e, 1, 1) + np.roll(e, -1, 1)) / 3
    dd = np.abs(princ(d - np.roll(d, 1, 1)))                  # IF change between frames
    f = np.fft.rfftfreq(n_fft, 1 / 16000)
    sel = (f[:, None] < 4000) & (xi > 10 ** -0.5) & (xi < 10) & (np.arange(T)[None, :] > 1) & (np.arange(T)[None, :] < T - 2)
    lab = (S > D)
    return coh3[sel], dd[sel], lab[sel], np.full(sel.sum(), m_[i][1] in ('alarm_siren', 'machine', 'phone_dialing', 'bell'))
def auc(score, lab):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(len(o))
    npos = lab.sum(); nneg = len(lab) - npos
    return (r[lab].sum() - npos * (npos - 1) / 2) / (npos * nneg)
with Pool(15) as pool: R = pool.map(feats, range(len(c_)))
coh = np.concatenate([r[0] for r in R]); dd = np.concatenate([r[1] for r in R]); lab = np.concatenate([r[2] for r in R]); ton = np.concatenate([r[3] for r in R])
print('ambiguous bins', len(lab), 'speech-dominated fraction %.3f' % lab.mean())
print('AUC coh3 (higher=speech): all %.3f | tonal-noise files %.3f | other files %.3f' % (auc(coh, lab), auc(coh[ton], lab[ton]), auc(coh[~ton], lab[~ton])))
print('AUC -dIF (smaller change=speech): all %.3f | tonal %.3f | other %.3f' % (auc(-dd, lab), auc(-dd[ton], lab[ton]), auc(-dd[~ton], lab[~ton])))
for q in (0.3, 0.5, 0.7, 0.9):
    m = coh < q
    print(f'coh3<{q}: frac {m.mean():.3f}, speech-dominated among them {lab[m].mean():.3f} (vs {lab.mean():.3f} overall)')

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from multiprocessing import Pool
from nr import core as C, pipeline as PL
from run2 import BP
c_, n_, m_ = ek.load('validation')
def gain_of(x):
    """Recover the effective gain of the pipeline: run on noisy, compare STFTs."""
    p = dict(BP); n_fft, hop = p['n_fft'], p['hop']
    Z, meta = C.stft(x, n_fft, hop); y = PL.enhance(x, **p); Y = C.stft(y, n_fft, hop)[0]
    return np.clip(np.abs(Y) / (np.abs(Z) + 1e-12), 0, 1), meta, n_fft, hop
def job(i, band):
    G, meta, n_fft, hop = gain_of(n_[i])
    S = C.stft(c_[i], n_fft, hop)[0]; D = C.stft(n_[i]-c_[i], n_fft, hop)[0]
    f = np.fft.rfftfreq(n_fft, 1/16000)
    Gd = G.copy()
    if band: Gd[(f >= band[0]) & (f < band[1])] = 0
    return ek.pesq_wb(c_[i], C.istft(G*S, meta) + C.istft(Gd*D, meta))
for band in [None, (0, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 8001)]:
    with Pool(15) as p: r = p.starmap(job, [(i, band) for i in range(len(c_))])
    print(band, round(float(np.mean(r)), 4), flush=True)

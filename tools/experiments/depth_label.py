import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, evalkit as ek
from multiprocessing import Pool
from nr import refnoise as R
bank = R.load_bank(os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'bank_train.npz'))
R.cancel_reference(np.random.default_rng(0).standard_normal(48000) * 0.01, bank)
c_, n_, m_ = ek.load('validation')
def f(i):
    x = n_[i]; d = x - c_[i]
    found, cl = R.find_matches(x, bank, min_ncc=0.0, max_matches=1, return_cands=True, max_depth=99)
    out = []
    for j, rel, nc, L, depth, lag in cl:
        if L < 4000: continue
        T = bank[j][1].astype(np.float64); n = len(x)
        start, end = max(0, -lag), min(n, len(T) - lag)
        seg = T[start + lag:end + lag]; xs = x[start:end]
        g = float(np.dot(xs, seg) / np.dot(seg, seg))
        q_true = float(np.sum((d[start:end] - g * seg) ** 2) / np.sum(d[start:end] ** 2))   # residual noise share
        out.append((bank[j][0] == m_[i][2], nc, depth, q_true, float(m_[i][4])))
    return out
if __name__ == "__main__":
    with Pool(12) as p: Rr = p.map(f, range(len(c_)))
    A = np.array([o for r in Rr for o in r], float)
    exact = A[:, 3] < 0.1
    print('candidates', len(A), 'exact (true residual < 0.1)', int(exact.sum()))
    print('depth for exact  :', np.percentile(A[exact, 2], [50, 75, 90, 95, 99]).round(3))
    print('depth for others :', np.percentile(A[~exact, 2], [1, 5, 10, 25, 50]).round(3))
    for thr in (0.1, 0.2, 0.3, 0.5):
        acc = A[:, 2] < thr
        print(f'depth<{thr}: accept {int(acc.sum())}, exact among accepted {int((acc & exact).sum())}, non-exact accepted {int((acc & ~exact).sum())}, exact rejected {int((~acc & exact).sum())}')
    for snr in (0, 5, 10, 15):
        s_ = (A[:, 4] == snr) & exact
        if s_.any(): print('exact, SNR', snr, 'depth p90 %.3f' % np.percentile(A[s_, 2], 90))

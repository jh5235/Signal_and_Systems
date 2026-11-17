"""Parallel PESQ-WB / SNR evaluation harness on cached train/validation data."""
from __future__ import annotations
import os, sys, time, collections
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
import numpy as np
from pesq import pesq, PesqError

CACHE = os.environ.get("NR_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache"))
_DATA = {}

def load(split):
    if split not in _DATA:
        z = np.load(f"{CACHE}/{split}.npz")
        _DATA[split] = (z["clean"].astype(np.float64), z["noisy"].astype(np.float64), z["meta"])
    return _DATA[split]

def snr_db(c, e):
    return float(10 * np.log10(np.sum(c ** 2) / max(np.sum((c - e) ** 2), 1e-20)))

def pesq_wb(c, e):
    try:
        return float(pesq(16000, c, e, "wb", on_error=PesqError.RAISE_EXCEPTION))
    except Exception:
        return float("nan")

_G = {}
def _work(i):
    fn, kw, split = _G["fn"], _G["kw"], _G["split"]
    c, n, _ = load(split)
    y = fn(n[i], **kw)
    assert y.shape == n[i].shape
    return i, pesq_wb(c[i], y), snr_db(c[i], y)

def evaluate(fn, split="validation", idx=None, procs=15, **kw):
    c, n, meta = load(split)
    idx = list(range(len(c))) if idx is None else list(idx)
    _G.update(fn=fn, kw=kw, split=split)
    with Pool(procs) as p:
        res = p.map(_work, idx, chunksize=4)
    res.sort()
    P = np.array([r[1] for r in res]); S = np.array([r[2] for r in res])
    return dict(idx=idx, pesq=P, snr=S, meta=meta[idx])

def noisy_scores(split):
    """Cached input (noisy) PESQ/SNR."""
    path = f"{CACHE}/{split}_noisy_scores.npz"
    if os.path.exists(path):
        z = np.load(path); return z["pesq"], z["snr"]
    r = evaluate(lambda x: x, split)
    np.savez(path, pesq=r["pesq"], snr=r["snr"]); return r["pesq"], r["snr"]

def report(r, name="", by=("diff",), base=None):
    m = r["meta"]
    s = f"{name:30s} PESQ {np.nanmean(r['pesq']):.4f}  SNR {np.mean(r['snr']):6.2f}  nan {int(np.isnan(r['pesq']).sum())}"
    if base is not None:
        s += f"  dPESQ {np.nanmean(r['pesq'] - base[0][r['idx']]):+.4f}"
    lines = [s]
    cols = {"cat": 1, "code": 2, "diff": 3, "snr": 4}
    for b in by:
        g = collections.defaultdict(list)
        for k, (p, q) in enumerate(zip(r["pesq"], r["snr"])):
            g[m[k][cols[b]]].append((p, q))
        lines.append("   " + " | ".join(f"{k}:{np.nanmean([a for a,_ in v]):.3f}" for k, v in sorted(g.items())))
    return "\n".join(lines)

"""Validation evaluation of the submitted algorithm (PESQ-WB 16 kHz + SNR), like the organisers' notebook.

    python tools/evaluate.py --data ../noise_reduction_challenge --split validation [--out results]

Reads <data>/<split>/samples.csv, enhances every noisy file with main.enhance_signal (noisy input only),
and compares with the clean reference. Writes per-file and summary CSVs.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from pesq import PesqError, pesq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nr.pipeline import enhance_signal, enhance_signal_ref  # noqa: E402
BANK = None


def _init_worker(bank_path):
    """Windows(spawn)에서는 워커가 모듈을 새로 불러와 전역 BANK가 비므로, 경로를 받아 워커마다 뱅크를 직접 로드한다."""
    global BANK
    if bank_path:
        from nr.refnoise import load_bank
        BANK = load_bank(bank_path)


def snr_db(c, e):
    return float(10 * np.log10(np.sum(c ** 2) / max(np.sum((c - e) ** 2), 1e-20)))


def pesq_wb(c, e):
    try:
        return float(pesq(16000, c, e, mode="wb", on_error=PesqError.RAISE_EXCEPTION))
    except Exception:
        return float("nan")


def work(args):
    data, row = args
    c, fs = sf.read(str(data / row["clean_path"]), dtype="float64")
    n, _ = sf.read(str(data / row["noisy_path"]), dtype="float64")
    if BANK is not None:
        y = enhance_signal_ref(n, fs, bank=BANK)
    else:
        y = enhance_signal(n, fs)
    return dict(segment_id=row["segment_id"], noise_category=row["noise_category"], noise_code=row["noise_code"],
                noise_difficulty=row["noise_difficulty"], target_snr_db=row["target_snr_db"],
                pesq_noisy=pesq_wb(c, n), pesq=pesq_wb(c, y), snr_noisy=snr_db(c, n), snr=snr_db(c, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "noise_reduction_challenge"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--bank", default="", help="reference-noise bank (.npz) -> evaluate the reference version")
    a = ap.parse_args()
    global BANK
    if a.bank:
        from nr.refnoise import load_bank, cancel_reference
        BANK = load_bank(a.bank)
        cancel_reference(np.random.default_rng(0).standard_normal(48000) * 0.01, BANK)   # warm caches before fork
    data = Path(a.data)
    with open(data / a.split / "samples.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with Pool(a.procs, initializer=_init_worker, initargs=(a.bank,)) as p:
        res = p.map(work, [(data, r) for r in rows], chunksize=4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.split}_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    lines = []
    def summ(name, rs):
        P = np.array([r["pesq"] for r in rs]); P0 = np.array([r["pesq_noisy"] for r in rs])
        S = np.array([r["snr"] for r in rs]); S0 = np.array([r["snr_noisy"] for r in rs])
        lines.append([name, len(rs), f"{np.nanmean(P0):.4f}", f"{np.nanmean(P):.4f}", f"{np.nanmean(P - P0):+.4f}",
                      f"{np.mean(S0):.2f}", f"{np.mean(S):.2f}", int(np.isnan(P).sum())])
    summ("ALL", res)
    for key in ("noise_difficulty", "target_snr_db", "noise_category"):
        g = defaultdict(list)
        for r in res: g[r[key]].append(r)
        for k in sorted(g): summ(f"{key}={k}", g[k])
    hdr = ["group", "n", "pesq_noisy", "pesq", "delta_pesq", "snr_noisy", "snr", "pesq_nan"]
    with open(out / f"{a.split}_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(lines)
    for l in lines:
        print("{:42s} {:>4} {:>8} {:>8} {:>8} {:>7} {:>7} {:>3}".format(*map(str, l)))


if __name__ == "__main__":
    main()

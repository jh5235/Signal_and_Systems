"""Validation score of the packaged model (03_model) for a given profile.

Usage: python evaluate_model.py [--profile optuna|pesq|balanced] [--jobs 14]
Writes 02_result/validation_<profile>_scores.csv
"""
import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_model"))
from dsp import enhance  # noqa: E402

DATA = Path(os.environ.get("NR_DATA", HERE.parents[2] / "00_원자료" / "noise_reduction_challenge"))


def score(a):
    c, n, prof = a
    s, sr = sf.read(DATA / c, dtype="float64")
    x, _ = sf.read(DATA / n, dtype="float64")
    y = np.clip(np.nan_to_num(enhance(x, sr=sr, profile=prof)), -1, 1)
    snr = lambda e: 10 * np.log10(np.sum(s ** 2) / max(np.sum((s - e) ** 2), 1e-20))
    return pesq(sr, s, x, "wb"), snr(x), pesq(sr, s, y, "wb"), snr(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="optuna", choices=["optuna", "pesq", "balanced"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    df = pd.read_csv(DATA / "validation" / "samples.csv", encoding="utf-8-sig")
    with Pool(a.jobs) as p:
        r = p.map(score, [(c, n, a.profile) for c, n in zip(df.clean_path, df.noisy_path)], chunksize=4)
    df[["pesq_noisy", "snr_noisy", "pesq", "snr"]] = np.array(r)
    out = HERE.parent / "02_result"
    df.drop(columns=["clean_path", "noisy_path"]).to_csv(out / f"validation_{a.profile}_scores.csv", index=False)
    print(f"[validation / {a.profile}] files={len(df)}")
    print(f"  Noisy   : PESQ {df.pesq_noisy.mean():.4f} | SNR {df.snr_noisy.mean():.2f} dB")
    print(f"  Enhanced: PESQ {df.pesq.mean():.4f} | SNR {df.snr.mean():.2f} dB")
    print(df.groupby("target_snr_db")[["pesq_noisy", "pesq", "snr"]].mean().round(3).to_string())
    print(df.groupby("noise_category")[["pesq_noisy", "pesq"]].mean().round(3).to_string())

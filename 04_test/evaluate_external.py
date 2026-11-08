"""외부 데이터(VoiceBank-DEMAND 16k, test 824개)로 03_model 검증.

데이터: https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k (CC-BY-4.0)
Usage : python evaluate_external.py [--profiles optuna pesq balanced] [--jobs 14]
결과  : result/external_<profile>_scores.csv, result/summary.csv
"""
import argparse, io, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "03_model"))
from dsp import enhance  # noqa: E402

REPO = "JacobLinCool/VoiceBank-DEMAND-16k"
PARQUET = HERE / "data" / "data" / "test-00000-of-00001.parquet"
_DF = None


def fetch():
    if PARQUET.is_file():
        return
    from huggingface_hub import hf_hub_download
    for f in ("data/test-00000-of-00001.parquet", "license.txt", "README.md"):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=HERE / "data")


def _init():
    global _DF
    _DF = pd.read_parquet(PARQUET)


def snr_db(s, y):
    return 10 * np.log10(np.sum(s ** 2) / max(np.sum((s - y) ** 2), 1e-20))


def score(a):
    i, prof = a
    s, sr = sf.read(io.BytesIO(_DF.clean[i]["bytes"]), dtype="float64")
    x, _ = sf.read(io.BytesIO(_DF.noisy[i]["bytes"]), dtype="float64")
    n = min(len(s), len(x)); s, x = s[:n], x[:n]
    y = x if prof == "noisy" else np.clip(np.nan_to_num(enhance(x, sr=sr, profile=prof)), -1, 1)
    try:
        p = pesq(sr, s, y, "wb")
    except Exception:
        p = np.nan
    return p, snr_db(s, y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["optuna", "pesq", "balanced"])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    fetch()
    ids = pd.read_parquet(PARQUET, columns=["id"]).id.tolist()
    out = HERE / "result"; out.mkdir(exist_ok=True)
    rows = []
    with Pool(a.jobs, initializer=_init) as pool:
        for prof in ["noisy"] + a.profiles:
            r = np.array(pool.map(score, [(i, prof) for i in range(len(ids))], chunksize=4))
            pd.DataFrame({"id": ids, "pesq": r[:, 0], "snr": r[:, 1]}).to_csv(out / f"external_{prof}_scores.csv", index=False)
            rows.append(dict(profile=prof, files=len(ids), pesq=np.nanmean(r[:, 0]), snr=np.nanmean(r[:, 1]), pesq_failed=int(np.isnan(r[:, 0]).sum())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(out / "summary.csv", index=False)

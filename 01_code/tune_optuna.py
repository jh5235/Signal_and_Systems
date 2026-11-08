"""Optuna (TPE) tuning of the Log-MMSE DSP denoiser (denoise-dsp 1회차 제출물).

Objective : mean PESQ-WB on a fixed random subset of the train split.
Check     : baseline ("pesq" profile) and best trial are scored on validation.
Usage     : python tune_optuna.py [--trials 300] [--subset 400] [--jobs 14] [--study NAME]
Resume    : same --study name continues from 02_result/optuna.db
"""
import argparse, json, os, time, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import optuna
from pesq import pesq

from dsp import PROFILES, enhance

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("NR_DATA", HERE.parents[2] / "00_원자료" / "noise_reduction_challenge"))
OUT = HERE.parent / "02_result"
SR = 16000
_D = {}


def load(split, idx=None):
    df = pd.read_csv(DATA / split / "samples.csv", encoding="utf-8-sig")
    if idx is not None:
        df = df.iloc[idx].reset_index(drop=True)
    clean = [sf.read(DATA / p, dtype="float64")[0] for p in df.clean_path]
    noisy = [sf.read(DATA / p, dtype="float64")[0] for p in df.noisy_path]
    return df, clean, noisy


def _init(clean, noisy):
    _D["c"], _D["n"] = clean, noisy


def _one(args):
    i, kw = args
    c, n = _D["c"][i], _D["n"][i]
    try:
        y = np.clip(np.nan_to_num(enhance(n, sr=SR, profile="pesq", **kw)), -1, 1)
        p = pesq(SR, c, y, "wb")
    except Exception:
        p = np.nan
    snr = 10 * np.log10(np.sum(c ** 2) / max(np.sum((c - y) ** 2), 1e-20)) if p == p else np.nan
    return p, snr


def score(pool, n, kw):
    r = np.array(pool.map(_one, [(i, kw) for i in range(n)], chunksize=4))
    return float(np.nanmean(r[:, 0])), float(np.nanmean(r[:, 1]))


def suggest(t):
    return dict(
        n_fft=t.suggest_categorical("n_fft", [768, 1024, 1536, 2048]),
        hop=t.suggest_categorical("hop", [128, 192, 256]),
        frac=t.suggest_float("frac", 0.03, 0.3),
        w=t.suggest_float("w", 0.0, 1.0),
        a_pow=t.suggest_float("a_pow", 0.8, 0.98),
        xi_h1_db=t.suggest_float("xi_h1_db", 5.0, 25.0),
        noise_scale=t.suggest_float("noise_scale", 0.8, 2.5),
        a=t.suggest_float("a", 0.85, 0.99),
        xi_min_db=t.suggest_float("xi_min_db", -35.0, -5.0),
        gmin_db=t.suggest_float("gmin_db", -45.0, -8.0),
        hp_hz=t.suggest_float("hp_hz", 60.0, 400.0),
        hp_order=t.suggest_int("hp_order", 2, 6),
        hp_floor_db=t.suggest_float("hp_floor_db", -40.0, -5.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--subset", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--study", default="dsp_tpe")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    idx = np.random.RandomState(1).permutation(1296)[: a.subset]   # same subset as original coordinate descent
    _, tc, tn = load("train", idx)
    vdf, vc, vn = load("validation")
    base = {k: v for k, v in PROFILES["pesq"].items()}

    study = optuna.create_study(
        study_name=a.study, storage=f"sqlite:///{OUT / 'optuna.db'}", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed, multivariate=True, n_startup_trials=30))
    if not study.trials:
        study.enqueue_trial({k: v for k, v in base.items()})   # warm start from the submitted profile

    with Pool(a.jobs, initializer=_init, initargs=(tc, tn)) as pool:
        def objective(t):
            p, s = score(pool, len(tc), suggest(t))
            t.set_user_attr("snr", s)
            return p
        todo = a.trials - len(study.trials)
        t0 = time.time()
        if todo > 0:
            study.optimize(objective, n_trials=todo, gc_after_trial=True,
                           callbacks=[lambda st, tr: print(
                               f"trial {tr.number:3d} PESQ {tr.value:.4f} SNR {tr.user_attrs['snr']:.2f} | best {st.best_value:.4f} | {time.time()-t0:.0f}s",
                               flush=True)])

    best = {**base, **study.best_params}
    print("BEST train PESQ", round(study.best_value, 4)); print(json.dumps(best, indent=1))
    (OUT / "best_params.json").write_text(json.dumps(best, indent=1))
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)

    with Pool(a.jobs, initializer=_init, initargs=(vc, vn)) as pool:
        for name, kw in (("baseline(pesq profile)", {}), ("optuna best", best)):
            p, s = score(pool, len(vc), kw)
            print(f"validation [{name}] PESQ {p:.4f} SNR {s:.2f} dB", flush=True)


if __name__ == "__main__":
    main()

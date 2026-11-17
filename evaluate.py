"""Run the exact submission path on a public split and score the saved WAVs.
Clean WAVs are read here only to compute PESQ-WB and SNR (pip install pesq)."""
import argparse, csv, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf


def score(task):
    from pesq import pesq
    clean_p, noisy_p, out_p = task
    c, sr = sf.read(clean_p, dtype="float64"); n, _ = sf.read(noisy_p, dtype="float64")
    o, sro = sf.read(out_p, dtype="float64")
    assert sr == sro == 16000 and c.shape == o.shape
    snr = lambda y: float(10 * np.log10(np.sum(c ** 2) / max(np.sum((c - y) ** 2), 1e-20)))
    return float(pesq(sr, c, n, "wb")), snr(n), float(pesq(sr, c, o, "wb")), snr(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="folder holding train/ and validation/")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--independent", action="store_true")
    ap.add_argument("--save", default="results")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.data).resolve(); here = Path(__file__).resolve().parent
    rows = list(csv.DictReader(open(root / a.split / "samples.csv", encoding="utf-8-sig")))
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(here / "enhance.py"), "--input", str(root / a.split / "noisy"),
               "--output", tmp, "--jobs", str(a.jobs)]
        subprocess.run(cmd, check=True)
        tasks = [(root / r["clean_path"], root / r["noisy_path"], Path(tmp) / Path(r["noisy_path"]).name) for r in rows]
        with ProcessPoolExecutor(a.jobs) as ex:
            sc = list(ex.map(score, tasks, chunksize=4))
    names = ("pesq_noisy", "snr_noisy", "pesq", "snr")
    for r, v in zip(rows, sc):
        r.update(zip(names, v))
    Path(a.save).mkdir(exist_ok=True)
    dst = Path(a.save) / f"{a.split}_scores.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    m = {k: float(np.mean([r[k] for r in rows])) for k in names}
    print(f"[{a.split}] files={len(rows)}  Noisy PESQ-WB {m['pesq_noisy']:.4f} SNR {m['snr_noisy']:.2f} dB"
          f"  ->  Output PESQ-WB {m['pesq']:.4f} SNR {m['snr']:.2f} dB   ({dst})")


if __name__ == "__main__":
    main()

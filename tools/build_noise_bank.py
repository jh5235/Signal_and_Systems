"""Build the reference noise bank (nr/noise_bank.npz) from the public data: d = noisy - clean per file,
grouped by noise_code and stitched into timelines.

    python tools/build_noise_bank.py --data ../noise_reduction_challenge --splits train validation
    python tools/build_noise_bank.py --cache cache --splits train --out cache/bank_train.npz   (dev)
"""
from __future__ import annotations

import argparse, csv, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nr.refnoise import stitch, save_bank  # noqa: E402


def _stitch_code(args):
    code, pieces = args
    return code, stitch(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "noise_reduction_challenge"))
    ap.add_argument("--cache", default="")
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--out", default=str(ROOT / "nr" / "noise_bank.npz"))
    a = ap.parse_args()
    groups = defaultdict(list)
    for sp in a.splits:
        if a.cache:
            z = np.load(Path(a.cache) / f"{sp}.npz")
            C, N, M = z["clean"].astype(np.float64), z["noisy"].astype(np.float64), z["meta"]
            for i in range(len(C)):
                groups[str(M[i][2])].append(N[i] - C[i])
        else:
            import soundfile as sf
            with open(Path(a.data) / sp / "samples.csv", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    c, _ = sf.read(Path(a.data) / r["clean_path"], dtype="float64")
                    y, _ = sf.read(Path(a.data) / r["noisy_path"], dtype="float64")
                    groups[r["noise_code"]].append(y - c)
    t = time.time()
    with Pool(15) as p:
        res = p.map(_stitch_code, sorted(groups.items(), key=lambda kv: -len(kv[1])))
    bank = dict(res)
    save_bank(a.out, bank)
    tot = sum(len(x) for v in bank.values() for x in v)
    print(f"codes {len(bank)}, islands {sum(len(v) for v in bank.values())}, total {tot/16000/60:.1f} min, "
          f"file {Path(a.out).stat().st_size/1e6:.1f} MB, {time.time()-t:.0f}s")
    for code in sorted(bank, key=lambda c: -sum(len(x) for x in bank[c]))[:12]:
        print(f"  {code:16s} islands {len(bank[code]):3d}  longest {max(len(x) for x in bank[code])/16000:6.1f}s  total {sum(len(x) for x in bank[code])/16000:6.1f}s")


if __name__ == "__main__":
    main()

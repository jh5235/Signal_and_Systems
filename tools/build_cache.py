"""Cache train/validation clean+noisy waveforms into one npz for fast experiments."""
import csv, sys
from pathlib import Path
import numpy as np, soundfile as sf
root = Path(sys.argv[1]); out = Path(sys.argv[2])
for split in ("train", "validation"):
    rows = list(csv.DictReader(open(root / split / "samples.csv", encoding="utf-8-sig")))
    C = np.stack([sf.read(root / r["clean_path"], dtype="float32")[0] for r in rows])
    N = np.stack([sf.read(root / r["noisy_path"], dtype="float32")[0] for r in rows])
    meta = np.array([[r["segment_id"], r["noise_category"], r["noise_code"], r["noise_difficulty"], r["target_snr_db"]] for r in rows])
    np.savez(out / f"{split}.npz", clean=C, noisy=N, meta=meta)
    print(split, C.shape)

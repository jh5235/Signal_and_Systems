"""Denoise every WAV in an input folder and write same-named WAVs to an output folder.

Usage:
    python enhance.py                       # ./input -> ./output
    python enhance.py --input IN --output OUT [--profile optuna|pesq|balanced]
Output keeps the input's sampling rate, channel count, number of samples and
sample format (subtype).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp import enhance


def process_file(src: Path, dst: Path, profile: str = "optuna"):
    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)   # [samples, channels]
    y = np.stack([enhance(x[:, c], sr=sr, profile=profile) for c in range(x.shape[1])], axis=1)
    y = np.clip(np.nan_to_num(y), -1.0, 1.0)
    if info.channels == 1:
        y = y[:, 0]
    subtype = info.subtype if info.subtype in sf.available_subtypes(info.format) else None
    sf.write(str(dst), y, sr, subtype=subtype, format=info.format)
    out = sf.info(str(dst))
    assert (out.samplerate, out.channels, out.frames) == (info.samplerate, info.channels, info.frames)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="input folder (default: ./input)")
    ap.add_argument("--output", default=None, help="output folder (default: ./output)")
    ap.add_argument("--profile", default="optuna", choices=["optuna", "pesq", "balanced"],
                    help="optuna: Optuna-tuned (default) / pesq: original submission / balanced: keep more low-frequency speech")
    a = ap.parse_args()
    inp = Path(a.input) if a.input else (Path.cwd() / "input" if (Path.cwd() / "input").is_dir() else here / "input")
    out = Path(a.output) if a.output else inp.parent / "output"
    files = sorted(p for p in inp.iterdir() if p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit(f"No WAV files in {inp}")
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        process_file(f, out / f.name, a.profile)
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
    print(f"Done: {len(files)} files -> {out}")


if __name__ == "__main__":
    main()

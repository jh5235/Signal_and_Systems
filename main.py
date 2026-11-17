"""Speech denoising for the 2026 Signals & Systems challenge (classical DSP only, no ML/DL).

Usage:
    python main.py                       # input/ -> output/ (relative to this file)
    python main.py --input IN --output OUT [--jobs N]

Every WAV file under the input folder is enhanced and written to the output folder with the same
relative path and file name, the same sampling rate, channel count and number of samples.
Files are independent, so they can be processed in parallel (--jobs, default: up to 4 processes).
"""
from __future__ import annotations

import os
# one BLAS/FFT thread per process: files are processed in parallel by worker processes instead
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from nr.pipeline import enhance_signal, enhance_signal_ref
from nr.refnoise import BANK_PATH

HERE = Path(__file__).resolve().parent
USE_REF = BANK_PATH.exists()        # reference-noise version when the noise bank file is present


def process_file(src: Path, dst: Path, use_ref: bool = None) -> None:
    use_ref = USE_REF if use_ref is None else use_ref
    info = sf.info(str(src))
    x, fs = sf.read(str(src), dtype="float64", always_2d=True)
    y = np.empty_like(x)
    fn = enhance_signal_ref if use_ref else enhance_signal
    for ch in range(x.shape[1]):
        y[:, ch] = fn(x[:, ch], fs)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    subtype = info.subtype if info.subtype in sf.available_subtypes("WAV") else "PCM_16"
    if subtype.startswith("PCM") or subtype.startswith("ULAW") or subtype.startswith("ALAW"):
        y = np.clip(y, -1.0, 1.0 - 2.0 ** -15)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y if info.channels > 1 else y[:, 0], fs, subtype=subtype)


def safe_process(src: Path, dst: Path, use_ref: bool = None):
    """Enhance one file; on any error retry without the reference bank, then copy the input unchanged."""
    try:
        process_file(src, dst, use_ref)
        return None
    except Exception:
        try:
            process_file(src, dst, False)
            return None
        except Exception as e:  # never stop the batch
            pass
    x, fs = sf.read(str(src), dtype="float64")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, fs, subtype=sf.info(str(src)).subtype)
    return f"{src}: could not be enhanced"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "input"), help="input folder (default: ./input)")
    ap.add_argument("--output", default=str(HERE / "output"), help="output folder (default: ./output)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="parallel worker processes (default: min(4, CPU count); 1 = sequential)")
    ap.add_argument("--no-ref", action="store_true",
                    help="do not use the reference noise bank (pure single-file signal processing)")
    args = ap.parse_args(argv)
    use_ref = USE_REF and not args.no_ref
    print(f"mode: {'reference-noise bank + signal processing' if use_ref else 'signal processing only'}")
    src_root, dst_root = Path(args.input), Path(args.output)
    files = sorted(p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        print(f"No WAV files found in {src_root}", file=sys.stderr)
        return 1
    t0 = time.time()
    jobs = [(f, dst_root / f.relative_to(src_root)) for f in files]
    warnings = []
    done = 0
    if args.jobs <= 1:
        for src, dst in jobs:
            w = safe_process(src, dst, use_ref)
            if w:
                warnings.append(w)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time() - t0:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(safe_process, src, dst, use_ref) for src, dst in jobs]
            for fut in as_completed(futs):
                w = fut.result()
                if w:
                    warnings.append(w)
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"[{done}/{len(jobs)}] {time.time() - t0:.1f}s", flush=True)
    for w in warnings:
        print(f"WARNING: {w}; input copied unchanged", file=sys.stderr)
    print(f"Done: {len(files)} files -> {dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

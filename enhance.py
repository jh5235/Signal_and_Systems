"""Speech denoising with classical signal processing only (no deep / machine learning).

Usage:
    python enhance.py --input input --output output

Every WAV in the input folder gives one same-named WAV in the output folder with the same
sampling rate, channel count and number of samples.  Every file is processed using that
noisy file alone: no clean speech, no labels, no stored noise and no other file is used.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path



def main():
    from denoise.pipeline import process_folder
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help="input WAV folder (default: ./input)")
    ap.add_argument("--output", default=None, help="output WAV folder (default: ./output)")
    ap.add_argument("--independent", action="store_true",
                    help="accepted for compatibility; every file is always processed alone")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: min(4, CPUs))")
    args = ap.parse_args()
    inp = Path(args.input) if args.input else (
        Path.cwd() / "input" if (Path.cwd() / "input").is_dir() else here / "input")
    out = Path(args.output) if args.output else inp.parent / "output"
    print("mode: each file alone (noisy input only); signal processing only", flush=True)
    process_folder(inp, out, cross_file=False, jobs=args.jobs)


if __name__ == "__main__":
    main()

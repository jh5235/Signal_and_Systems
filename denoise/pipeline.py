"""Folder-level processing: every input WAV -> one same-named output WAV.

Only the noisy WAV files of the input folder are read.  No clean speech, metadata,
noise labels, stored noise waveforms or trained models are used.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from . import final, repeat


def _finish_one(args):
    y, y1, quality, sr = args
    return final.finish(y, y1, quality, sr)


def _single_channels(args):
    x, sr = args                       # x: (frames, channels)
    return np.stack([final.single_file(x[:, c], sr) for c in range(x.shape[1])], axis=1)


def _save(path, info, y, out_dir, rel=None):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape != (info.frames, info.channels):
        raise RuntimeError(f"shape changed for {path}")
    dst = Path(out_dir) / (rel if rel is not None else path.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    data = np.clip(np.nan_to_num(y), -1.0, 1.0)
    try:
        sf.write(tmp, data[:, 0] if info.channels == 1 else data, info.samplerate,
                 format=info.format, subtype=info.subtype)
        chk = sf.info(tmp)
        if (chk.samplerate, chk.channels, chk.frames) != (info.samplerate, info.channels, info.frames):
            raise RuntimeError(f"output format mismatch for {path}")
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def process_folder(input_dir, output_dir, cross_file=True, jobs=None, match_budget_s=1200.0,
                   log=print):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_dir}")
    # WAV files in sub-folders are processed too and keep their relative path
    paths = sorted((p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
                   key=lambda p: str(p.relative_to(input_dir)))
    if not paths:
        raise ValueError(f"no WAV files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or min(4, os.cpu_count() or 1)
    infos = [sf.info(p) for p in paths]
    audio = []
    for p in paths:
        x, _ = sf.read(p, dtype="float64", always_2d=True)
        audio.append(np.nan_to_num(x))
    # repeated-noise search is defined for 16 kHz mono recordings of at least one second
    group = [k for k, (i, x) in enumerate(zip(infos, audio))
             if i.samplerate == 16000 and i.channels == 1 and i.frames >= 16000 and np.any(x)]
    t0 = time.time()
    pool = None
    try:
        if jobs > 1 and len(paths) > 1:
            try:
                pool = ProcessPoolExecutor(jobs)
            except Exception:
                pool = None

        def pmap(fn, items):
            items = list(items)
            if pool is None or len(items) < 2:
                return [fn(a) for a in items]
            return list(pool.map(fn, items, chunksize=max(1, len(items) // (jobs * 8))))

        outputs = [None] * len(paths)
        n_cancel = 0
        if group:
            ys = [audio[k][:, 0] for k in group]
            try:
                y1, info = repeat.cancel_repeated_noise(
                    ys, sr=16000, cross=cross_file, deadline=t0 + match_budget_s,
                    map_fn=pmap, threads=jobs)
            except Exception as exc:            # never lose the batch because of this stage
                log(f"WARNING: repeated-noise stage skipped ({exc})")
                y1, info = ys, [dict(quality=None) for _ in ys]
            n_cancel = sum(d["quality"] is not None for d in info)
            log(f"repeated noise found in {n_cancel}/{len(group)} files "
                f"({'whole input folder' if cross_file else 'inside each file only'}), "
                f"{time.time() - t0:.0f}s")
            res = pmap(_finish_one, [(y, a, d["quality"], 16000) for y, a, d in zip(ys, y1, info)])
            for k, r in zip(group, res):
                outputs[k] = r
        rest = [k for k in range(len(paths)) if outputs[k] is None]
        for k, r in zip(rest, pmap(_single_channels, [(audio[k], infos[k].samplerate) for k in rest])):
            outputs[k] = r
    finally:
        if pool is not None:
            pool.shutdown()
    for p, i, y in zip(paths, infos, outputs):
        _save(p, i, y, output_dir, p.relative_to(input_dir))
    log(f"Done: {len(paths)} WAV files -> {output_dir} ({time.time() - t0:.0f}s)")
    return dict(files=len(paths), cancelled=n_cancel)

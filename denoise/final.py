"""Final single-file enhancement stage and splicing of cancelled / non-cancelled parts."""
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import dsp, ridge

LONG_STFT_WEIGHT = 0.25
NO_COPY = dict(gain_time_bins=5)
# After repeated-noise subtraction the residual noise is small: less over-subtraction.
CANCELLED = dict(noise_scale=1.0, noise_scale_st=1.0, gmin_db=-12.0)
OUTPUT_GAIN = 1.5
SPLICE_THRESHOLD = 0.4


def apply_gain(y, gain=OUTPUT_GAIN):
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return y * min(gain, 0.999 / peak) if peak > 0 else y


def single_file(y, sr, gain=True):
    """Noisy-only single-file enhancement of one channel (uses that channel alone)."""
    if sr == 16000 and len(y) >= 3072:
        from . import single_rules
        out = single_rules.enhance_single_no_match(np.asarray(y, dtype=np.float64), _config())
        return np.asarray(out, dtype=np.float64).reshape(-1)
    return apply_gain(dsp.enhance(y, sr=sr, profile="pesq", output_gain=0))


_CONFIG = []


def _config():
    if not _CONFIG:
        from . import single_rules
        _CONFIG.append(single_rules.BatchConfig())
    return _CONFIG[0]


CANCEL_MODE = "mix"


def cancelled(y1, sr, gain=True):
    """Enhancement of a recording whose repeated noise has already been subtracted."""
    a = None
    if CANCEL_MODE in ("ns1", "mix"):
        a = apply_gain(dsp.enhance(y1, sr=sr, profile="pesq", output_gain=0, **CANCELLED))
    if CANCEL_MODE in ("rules", "mix") and sr == 16000 and len(y1) >= 3072:
        b = single_file(y1, sr)
        a = b if a is None else 0.5 * (a + b)
    return a


def splice_weight(quality, sr, threshold=SPLICE_THRESHOLD, fade_s=0.03):
    """1 where the copies explain the noise (use the cancelled path), 0 elsewhere."""
    w = (np.asarray(quality) >= threshold).astype(np.float64)
    k = max(1, int(round(fade_s * sr)))
    return uniform_filter1d(uniform_filter1d(w, k, mode="nearest"), k, mode="nearest")


def finish(y, y1, quality, sr):
    """Cancelled path where the noise copies explain the noise, single-file path elsewhere."""
    if quality is None:
        return single_file(y, sr)
    w = splice_weight(quality, sr)
    if w.max() <= 0.0:
        return single_file(y, sr)
    a = cancelled(y1, sr)
    if w.min() >= 1.0:
        return a
    return w * a + (1.0 - w) * single_file(y, sr)

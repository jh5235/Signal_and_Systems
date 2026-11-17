"""Repeated-noise cancellation for a set of noisy recordings (noisy input only).

Stages
  1. search for repeated noise waveforms (inside one recording and, optionally, between
     the recordings of the same input folder) and verification of every candidate,
  2. first-pass speech estimate of every involved recording (single-file Log-MMSE),
  3. MMSE/EM estimate of each recording's noise waveform from its other copies and
     time-domain subtraction.

Nothing but the noisy waveforms given to `cancel_repeated_noise` is used: no clean
speech, no labels, no stored noise examples and no trained model.
"""
import numpy as np

from . import dsp, xcancel, xmatch

DEFAULTS = dict(
    thr=5.0, thr_self=5.5,                      # detection thresholds (local sigma units)
    min_frac=0.07, min_exact=0.010,             # acceptance of a copy in another recording
    min_frac_self=0.09, min_exact_self=0.015,   # ... inside the same recording
    min_overlap=4000,
    tone_guard=0.02,    # recordings dominated by stable sinusoids keep the single-file path
    narrow_guard=0.85,  # ... as do recordings whose low-band noise sits in a few spectral lines
    max_copies=10,
    n_fft=1024, hop=256, iters=20,
    outer=1,            # re-estimate the speech weights from the cancelled signals and repeat
)


def first_pass(x, sr=16000):
    """Mild single-file speech estimate; it only initialises the copy weights."""
    return dsp.enhance(x, sr=sr, profile="research_base", hp_hz=0, sin_max=0, imp_min_frames=0,
                       output_gain=1, gain_freq_bins=3, gain_time_bins=3)


def stable_tone_share(x, sr=16000):
    """Share of the energy removed by the stable-sinusoid fit of the single-file path."""
    e = float(np.sum(x * x))
    if e <= 0 or len(x) < 1536:
        return 0.0
    xt = dsp.remove_stable_tones(x, sr, dsp.PROFILES["pesq"])
    return float(np.sum((x - xt) ** 2)) / e


def narrowband_share(x, sr=16000):
    """Share of the quiet-frame noise power below 1 kHz held by its five strongest bins."""
    if len(x) < 1536:
        return 0.0
    from . import single_rules
    return float(single_rules.quiet_psd_features(x, sr)[1])


def _verify_one(args):
    yi, yj, tau, same = args
    return xmatch.verify(yi, yj, tau, same=same)


def select_copies(ys, cands, p, map_fn=map):
    """copies[i] = [(j, tau, a, quality)], meaning noise_j(t - tau) = a * noise_i(t)."""
    copies = [[] for _ in ys]
    res = list(map_fn(_verify_one, [(ys[i], ys[j], tau, i == j) for (i, j, tau, z) in cands]))
    for (i, j, tau, z), v in zip(cands, res):
        if v is None:
            continue
        same = i == j
        if v["frac"] < (p["min_frac_self"] if same else p["min_frac"]):
            continue
        if v["exact"] < (p["min_exact_self"] if same else p["min_exact"]):
            continue
        a, t, q = v["a"], v["tau"], v["frac"] * v["overlap"]
        if same:
            copies[i].append((i, t, 1.0, q))
            copies[i].append((i, -t, 1.0, q))
        else:
            copies[i].append((j, t, a, q))
            copies[j].append((i, -t, 1.0 / a, q))
    out = []
    for lst in copies:
        # prefer loud-noise copies (large a scales the foreign speech down) and long overlaps
        lst = sorted(lst, key=lambda c: -(min(c[2], 4.0) ** 2) * c[3])
        out.append([(c[0], c[1], c[2]) for c in lst[:p["max_copies"]]])
    return out


def _estimate_one(args):
    y, s_hat, cps, ys_sub, shat_sub, kw = args
    return xcancel.estimate_noise(y, s_hat, cps, ys_sub, shat_sub, **kw)


def cancel_repeated_noise(ys, sr=16000, cross=True, deadline=None, params=None, map_fn=map,
                          threads=4):
    """Subtract repeated noise from every recording.

    Returns (y1 list, info list); info[i] = dict(n_copies, coverage, quality).
    quality is a per-sample array (None when the recording has no usable copy, in which
    case y1[i] is the unchanged input).
    """
    p = dict(DEFAULTS, **(params or {}))
    ys = [np.asarray(y, dtype=np.float64) for y in ys]
    cands = xmatch.find_candidates(ys, sr=sr, thr=p["thr"], thr_self=p["thr_self"], cross=cross,
                                   min_overlap=p["min_overlap"], min_self_lag=p["min_overlap"],
                                   deadline=deadline, threads=threads)
    copies = select_copies(ys, cands, p, map_fn)
    active = [i for i, c in enumerate(copies) if c]
    if p["tone_guard"] and active:
        share = list(map_fn(stable_tone_share, [ys[i] for i in active]))
        active = [i for i, s in zip(active, share) if s <= p["tone_guard"]]
    if p["narrow_guard"] and active:
        share = list(map_fn(narrowband_share, [ys[i] for i in active]))
        active = [i for i, s in zip(active, share) if s <= p["narrow_guard"]]
    info = [dict(n_copies=0, coverage=0.0, quality=None) for _ in ys]
    y1 = [y.copy() for y in ys]
    if not active:
        return y1, info
    needed = sorted(set(active) | {c[0] for i in active for c in copies[i]})
    shat = dict(zip(needed, map_fn(first_pass, [ys[i] for i in needed])))
    kw = dict(n_fft=p["n_fft"], hop=p["hop"], iters=p["iters"])
    for it in range(p["outer"]):
        jobs = []
        for i in active:
            src = sorted({c[0] for c in copies[i]})
            jobs.append((ys[i], shat[i], copies[i], {j: ys[j] for j in src},
                         {j: shat[j] for j in src}, kw))
        for i, (nw, quality, vr) in zip(active, map_fn(_estimate_one, jobs)):
            y1[i] = ys[i] - nw
            info[i] = dict(n_copies=len(copies[i]), coverage=float(np.mean(quality > 0.5)),
                           quality=quality)
        if it + 1 < p["outer"]:
            for i, sh in zip(active, map_fn(first_pass, [y1[i] for i in active])):
                shat[i] = sh
    return y1, info

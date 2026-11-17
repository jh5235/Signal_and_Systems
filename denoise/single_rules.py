# NOTE: this package only calls enhance_single_no_match() and its helpers from this module
# (single-file rules: two STFT resolutions, residual-tone and band post-filters).
# The folder-level functions further below are not used by enhance.py.
"""Noisy-only, traditional DSP batch cancellation of repeated background noise.

Only WAVs in the input folder enter the matching and template stages. Clean,
noise labels, metadata CSVs, stored examples, and learned models are not read.
The validated parameter constants live in BatchConfig below.
"""

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import dsp, ridge

EnhanceFn = Callable[[np.ndarray, int, str, int | None, float | None], np.ndarray]
ResidualFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class BatchConfig:
    """Fixed submission parameters checked on public Train and Validation."""

    beta: float = 1.0
    corr_threshold: float = 0.4
    max_refs: int = 3
    min_overlap_s: float = 1.0
    refine_radius_samples: int = 50
    max_match_files: int = 256
    max_match_samples: int = 20_000_000
    # If set, a file with stronger framewise low-band tonality is not
    # corrected. It can still serve as a reference for another file, matching
    # the offline guard evaluation's whole-folder pair construction.
    frame_tone_guard: float | None = 0.65
    quiet_psd_flatness_guard: float | None = 0.93
    quiet_psd_top5_guard: float | None = 0.95
    # Conservative subtraction for strongly tonal targets with consistent
    # multi-reference alignment; white noise and dial tones remain protected.
    rescue_beta: float = 0.75
    rescue_beta_strong: float = 1.0
    rescue_low300_min: float = 0.20
    rescue_low300_max: float = 0.85
    rescue_corr_threshold: float = 0.45
    rescue_lag_tolerance: int = 16
    rescue_max_neighbors: int = 12
    rescue_recompute_low_cutoff: bool = True
    wind_shelf_db: float = 23.0
    # Train speaker-disjoint, Validation-confirmed final-stage refinements.
    no_match_gain_time_bins: int = 5
    no_match_long_gain_time_bins: int = 11
    no_match_long_stft_weight: float = 0.25
    matched_gmin_db: float = -20.0
    matched_dynamic_shelf: bool = True
    unmatched_ridge_strength: float = 0.60
    unmatched_ridge_adaptive: bool = True
    no_match_ridge_residual_scale: float = 16.0
    no_match_ridge_strength: float = 0.60
    no_match_ridge_dense_strength: float = 0.80
    no_match_ridge_dense_fraction: float = 0.005
    no_match_ridge_protect_highmask: bool = True
    no_match_midband_boost: float = 0.10
    no_match_midband_dynamic_db: float = 15.0
    no_match_active_highband_boost: float = 0.50
    no_match_quiet_low_cv_max: float = 0.90
    no_match_quiet_highband_suppress: float = 0.50
    guard_active_highband_boost: float = 0.25
    matched_active_highband_boost: float = 0.25
    matched_band_crossover_hz: int = 1500
    matched_band_scale_cap: float = 0.50
    matched_pink_guard: bool = True
    profile: str = "pesq"

    def __post_init__(self):
        if not 0 <= self.beta <= 2:
            raise ValueError("beta must be in [0, 2]")
        if not 0 < self.corr_threshold <= 1:
            raise ValueError("corr_threshold must be in (0, 1]")
        if self.max_refs < 1 or self.refine_radius_samples < 0:
            raise ValueError("max_refs must be positive and refinement radius nonnegative")
        if self.min_overlap_s <= 0 or self.max_match_files < 2 or self.max_match_samples < 2:
            raise ValueError("invalid overlap or matching resource limit")
        if self.frame_tone_guard is not None and not 0 <= self.frame_tone_guard <= 1:
            raise ValueError("frame_tone_guard must be in [0, 1] or None")
        for name in ("quiet_psd_flatness_guard", "quiet_psd_top5_guard"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1] or None")
        if not 0 <= self.rescue_beta <= 2:
            raise ValueError("rescue_beta must be in [0, 2]")
        if not 0 <= self.rescue_beta_strong <= 2:
            raise ValueError("rescue_beta_strong must be in [0, 2]")
        if not 0 <= self.rescue_low300_min < self.rescue_low300_max <= 1:
            raise ValueError("invalid rescue_low300 interval")
        if not 0 < self.rescue_corr_threshold <= 1:
            raise ValueError("rescue_corr_threshold must be in (0, 1]")
        if self.rescue_lag_tolerance < 0 or self.rescue_max_neighbors < 2:
            raise ValueError("invalid rescue lag tolerance or neighbor limit")
        if self.rescue_beta and self.frame_tone_guard is None:
            raise ValueError("rescue_beta requires frame_tone_guard")
        if self.rescue_beta and (self.quiet_psd_flatness_guard is None or
                                 self.quiet_psd_top5_guard is None):
            raise ValueError("rescue_beta requires both quiet-PSD protection guards")
        if not 0 <= self.wind_shelf_db <= 60:
            raise ValueError("wind_shelf_db must be in [0, 60]")
        if self.no_match_gain_time_bins < 1 or self.no_match_gain_time_bins % 2 != 1:
            raise ValueError("no_match_gain_time_bins must be positive and odd")
        if self.no_match_long_gain_time_bins < 1 or self.no_match_long_gain_time_bins % 2 != 1:
            raise ValueError("no_match_long_gain_time_bins must be positive and odd")
        if not 0 <= self.no_match_long_stft_weight <= 1:
            raise ValueError("no_match_long_stft_weight must be in [0, 1]")
        if not -80 <= self.matched_gmin_db <= 0:
            raise ValueError("matched_gmin_db must be in [-80, 0]")
        if not isinstance(self.matched_dynamic_shelf, bool):
            raise ValueError("matched_dynamic_shelf must be a boolean")
        if not 0 <= self.unmatched_ridge_strength <= 1:
            raise ValueError("unmatched_ridge_strength must be in [0, 1]")
        if not 0 < self.no_match_ridge_residual_scale <= 32:
            raise ValueError("no_match_ridge_residual_scale must be in (0, 32]")
        if not 0 <= self.no_match_ridge_strength <= 1:
            raise ValueError("no_match_ridge_strength must be in [0, 1]")
        if not 0 <= self.no_match_ridge_dense_strength <= 1:
            raise ValueError("no_match_ridge_dense_strength must be in [0, 1]")
        if not 0 <= self.no_match_ridge_dense_fraction <= 1:
            raise ValueError("no_match_ridge_dense_fraction must be in [0, 1]")
        if not isinstance(self.no_match_ridge_protect_highmask, bool):
            raise ValueError("no_match_ridge_protect_highmask must be a boolean")
        if not 0 <= self.no_match_midband_boost <= 1:
            raise ValueError("no_match_midband_boost must be in [0, 1]")
        if not np.isfinite(self.no_match_midband_dynamic_db):
            raise ValueError("no_match_midband_dynamic_db must be finite")
        if not 0 <= self.no_match_active_highband_boost <= 1:
            raise ValueError("no_match_active_highband_boost must be in [0, 1]")
        if self.no_match_quiet_low_cv_max < 0:
            raise ValueError("no_match_quiet_low_cv_max must be nonnegative")
        if not 0 <= self.no_match_quiet_highband_suppress <= 1:
            raise ValueError("no_match_quiet_highband_suppress must be in [0, 1]")
        if not 0 <= self.guard_active_highband_boost <= 1:
            raise ValueError("guard_active_highband_boost must be in [0, 1]")
        if not 0 <= self.matched_active_highband_boost <= 1:
            raise ValueError("matched_active_highband_boost must be in [0, 1]")
        if not 100 <= self.matched_band_crossover_hz <= 4000:
            raise ValueError("matched_band_crossover_hz must be in [100, 4000]")
        if not 0 <= self.matched_band_scale_cap <= 1:
            raise ValueError("matched_band_scale_cap must be in [0, 1]")
        if not isinstance(self.matched_pink_guard, bool):
            raise ValueError("matched_pink_guard must be a boolean")
        if self.profile not in dsp.PROFILES:
            raise ValueError(f"unknown profile: {self.profile}")


@dataclass
class BatchResult:
    input_files: int = 0
    written_files: int = 0
    matched_files: int = 0
    matched_pairs: int = 0
    rescued_files: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    references_by_file: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class InputFile:
    path: Path
    info: sf._SoundFileInfo


def default_enhance(x: np.ndarray, sr: int, profile: str,
                    original_cutoff_hz: int | None = None,
                    shelf_db: float | None = None,
                    parameter_overrides: dict | None = None) -> np.ndarray:
    overrides = {} if shelf_db is None else {"shelf_db": shelf_db}
    if parameter_overrides:
        overrides.update(parameter_overrides)
    if original_cutoff_hz is None:
        return dsp.enhance(x, sr=sr, profile=profile, **overrides)
    # The validated algorithm selects low-band attenuation from the ORIGINAL
    # Noisy file, before coherent subtraction changes its spectrum.
    y = dsp.enhance(x, sr=sr, profile=profile, low_cutoff_rule=False,
                    **overrides)
    if original_cutoff_hz:
        p = dsp.PROFILES[profile]
        y = dsp.attenuate_low_band(y, sr, original_cutoff_hz,
                                   p["low_attenuation"], p["low_order"])
    return y


def original_low_cutoff(x: np.ndarray, sr: int, profile: str) -> int:
    p = dsp.PROFILES[profile]
    if not p.get("low_cutoff_rule"):
        return 0
    scale = sr / 16000
    n_fft = max(32, round(p["n_fft"] * scale))
    hop = max(8, round(p["hop"] * scale))
    return dsp.choose_low_cutoff(x, sr, n_fft, hop, p["frac"])


def preliminary_residual(x: np.ndarray, sr: int) -> np.ndarray:
    """The same noisy-only residual estimate as crossfile_cancel.py."""
    speech = dsp.enhance(x, sr=sr, profile="research_base", hp_hz=0,
                         sin_max=0, imp_min_frames=0, output_gain=1,
                         gain_freq_bins=3, gain_time_bins=3)
    return x - speech


def frame_tone_median(x: np.ndarray, sr: int) -> float:
    """Exact noisy-only moving-tone feature from frame_tone_guard.py."""
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    _, _, z = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          nfft=n_fft, boundary="zeros", padded=True)
    power = np.abs(z[np.fft.rfftfreq(n_fft, 1 / sr) < 1000]) ** 2
    top5 = np.partition(power, -5, axis=0)[-5:].sum(0)
    ratio = top5 / np.maximum(power.sum(0), 1e-12)
    return float(np.median(ratio[2:-2] if len(ratio) > 8 else ratio))


def quiet_psd_features(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Original-Noisy flatness and top-five share from rule_features.describe."""
    n_fft = round(1536 * sr / 16000)
    hop = round(192 * sr / 16000)
    power = np.abs(dsp._stft(x, n_fft, hop)) ** 2
    floor = dsp.noise_fixed(power, .1)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = floor[(f >= 100) & (f < 800)]
    under1k = floor[f < 1000]
    flatness = float(np.exp(np.mean(np.log(np.maximum(band, dsp.EPS)))) /
                     np.mean(band))
    top5 = float(np.sort(under1k)[-5:].sum() /
                 max(float(under1k.sum()), dsp.EPS))
    return flatness, top5


def rescue_beta_for(x: np.ndarray, sr: int, config: BatchConfig) -> float:
    """Select stronger coherent subtraction from original Noisy low-band PSD."""
    if config.rescue_beta != .75:
        return config.rescue_beta
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    floor = dsp.noise_fixed(np.abs(dsp._stft(x, n_fft, hop)) ** 2, .1)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    low300 = float(floor[freq < 300].sum() /
                   max(float(floor.sum()), dsp.EPS))
    if config.rescue_low300_min < low300 < config.rescue_low300_max:
        return config.rescue_beta_strong
    return config.rescue_beta


def is_pink_like_noisy(x: np.ndarray, sr: int) -> bool:
    """Protect pink-like broad noise using only the original Noisy PSD.

    The narrow rule was fixed on public Train before Validation comparison.
    It is a spectral-shape guard, not a noise label or an external classifier.
    """
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    floor = dsp.noise_fixed(np.abs(dsp._stft(x, n_fft, hop)) ** 2, .1)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freq >= 300) & (freq <= min(6000, sr / 2))
    if band.sum() < 3:
        return False
    logf = np.log(freq[band])
    logp = np.log(np.maximum(floor[band], dsp.EPS))
    a, b = logf - logf.mean(), logp - logp.mean()
    slope = float(np.dot(a, b) / max(np.dot(a, a), dsp.EPS))
    flatness = float(np.exp(logp.mean()) /
                     max(float(floor[band].mean()), dsp.EPS))
    low300 = float(floor[freq < 300].sum() /
                   max(float(floor.sum()), dsp.EPS))
    return -1.2 <= slope <= -.8 and flatness > .6 and .35 < low300 < .6


def lowband_dynamic_db(x: np.ndarray, sr: int) -> float:
    """Noisy-only 80–600 Hz frame-energy q90/q10 ratio in decibels."""
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    power = np.abs(dsp._stft(x, n_fft, hop)) ** 2
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    low = power[(freq >= 80) & (freq < 600)].sum(axis=0)
    low = low[2:-2] if len(low) > 8 else low
    return float(10 * np.log10(np.quantile(low, .9) /
                               max(np.quantile(low, .1), dsp.EPS)))


def quiet_low_power_cv(x: np.ndarray, sr: int) -> float:
    """Variation of 80–600 Hz energy in the quietest Noisy STFT frames."""
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    power = np.abs(dsp._stft(x, n_fft, hop)) ** 2
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    low = power[(freq >= 80) & (freq < 600)].sum(axis=0)
    full = power.sum(axis=0)
    inner = slice(2, -2)
    selected = (np.argsort(full[inner], kind="stable")
                [:max(1, round(.2 * (len(full) - 4)))] + 2)
    return float(np.std(low[selected]) /
                 max(np.mean(low[selected]), dsp.EPS))


def corrected_shelf_db(x: np.ndarray, sr: int, config: BatchConfig) -> float | None:
    """Train-selected shelf for one PSD shape, measured on original Noisy.

    This is called only after a main-branch match and all protection guards
    have passed. A ``None`` result leaves the standard 29 dB shelf unchanged.
    """
    n_fft = round(1536 * sr / 16000)
    hop = round(192 * sr / 16000)
    power = np.abs(dsp._stft(x, n_fft, hop)) ** 2
    floor = dsp.noise_fixed(power, .1)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    total = max(float(floor.sum()), dsp.EPS)
    low300 = float(floor[f < 300].sum() / total)
    mid300_600 = float(floor[(f >= 300) & (f < 600)].sum() / total)
    under1k = floor[f < 1000]
    top5 = float(np.sort(under1k)[-5:].sum() /
                 max(float(under1k.sum()), dsp.EPS))
    if low300 < .20 and mid300_600 > .30 and .20 < top5 < .40:
        return config.wind_shelf_db
    return None


def matched_highband_restore_allowed(x: np.ndarray, sr: int,
                                     shelf_db: float | None) -> bool:
    """Protect wind-like and wide-band transient inputs from high-band gain."""
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    floor = dsp.noise_fixed(np.abs(dsp._stft(x, n_fft, hop)) ** 2, .1)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    total = max(float(floor.sum()), dsp.EPS)
    low300 = float(floor[freq < 300].sum() / total)
    high1000 = float(floor[freq >= 1000].sum() / total)
    under1k = floor[freq < 1000]
    top5 = float(np.sort(under1k)[-5:].sum() /
                 max(float(under1k.sum()), dsp.EPS))
    windlike = .30 < low300 < .65 and high1000 < .20 and top5 < .20
    low_shelf = shelf_db is not None and high1000 < .20
    if windlike or low_shelf:
        return False
    if use_dynamic_shelf(x, sr):
        power = np.abs(dsp._stft(x, n_fft, hop)) ** 2
        high = power[(freq >= 800) & (freq < 5000)].sum(axis=0)
        mid = power[(freq >= 300) & (freq < 2000)].sum(axis=0)
        high_db = 10 * np.log10(np.quantile(high, .9) /
                                max(np.quantile(high, .1), 1e-15))
        mid_db = 10 * np.log10(np.quantile(mid, .9) /
                               max(np.quantile(mid, .1), 1e-15))
        if high_db - mid_db >= -2:
            return False
    return True


def use_raw_template_scale(x: np.ndarray, sr: int, nrefs: int) -> bool:
    """Select noise-dominated low-band inputs for direct Noisy/reference LS.

    The rule uses only the original Noisy spectrum. Three independent aligned
    references are required so that speech remaining in a single reference
    does not set the correction amplitude by itself.
    """
    if nrefs < 3:
        return False
    n_fft = max(32, round(1536 * sr / 16000))
    hop = max(8, round(192 * sr / 16000))
    floor = dsp.noise_fixed(np.abs(dsp._stft(x, n_fft, hop)) ** 2, .1)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    total = max(float(floor.sum()), dsp.EPS)
    low300 = float(floor[f < 300].sum() / total)
    high1000 = float(floor[f >= 1000].sum() / total)
    under1k = floor[f < 1000]
    top5 = float(np.sort(under1k)[-5:].sum() /
                 max(float(under1k.sum()), dsp.EPS))
    return .60 < low300 < .75 and top5 > .48 and high1000 < .10


def use_dynamic_shelf(x: np.ndarray, sr: int) -> bool:
    """Train-selected smooth-noise shape, measured on original Noisy only."""
    if sr != 16000:
        return False
    n_fft, hop = 1536, 192
    floor = dsp.noise_fixed(np.abs(dsp._stft(x, n_fft, hop)) ** 2, .1)
    freq = np.fft.rfftfreq(n_fft, 1 / sr)
    high1000 = float(floor[freq >= 1000].sum() /
                     max(float(floor.sum()), dsp.EPS))
    under1k = floor[freq < 1000]
    top5 = float(np.sort(under1k)[-5:].sum() /
                 max(float(under1k.sum()), dsp.EPS))
    return high1000 < .20 and top5 < .20


def dynamic_shelf_activity(x: np.ndarray, sr: int) -> np.ndarray:
    """Noisy-derived 0–1 high-band activity mask for the matched shelf blend."""
    sos = signal.butter(4, 800, "highpass", fs=sr, output="sos")
    high = signal.sosfiltfilt(sos, x)
    width = max(1, round(.025 * sr))
    rms = np.sqrt(np.maximum(uniform_filter1d(high * high, width,
                                               mode="nearest"), 1e-14))
    margin = min(len(rms) // 4, round(.15 * sr))
    interior = rms[margin:-margin] if margin else rms
    floor = max(float(np.percentile(interior, 20)), 1e-8)
    excess_db = 20 * np.log10(np.maximum(rms / floor, 1e-8))
    excess_db = uniform_filter1d(excess_db, width, mode="nearest")
    return np.clip((excess_db - 6) / 6, 0, 1)


def input_midband_activity(x: np.ndarray, sr: int) -> np.ndarray:
    """Speech-activity proxy from 300–2000 Hz waveform energy."""
    sos = signal.butter(4, (300, 2000), "bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x)
    width = max(1, round(.025 * sr))
    env = np.sqrt(np.maximum(uniform_filter1d(band * band, width,
                                               mode="nearest"), 1e-14))
    margin = round(.15 * sr)
    inner = env[margin:-margin] if len(env) > 2 * margin else env
    floor = max(float(np.percentile(inner, 20)), 1e-8)
    db = 20 * np.log10(np.maximum(env / floor, 1e-8))
    db = uniform_filter1d(db, width, mode="nearest")
    return np.clip((db - 6) / 6, 0, 1)


def no_match_highband_postfilter(noisy: np.ndarray, enhanced: np.ndarray,
                                 config: BatchConfig) -> np.ndarray:
    """Restore active-band detail or attenuate quiet-band residue.

    The quiet-low CV gate is computed from the current Noisy WAV alone. Both
    activity estimates come from that WAV or its enhanced output, with no
    clean reference, metadata, learned model, or stored calibration signal.
    """
    if not (config.no_match_active_highband_boost or
            config.no_match_quiet_highband_suppress):
        return enhanced
    cv = quiet_low_power_cv(noisy, 16000)
    if cv < config.no_match_quiet_low_cv_max:
        if not config.no_match_active_highband_boost:
            return enhanced
        sos = signal.butter(4, (700, 4500), "bandpass", fs=16000,
                            output="sos")
        high = signal.sosfiltfilt(sos, enhanced)
        activity = input_midband_activity(noisy, 16000)
        return enhanced + config.no_match_active_highband_boost * activity * high
    if not config.no_match_quiet_highband_suppress:
        return enhanced
    sos = signal.butter(4, (700, 4000), "bandpass", fs=16000, output="sos")
    high = signal.sosfiltfilt(sos, enhanced)
    # Blend the original Noisy and enhanced activity estimates. The former
    # preserves weak speech cues; the latter rejects residual background.
    activity = .5 * (input_midband_activity(noisy, 16000) +
                     input_midband_activity(enhanced, 16000))
    return enhanced - config.no_match_quiet_highband_suppress * (1 - activity) * high


def guard_reason(x: np.ndarray, sr: int, config: BatchConfig) -> str | None:
    """Apply quiet-PSD protection before considering tonal rescue.

    White/flat-spectrum files can also have high frame tonality; rescuing them
    harmed held-out Train examples. The PSD guards therefore take precedence.
    """
    if (config.quiet_psd_flatness_guard is not None or
            config.quiet_psd_top5_guard is not None):
        flatness, top5 = quiet_psd_features(x, sr)
        if (config.quiet_psd_flatness_guard is not None and
                flatness > config.quiet_psd_flatness_guard):
            return "flatness_guard"
        if (config.quiet_psd_top5_guard is not None and
                top5 > config.quiet_psd_top5_guard):
            return "top5_guard"
    if (config.frame_tone_guard is not None and
            frame_tone_median(x, sr) > config.frame_tone_guard):
        return "tone_guard"
    return None


def downsample(x: np.ndarray) -> np.ndarray:
    return signal.resample_poly(x, 1, 4)


def max_corr(a: np.ndarray, b: np.ndarray, min_overlap_s: float) -> tuple[float, float]:
    """Absolute normalized cross-correlation and lag in seconds at 4 kHz."""
    if len(a) != len(b):
        raise ValueError("matching requires equal sample counts")
    a = a - a.mean()
    b = b - b.mean()
    n = len(a)
    lags = np.arange(-n + 1, n)
    numerator = signal.correlate(a, b, mode="full", method="fft")
    ea = np.cumsum(np.r_[0.0, a * a])
    eb = np.cumsum(np.r_[0.0, b * b])
    starts_a = np.maximum(0, lags)
    starts_b = np.maximum(0, -lags)
    lengths = n - np.abs(lags)
    den = np.sqrt((ea[starts_a + lengths] - ea[starts_a]) *
                  (eb[starts_b + lengths] - eb[starts_b]))
    valid = lengths >= int(4000 * min_overlap_s)
    corr = np.divide(np.abs(numerator), den, out=np.zeros_like(numerator),
                     where=den > 1e-12)
    corr[~valid] = 0.0
    idx = int(np.argmax(corr))
    return float(corr[idx]), float(lags[idx] / 4000)


def overlap_slices(n: int, lag: int) -> tuple[slice, slice]:
    if lag >= 0:
        return slice(lag, n), slice(0, n - lag)
    return slice(0, n + lag), slice(-lag, n)


def refine_lag(target: np.ndarray, reference: np.ndarray, coarse_lag: int,
               min_overlap_s: float, radius: int) -> tuple[float, int]:
    n = len(target)
    min_overlap = int(round(min_overlap_s * 16000))
    best = (-np.inf, coarse_lag)
    for lag in range(coarse_lag - radius, coarse_lag + radius + 1):
        if abs(lag) > n - min_overlap:
            continue
        a, b = overlap_slices(n, lag)
        x, y = target[a], reference[b]
        score = float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y) + 1e-12))
        if score > best[0]:
            best = (score, lag)
    return best


def match_pairs(residuals: list[np.ndarray], config: BatchConfig):
    """Exact all-pairs search; no unvalidated feature prefilter is applied."""
    coarse = [downsample(x) for x in residuals]
    matches = defaultdict(list)
    pairs = []
    for i, j in combinations(range(len(residuals)), 2):
        corr, lag_s = max_corr(coarse[i], coarse[j], config.min_overlap_s)
        if corr < config.corr_threshold:
            continue
        coarse_lag = int(round(lag_s * 16000))
        refined, lag = refine_lag(residuals[i], residuals[j], coarse_lag,
                                  config.min_overlap_s, config.refine_radius_samples)
        if refined < config.corr_threshold:
            continue
        matches[i].append((refined, j, lag))
        matches[j].append((refined, i, -lag))
        pairs.append((i, j, corr, refined, lag))
    return matches, pairs


def template_for(i: int, residuals: list[np.ndarray], matches, max_refs: int,
                 raw_target: np.ndarray | None = None):
    """Align, scale, taper, and average top correlated residuals."""
    target = residuals[i]
    n = len(target)
    accum = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    selected = sorted(matches.get(i, []), reverse=True)[:max_refs]
    for _, j, lag in selected:
        a, b = overlap_slices(n, lag)
        ref = residuals[j][b]
        tar = target[a]
        denom = float(np.dot(ref, ref))
        if raw_target is None:
            alpha = float(np.clip(np.dot(tar, ref) / max(denom, 1e-12), 0, 2))
        else:
            alpha = float(np.clip(np.dot(raw_target[a], ref) /
                                  max(denom, 1e-12), 0, 2.5))
        if alpha == 0:
            continue
        w = np.ones(len(ref), dtype=np.float64)
        ramp = min(160, len(ref) // 2)
        if ramp:
            w[:ramp] = np.linspace(0, 1, ramp)
            w[-ramp:] = np.linspace(1, 0, ramp)
        accum[a] += alpha * ref * w
        count[a] += w
    return accum / np.maximum(count, 1.0), len(selected)


class BandTemplateProvider:
    """Two-band least-squares scaling of aligned Noisy-derived residuals."""

    def __init__(self, residuals: list[np.ndarray], sr: int,
                 crossover_hz: int, cap_ratio: float):
        self.residuals = residuals
        self.sos = signal.butter(4, crossover_hz, btype="lowpass",
                                 fs=sr, output="sos")
        self.cap_ratio = cap_ratio
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _partition(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = signal.sosfiltfilt(self.sos, x)
        return low, x - low

    def _residual_bands(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self.cache:
            self.cache[i] = self._partition(self.residuals[i])
        return self.cache[i]

    def template_for(self, i: int, matches, max_refs: int,
                     raw_target: np.ndarray | None = None):
        target = self.residuals[i]
        n = len(target)
        accum = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)
        selected = sorted(matches.get(i, []), reverse=True)[:max_refs]
        target_bands = (self._partition(raw_target) if raw_target is not None
                        else self._residual_bands(i))
        for _, j, lag in selected:
            a, b = overlap_slices(n, lag)
            ref = self.residuals[j][b]
            denom = float(np.dot(ref, ref))
            max_alpha = 2.5 if raw_target is not None else 2.0
            global_target = raw_target[a] if raw_target is not None else target[a]
            global_alpha = float(np.clip(np.dot(global_target, ref) /
                                         max(denom, 1e-12), 0, max_alpha))
            if global_alpha == 0:
                continue
            scaled = np.zeros(len(ref), dtype=np.float64)
            for target_band, reference_band in zip(
                    target_bands, self._residual_bands(j)):
                u, v = target_band[a], reference_band[b]
                local_alpha = float(np.clip(np.dot(u, v) /
                                            max(np.dot(v, v), 1e-12),
                                            0, max_alpha))
                relative = np.clip(local_alpha / global_alpha - 1,
                                   -self.cap_ratio, self.cap_ratio)
                scaled += global_alpha * (1 + relative) * v
            w = np.ones(len(ref), dtype=np.float64)
            ramp = min(160, len(ref) // 2)
            if ramp:
                w[:ramp] = np.linspace(0, 1, ramp)
                w[-ramp:] = np.linspace(1, 0, ramp)
            accum[a] += scaled * w
            count[a] += w
        return accum / np.maximum(count, 1.0), len(selected)


def rescue_max_corr_4k(a: np.ndarray, b: np.ndarray,
                       min_overlap: int = 4000) -> tuple[float, int]:
    """Positive coarse correlation in the independent triangle experiment."""
    n = len(a)
    a = a - a.mean()
    b = b - b.mean()
    lags = np.arange(-n + 1, n)
    num = signal.correlate(a, b, mode="full", method="fft")
    ea = np.cumsum(np.r_[0.0, a * a])
    eb = np.cumsum(np.r_[0.0, b * b])
    sa = np.maximum(lags, 0)
    sb = np.maximum(-lags, 0)
    length = n - np.abs(lags)
    den = np.sqrt((ea[sa + length] - ea[sa]) *
                  (eb[sb + length] - eb[sb]))
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    corr[length < min_overlap] = 0
    k = int(np.argmax(corr))
    return float(corr[k]), int(lags[k]) * 4


def rescue_refine_lag(a: np.ndarray, b: np.ndarray,
                      lag: int) -> tuple[float, int]:
    """Independent 16 kHz lag refinement within ±5 samples."""
    best = (-1.0, lag)
    for shift in range(lag - 5, lag + 6):
        u, v = overlap_slices(len(a), shift)
        x, y = a[u], b[v]
        if len(x) < 16000:
            continue
        score = float(np.dot(x, y) /
                      np.sqrt(max(np.dot(x, x) * np.dot(y, y), 1e-20)))
        if score > best[0]:
            best = (score, shift)
    return best


def rescue_edges(residuals: list[np.ndarray], threshold: float):
    """Exact positive-correlation graph used by multi_complex.py."""
    coarse = [downsample(x) for x in residuals]
    edges = {}
    for i, j in combinations(range(len(residuals)), 2):
        corr, lag = rescue_max_corr_4k(coarse[i], coarse[j])
        if corr >= threshold:
            score, lag = rescue_refine_lag(residuals[i], residuals[j], lag)
            if score >= threshold:
                edges[(i, j)] = (score, lag)
    return edges


def rescue_edge(edges, i: int, j: int):
    if i < j:
        return edges.get((i, j))
    out = edges.get((j, i))
    return (out[0], -out[1]) if out is not None else None


def consistent_rescue_refs(edges, target: int, n: int,
                           lag_tolerance: int = 16,
                           max_neighbors: int = 12) -> list[int]:
    """Largest mutually consistent reference clique around a target.

    The 12-neighbor cap prevents combinatorial blowup on pathological input
    folders. It does not affect the validated Val set (max degree four).
    """
    neighbors = [j for j in range(n) if j != target and rescue_edge(edges, target, j)]
    if len(neighbors) > max_neighbors:
        neighbors = sorted(neighbors,
                           key=lambda j: (-rescue_edge(edges, target, j)[0], j))[:max_neighbors]
    best = []
    best_score = -1.0
    for k in range(2, min(len(neighbors), 6) + 1):
        for subset in combinations(neighbors, k):
            valid = True
            score = float(k)
            for j in subset:
                score += rescue_edge(edges, target, j)[0]
            for j, h in combinations(subset, 2):
                ejh = rescue_edge(edges, j, h)
                if ejh is None or abs(rescue_edge(edges, target, h)[1] -
                                      rescue_edge(edges, target, j)[1] - ejh[1]) > lag_tolerance:
                    valid = False
                    break
                score += 0.1 * ejh[0]
            if valid and score > best_score:
                best = list(subset)
                best_score = score
    return best


def rescue_time_template(residuals: list[np.ndarray], edges,
                         target: int, refs: list[int]) -> np.ndarray:
    """Aligned residual average with positive LS scale and 10 ms edge ramps."""
    n = len(residuals[target])
    accum = np.zeros(n)
    count = np.zeros(n)
    for j in refs:
        lag = rescue_edge(edges, target, j)[1]
        aligned = np.zeros(n)
        valid = np.zeros(n, dtype=bool)
        a, b = overlap_slices(n, lag)
        aligned[a] = residuals[j][b]
        valid[a] = True
        ri = residuals[target][valid]
        rj = aligned[valid]
        scale = float(np.clip(np.dot(ri, rj) /
                              max(np.dot(rj, rj), 1e-20), 0, 2))
        weight = valid.astype(float)
        positions = np.flatnonzero(valid)
        if len(positions) > 320:
            start, end = int(positions[0]), int(positions[-1]) + 1
            weight[start:start + 160] = np.linspace(0, 1, 160)
            weight[end - 160:end] = np.linspace(1, 0, 160)
        accum += scale * aligned * weight
        count += weight
    return accum / np.maximum(count, 1)


def _load(record: InputFile) -> np.ndarray:
    x, sr = sf.read(record.path, dtype="float64", always_2d=True)
    if sr != record.info.samplerate or x.shape != (record.info.frames, record.info.channels):
        raise RuntimeError(f"Input WAV changed during batch processing: {record.path}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _save(record: InputFile, y: np.ndarray, output_dir: Path):
    if y.shape != (record.info.frames, record.info.channels):
        raise RuntimeError(f"Enhancer changed input shape: {record.path}")
    # The same-named final file replaces any previous result only after its
    # temporary WAV has passed format, channel, and length verification.
    dst = output_dir / record.path.name
    tmp = output_dir / (record.path.name + ".tmp")
    subtype = record.info.subtype
    try:
        write_y = y[:, 0] if record.info.channels == 1 else y
        sf.write(tmp, np.clip(np.nan_to_num(write_y), -1.0, 1.0),
                 record.info.samplerate, format=record.info.format, subtype=subtype)
        info = sf.info(tmp)
        actual = (info.samplerate, info.channels, info.frames, info.format, info.subtype)
        wanted = (record.info.samplerate, record.info.channels, record.info.frames,
                  record.info.format, record.info.subtype)
        if actual != wanted:
            raise RuntimeError(f"Output WAV format mismatch: {record.path} -> {dst}")
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _enhance_channels(x: np.ndarray, sr: int, profile: str, fn: EnhanceFn,
                      original_cutoff_hz: int | None = None,
                      shelf_db: float | None = None,
                      parameter_overrides: dict | None = None):
    # Keep the original five-argument callback contract for research callers.
    # The validated production enhancer alone receives the branch parameters.
    def one(channel):
        if fn is default_enhance and parameter_overrides:
            return fn(channel, sr, profile, original_cutoff_hz, shelf_db,
                      parameter_overrides)
        return fn(channel, sr, profile, original_cutoff_hz, shelf_db)
    y = np.stack([one(x[:, c])
                  for c in range(x.shape[1])], axis=1)
    return y


def enhance_single_no_match(x: np.ndarray, config: BatchConfig) -> np.ndarray:
    """Use the validated no-reference path for a lone eligible Noisy WAV."""
    short = _enhance_channels(
        x[:, None], 16000, config.profile, default_enhance,
        parameter_overrides={"gain_time_bins": config.no_match_gain_time_bins})
    y = short
    if config.no_match_long_stft_weight:
        long = _enhance_channels(
            x[:, None], 16000, config.profile, default_enhance,
            parameter_overrides={
                "n_fft": 3072, "hop": 384,
                "gain_time_bins": config.no_match_long_gain_time_bins})
        w = config.no_match_long_stft_weight
        y = (1 - w) * short + w * long
    if config.unmatched_ridge_strength:
        if config.no_match_ridge_protect_highmask:
            y[:, 0] = ridge.suppress_highband_ridges_compensated(
                x, y[:, 0], 16000, config.no_match_ridge_strength,
                config.no_match_ridge_residual_scale,
                dense_strength=config.no_match_ridge_dense_strength,
                dense_fraction=config.no_match_ridge_dense_fraction)
        else:
            y[:, 0] = ridge.suppress_highband_ridges(
                x, y[:, 0], 16000, config.no_match_ridge_strength,
                config.no_match_ridge_residual_scale)
    if (config.no_match_midband_boost and
            lowband_dynamic_db(x, 16000) < config.no_match_midband_dynamic_db):
        sos = signal.butter(4, (300, 700), "bandpass", fs=16000,
                            output="sos")
        y[:, 0] += config.no_match_midband_boost * signal.sosfiltfilt(
            sos, y[:, 0])
    y[:, 0] = no_match_highband_postfilter(x, y[:, 0], config)
    return y


def process_batch(input_dir: Path, output_dir: Path, config: BatchConfig = BatchConfig(),
                  *, enhance_fn: EnhanceFn = default_enhance,
                  residual_fn: ResidualFn = preliminary_residual) -> BatchResult:
    """Create exactly one same-named output for every top-level input WAV.

    Matching is attempted only for 16 kHz mono files of identical length and
    at least ``min_overlap_s`` duration. Other WAVs use independent enhancement.
    The result depends on the complete input folder, so evaluation must use
    the same folder composition as submission.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input and output folders must differ")
    paths = sorted((p for p in input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".wav"), key=lambda p: p.name)
    if not paths:
        raise ValueError(f"No WAV files in {input_dir}")
    records = []
    for path in paths:
        info = sf.info(path)
        if info.format not in ("WAV", "WAVEX"):
            raise ValueError(f"Expected WAV container: {path}")
        records.append(InputFile(path, info))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult(input_files=len(records))
    groups = defaultdict(list)
    for i, record in enumerate(records):
        info = record.info
        eligible = (config.beta > 0 and config.profile == "pesq" and
                    info.samplerate == 16000 and info.channels == 1 and
                    info.frames >= round(config.min_overlap_s * info.samplerate))
        if eligible:
            groups[(info.samplerate, info.channels, info.frames)].append(i)
        else:
            reason = ("beta_zero" if config.beta == 0 else
                      "non_pesq_profile" if config.profile != "pesq" else
                      "unsupported_format_or_short")
            result.fallback_reasons[reason] += 1

    written = set()
    for _, ids in sorted(groups.items()):
        if len(ids) < 2:
            result.fallback_reasons["singleton_group"] += len(ids)
            continue
        total_samples = sum(records[i].info.frames for i in ids)
        if len(ids) > config.max_match_files or total_samples > config.max_match_samples:
            result.fallback_reasons["resource_limit"] += len(ids)
            continue
        noisy = [_load(records[i])[:, 0] for i in ids]
        residuals = [residual_fn(x, 16000) for x in noisy]
        if any(r.shape != x.shape or not np.isfinite(r).all()
               for r, x in zip(residuals, noisy)):
            raise RuntimeError("Residual estimator returned invalid data")
        matches, pairs = match_pairs(residuals, config)
        result.matched_pairs += len(pairs)
        band_provider = (BandTemplateProvider(
            residuals, 16000, config.matched_band_crossover_hz,
            config.matched_band_scale_cap)
            if pairs and config.profile == "pesq" and
            config.matched_band_scale_cap else None)
        # The rescue graph uses a stricter, positive-only coarse match and
        # narrower fine-lag search. Keep it separate from the validated main
        # matcher so neither branch silently changes the other's results.
        tonal_targets = ([(i, x) for i, x in enumerate(noisy) if
                          matches.get(i) and
                          frame_tone_median(x, 16000) > config.frame_tone_guard]
                         if config.rescue_beta else [])
        rescue_graph = (rescue_edges(residuals, config.rescue_corr_threshold)
                        if tonal_targets else {})
        for local_i, global_i in enumerate(ids):
            x = noisy[local_i]
            original_noisy = x
            cutoff = None
            shelf_db = None
            parameter_overrides = None
            refs = matches.get(local_i, [])
            corrected = False
            regular_corrected = False
            no_match = not refs
            dynamic_shelf = False
            if not refs:
                result.fallback_reasons["no_match"] += 1
                if config.profile == "pesq":
                    parameter_overrides = {"gain_time_bins": config.no_match_gain_time_bins}
            else:
                guard = guard_reason(x, 16000, config)
                rescue_refs = (consistent_rescue_refs(
                    rescue_graph, local_i, len(ids),
                    config.rescue_lag_tolerance, config.rescue_max_neighbors)
                    if guard == "tone_guard" and config.rescue_beta else [])
                if rescue_refs:
                    template = rescue_time_template(residuals, rescue_graph,
                                                    local_i, rescue_refs)
                    if not config.rescue_recompute_low_cutoff:
                        cutoff = original_low_cutoff(x, 16000, config.profile)
                    x = x - rescue_beta_for(x, 16000, config) * template
                    corrected = True
                    result.references_by_file[records[global_i].path.name] = len(rescue_refs)
                    result.matched_files += 1
                    result.rescued_files += 1
                elif guard:
                    result.fallback_reasons[guard] += 1
                else:
                    cutoff = original_low_cutoff(x, 16000, config.profile)
                    shelf_db = corrected_shelf_db(x, 16000, config)
                    nrefs = min(len(refs), config.max_refs)
                    raw_scale = (use_raw_template_scale(x, 16000, nrefs)
                                 if config.profile == "pesq" else False)
                    if (band_provider is not None and
                            (not config.matched_pink_guard or
                             not is_pink_like_noisy(original_noisy, 16000))):
                        template, nrefs = band_provider.template_for(
                            local_i, matches, config.max_refs,
                            raw_target=x if raw_scale else None)
                    else:
                        template, nrefs = template_for(
                            local_i, residuals, matches, config.max_refs,
                            raw_target=x if raw_scale else None)
                    x = x - config.beta * template
                    corrected = True
                    regular_corrected = True
                    if config.profile == "pesq":
                        parameter_overrides = {"gmin_db": config.matched_gmin_db}
                        dynamic_shelf = (config.matched_dynamic_shelf and
                                         shelf_db is None and
                                         enhance_fn is default_enhance and
                                         use_dynamic_shelf(original_noisy, 16000))
                    result.references_by_file[records[global_i].path.name] = nrefs
                    result.matched_files += 1
            y = _enhance_channels(x[:, None], 16000, config.profile,
                                  enhance_fn, cutoff, shelf_db,
                                  parameter_overrides)
            if dynamic_shelf:
                shallow = _enhance_channels(x[:, None], 16000, config.profile,
                                            enhance_fn, cutoff, 23.0,
                                            parameter_overrides)
                activity = dynamic_shelf_activity(x, 16000)
                y += activity[:, None] * (shallow - y)
            if (no_match and enhance_fn is default_enhance and
                    config.profile == "pesq" and config.no_match_long_stft_weight):
                # Two independently reconstructed STFT outputs use only this
                # Noisy file. The longer window helps stable residual noise.
                long_y = _enhance_channels(
                    original_noisy[:, None], 16000, config.profile,
                    enhance_fn, parameter_overrides={
                        "n_fft": 3072, "hop": 384,
                        "gain_time_bins": config.no_match_long_gain_time_bins})
                weight = config.no_match_long_stft_weight
                y = (1 - weight) * y + weight * long_y
            if (not corrected and enhance_fn is default_enhance and
                    config.profile == "pesq" and config.unmatched_ridge_strength):
                if no_match:
                    if config.no_match_ridge_protect_highmask:
                        y[:, 0] = ridge.suppress_highband_ridges_compensated(
                            original_noisy, y[:, 0], 16000,
                            config.no_match_ridge_strength,
                            config.no_match_ridge_residual_scale,
                            dense_strength=config.no_match_ridge_dense_strength,
                            dense_fraction=config.no_match_ridge_dense_fraction)
                    else:
                        y[:, 0] = ridge.suppress_highband_ridges(
                            original_noisy, y[:, 0], 16000,
                            config.no_match_ridge_strength,
                            config.no_match_ridge_residual_scale)
                elif config.unmatched_ridge_adaptive:
                    y[:, 0] = ridge.suppress_highband_ridges_auto(
                        original_noisy, y[:, 0], 16000)
                else:
                    y[:, 0] = ridge.suppress_highband_ridges(
                        original_noisy, y[:, 0], 16000,
                        config.unmatched_ridge_strength)
            if (no_match and enhance_fn is default_enhance and
                    config.profile == "pesq" and config.no_match_midband_boost and
                    lowband_dynamic_db(original_noisy, 16000) <
                    config.no_match_midband_dynamic_db):
                sos = signal.butter(4, (300, 700), "bandpass", fs=16000,
                                    output="sos")
                y[:, 0] += config.no_match_midband_boost * signal.sosfiltfilt(
                    sos, y[:, 0])
            if no_match and enhance_fn is default_enhance and config.profile == "pesq":
                y[:, 0] = no_match_highband_postfilter(original_noisy, y[:, 0],
                                                       config)
            if (refs and not corrected and enhance_fn is default_enhance and
                    config.profile == "pesq" and config.guard_active_highband_boost):
                activity = input_midband_activity(original_noisy, 16000)
                sos = signal.butter(4, (800, 5000), "bandpass", fs=16000,
                                    output="sos")
                high = signal.sosfiltfilt(sos, y[:, 0])
                y[:, 0] += config.guard_active_highband_boost * activity * high
            if (regular_corrected and enhance_fn is default_enhance and
                    config.profile == "pesq" and
                    config.matched_active_highband_boost and
                    matched_highband_restore_allowed(original_noisy, 16000,
                                                     shelf_db)):
                activity = input_midband_activity(original_noisy, 16000)
                sos = signal.butter(4, 800, "highpass", fs=16000,
                                    output="sos")
                high = signal.sosfiltfilt(sos, y[:, 0])
                y[:, 0] += config.matched_active_highband_boost * activity * high
            _save(records[global_i], y, output_dir)
            result.written_files += 1
            written.add(global_i)
        # Free group arrays before the next group. No corrected waveforms from
        # earlier groups remain in memory.
        del noisy, residuals

    for i, record in enumerate(records):
        if i in written:
            continue
        x = _load(record)
        lone_no_match = (enhance_fn is default_enhance and config.beta > 0 and
                         config.profile == "pesq" and
                         record.info.samplerate == 16000 and
                         record.info.channels == 1 and
                         record.info.frames >= round(config.min_overlap_s * 16000))
        if lone_no_match:
            y = enhance_single_no_match(x[:, 0], config)
        else:
            y = _enhance_channels(x, record.info.samplerate, config.profile,
                                  enhance_fn)
        if (not lone_no_match and enhance_fn is default_enhance and config.beta > 0 and
                config.profile == "pesq" and
                config.unmatched_ridge_strength and record.info.samplerate == 16000 and
                record.info.channels == 1):
            if config.unmatched_ridge_adaptive:
                y[:, 0] = ridge.suppress_highband_ridges_auto(
                    x[:, 0], y[:, 0], 16000)
            else:
                y[:, 0] = ridge.suppress_highband_ridges(
                    x[:, 0], y[:, 0], 16000, config.unmatched_ridge_strength)
        _save(record, y, output_dir)
        result.written_files += 1
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=.4)
    ap.add_argument("--max-refs", type=int, default=3)
    ap.add_argument("--frame-tone-guard", type=float, default=0.65)
    ap.add_argument("--quiet-psd-flatness-guard", type=float, default=0.93)
    ap.add_argument("--quiet-psd-top5-guard", type=float, default=0.95)
    ap.add_argument("--rescue-beta", type=float, default=0.75)
    ap.add_argument("--rescue-fixed-original-cutoff", action="store_true")
    ap.add_argument("--no-dynamic-shelf", action="store_true")
    ap.add_argument("--profile", choices=("pesq", "balanced"), default="pesq")
    args = ap.parse_args()
    config = BatchConfig(beta=args.beta, corr_threshold=args.threshold,
                         max_refs=args.max_refs, frame_tone_guard=args.frame_tone_guard,
                         quiet_psd_flatness_guard=args.quiet_psd_flatness_guard,
                         quiet_psd_top5_guard=args.quiet_psd_top5_guard,
                         rescue_beta=args.rescue_beta,
                         rescue_recompute_low_cutoff=not args.rescue_fixed_original_cutoff,
                         matched_dynamic_shelf=not args.no_dynamic_shelf,
                         profile=args.profile)
    result = process_batch(args.input, args.output, config)
    print(f"Wrote {result.written_files}/{result.input_files} WAVs; "
          f"matched files={result.matched_files}, pairs={result.matched_pairs}; "
          f"rescued files={result.rescued_files}; "
          f"fallbacks={dict(result.fallback_reasons)}")


if __name__ == "__main__":
    main()

"""Traditional DSP speech enhancement (no deep learning / no machine learning).

Pipeline (per channel):
  0. Impulse pre-suppression on a short STFT (only for files with many broadband transients,
     e.g. clapping, footsteps, door noise)
  1. STFT (Hann window)
  2. Noise PSD estimate = geometric mean of
       (a) mean of the lowest-energy frames (fixed per file), and
       (b) the SPP-based unbiased MMSE noise tracker of Gerkmann & Hendriks (2012), run
           forward and backward in time (offline processing) and geometrically averaged,
     scaled by an over-estimation factor. The weight of (a) vs. (b) and the over-estimation
     factor are interpolated by a noise-stationarity index measured on the input
     (mean |log(tracked / fixed)|): stationary broadband noise -> rely on the fixed estimate
     and subtract more; fluctuating noise -> rely on the tracker.
     Stationary tonal lines (e.g. sirens, bells) found from the per-bin temporal median are
     added to the noise estimate and attenuated with a lower gain floor.
  3. Decision-directed a priori SNR (Ephraim & Malah 1984)
  4. Log-MMSE spectral amplitude gain (Ephraim & Malah 1985) with gain floor
  5. Smooth the STFT gain in frequency and time, then apply a smooth high-pass
     weighting (low-frequency noise suppression)
  6. ISTFT with the noisy phase
  7. Optional zero-phase high-frequency shelf and temporal noise gate
  8. Input-noise-PSD-guided zero-phase low-band attenuation
All parameters below were chosen by grid / coordinate search on the public
train/validation data; nothing is learned or stored besides these constants.
"""
import numpy as np
from scipy import signal
from scipy.special import exp1
from scipy.ndimage import uniform_filter1d, median_filter

EPS = 1e-12

# Parameters for 16 kHz input (hop / n_fft are scaled for other sampling rates).
# "pesq": maximises PESQ-WB (ranking metric); strong low-frequency attenuation.
# "balanced": keeps more low-frequency speech energy (higher SNR, slightly lower PESQ).
PROFILES = {
    "pesq": dict(
        n_fft=1536, hop=192,
        frac=0.1,            # fraction of lowest-energy frames for the fixed noise estimate
        w=0.3,               # geometric-mean weight of fixed vs. tracked noise estimate (fluctuating noise)
        a_pow=0.9,           # SPP tracker smoothing
        xi_h1_db=15.0,       # SPP tracker: fixed a priori SNR under speech presence
        bidir=True,          # forward + backward SPP tracking, geometric mean
        noise_scale=1.5,     # noise over-estimation factor (fluctuating noise)
        # stationarity adaptation: index v = mean |log(tracked/fixed)|; v <= st_lo -> stationary
        st_lo=0.45, st_hi=0.6,   # linear ramp between (w, noise_scale) and (w_st, noise_scale_st)
        w_st=0.8,            # weight of the fixed estimate for stationary noise
        noise_scale_st=2.2,  # over-estimation factor for stationary noise
        a=0.92,              # decision-directed smoothing
        xi_min_db=-25.0,     # a priori SNR floor
        gmin_db=-30.0,       # gain floor
        hp_hz=240.0,         # corner of the high-pass gain weighting (0 = off)
        hp_order=4,          # Butterworth-like order of the weighting
        hp_floor_db=-20.0,   # weighting never attenuates more than this
        # stationary tonal noise (sirens, bells, dial tones)
        tone_thr_db=10.0,    # peak of per-bin temporal median above its spectral median
        tone_min_bins=30,    # apply only if at least this many tonal bins are found
        tone_scale=2.0,      # noise PSD in tonal bins >= tone_scale * temporal median
        tone_gmin_db=-50.0,  # gain floor in tonal bins
        # stable sinusoids (dial tones, electronic sirens): whole-block least-squares fit and subtraction
        sin_thr_db=20.0,     # spectral peak above the local spectral median (whole-block spectrum)
        sin_max=40,          # at most this many sinusoids
        sin_gate_db=8.0,     # keep only if the quietest 10% frames lose at least this much energy
        # impulsive noise (short STFT 256/64)
        imp_thr_db=8.0,      # sub-band jump above the running temporal median (15 frames)
        imp_bands=3,         # ... in at least this many of 4 sub-bands -> transient frame
        imp_min_frames=30,   # apply only if at least this many transient frames
        imp_gate_db=12.0,    # and at least imp_gate_frames frames jump > imp_gate_db
        imp_gate_frames=6,   #    in all 4 sub-bands
    ),
}
# Keep the preceding Log-MMSE configuration available for reproducible experiments.
PROFILES["research_base"] = dict(PROFILES["pesq"])
PROFILES["pesq"] = dict(
    PROFILES["research_base"],
    hp_hz=400.0, hp_order=6, hp_floor_db=-20.0,
    gain_freq_bins=3, gain_time_bins=3,
    shelf_hz=625.0, shelf_db=29.0, shelf_order=10,
    gate_threshold_db=6.0, gate_min_gain=0.65,
    gate_tone_fraction=0.60,
    output_gain=1.5,
    low_cutoff_rule=True, low_attenuation=0.5, low_order=4,
)
PROFILES["balanced"] = dict(PROFILES["research_base"], hp_hz=120.0, hp_floor_db=-200.0)
PARAMS = PROFILES["pesq"]


def _stft(x, n_fft, hop):
    return signal.stft(x, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
                       boundary="zeros", padded=True)[2]


def _istft(Z, length, n_fft, hop):
    _, y = signal.istft(Z, window="hann", nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft,
                        input_onesided=True, boundary=True)
    y = y[:length]
    return np.pad(y, (0, length - len(y))) if len(y) < length else y


def noise_fixed(P, frac):
    """Mean power of the lowest-energy complete frames (boundary frames excluded)."""
    T = P.shape[1]
    inner = np.arange(2, T - 2) if T > 8 else np.arange(T)
    k = max(1, int(np.ceil(frac * len(inner))))
    sel = inner[np.argsort(P[:, inner].mean(0), kind="stable")[:k]]
    est = uniform_filter1d(P[:, sel].mean(1), 3, mode="nearest")
    return np.maximum(est, EPS)


def noise_spp_tracker(P, init, xi_h1_db, a_pow, a_p=0.9):
    """Gerkmann & Hendriks (2012), IEEE TASLP 20(4), 1383-1393."""
    lam = init.copy()
    xi = 10 ** (xi_h1_db / 10)
    pbar = np.full(P.shape[0], 0.5)
    out = np.empty_like(P)
    for t in range(P.shape[1]):
        y = P[:, t]
        g = y / np.maximum(lam, EPS)
        p = 1.0 / (1.0 + (1 + xi) * np.exp(np.minimum(-g * xi / (1 + xi), 50)))
        pbar = a_p * pbar + (1 - a_p) * p
        p = np.where(pbar > 0.99, np.minimum(p, 0.99), p)   # avoid stagnation
        lam = a_pow * lam + (1 - a_pow) * ((1 - p) * y + p * lam)
        out[:, t] = lam
    return np.maximum(out, EPS)


def stationarity(fixed, tracked, lo, hi):
    """0 (fluctuating) .. 1 (stationary): how far the tracked noise PSD strays from the fixed one."""
    T = tracked.shape[1]
    tr = tracked[:, 2:T - 2] if T > 8 else tracked
    v = np.mean(np.abs(np.log(tr / fixed[:, None])))
    return float(np.clip((hi - v) / (hi - lo), 0.0, 1.0))


def tonal_lines(P, thr_db, width=31, q=0.5, dil=1):
    """Bins holding stationary tonal noise: the per-bin temporal median is a spectral peak.
    Speech harmonics move with pitch, so they do not form peaks in the temporal median."""
    m = np.median(P, axis=1) if q == 0.5 else np.quantile(P, q, axis=1)
    ldb = 10 * np.log10(m + EPS)
    tone = (ldb - median_filter(ldb, size=width, mode="nearest")) > thr_db
    tone = np.convolve(tone.astype(float), np.ones(2 * dil + 1), "same") > 0
    return tone, m


def suppress_impulses(x, sr, p):
    """Attenuate short broadband transients back to the running temporal median level.
    Returns x unchanged unless the file contains many such transients (impulsive noise)."""
    scale = sr / 16000.0
    n_fft, hop = max(32, int(round(256 * scale))), max(8, int(round(64 * scale)))
    Z = _stft(x, n_fft, hop)
    P = np.abs(Z) ** 2
    med = median_filter(P, size=(1, 15), mode="nearest")
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = (300, 1000, 2500, 4500, 8000)
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (f >= lo) & (f < hi)
        ratios.append(10 * np.log10((P[b].sum(0) + EPS) / (med[b].sum(0) + EPS)))
    ratios = np.array(ratios)
    fl = (ratios > p["imp_thr_db"]).sum(0) >= p["imp_bands"]
    strong = ((ratios > p["imp_gate_db"]).sum(0) >= 4).sum()
    if fl.sum() < p["imp_min_frames"] or strong < p["imp_gate_frames"]:
        return x
    g = np.ones_like(P)
    g[:, fl] = np.sqrt(np.minimum(1.0, med[:, fl] / (P[:, fl] + EPS)))
    return _istft(g * Z, len(x), n_fft, hop)


def _fit_sinusoids(x, sr, thr_db, n_max):
    """Least-squares fit of the strongest stable sinusoids found in the whole-block spectrum."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n * 4)))
    L = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(n), nfft)) + EPS)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    step = max(1, nfft // 32768)                      # coarse floor estimate (~60 Hz median)
    k_w = max(3, int(60.0 / (sr / nfft) / step) | 1)
    base = np.repeat(median_filter(L[::step], size=k_w, mode="nearest"), step)[:len(L)]
    d = L - base
    k = np.arange(2, len(L) - 2)
    k = k[(d[k] > thr_db) & (L[k] >= L[k - 1]) & (L[k] >= L[k + 1]) & (f[k] > 40)]
    sel = []
    for i in k[np.argsort(-L[k], kind="stable")]:
        if all(abs(f[i] - f[j]) > 6.0 for j in sel):
            sel.append(i)
            if len(sel) >= n_max:
                break
    if not sel:
        return np.zeros(n)
    sel = np.array(sel)
    a, b, c = L[sel - 1], L[sel], L[sel + 1]
    den = a - 2 * b + c
    fr = f[sel] + np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1), 0) * sr / nfft
    t = np.arange(n) / sr
    A = np.concatenate([np.cos(2 * np.pi * fr[None] * t[:, None]), np.sin(2 * np.pi * fr[None] * t[:, None])], 1)
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    return A @ coef


def remove_stable_tones(x, sr, p):
    """Subtract stationary sinusoids (e.g. dial tones) estimated over ~3 s blocks (50% overlap-add).
    A block is changed only if the fit removes energy from its quietest frames, i.e. the sinusoids
    are present where speech is absent (speech harmonics fail this test)."""
    n = len(x)
    L = int(3.0 * sr)
    starts = [0] if n <= int(1.5 * L) else list(range(0, n - L // 2, L // 2))
    tones = np.zeros(n); wsum = np.zeros(n)
    nf, hp = max(32, int(round(1536 * sr / 16000))), max(8, int(round(192 * sr / 16000)))
    for s0 in starts:
        seg = x[s0:s0 + L] if len(starts) > 1 else x
        m = len(seg)
        w = np.ones(m) if len(starts) == 1 else np.hanning(m + 2)[1:-1]
        wsum[s0:s0 + m] += w
        if m < nf:
            continue
        tn = _fit_sinusoids(seg, sr, p["sin_thr_db"], p["sin_max"])
        e0 = (np.abs(_stft(seg, nf, hp)) ** 2).sum(0)
        e1 = (np.abs(_stft(seg - tn, nf, hp)) ** 2).sum(0)
        inner = slice(2, -2) if len(e0) > 8 else slice(None)
        e0, e1 = e0[inner], e1[inner]
        q = np.argsort(e0)[:max(1, len(e0) // 10)]
        if 10 * np.log10((e0[q].sum() + EPS) / (e1[q].sum() + EPS)) >= p["sin_gate_db"]:
            tones[s0:s0 + m] += w * tn
    return x - tones / np.maximum(wsum, EPS)


def logmmse_gain(xi, gamma):
    """Ephraim & Malah (1985) log-spectral amplitude gain."""
    v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
    return xi / (1 + xi) * np.exp(0.5 * exp1(v))


def output_shelf(y, sr, cutoff_hz, depth_db, order):
    """Keep the low-pass output and a small fraction of the residual high band.

    Filtering forward and backward avoids shifting speech in time. The cutoff
    is kept below Nyquist for input WAV files with a different sample rate.
    """
    if len(y) < 2:
        return y.copy()
    cutoff = min(float(cutoff_hz), 0.45 * sr)
    sos = signal.butter(int(order), cutoff, "lowpass", fs=sr, output="sos")
    try:
        low = signal.sosfiltfilt(sos, y)
    except ValueError:  # a very short input cannot accommodate the default padding
        low = signal.sosfiltfilt(sos, y, padlen=len(y) - 1)
    return low + 10 ** (-depth_db / 20) * (y - low)


def temporal_gate(y, sr, threshold_db, min_gain):
    """Attenuate quiet output intervals using an envelope from this file only."""
    window = max(1, round(0.04 * sr))
    rms = np.sqrt(np.maximum(uniform_filter1d(y * y, window, mode="nearest"), 1e-16))
    margin = min(len(rms) // 4, round(0.1 * sr))
    inner = rms[margin:-margin] if margin else rms
    floor = max(float(np.percentile(inner, 20)), 1e-8)
    level_db = 20 * np.log10(np.maximum(rms, 1e-8) / floor)
    activity = np.clip((level_db - threshold_db) / 6.0, 0, 1)
    gain = min_gain + (1 - min_gain) * activity
    gain = uniform_filter1d(gain, max(1, round(0.06 * sr)), mode="nearest")
    return y * gain


def narrowband_noise_fraction(x, sr, n_fft, hop, frac):
    """Fraction of sub-1 kHz noise power in its five strongest STFT bins.

    The feature is computed on the original noisy input, before any tone or
    impulse suppression. Strong persistent lines indicate that the temporal
    gate could also remove quiet speech in such recordings.
    """
    xp = np.pad(x, (0, n_fft - len(x))) if len(x) < n_fft else x
    power = np.abs(_stft(xp, n_fft, hop)) ** 2
    noise_psd = noise_fixed(power, frac)
    low = noise_psd[np.fft.rfftfreq(n_fft, 1.0 / sr) < 1000]
    return float(np.sort(low)[-5:].sum() / max(low.sum(), EPS))


def choose_low_cutoff(noisy, sr, n_fft, hop, frac):
    """Choose a low-band cutoff from the input's quiet-frame noise spectrum.

    These fixed thresholds were selected on public Train files. No clean
    waveform, noise label, stored model, or validation score enters this rule.
    """
    xp = np.pad(noisy, (0, n_fft - len(noisy))) if len(noisy) < n_fft else noisy
    P = np.abs(_stft(xp, n_fft, hop)) ** 2
    floor = noise_fixed(P, frac)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = max(float(floor.sum()), EPS)
    low = float(floor[f < 300].sum() / total)
    mid = float(floor[(f >= 300) & (f < 600)].sum() / total)
    under1k = floor[f < 1000]
    top5 = float(np.sort(under1k)[-5:].sum() / max(float(under1k.sum()), EPS))
    active = low > 0.30 or (mid > 0.25 and top5 < 0.40)
    if active:
        return 450 if low > 0.75 else 350
    return 200 if 0.12 < low < 0.25 else 0


def attenuate_low_band(y, sr, cutoff_hz, attenuation, order):
    """Subtract a fraction of the zero-phase low-pass component."""
    if cutoff_hz <= 0 or len(y) < 2:
        return y
    cutoff = min(float(cutoff_hz), 0.45 * sr)
    sos = signal.butter(int(order), cutoff, "lowpass", fs=sr, output="sos")
    try:
        low = signal.sosfiltfilt(sos, y)
    except ValueError:
        low = signal.sosfiltfilt(sos, y, padlen=len(y) - 1)
    out = y - float(attenuation) * low
    peak = np.max(np.abs(out))
    return out * min(1.0, 0.999 / peak) if peak else out


def enhance(x, sr=16000, profile="pesq", **overrides):
    """Denoise a 1-D float waveform; output has exactly the same length."""
    p = dict(PROFILES[profile], **overrides)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return x.copy()
    scale = sr / 16000.0
    n_fft = max(32, int(round(p["n_fft"] * scale)))
    hop = max(8, int(round(p["hop"] * scale)))
    n = len(x)
    low_cutoff = choose_low_cutoff(x, sr, n_fft, hop, p["frac"]) if p.get("low_cutoff_rule") else 0
    gate_enabled = bool(p.get("gate_threshold_db"))
    if gate_enabled and p.get("gate_tone_fraction"):
        gate_enabled = (narrowband_noise_fraction(x, sr, n_fft, hop, p["frac"])
                        < p["gate_tone_fraction"])
    if p.get("sin_max") and n >= n_fft:
        x = remove_stable_tones(x, sr, p)
    if p.get("imp_min_frames") and n >= max(32, int(round(256 * scale))):
        x = suppress_impulses(x, sr, p)
    xp = np.pad(x, (0, n_fft - n)) if n < n_fft else x
    Z = _stft(xp, n_fft, hop)
    P = np.abs(Z) ** 2
    F, T = P.shape
    fixed = noise_fixed(P, p["frac"])
    tracked = noise_spp_tracker(P, fixed, p["xi_h1_db"], p["a_pow"])
    if p.get("bidir"):
        back = noise_spp_tracker(P[:, ::-1], fixed, p["xi_h1_db"], p["a_pow"])[:, ::-1]
        tracked = np.sqrt(tracked * back)
    w, ns = p["w"], p["noise_scale"]
    if p.get("st_hi"):
        st = stationarity(fixed, tracked, p["st_lo"], p["st_hi"])
        w = w + st * (p["w_st"] - w)
        ns = ns * (p["noise_scale_st"] / ns) ** st
    lam = np.exp(w * np.log(fixed)[:, None] + (1 - w) * np.log(tracked)) * ns
    tone = np.zeros(F, bool)
    if p.get("tone_min_bins"):
        tone, med = tonal_lines(P, p["tone_thr_db"], p.get("tone_width", 31), p.get("tone_q", 0.5), p.get("tone_dil", 1))
        if tone.sum() >= p["tone_min_bins"]:
            lam[tone] = np.maximum(lam[tone], p["tone_scale"] * med[tone, None])
        else:
            tone[:] = False
    gmin_tone = 10 ** (p.get("tone_gmin_db", 0.0) / 20)
    gam = np.minimum(P / lam, 1e4)
    xi_min = 10 ** (p["xi_min_db"] / 10)
    gmin = 10 ** (p["gmin_db"] / 20)
    G = np.empty_like(P)
    prev = np.zeros(F)
    for t in range(T):
        ml = np.maximum(gam[:, t] - 1, 0)
        xi = ml if t == 0 else p["a"] * prev / lam[:, t] + (1 - p["a"]) * ml
        g = np.clip(logmmse_gain(np.maximum(xi, xi_min), gam[:, t]), gmin, 1.0)
        if tone.any():   # deeper attenuation allowed in tonal-noise bins
            g_t = np.clip(logmmse_gain(np.maximum(xi[tone], 1e-6), gam[tone, t]), gmin_tone, g[tone])
            g[tone] = g_t
        G[:, t] = g
        prev = g * g * P[:, t]
    if p.get("gain_freq_bins", 0) > 1:
        G = uniform_filter1d(G, int(p["gain_freq_bins"]), axis=0, mode="nearest")
    if p.get("gain_time_bins", 0) > 1:
        G = uniform_filter1d(G, int(p["gain_time_bins"]), axis=1, mode="nearest")
    if p["hp_hz"]:
        f = np.fft.rfftfreq(n_fft, 1.0 / sr)
        h = 1.0 / np.sqrt(1 + (p["hp_hz"] / np.maximum(f, 1.0)) ** (2 * p["hp_order"]))
        G *= np.maximum(h, 10 ** (p["hp_floor_db"] / 20))[:, None]
    y = _istft(G * Z, len(xp), n_fft, hop)[:n]
    if p.get("shelf_hz"):
        y = output_shelf(y, sr, p["shelf_hz"], p["shelf_db"], p["shelf_order"])
    if gate_enabled:
        y = temporal_gate(y, sr, p["gate_threshold_db"], p["gate_min_gain"])
    if p.get("output_gain"):
        peak = np.max(np.abs(y))
        if peak:
            y *= min(float(p["output_gain"]), 0.999 / peak)
    if low_cutoff:
        y = attenuate_low_band(y, sr, low_cutoff, p["low_attenuation"], p["low_order"])
    return y

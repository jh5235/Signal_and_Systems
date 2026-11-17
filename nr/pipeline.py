"""Full enhancement pipeline (classical DSP only).

1. STFT (1024-pt Hann, hop 256 @16 kHz)
2. Noise PSD: geometric mean of a static estimate (lowest-energy frames) and a bidirectional
   SPP-based MMSE tracker (Gerkmann & Hendriks 2012)
3. Optional recording-floor preservation: the estimated noise floor that looks like the
   talker's own recording floor is kept in the output (it is part of the reference signal)
4. A priori SNR: non-causal decision-directed (forward/backward geometric mean)
5. Two-step noise reduction (TSNR) + harmonic regeneration (HRNR), Plapous et al. 2006
6. Log-MMSE (Ephraim & Malah 1985) gain with floor
"""
from __future__ import annotations

import numpy as np

from . import core as C

DEFAULTS = dict(
    n_fft=1024, hop=256, sfrac=0.1, a=0.98, gmin_db=-30.0, xi_min_db=-25.0,
    rho=0.5, tsnr_mix=0.5, nmode="geo", spp_alpha=0.8, xi_h1=15.0, nscale=1.0,
    floor=False, floor_pct=5.0, floor_cap_db=-40.0, floor_keep=True, floor_sub=1.0,
    floor_band=(300.0, 7000.0), hrnr=True, ncausal=True,
    low_hz=0.0, gmin_low_db=-40.0,
    mod_bands=6, mod_q=0.3, mod_med=5, mod_max_db=20.0, mod_min_db=-3.0, mod_mix=1.0,
)

# Spectral shape of a typical recording noise floor: a smooth -4 dB/octave slope
# (pink-like microphone/room noise). Only the level is estimated per file.
FLOOR_SLOPE_DB_PER_OCT = -4.0


def floor_shape(n_bins: int, fs: int = 16000) -> np.ndarray:
    f = np.maximum(np.linspace(0, fs / 2, n_bins), 100.0)
    return (f / 1000.0) ** (FLOOR_SLOPE_DB_PER_OCT / (10 * np.log10(2)))


def modulated_noise(P, N0, p):
    """Noise = fixed spectral shape N0 x band-wise time-varying level g_b(t).

    g_b(t) is a low quantile over the bins of band b of P/N0 (speech occupies only part of the bins),
    divided by the same quantile of an exponential variable (noise periodogram statistics)."""
    from scipy.ndimage import median_filter
    F, T = P.shape
    R = P / N0[:, None]
    edges = np.unique(np.round(np.geomspace(4, F, p["mod_bands"] + 1)).astype(int))
    edges[0] = 0
    cq = -np.log(1 - p["mod_q"])
    Nm = np.empty_like(P)
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = np.quantile(R[lo:hi], p["mod_q"], axis=0) / cq
        if p["mod_med"] > 1:
            g = median_filter(g, p["mod_med"], mode="nearest")
        g = np.clip(g, 10 ** (p["mod_min_db"] / 10), 10 ** (p["mod_max_db"] / 10))
        Nm[lo:hi] = N0[lo:hi, None] * g[None, :]
    return Nm


def estimate_noise(P, p):
    N0 = C.init_noise(P, p["sfrac"])
    S0 = np.tile(N0[:, None], (1, P.shape[1]))
    if p["nmode"] == "static":
        return S0, N0
    if p["nmode"].startswith("mod"):
        M = modulated_noise(P, N0, p)
        M = M ** p["mod_mix"] * S0 ** (1 - p["mod_mix"])
        if p["nmode"] == "mod":
            return M, N0
        a = C.noise_mmse_spp(P, N0, p["xi_h1"], p["spp_alpha"])
        b = C.noise_mmse_spp(P, N0, p["xi_h1"], p["spp_alpha"], backward=True)
        return np.sqrt(np.sqrt(a * b) * M), N0
    a = C.noise_mmse_spp(P, N0, p["xi_h1"], p["spp_alpha"])
    b = C.noise_mmse_spp(P, N0, p["xi_h1"], p["spp_alpha"], backward=True)
    T = np.sqrt(a * b)
    if p["nmode"] == "tracker":
        return T, N0
    return np.sqrt(T * S0), N0


def estimate_floor(P, N0, p, n_fft, fs=16000):
    """Recording floor = c * shape, c from a low percentile of N0/shape, capped relative to speech peak."""
    shp = floor_shape(len(N0))
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    band = (f > p["floor_band"][0]) & (f < p["floor_band"][1])
    r = np.log10(N0[band] / shp[band] + 1e-30) * 10
    c_db = np.percentile(r, p["floor_pct"])
    peak_db = 10 * np.log10(P.sum(0).max() + 1e-30)
    shape_total_db = 10 * np.log10(shp.sum())
    c_db = min(c_db, peak_db + p["floor_cap_db"] - shape_total_db)
    return shp * 10 ** (c_db / 10)


_MS_BIAS = {}


def _ms_bias(n_fft, hop, fw, tw):
    """mean / min ratio of the smoothed periodogram of white noise (3-s signal), per configuration."""
    key = (n_fft, hop, fw, tw)
    if key not in _MS_BIAS:
        from scipy.ndimage import uniform_filter
        rng = np.random.default_rng(1)
        vals = []
        for _ in range(6):
            Zw, _ = C.stft(rng.standard_normal(48000), n_fft, hop)
            Pw = np.abs(Zw) ** 2
            Sw = uniform_filter(Pw, (fw, tw), mode="nearest")
            vals.append(np.median(Pw.mean(1) / Sw[:, 2:-2].min(1)))
        _MS_BIAS[key] = float(np.mean(vals))
    return _MS_BIAS[key]


def min_stat_cap(P, n_fft, hop, fw=3, tw=5):
    """Per-bin noise floor from the minimum over time of the (f,t)-smoothed periodogram, bias-compensated
    to the mean of stationary noise. Short speech pauses/closures reveal the true floor in every band."""
    from scipy.ndimage import uniform_filter
    S = uniform_filter(P, (fw, tw), mode="nearest")
    return S[:, 2:-2].min(1) * _ms_bias(n_fft, hop, fw, tw)


def _prior_snr(Z, meta, P, N, p, g, g_dd, wprior=None):
    """Non-causal DD + TSNR + HRNR a priori SNR; returns (xi_final, gamma, G).
    wprior: optional per-bin weight applied to the intermediate a priori SNRs."""
    n_fft, hop = p["n_fft"], p["hop"]
    gamma = P / N
    xmin = 10 ** (p["xi_min_db"] / 10)
    xi, _, _ = C.dd_prior_snr(P, N, p["a"], p["xi_min_db"], g_dd)
    if p["ncausal"]:
        xb, _, _ = C.dd_prior_snr(P[:, ::-1], N[:, ::-1], p["a"], p["xi_min_db"], g_dd)
        xi = np.sqrt(xi * xb[:, ::-1])
    if wprior is not None:
        xi = np.maximum(xi * wprior, xmin)
    G = g(xi, gamma)
    if p["hrnr"]:
        xi2 = np.maximum(G ** 2 * gamma, xmin) ** p["tsnr_mix"] * xi ** (1 - p["tsnr_mix"])
        if wprior is not None:
            xi2 = np.maximum(xi2 * wprior, xmin)
        G = g(xi2, gamma)
        s = C.istft(G * Z, meta)
        H = C.stft(np.maximum(s, 0), n_fft, hop)[0]
        xi = np.maximum((p["rho"] * G ** 2 * P + (1 - p["rho"]) * np.abs(H) ** 2) / N, xmin)
        G = g(xi, gamma)
    return xi, gamma, G


def enhance(x: np.ndarray, period=None, rep_mix=1.0, rep_scale=1.0, comb=None, esch=None, rep_smooth=0.0, rep_max=0.0, rep_complex=0.0, rep_w=1.0, beat=None, rep_mmse=False, rep_mmse_noise="N", rep_eps=1e-2, rep_post=0.0, rep_mmse_w=1.0, rep_sp=None, rep_iter=0, rep_iter_smooth=False, rep_em=None, omlsa=None, lt_corr=None, ms_cap=None, band_cap=None, consist=None, lt_prior=None, **kw) -> np.ndarray:
    p = dict(DEFAULTS, **kw)
    n_fft, hop = p["n_fft"], p["hop"]
    Z, meta = C.stft(x, n_fft, hop)
    P = np.abs(Z) ** 2 + 1e-20
    N, N0 = estimate_noise(P, p)
    N = N * p["nscale"]
    if beat and not period:
        from .repet import beat_period, repeat_min_frames
        lo = max(2, int(beat.get("min_s", 0.4) * 16000 / hop)); hi = int(beat.get("max_s", 1.6) * 16000 / hop)
        bp, bpk, bpr = beat_period(P, lo, hi)
        if bp and bpr > beat.get("thr", 3.0):
            R = repeat_min_frames(P, bp, beat.get("reps", 2)) * beat.get("scale", 1.0) + 1e-20
            R = np.where(np.isfinite(R), R, N)
            mix = beat.get("mix", 0.5)
            N = R ** mix * N ** (1 - mix)
    if period:
        from .repet import shifted_stfts, repeat_noise_psd
        Zs, _ = shifted_stfts(x, int(period), lambda v: C.stft(v, n_fft, hop))
        R = repeat_noise_psd(Zs) * rep_scale + 1e-20
        if rep_smooth:
            for t in range(1, R.shape[1]):
                R[:, t] = rep_smooth * R[:, t - 1] + (1 - rep_smooth) * R[:, t]
        cover = np.sum([~np.isnan(z[0]) for z in Zs[1:]], axis=0) > 0
        Nr = np.where(cover[None, :], R, N)
        if rep_max:
            Nr = np.minimum(Nr, rep_max * N)
        N = Nr ** rep_mix * N ** (1 - rep_mix)
    if ms_cap:
        cap = min_stat_cap(P, n_fft, hop, ms_cap.get("fw", 3), ms_cap.get("tw", 5)) * ms_cap.get("kappa", 2.0)
        N = np.minimum(N, cap[:, None])
        N0 = np.minimum(N0, cap)
    F = None
    if p["floor"]:
        F = estimate_floor(P, N0, p, n_fft)
        if p["floor_sub"]:
            N = np.maximum(N - p["floor_sub"] * F[:, None], 0.1 * N)
    g = {"logmmse": C.gain_logmmse, "mapsg": C.gain_map_sg, "wiener": C.gain_wiener,
         "mix": lambda a, b: C.gain_mix(a, b, p.get("gmix", 0.5)),
         "beta": lambda a, b: C.gain_beta(a, b, p.get("beta", 0.5))}[p.get("gain", "logmmse")]
    g_dd = C.gain_logmmse if p.get("dd_logmmse", True) else g
    if lt_corr:
        # long-term per-bin noise spectrum from an SPP-weighted average of the noisy power
        xi0, gamma0, G0 = _prior_snr(Z, meta, P, N, p, g, g_dd)
        fs_ = lt_corr.get("fsm", 1)
        if fs_ > 1:      # neighbours decide whether a bin is noise-dominated (less self-selection bias)
            from scipy.ndimage import uniform_filter1d
            xi0 = np.exp(uniform_filter1d(np.log(xi0), fs_, axis=0, mode="nearest"))
            gamma0 = uniform_filter1d(gamma0, fs_, axis=0, mode="nearest")
        w = (1 - C.spp(xi0, gamma0, lt_corr.get("q0", 0.5))) ** lt_corr.get("q", 3.0)
        Nlt = (w * P).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1e-9)
        cmax = 10 ** (lt_corr.get("clip_db", 20.0) / 10)
        corr = np.clip(Nlt / N.mean(1, keepdims=True), 1 / cmax, cmax) ** lt_corr.get("mix", 1.0)
        N = N * corr
        if lt_corr.get("tpl_fix") is not None:          # experimental: external spectral template
            N = N * lt_corr["tpl_fix"](N.mean(1), N.shape[0])[:, None]
        if band_cap:
            # bands whose 'noise' estimate is far above the minimum-statistics floor AND whose energy
            # follows the speech envelope of the other bands carry (almost) no added noise
            ms = min_stat_cap(P, n_fft, hop, 3, 5)
            fr = np.fft.rfftfreq(n_fft, 1 / 16000)
            edges_hz = np.geomspace(60, 8000, band_cap.get("n_bands", 21) + 1)
            S_hat = (G0 ** 2) * P
            Nlt1 = Nlt[:, 0]
            for lo, hi in zip(edges_hz[:-1], edges_hz[1:]):
                b = (fr >= lo) & (fr < hi)
                if not b.any():
                    continue
                Eb = np.log(P[b].sum(0)); ref = np.log(S_hat[~b].sum(0) + 1e-20)
                rho = np.corrcoef(Eb[2:-2], ref[2:-2])[0, 1]
                ratio = 10 * np.log10(Nlt1[b].mean() / ms[b].mean())
                if ratio > band_cap.get("ratio_db", 6.0) and rho > band_cap.get("rho", 0.6):
                    N[b] = np.minimum(N[b], band_cap.get("kappa", 1.0) * ms[b, None])
    wpri = None
    if lt_prior and lt_prior.get("early"):
        lts0 = np.maximum(P.mean(1) - N.mean(1), 0.0) / N.mean(1)
        if lt_prior.get("fsm", 1) > 1:
            from scipy.ndimage import uniform_filter1d
            lts0 = uniform_filter1d(lts0, lt_prior["fsm"], mode="nearest")
        c0 = 10 ** (lt_prior.get("c_db", -5.0) / 10)
        wpri = ((lts0 / (lts0 + c0)) ** lt_prior.get("gamma_early", lt_prior.get("gamma", 1.0)))[:, None]
    xi3, gamma, G = _prior_snr(Z, meta, P, N, p, g, g_dd, wpri)
    xi = xi3
    if lt_prior:
        # long-term speech-to-noise ratio per bin acts as a prior: bins where the talker is (long-term)
        # much weaker than the noise rarely carry speech, so their a priori SNR is shrunk
        from scipy.ndimage import uniform_filter1d, gaussian_filter1d
        sig = lt_prior.get("sig_s", 0.0)
        if sig:          # local (segmental) long-term SNR, Gaussian window in time
            st = sig * 16000 / hop
            Pm = gaussian_filter1d(P, st, axis=1, mode="nearest"); Nm_ = gaussian_filter1d(N, st, axis=1, mode="nearest")
        else:
            Pm = P.mean(1, keepdims=True); Nm_ = N.mean(1, keepdims=True)
        lts = np.maximum(Pm - Nm_, 0.0) / Nm_
        if lt_prior.get("fsm", 1) > 1:
            lts = uniform_filter1d(lts, lt_prior["fsm"], axis=0, mode="nearest")
        c_ = 10 ** (lt_prior.get("c_db", -5.0) / 10)
        wgt = (lts / (lts + c_)) ** lt_prior.get("gamma", 1.0)
        xi = np.maximum(xi * wgt, 10 ** (p["xi_min_db"] / 10))
        G = g(xi, gamma)
    if omlsa:
        xi_f = xi3 if p["hrnr"] else xi
        q = C.absence_prob(xi_f, **{k: v for k, v in omlsa.items() if k not in ("gmin_db", "mode")})
        v = np.minimum(xi_f * gamma / (1 + xi_f), 50)
        pp = 1.0 / (1 + q / (1 - q) * (1 + xi_f) * np.exp(-v))
        gm = 10 ** (omlsa.get("gmin_db", p["gmin_db"]) / 20)
        if omlsa.get("mode", "omlsa") == "omlsa":
            G = G ** pp * gm ** (1 - pp)
        else:                      # soft: G * p
            G = G * pp
    gmin = np.full(G.shape[0], 10 ** (p["gmin_db"] / 20))
    if p["low_hz"]:
        gmin[np.fft.rfftfreq(n_fft, 1 / 16000) < p["low_hz"]] = 10 ** (p["gmin_low_db"] / 20)
    G = np.maximum(G, gmin[:, None])
    if esch:
        # Esch & Vary (2009): smooth the gain across frequency only in frames with low a-posteriori SNR
        from scipy.ndimage import uniform_filter1d
        snr_fr = (G ** 2 * P).sum(0) / N.sum(0)
        L = np.where(snr_fr < esch.get("thr", 0.4), esch.get("width", 7), 1).astype(int)
        Gs = G.copy()
        for w in np.unique(L):
            if w > 1:
                cols = L == w
                Gs[:, cols] = uniform_filter1d(G[:, cols], w, axis=0, mode="nearest")
        G = np.maximum(Gs, gmin[:, None])
    if comb:
        f0, st = C.harmonic_f0(G * np.sqrt(P), 16000, n_fft, fmax=comb.get("fmax_det", 3000.0))
        W = C.comb_weight(f0, st, G.shape[0], 16000, n_fft, fmax=comb.get("fmax", 3500.0),
                          depth=comb.get("depth", 0.5), s0=comb.get("s0", 0.5), s1=comb.get("s1", 1.5),
                          width=comb.get("width", 1.0))
        G = np.maximum(G * W, gmin[:, None])
    if F is not None and p["floor_keep"]:
        G = np.maximum(G, np.sqrt(np.minimum(1.0, F[:, None] / P)))
    if period and rep_em:
        from .repet import repeat_em, repeat_em2, repeat_fraction
        Sp = (G * np.abs(Z)) ** 2 if rep_sp is None else rep_sp
        sf = lambda v: C.stft(v, n_fft, hop)
        if rep_em.get("v2"):
            cfg = {k: v for k, v in rep_em.items() if k not in ("v2", "u_max", "u_fixed")}
            u = rep_em.get("u_fixed")
            if u is None:
                u = repeat_fraction(x, int(period), Sp, N, sf, hop, n_fft)
            if u > rep_em.get("u_max", 0.6):
                return C.istft(G * Z, meta)
            Shat, cov = repeat_em2(x, int(period), Sp, N, n_fft, hop, sf, u=u, **cfg)
        else:
            Shat, cov = repeat_em(x, int(period), Sp, N, n_fft, hop, sf, **rep_em)
        mag = np.minimum(np.abs(Shat), np.abs(Z))
        Shat = mag * np.exp(1j * np.angle(Shat))
        if F is not None and p["floor_keep"]:
            fl = np.sqrt(np.minimum(1.0, F[:, None] / P)) * np.abs(Z)
            Shat = np.where(np.abs(Shat) < fl, fl * np.exp(1j * np.angle(Z)), Shat)
        out = np.where(cov[None, :], Shat, G * Z)
        return C.istft(out, meta)
    if period and rep_mmse:
        from .repet import repeat_mmse
        Sp = (G * np.abs(Z)) ** 2 if rep_sp is None else rep_sp
        lamN = N
        Shat, cov = repeat_mmse(x, int(period), Sp, lamN, n_fft, hop, lambda v: C.stft(v, n_fft, hop),
                                eps=rep_eps)
        for _ in range(rep_iter):          # re-estimate speech variances from the MMSE output (EM-like)
            Sp2 = np.abs(Shat) ** 2
            if rep_iter_smooth:
                from scipy.ndimage import uniform_filter
                Sp2 = uniform_filter(Sp2, (1, 3), mode="nearest")
            Sp = np.where(cov[None, :], Sp2, Sp)
            Shat, cov = repeat_mmse(x, int(period), Sp, lamN, n_fft, hop, lambda v: C.stft(v, n_fft, hop),
                                    eps=rep_eps)
        mag = np.minimum(np.abs(Shat), np.abs(Z))
        if rep_post:
            # optional magnitude post-filter: never exceed the pipeline estimate by more than rep_post
            mag = np.minimum(mag, rep_post * G * np.abs(Z))
        Shat = mag * np.exp(1j * np.angle(Shat))
        if F is not None and p["floor_keep"]:
            fl = np.sqrt(np.minimum(1.0, F[:, None] / P)) * np.abs(Z)
            Shat = np.where(np.abs(Shat) < fl, fl * np.exp(1j * np.angle(Z)), Shat)
        out = np.where(cov[None, :], rep_mmse_w * Shat + (1 - rep_mmse_w) * G * Z, G * Z)
        return C.istft(out, meta)
    if period and rep_complex:
        # exact cancellation of a repeating background: S_hat = Y - Y_c for the copy with least energy
        others = np.stack([np.where(np.isnan(z), np.inf + 0j, z) for z in Zs[1:]])
        mags = np.abs(others)
        c = np.argmin(mags, axis=0)
        Yc = np.take_along_axis(others, c[None], 0)[0]
        Pc = np.abs(Yc) ** 2
        ok = np.isfinite(Pc) & (Pc <= rep_complex * P)
        Ssub = Z - np.where(ok, Yc, 0)
        # keep the magnitude within the pipeline estimate's neighbourhood
        mag = np.minimum(np.abs(Ssub), np.abs(Z))
        Ssub = mag * np.exp(1j * np.angle(Ssub))
        out = np.where(ok, rep_w * Ssub + (1 - rep_w) * G * Z, G * Z)
        return C.istft(out, meta)
    if consist:
        # STFT-consistency enforcement (Griffin-Lim style): keep the estimated magnitude, iterate the phase
        mag = G * np.abs(Z)
        ph = np.angle(Z)
        upd = None
        if consist.get("snr_db") is not None:      # only re-estimate the phase where the local SNR is low
            upd = (10 * np.log10(np.maximum(xi, 1e-10)) < consist["snr_db"])
        for _ in range(consist.get("iters", 5)):
            y_ = C.istft(mag * np.exp(1j * ph), meta)
            X_ = C.stft(y_, n_fft, hop)[0]
            ph_new = np.angle(X_)
            ph = ph_new if upd is None else np.where(upd, ph_new, ph)
        return C.istft(mag * np.exp(1j * ph), meta)
    return C.istft(G * Z, meta)


# Parameters selected on the training split (random search, tools/rsearch.py) and checked on validation
BEST = dict(a=0.982, gmin_db=-36.5, xi_min_db=-30.1, rho=0.5, tsnr_mix=0.62, sfrac=0.122, spp_alpha=0.835,
            xi_h1=12.1, nscale=1.05, floor_cap_db=-47.2, hop=256, n_fft=1280, floor=True, nmode="geo")
# Multi-resolution ensemble members (window lengths 40 / 80 / 128 ms)
ENSEMBLE = [dict(n_fft=1280, hop=256), dict(n_fft=640, hop=128), dict(n_fft=2048, hop=512)]
ENS_WEIGHTS = None  # equal weights
REP_THR = 6.0       # repeating-background detection threshold (band-wise correlation score)
REP_MIX = 0.6       # weight of the repetition-based noise PSD
REP_SUBMULT = 0.95  # prefer L/2, L/3 when they explain the repetition (almost) as well
# EM for an exactly repeating background (12 iterations, 3x3 smoothing, Wiener post-stage)
REP_EM = dict(iters=12, wf=3, wt=3, noise_wt=9, noise_wf=5, post="wiener")
# per-bin long-term noise spectrum from an SPP-weighted average (q = weight exponent)
LT_CORR = dict(q=3.0)
# phase-coherent cancellation of stable tones (complex demodulation, envelope bandwidth 0.15 Hz)
TONAL = dict(stab=0.3, prom_db=8.0, bw_hz=0.15)
# long-term speech-to-noise ratio per bin used as a prior on the a priori SNR
LT_PRIOR = dict(c_db=-2.5, fsm=5)


def enhance_signal(x: np.ndarray, fs: int = 16000, **kw) -> np.ndarray:
    """Robust entry point: any length / sampling rate; returns an array with the same shape."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x):
        return x.copy()
    p = dict(BEST, **kw)
    if TONAL is not None:
        from .tonal import cancel_tones
        x = cancel_tones(x, fs, fmax=min(7900.0, 0.49 * fs), **TONAL)[0]
    if fs != 16000 or x.size < 4 * 2048:
        # non-standard input: single resolution, window ~80 ms (shorter for very short files)
        n = int(round(0.08 * fs))
        while x.size < 4 * n and n > 64:
            n //= 2
        p.update(n_fft=n, hop=max(1, n // 5))
        if fs != 16000:
            p["floor"] = False
        y = enhance(x, lt_corr=LT_CORR, lt_prior=LT_PRIOR, **p)
    else:
        y = enhance_auto(x, rep_thr=REP_THR, rep_mix_auto=REP_MIX, ens=ENSEMBLE, det="both",
                         submult=REP_SUBMULT, rep_em=REP_EM, lt_corr=LT_CORR, lt_prior=LT_PRIOR, **p)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y[: x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))


def enhance_ensemble(x: np.ndarray, members=None, weights=None, mode="wave", **kw) -> np.ndarray:
    """Average of several enhanced versions (different resolutions / noise estimators).

    Speech components add coherently while uncorrelated processing artefacts partially cancel."""
    members = members or [dict(n_fft=1024, hop=256), dict(n_fft=512, hop=128)]
    if weights is None:
        weights = ENS_WEIGHTS if (ENS_WEIGHTS is not None and len(ENS_WEIGHTS) == len(members)) else None
    weights = np.ones(len(members)) / len(members) if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum()
    ys = [enhance(x, **dict(kw, **m)) for m in members]
    return sum(w * y for w, y in zip(weights, ys))


def enhance_iter(x, passes=2, second=None, **kw):
    y = enhance(x, **kw)
    for _ in range(passes - 1):
        y = enhance(y, **dict(kw, **(second or {})))
    return y


def detect_repetition(x, det="both", rep_thr=6.0, submult=0.95, rc_thr=0.0, rc_rescue=0.0, return_info=False, **kw):
    """Period of an exactly repeating background.

    Candidates: band-wise long-lag correlation peaks of the noisy input and of the residual x - s_hat
    of a first enhancement pass (speech removed -> clearer peak). A candidate is accepted when its
    band score exceeds rep_thr, or when the residual itself correlates strongly at that lag
    (rc_rescue; e.g. tonal backgrounds whose correlation peaks are not prominent but exact).
    Sub-multiples L/2, L/3 are preferred when they correlate (almost) as well."""
    from .repet import detect_period_bands, long_lag_corr
    cands = []
    if det in ("noisy", "both"):
        L, pk, score, band = detect_period_bands(x)
        if L is not None:
            cands.append((L, score))
    r = None
    if det in ("residual", "both"):
        r = x - enhance(x, **kw)
        L, pk, score, band = detect_period_bands(r)
        if L is not None:
            cands.append((L, score))
    if not cands:
        return (None, 0.0, 0.0) if return_info else None
    if r is None:
        r = x
    c = long_lag_corr(r - r.mean(), 1000, 3000)
    best = None
    def seg_corr(a_lag, span):
        """correlation of r[t] and r[t + a_lag] for t in [0, span)"""
        u = r[:span] - r[:span].mean(); v = r[a_lag:a_lag + span] - r[a_lag:a_lag + span].mean()
        return float(np.dot(u, v) / np.sqrt(np.dot(u, u) * np.dot(v, v) + 1e-30))
    n = len(r)
    # cross-confirmation: if one candidate is (about) an integer multiple of another one, the smaller
    # lag is the period (the larger is just its multiple); keep the better score of the two.
    if len(cands) == 2:
        (L1, s1), (L2, s2) = sorted(cands)
        for k in (2, 3):
            if abs(L2 - k * L1) <= 4 * k:
                cands = [(L1, max(s1, s2))]
                break
    for L, score in cands:
        if submult:
            span = n - L                    # region where the lag-L comparison is available
            ref = seg_corr(L, span) if span > 1000 else c[L]
            for k in (3, 2):
                Lk = int(round(L / k))
                if Lk < 4000:
                    continue
                lo, hi = Lk - 3, Lk + 4
                j2 = int(np.argmax(c[lo:hi])) + lo
                # same time span as the full-period comparison (fair test of exactness)
                if span > 1000 and seg_corr(j2, span) >= submult * ref:
                    L = j2
                    break
        ok = (score > rep_thr and c[L] >= rc_thr) or (rc_rescue and c[L] >= rc_rescue)
        if ok and (best is None or (score, c[L]) > (best[1], best[2])):
            best = (L, score, float(c[L]))
    if best is None:
        return (None, max(s for _, s in cands), 0.0) if return_info else None
    return best if return_info else best[0]


def enhance_auto(x, rep_thr=8.0, rep_mix_auto=0.6, ens=None, v2_thr=0.0, det="noisy", rc_thr=0.0, rc_rescue=0.0, submult=0.0, **kw):
    """Detect a repeating background in the input itself; use it for the noise PSD when found."""
    from .repet import detect_period_bands, detect_period_v2
    if det == "noisy":
        L, pk, score, band = detect_period_bands(x)
        period = L if (L is not None and score > rep_thr) else None
    else:
        base = {k: v for k, v in kw.items() if k not in ("rep_em", "rep_mmse")}
        period = detect_repetition(x, det, rep_thr, submult=submult, rc_thr=rc_thr, rc_rescue=rc_rescue, **base)
    if period is None and v2_thr:
        L2, ratio = detect_period_v2(x)
        if L2 is not None and ratio < v2_thr:
            period = L2
    if ens:
        return enhance_ensemble(x, members=ens, period=period, rep_mix=rep_mix_auto, **kw)
    return enhance(x, period=period, rep_mix=rep_mix_auto, **kw)


def enhance_tonal(x, tonal=None, tonal2=None, inner="enhance", **kw):
    """Cancel persistent tones phase-coherently first, then run the regular enhancement."""
    from .tonal import cancel_tones
    y, freqs = cancel_tones(x, **(tonal or {}))
    if tonal2:
        y, f2 = cancel_tones(y, **tonal2)
    return globals()[inner](y, **kw)


def enhance_mfmvdr(x, n_fft_mf=256, hop_mf=32, taps=5, alpha=0.9, mix=1.0, post="logmmse", gmin_mf=-30.0,
                   **kw):
    """MFMVDR + post-filter, using the pipeline's noise estimator on the MFMVDR STFT grid."""
    from .mfmvdr import mfmvdr
    p = dict(DEFAULTS, **kw); p.update(n_fft=n_fft_mf, hop=hop_mf)
    def N_fn(P):
        N, N0 = estimate_noise(P, p)
        return N * p["nscale"]
    return mfmvdr(x, N_fn, n_fft_mf, hop_mf, taps, alpha, post=post, gmin_db=gmin_mf, mix=mix)


# ---------------------------------------------------------------------------
# Reference-noise version (uses the noise bank built from the public training data)
# ---------------------------------------------------------------------------
REF = dict(full_thr=0.98, post=None)


def enhance_signal_ref(x: np.ndarray, fs: int = 16000, bank=None, full_thr=None, post=None, **kw) -> np.ndarray:
    """Cancel matched reference noise first; the regular pipeline handles what the bank does not cover."""
    from .refnoise import cancel_reference, load_bank
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.any(x) or fs != 16000 or x.size < 8000:
        return enhance_signal(x, fs, **kw)
    full_thr = REF["full_thr"] if full_thr is None else full_thr
    post = REF["post"] if post is None else post
    bank = bank if bank is not None else load_bank()
    y1, cov, matches = cancel_reference(x, bank)
    if not matches:
        return enhance_signal(x, fs, **kw)
    if REF.get("resid_post"):
        y1 = residual_postfilter(x, y1, cov, matches, **REF["resid_post"])
    worst = max(m[6] for m in matches)
    if cov.mean() >= full_thr and worst <= REF.get("clean_depth", 1.0):
        if post == "tonal":            # remove stable tones the reference did not cancel exactly
            from .tonal import cancel_tones
            return cancel_tones(y1, fs, **TONAL)[0]
        if post:                       # optional light clean-up of the residual
            return enhance(y1, **dict(BEST, **post))
        return y1
    mode = REF.get("partial_mode", "splice")
    if mode == "enh_resid":            # the regular pipeline on the residual (noise largely removed)
        return enhance_signal(y1, fs, **kw)
    base = enhance_signal(x, fs, **kw)
    return cov * y1 + (1.0 - cov) * base


def residual_postfilter(x, y1, cov, matches, kappa=1.0, n_fft=1024, hop=256, gmin_db=-25.0, a=0.98, smooth=3):
    """After reference subtraction the remaining noise is roughly depth * |subtracted noise|^2.
    A Log-MMSE gain with that residual-noise PSD cleans imperfect matches and is ~1 for exact ones."""
    from scipy.ndimage import uniform_filter
    nhat = x - y1                                       # subtracted noise waveform
    depth = max(m[6] for m in matches)
    Zy, meta = C.stft(y1, n_fft, hop)
    Zn = C.stft(nhat, n_fft, hop)[0]
    Py = np.abs(Zy) ** 2 + 1e-20
    Nr = kappa * depth * uniform_filter(np.abs(Zn) ** 2, (smooth, smooth), mode="nearest") + 1e-20
    xi, gamma, G = C.dd_prior_snr(Py, Nr, a, -30.0, C.gain_logmmse)
    G = np.maximum(G, 10 ** (gmin_db / 20))
    out = C.istft(G * Zy, meta)
    return cov * out + (1.0 - cov) * y1

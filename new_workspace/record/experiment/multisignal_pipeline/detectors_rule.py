from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind, norm
from scipy.signal import savgol_filter

from data import _clean
from protocol import (
    EXPECTED_OVULATION_FRACTION,
    HMM_TRANSITION_STAY_PROB,
    MAX_RIGHT_MARGIN_DAYS,
    MIN_CYCLE_LEN_FOR_DETECTION,
    MIN_DETECTION_DAY,
    MIN_EXPECTED_OVULATION_DAY,
    MULTI_SIGNAL_CUSUM_THRESHOLD,
    POSITION_PRIOR_WIDTH,
    SAVGOL_FALLBACK_SIGMA,
    SINGLE_SIGNAL_CUSUM_THRESHOLD,
)


# =====================================================================
# A. RULE-BASED METHODS (split out)
# =====================================================================


def detect_ttest_optimal(
    cs,
    sig_key,
    sigma=2.0,
    invert=False,
    pw=POSITION_PRIOR_WIDTH,
    frac=EXPECTED_OVULATION_FRACTION,
):
    """T-test with Gaussian position prior."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        best_ws, best_sp, best_stat = -np.inf, None, 0
        for sp in range(MIN_DETECTION_DAY, n - MAX_RIGHT_MARGIN_DAYS):
            diff = np.mean(t[sp:]) - np.mean(t[:sp])
            if diff <= 0:
                continue
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
            except:
                continue
            if np.isnan(stat):
                continue
            pp = np.exp(-0.5 * ((sp - exp) / pw) ** 2)
            if stat * pp > best_ws:
                best_ws = stat * pp
                best_sp = sp
                best_stat = stat
        if best_sp is not None:
            det[sgk] = best_sp
            conf[sgk] = min(1.0, max(0, best_stat / 5))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_ttest_prefix_daily(
    cs,
    sig_key,
    sigma=2.0,
    invert=False,
    pw=POSITION_PRIOR_WIDTH,
    frac=EXPECTED_OVULATION_FRACTION,
):
    """
    Prefix-based ongoing-cycle T-test.

    For cycle day d, only prefix days [0, d) are visible.
    The detector may return:
      - None: no ovulation estimate available yet
      - ov_est <= d-1: a retrospective ovulation estimate within the visible prefix
    """
    det_by_day, conf_by_day = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        daily_det = [None] * n
        daily_conf = [0.0] * n
        if raw is None or np.isnan(raw).all():
            det_by_day[sgk] = daily_det
            conf_by_day[sgk] = daily_conf
            continue

        for prefix_len in range(1, n + 1):
            if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
                continue
            t = _clean(raw[:prefix_len], sigma=sigma)
            if invert:
                t = -t
            exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
            best_ws, best_sp, best_stat = -np.inf, None, 0.0
            for sp in range(MIN_DETECTION_DAY, prefix_len - MAX_RIGHT_MARGIN_DAYS):
                diff = np.mean(t[sp:prefix_len]) - np.mean(t[:sp])
                if diff <= 0:
                    continue
                try:
                    stat, _ = ttest_ind(t[sp:prefix_len], t[:sp], alternative="greater")
                except:
                    continue
                if np.isnan(stat):
                    continue
                pp = np.exp(-0.5 * ((sp - exp) / pw) ** 2)
                if stat * pp > best_ws:
                    best_ws = stat * pp
                    best_sp = sp
                    best_stat = stat
            if best_sp is not None:
                daily_det[prefix_len - 1] = int(best_sp)
                daily_conf[prefix_len - 1] = min(1.0, max(0, best_stat / 5))

        det_by_day[sgk] = daily_det
        conf_by_day[sgk] = daily_conf
    return det_by_day, conf_by_day


def detect_cusum(
    cs,
    sig_key,
    sigma=2.0,
    invert=False,
    threshold=SINGLE_SIGNAL_CUSUM_THRESHOLD,
    frac=EXPECTED_OVULATION_FRACTION,
):
    """CUSUM changepoint detection on single signal."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        mu = np.mean(t)
        s_pos = np.zeros(n)
        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i - 1] + (t[i] - mu) - threshold * np.std(t))
        alarm_pts = np.where(s_pos > 0)[0]
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        if len(alarm_pts) > 0:
            dists = np.abs(alarm_pts - exp)
            best = alarm_pts[np.argmin(dists)]
            det[sgk] = int(best)
            conf[sgk] = min(1.0, s_pos[best] / (3 * np.std(t) + 1e-8))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_cusum_prefix_daily(
    cs,
    sig_key,
    sigma=2.0,
    invert=False,
    threshold=SINGLE_SIGNAL_CUSUM_THRESHOLD,
    frac=EXPECTED_OVULATION_FRACTION,
):
    """Prefix-based ongoing-cycle CUSUM on a single signal."""
    det_by_day, conf_by_day = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        daily_det = [None] * n
        daily_conf = [0.0] * n
        if raw is None or np.isnan(raw).all():
            det_by_day[sgk] = daily_det
            conf_by_day[sgk] = daily_conf
            continue

        for prefix_len in range(1, n + 1):
            if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
                continue
            t = _clean(raw[:prefix_len], sigma=sigma)
            if invert:
                t = -t
            mu = np.mean(t)
            std_t = np.std(t)
            s_pos = np.zeros(prefix_len)
            for i in range(1, prefix_len):
                s_pos[i] = max(0, s_pos[i - 1] + (t[i] - mu) - threshold * std_t)
            alarm_pts = np.where(s_pos > 0)[0]
            if len(alarm_pts) == 0:
                continue
            exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
            best = alarm_pts[np.argmin(np.abs(alarm_pts - exp))]
            daily_det[prefix_len - 1] = int(best)
            daily_conf[prefix_len - 1] = min(1.0, s_pos[best] / (3 * std_t + 1e-8))

        det_by_day[sgk] = daily_det
        conf_by_day[sgk] = daily_conf
    return det_by_day, conf_by_day


def detect_bayesian_biphasic(
    cs, sig_key, sigma=2.0, invert=False, frac=EXPECTED_OVULATION_FRACTION
):
    """Bayesian biphasic step-function fitting (SSE minimization + position prior)."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        best_score, best_sp = np.inf, n // 2
        for sp in range(MIN_DETECTION_DAY, n - MAX_RIGHT_MARGIN_DAYS):
            m1, m2 = np.mean(t[:sp]), np.mean(t[sp:])
            if m2 <= m1:
                continue
            sse = np.sum((t[:sp] - m1) ** 2) + np.sum((t[sp:] - m2) ** 2)
            pos_pen = 0.5 * ((sp - exp) / POSITION_PRIOR_WIDTH) ** 2
            score = sse + pos_pen
            if score < best_score:
                best_score = score
                best_sp = sp
        det[sgk] = best_sp
        m1, m2 = np.mean(t[:best_sp]), np.mean(t[best_sp:])
        shift = m2 - m1
        conf[sgk] = min(1.0, max(0, shift / (np.std(t) + 1e-8)))
    return det, conf


def detect_hmm_2state(
    cs, sig_key, sigma=2.0, invert=False, frac=EXPECTED_OVULATION_FRACTION
):
    """2-state Gaussian HMM: state 0=follicular(low), state 1=luteal(high)."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        half = n // 2
        mu0, mu1 = np.mean(t[:half]), np.mean(t[half:])
        if mu1 < mu0:
            mu0, mu1 = mu1, mu0
        s0, s1 = max(np.std(t[:half]), 0.01), max(np.std(t[half:]), 0.01)
        trans_stay = HMM_TRANSITION_STAY_PROB

        for _ in range(20):
            log_e0 = norm.logpdf(t, mu0, s0)
            log_e1 = norm.logpdf(t, mu1, s1)
            alpha = np.zeros((n, 2))
            alpha[0, 0] = 0.8
            alpha[0, 1] = 0.2
            for i in range(1, n):
                a00 = alpha[i - 1, 0] * trans_stay
                a10 = alpha[i - 1, 1] * (1 - trans_stay)
                a01 = alpha[i - 1, 0] * (1 - trans_stay)
                a11 = alpha[i - 1, 1] * trans_stay
                alpha[i, 0] = (a00 + a10) * np.exp(log_e0[i])
                alpha[i, 1] = (a01 + a11) * np.exp(log_e1[i])
                s = alpha[i].sum()
                if s > 0:
                    alpha[i] /= s

            states = np.argmax(alpha, axis=1)
            g0 = t[states == 0]
            g1 = t[states == 1]
            if len(g0) > 2 and len(g1) > 2:
                mu0, mu1 = np.mean(g0), np.mean(g1)
                s0, s1 = max(np.std(g0), 0.01), max(np.std(g1), 0.01)
                if mu1 < mu0:
                    mu0, mu1 = mu1, mu0
                    s0, s1 = s1, s0

        transition = None
        for i in range(1, n):
            if states[i - 1] == 0 and states[i] == 1:
                transition = i
                break
        if transition is not None:
            det[sgk] = transition
            conf[sgk] = min(1.0, abs(mu1 - mu0) / (s0 + s1 + 1e-8))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_savgol_gradient(
    cs, sig_key, sigma=0, invert=False, frac=EXPECTED_OVULATION_FRACTION
):
    """Savitzky-Golay smoothing + maximum gradient detection."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=0)
        if invert:
            t = -t
        wl = min(11, n - 1 if n % 2 == 0 else n - 2)
        if wl < 5:
            wl = 5
        if wl % 2 == 0:
            wl += 1
        try:
            ts = savgol_filter(t, window_length=wl, polyorder=3)
        except:
            ts = gaussian_filter1d(t, sigma=SAVGOL_FALLBACK_SIGMA)
        grad = np.gradient(ts)
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        search_lo = max(3, int(exp - 8))
        search_hi = min(n - 2, int(exp + 8))
        if search_lo >= search_hi:
            search_lo, search_hi = 3, n - 2
        region = grad[search_lo : search_hi + 1]
        if len(region) == 0:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        peak_idx = search_lo + np.argmax(region)
        det[sgk] = int(peak_idx)
        conf[sgk] = min(1.0, max(0, grad[peak_idx] / (np.std(grad) + 1e-8)))
    return det, conf


# =====================================================================
# Multi-signal fused rules (split out)
# =====================================================================


def detect_multi_signal_fused_ttest(
    cs,
    sigs,
    sigma=2.0,
    inverts=None,
    frac=EXPECTED_OVULATION_FRACTION,
    pw=POSITION_PRIOR_WIDTH,
):
    """Fused multi-signal t-test: z-score normalize each signal, average, then t-test."""
    det, conf = {}, {}
    if inverts is None:
        inverts = [False] * len(sigs)
    for sgk, data in cs.items():
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        valid_sigs = []
        for sk, inv in zip(sigs, inverts):
            raw = data.get(sk)
            if raw is not None and not np.isnan(raw).all():
                t = _clean(raw, sigma=sigma)
                if inv:
                    t = -t
                std = np.std(t)
                if std > 1e-8:
                    t = (t - np.mean(t)) / std
                valid_sigs.append(t)
        if len(valid_sigs) < 1 or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        fused = np.mean(valid_sigs, axis=0)
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        best_ws, best_sp, best_stat = -np.inf, None, 0
        for sp in range(MIN_DETECTION_DAY, n - MAX_RIGHT_MARGIN_DAYS):
            diff = np.mean(fused[sp:]) - np.mean(fused[:sp])
            if diff <= 0:
                continue
            try:
                stat, _ = ttest_ind(
                    fused[sp:], fused[:sp], alternative="greater"
                )
            except:
                continue
            if np.isnan(stat):
                continue
            pp = np.exp(-0.5 * ((sp - exp) / pw) ** 2)
            if stat * pp > best_ws:
                best_ws = stat * pp
                best_sp = sp
                best_stat = stat
        if best_sp is not None:
            det[sgk] = best_sp
            conf[sgk] = min(1.0, max(0, best_stat / 5))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_multi_signal_fused_ttest_prefix_daily(
    cs,
    sigs,
    sigma=2.0,
    inverts=None,
    frac=EXPECTED_OVULATION_FRACTION,
    pw=POSITION_PRIOR_WIDTH,
):
    """Prefix-based ongoing-cycle fused multi-signal t-test."""
    det_by_day, conf_by_day = {}, {}
    if inverts is None:
        inverts = [False] * len(sigs)
    for sgk, data in cs.items():
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        daily_det = [None] * n
        daily_conf = [0.0] * n
        for prefix_len in range(1, n + 1):
            if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
                continue
            valid_sigs = []
            for sk, inv in zip(sigs, inverts):
                raw = data.get(sk)
                if raw is None or np.isnan(raw).all():
                    continue
                t = _clean(raw[:prefix_len], sigma=sigma)
                if inv:
                    t = -t
                std = np.std(t)
                if std > 1e-8:
                    t = (t - np.mean(t)) / std
                valid_sigs.append(t)
            if not valid_sigs:
                continue
            fused = np.mean(valid_sigs, axis=0)
            exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
            best_ws, best_sp, best_stat = -np.inf, None, 0.0
            for sp in range(MIN_DETECTION_DAY, prefix_len - MAX_RIGHT_MARGIN_DAYS):
                diff = np.mean(fused[sp:prefix_len]) - np.mean(fused[:sp])
                if diff <= 0:
                    continue
                try:
                    stat, _ = ttest_ind(
                        fused[sp:prefix_len],
                        fused[:sp],
                        alternative="greater",
                    )
                except:
                    continue
                if np.isnan(stat):
                    continue
                pp = np.exp(-0.5 * ((sp - exp) / pw) ** 2)
                if stat * pp > best_ws:
                    best_ws = stat * pp
                    best_sp = sp
                    best_stat = stat
            if best_sp is not None:
                daily_det[prefix_len - 1] = int(best_sp)
                daily_conf[prefix_len - 1] = min(1.0, max(0, best_stat / 5))

        det_by_day[sgk] = daily_det
        conf_by_day[sgk] = daily_conf
    return det_by_day, conf_by_day


def detect_multi_hmm(
    cs, sigs, sigma=2.0, inverts=None, frac=EXPECTED_OVULATION_FRACTION
):
    """Multi-signal 2-state HMM: joint emission from multiple normalized signals."""
    det, conf = {}, {}
    if inverts is None:
        inverts = [False] * len(sigs)
    for sgk, data in cs.items():
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        valid_ts = []
        for sk, inv in zip(sigs, inverts):
            raw = data.get(sk)
            if raw is not None and not np.isnan(raw).all():
                t = _clean(raw, sigma=sigma)
                if inv:
                    t = -t
                valid_ts.append(t)
        if len(valid_ts) < 2 or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        X = np.column_stack(valid_ts)
        d = X.shape[1]
        half = n // 2
        mu0 = np.mean(X[:half], axis=0)
        mu1 = np.mean(X[half:], axis=0)
        s0 = np.maximum(np.std(X[:half], axis=0), 0.01)
        s1 = np.maximum(np.std(X[half:], axis=0), 0.01)
        trans_stay = HMM_TRANSITION_STAY_PROB

        for _ in range(15):
            log_e0 = np.sum(
                [norm.logpdf(X[:, j], mu0[j], s0[j]) for j in range(d)], axis=0
            )
            log_e1 = np.sum(
                [norm.logpdf(X[:, j], mu1[j], s1[j]) for j in range(d)], axis=0
            )
            alpha = np.zeros((n, 2))
            alpha[0] = [0.8, 0.2]
            for i in range(1, n):
                a0 = alpha[i - 1, 0] * trans_stay + alpha[i - 1, 1] * (1 - trans_stay)
                a1 = alpha[i - 1, 0] * (1 - trans_stay) + alpha[i - 1, 1] * trans_stay
                alpha[i, 0] = a0 * np.exp(log_e0[i] - max(log_e0[i], log_e1[i]))
                alpha[i, 1] = a1 * np.exp(log_e1[i] - max(log_e0[i], log_e1[i]))
                s = alpha[i].sum()
                if s > 0:
                    alpha[i] /= s
            states = np.argmax(alpha, axis=1)
            g0 = X[states == 0]
            g1 = X[states == 1]
            if len(g0) > 2 and len(g1) > 2:
                mu0, mu1 = np.mean(g0, axis=0), np.mean(g1, axis=0)
                s0 = np.maximum(np.std(g0, axis=0), 0.01)
                s1 = np.maximum(np.std(g1, axis=0), 0.01)

        transition = None
        for i in range(1, n):
            if states[i - 1] == 0 and states[i] == 1:
                transition = i
                break
        if transition is not None:
            det[sgk] = transition
            shift = np.linalg.norm(mu1 - mu0) / (np.linalg.norm(s0 + s1) + 1e-8)
            conf[sgk] = min(1.0, shift)
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_multi_cusum_fused(
    cs,
    sigs,
    sigma=2.0,
    inverts=None,
    frac=EXPECTED_OVULATION_FRACTION,
    threshold=MULTI_SIGNAL_CUSUM_THRESHOLD,
):
    """Multi-signal fused CUSUM: z-score each signal, average, run CUSUM."""
    det, conf = {}, {}
    if inverts is None:
        inverts = [False] * len(sigs)
    for sgk, data in cs.items():
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        valid_ts = []
        for sk, inv in zip(sigs, inverts):
            raw = data.get(sk)
            if raw is not None and not np.isnan(raw).all():
                t = _clean(raw, sigma=sigma)
                if inv:
                    t = -t
                std = np.std(t)
                if std > 1e-8:
                    t = (t - np.mean(t)) / std
                valid_ts.append(t)
        if not valid_ts or n < MIN_CYCLE_LEN_FOR_DETECTION:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        fused = np.mean(valid_ts, axis=0)
        mu = np.mean(fused)
        std_f = max(np.std(fused), 1e-8)
        s_pos = np.zeros(n)
        for i in range(1, n):
            s_pos[i] = max(
                0, s_pos[i - 1] + (fused[i] - mu) - threshold * std_f
            )
        alarm_pts = np.where(s_pos > 0)[0]
        exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
        if len(alarm_pts) > 0:
            dists = np.abs(alarm_pts - exp)
            best = alarm_pts[np.argmin(dists)]
            det[sgk] = int(best)
            conf[sgk] = min(1.0, s_pos[best] / (3 * std_f))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_multi_cusum_fused_prefix_daily(
    cs,
    sigs,
    sigma=2.0,
    inverts=None,
    frac=EXPECTED_OVULATION_FRACTION,
    threshold=MULTI_SIGNAL_CUSUM_THRESHOLD,
):
    """Prefix-based ongoing-cycle fused multi-signal CUSUM."""
    det_by_day, conf_by_day = {}, {}
    if inverts is None:
        inverts = [False] * len(sigs)
    for sgk, data in cs.items():
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        daily_det = [None] * n
        daily_conf = [0.0] * n
        for prefix_len in range(1, n + 1):
            if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
                continue
            valid_ts = []
            for sk, inv in zip(sigs, inverts):
                raw = data.get(sk)
                if raw is None or np.isnan(raw).all():
                    continue
                t = _clean(raw[:prefix_len], sigma=sigma)
                if inv:
                    t = -t
                std = np.std(t)
                if std > 1e-8:
                    t = (t - np.mean(t)) / std
                valid_ts.append(t)
            if not valid_ts:
                continue
            fused = np.mean(valid_ts, axis=0)
            mu = np.mean(fused)
            std_f = max(np.std(fused), 1e-8)
            s_pos = np.zeros(prefix_len)
            for i in range(1, prefix_len):
                s_pos[i] = max(0, s_pos[i - 1] + (fused[i] - mu) - threshold * std_f)
            alarm_pts = np.where(s_pos > 0)[0]
            if len(alarm_pts) == 0:
                continue
            exp = max(MIN_EXPECTED_OVULATION_DAY, hcl * frac)
            best = alarm_pts[np.argmin(np.abs(alarm_pts - exp))]
            daily_det[prefix_len - 1] = int(best)
            daily_conf[prefix_len - 1] = min(1.0, s_pos[best] / (3 * std_f))

        det_by_day[sgk] = daily_det
        conf_by_day[sgk] = daily_conf
    return det_by_day, conf_by_day

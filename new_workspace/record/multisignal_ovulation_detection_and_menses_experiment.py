"""
Multi-Signal Ovulation Detection and Menstrual Prediction Experiment.

Uses temperature, HR, HRV from wearables to detect ovulation (rules, ML, 1D-CNN, stacking),
then predicts next menses via ovulation + luteal countdown. Compares to Oracle (LH truth) and calendar.

Data: new_workspace by default (processed_dataset/cycle_cleaned_ov.csv, processed_dataset/signals/*.csv).

Usage:
  cd new_workspace
  python record/multisignal_ovulation_detection_and_menses_experiment.py

  Or from repo root:
  python new_workspace/record/multisignal_ovulation_detection_and_menses_experiment.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind, norm
from scipy.signal import savgol_filter
from collections import defaultdict
warnings.filterwarnings("ignore")

# Use new_workspace data
NEW_WS = Path(__file__).resolve().parent.parent
MAIN_WS = NEW_WS.parent / "main_workspace"
PROCESSED = NEW_WS / "processed_dataset"
SIGNALS_DIR = PROCESSED / "signals"
CYCLE_OV_CSV = PROCESSED / "cycle_cleaned_ov.csv"

sys.path.insert(0, str(MAIN_WS))
import model.config as _config
_config.CYCLE_CSV = str(CYCLE_OV_CSV)
_config.WORKSPACE = str(NEW_WS)

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76
results_db = {}


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def _pr(tag, ae, prefix="  "):
    ae = np.array(ae, dtype=float)
    if len(ae) == 0:
        return {}
    r = {"n": len(ae), "mae": float(np.mean(ae)),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"{prefix}[{tag}] n={r['n']} MAE={r['mae']:.2f}"
          f" ±1d={r['acc_1d']:.1%} ±2d={r['acc_2d']:.1%}"
          f" ±3d={r['acc_3d']:.1%} ±5d={r['acc_5d']:.1%}")
    return r


def load_all_signals():
    """Load and aggregate ALL available signals from new_workspace processed_dataset/signals."""
    print("  Loading cycle structure...")
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    sig_base = os.path.join(WORKSPACE, "processed_dataset", "signals")

    print("  Loading nightly temperature...")
    ct = pd.read_csv(os.path.join(sig_base, "computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    print("  Loading nocturnal wrist temperature...")
    wt = pd.read_csv(os.path.join(sig_base, "wrist_temperature_cycle.csv"),
                      usecols=key + ["timestamp", "temperature_diff_from_baseline"])
    wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
    noct_wt = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
    noct_temp_daily = noct_wt.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
    noct_temp_daily.rename(columns={"temperature_diff_from_baseline": "noct_temp"}, inplace=True)

    print("  Loading resting heart rate...")
    rhr = pd.read_csv(os.path.join(sig_base, "resting_heart_rate_cycle.csv"),
                       usecols=key + ["value"])
    rhr_daily = rhr.groupby(key)["value"].mean().reset_index()
    rhr_daily.rename(columns={"value": "rhr"}, inplace=True)

    print("  Loading HRV details...")
    hrv = pd.read_csv(os.path.join(sig_base, "heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        rmssd_mean=("rmssd", "mean"),
        rmssd_std=("rmssd", "std"),
        lf_mean=("low_frequency", "mean"),
        hf_mean=("high_frequency", "mean"),
        hrv_coverage=("coverage", "mean"),
    ).reset_index()
    hrv_daily["lf_hf_ratio"] = hrv_daily["lf_mean"] / hrv_daily["hf_mean"].clip(lower=1)

    print("  Loading nocturnal HR (chunked, 0-6AM)...")
    hr_path = os.path.join(sig_base, "heart_rate_cycle.csv")
    hr_aggs = []
    for chunk in pd.read_csv(hr_path, chunksize=2_000_000,
                              usecols=key + ["timestamp", "bpm", "confidence"]):
        chunk["hour"] = pd.to_datetime(chunk["timestamp"]).dt.hour
        noct = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 6) & (chunk["confidence"] >= 1)]
        if len(noct) > 0:
            agg = noct.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
            hr_aggs.append(agg)
    hr_daily = pd.concat(hr_aggs).groupby(key).mean().reset_index()
    hr_daily.rename(columns={"mean": "noct_hr_mean", "std": "noct_hr_std",
                              "min": "noct_hr_min"}, inplace=True)

    print("  Merging all signals...")
    merged = cc.merge(ct_daily, on=key, how="left")
    merged = merged.merge(noct_temp_daily, on=key, how="left")
    merged = merged.merge(rhr_daily, on=key, how="left")
    merged = merged.merge(hrv_daily, on=key, how="left")
    merged = merged.merge(hr_daily, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))

    signal_cols = ["nightly_temperature", "noct_temp", "rhr",
                   "rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio",
                   "noct_hr_mean", "noct_hr_std", "noct_hr_min", "hrv_coverage"]

    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        entry = {"dic": (grp["day_in_study"] - cs).values, "id": grp["id"].values[0],
                 "cycle_len": n}
        for sc in signal_cols:
            entry[sc] = grp[sc].values if sc in grp.columns else np.full(n, np.nan)
        cycle_series[sgk] = entry

    sgk_order = (merged.groupby("small_group_key")["day_in_study"]
                 .min().reset_index().rename(columns={"day_in_study": "start"}))
    sgk_order = sgk_order.merge(
        merged[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    subj_order = {}
    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        subj_order[uid] = sgks
        past_lens = []
        for sgk in sgks:
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = np.mean(past_lens) if past_lens else 28.0
                cycle_series[sgk]["hist_cycle_std"] = np.std(past_lens) if len(past_lens) > 1 else 4.0
                past_lens.append(cycle_series[sgk]["cycle_len"])

    quality = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        raw = cycle_series[sgk]["nightly_temperature"]
        if np.isnan(raw).all():
            continue
        t = pd.Series(raw).interpolate(limit_direction="both").values
        ov = lh_ov_dict[sgk]
        n = len(t)
        if ov < 3 or ov + 2 >= n:
            continue
        pre = np.mean(t[max(0, ov - 5):ov])
        post = np.mean(t[ov + 2:min(n, ov + 7)])
        if post - pre >= 0.2:
            quality.add(sgk)

    labeled = [s for s in cycle_series if s in lh_ov_dict]
    print(f"  Cycles: {len(cycle_series)} | Labeled: {len(labeled)} | Quality: {len(quality)}")
    return lh_ov_dict, cycle_series, quality, subj_order, signal_cols


def _get_multi(data, sigs, sigma=1.5):
    """Get multiple cleaned signals stacked as (n_days, n_signals)."""
    arrays = []
    for sk in sigs:
        raw = data.get(sk)
        if raw is None or np.isnan(raw).all():
            return None
        arrays.append(_clean(raw, sigma=sigma))
    return np.column_stack(arrays)


# =====================================================================
# A. RULE-BASED METHODS
# =====================================================================

def detect_ttest_optimal(cs, sig_key, sigma=2.0, invert=False, pw=4.0, frac=0.575):
    """T-test with Gaussian position prior."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        exp = max(8, hcl * frac)
        best_ws, best_sp, best_stat = -np.inf, None, 0
        for sp in range(5, n - 3):
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


def detect_cusum(cs, sig_key, sigma=2.0, invert=False, threshold=1.0, frac=0.575):
    """CUSUM changepoint detection on single signal."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
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
        exp = max(8, hcl * frac)
        if len(alarm_pts) > 0:
            dists = np.abs(alarm_pts - exp)
            best = alarm_pts[np.argmin(dists)]
            det[sgk] = int(best)
            conf[sgk] = min(1.0, s_pos[best] / (3 * np.std(t) + 1e-8))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


def detect_bayesian_biphasic(cs, sig_key, sigma=2.0, invert=False, frac=0.575):
    """Bayesian biphasic step-function fitting (SSE minimization + position prior)."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        if invert:
            t = -t
        exp = max(8, hcl * frac)
        best_score, best_sp = np.inf, n // 2
        for sp in range(5, n - 3):
            m1, m2 = np.mean(t[:sp]), np.mean(t[sp:])
            if m2 <= m1:
                continue
            sse = np.sum((t[:sp] - m1) ** 2) + np.sum((t[sp:] - m2) ** 2)
            pos_pen = 0.5 * ((sp - exp) / 4.0) ** 2
            score = sse + pos_pen
            if score < best_score:
                best_score = score
                best_sp = sp
        det[sgk] = best_sp
        m1, m2 = np.mean(t[:best_sp]), np.mean(t[best_sp:])
        shift = m2 - m1
        conf[sgk] = min(1.0, max(0, shift / (np.std(t) + 1e-8)))
    return det, conf


def detect_hmm_2state(cs, sig_key, sigma=2.0, invert=False, frac=0.575):
    """2-state Gaussian HMM: state 0=follicular(low), state 1=luteal(high)."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
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
        trans_stay = 0.95

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


def detect_savgol_gradient(cs, sig_key, sigma=0, invert=False, frac=0.575):
    """Savitzky-Golay smoothing + maximum gradient detection."""
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
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
            ts = gaussian_filter1d(t, sigma=2.0)
        grad = np.gradient(ts)
        exp = max(8, hcl * frac)
        search_lo = max(3, int(exp - 8))
        search_hi = min(n - 2, int(exp + 8))
        if search_lo >= search_hi:
            search_lo, search_hi = 3, n - 2
        region = grad[search_lo:search_hi + 1]
        if len(region) == 0:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        peak_idx = search_lo + np.argmax(region)
        det[sgk] = int(peak_idx)
        conf[sgk] = min(1.0, max(0, grad[peak_idx] / (np.std(grad) + 1e-8)))
    return det, conf


def detect_multi_signal_fused_ttest(cs, sigs, sigma=2.0, inverts=None, frac=0.575, pw=4.0):
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
        if len(valid_sigs) < 1 or n < 12:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        fused = np.mean(valid_sigs, axis=0)
        exp = max(8, hcl * frac)
        best_ws, best_sp, best_stat = -np.inf, None, 0
        for sp in range(5, n - 3):
            diff = np.mean(fused[sp:]) - np.mean(fused[:sp])
            if diff <= 0:
                continue
            try:
                stat, _ = ttest_ind(fused[sp:], fused[:sp], alternative="greater")
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


def detect_multi_hmm(cs, sigs, sigma=2.0, inverts=None, frac=0.575):
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
        if len(valid_ts) < 2 or n < 12:
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
        trans_stay = 0.95

        for _ in range(15):
            log_e0 = np.sum([norm.logpdf(X[:, j], mu0[j], s0[j]) for j in range(d)], axis=0)
            log_e1 = np.sum([norm.logpdf(X[:, j], mu1[j], s1[j]) for j in range(d)], axis=0)
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


def detect_multi_cusum_fused(cs, sigs, sigma=2.0, inverts=None, frac=0.575, threshold=0.5):
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
        if not valid_ts or n < 12:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
            continue
        fused = np.mean(valid_ts, axis=0)
        mu = np.mean(fused)
        std_f = max(np.std(fused), 1e-8)
        s_pos = np.zeros(n)
        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i - 1] + (fused[i] - mu) - threshold * std_f)
        alarm_pts = np.where(s_pos > 0)[0]
        exp = max(8, hcl * frac)
        if len(alarm_pts) > 0:
            dists = np.abs(alarm_pts - exp)
            best = alarm_pts[np.argmin(dists)]
            det[sgk] = int(best)
            conf[sgk] = min(1.0, s_pos[best] / (3 * std_f))
        else:
            det[sgk] = int(round(frac * hcl))
            conf[sgk] = 0.0
    return det, conf


# =====================================================================
# B. ML METHODS (LOSO)
# =====================================================================

def extract_features_v2(data, sigma=1.5):
    """Comprehensive multi-signal feature extraction. No cycle_len leakage."""
    feats = {}
    hcl = data["hist_cycle_len"]
    feats["hist_clen"] = hcl
    feats["hist_cstd"] = data.get("hist_cycle_std", 4.0)
    exp_ov = hcl * 0.575
    n = data["cycle_len"]

    def _sig_feats(raw, prefix, sig=sigma, invert=False):
        if raw is None or np.isnan(raw).all():
            return {}
        t = _clean(raw, sigma=sig)
        if invert:
            t = -t
        f = {}
        f[f"{prefix}_mean"] = np.mean(t)
        f[f"{prefix}_std"] = np.std(t)
        f[f"{prefix}_range"] = np.ptp(t)
        f[f"{prefix}_skew"] = float(pd.Series(t).skew())
        f[f"{prefix}_kurt"] = float(pd.Series(t).kurtosis())
        f[f"{prefix}_nadir"] = int(np.argmin(t))
        f[f"{prefix}_nadir_dev"] = int(np.argmin(t)) - exp_ov
        grad = np.gradient(gaussian_filter1d(t, sigma=1.5))
        f[f"{prefix}_mgd"] = int(np.argmax(grad))
        f[f"{prefix}_mgd_dev"] = int(np.argmax(grad)) - exp_ov
        f[f"{prefix}_mgv"] = float(np.max(grad))
        for d in [8, 10, 12, 14, 16, 18, 20]:
            if 5 <= d < n - 3:
                try:
                    stat, _ = ttest_ind(t[d:], t[:d], alternative="greater")
                    f[f"{prefix}_tt{d}"] = float(stat) if not np.isnan(stat) else 0
                except:
                    f[f"{prefix}_tt{d}"] = 0
        best_sc, best_sp = -np.inf, n // 2
        for sp in range(5, n - 3):
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                if not np.isnan(stat) and stat > best_sc:
                    best_sc = stat
                    best_sp = sp
            except:
                continue
        f[f"{prefix}_bs"] = best_sp
        f[f"{prefix}_bs_dev"] = best_sp - exp_ov
        f[f"{prefix}_bst"] = max(best_sc, 0)
        pre_m = np.mean(t[:best_sp])
        post_m = np.mean(t[best_sp:])
        f[f"{prefix}_shift"] = post_m - pre_m
        h1 = np.mean(t[:n // 2])
        h2 = np.mean(t[n // 2:])
        f[f"{prefix}_half_diff"] = h2 - h1
        q1 = np.mean(t[:n // 4])
        q2 = np.mean(t[n // 4:n // 2])
        q3 = np.mean(t[n // 2:3 * n // 4])
        q4 = np.mean(t[3 * n // 4:])
        f[f"{prefix}_q1"] = q1
        f[f"{prefix}_q2"] = q2
        f[f"{prefix}_q3"] = q3
        f[f"{prefix}_q4"] = q4
        ts = pd.Series(t)
        for lag in [1, 3, 5]:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0
        rm3 = ts.rolling(3, min_periods=1).mean()
        rm7 = ts.rolling(7, min_periods=1).mean()
        cross = np.where(np.diff(np.sign(rm3 - rm7)))[0]
        if len(cross) > 0:
            dists_c = np.abs(cross - exp_ov)
            f[f"{prefix}_xover"] = int(cross[np.argmin(dists_c)])
        else:
            f[f"{prefix}_xover"] = n // 2
        return f

    feats.update(_sig_feats(data.get("nightly_temperature"), "nt"))
    feats.update(_sig_feats(data.get("noct_temp"), "noct"))
    feats.update(_sig_feats(data.get("noct_hr_mean"), "nhr"))
    feats.update(_sig_feats(data.get("rhr"), "rhr"))
    feats.update(_sig_feats(data.get("rmssd_mean"), "rmssd", invert=True))
    feats.update(_sig_feats(data.get("hf_mean"), "hf", invert=True))
    feats.update(_sig_feats(data.get("lf_hf_ratio"), "lfhf"))

    # Cross-signal features
    for (s1, p1, inv1), (s2, p2, inv2) in [
        (("nightly_temperature", "nt", False), ("rmssd_mean", "rmssd", True)),
        (("nightly_temperature", "nt", False), ("noct_hr_mean", "nhr", False)),
        (("rmssd_mean", "rmssd", True), ("noct_hr_mean", "nhr", False)),
    ]:
        r1 = data.get(s1)
        r2 = data.get(s2)
        if r1 is not None and r2 is not None and not np.isnan(r1).all() and not np.isnan(r2).all():
            t1 = _clean(r1, sigma=sigma)
            t2 = _clean(r2, sigma=sigma)
            if inv1:
                t1 = -t1
            if inv2:
                t2 = -t2
            corr = np.corrcoef(t1, t2)[0, 1]
            feats[f"xcorr_{p1}_{p2}"] = float(corr) if not np.isnan(corr) else 0
            bs1 = feats.get(f"{p1}_bs", n // 2)
            bs2 = feats.get(f"{p2}_bs", n // 2)
            feats[f"bs_diff_{p1}_{p2}"] = abs(bs1 - bs2)
            feats[f"bs_mean_{p1}_{p2}"] = (bs1 + bs2) / 2

    for k in feats:
        if isinstance(feats[k], float) and (np.isnan(feats[k]) or np.isinf(feats[k])):
            feats[k] = 0.0
    return feats


def ml_detect_loso(cs, lh, model_type="ridge"):
    """LOSO ML detection with comprehensive features."""
    labeled = [s for s in cs if s in lh]
    all_f, all_t, all_id, all_s = [], [], [], []
    for sgk in labeled:
        feats = extract_features_v2(cs[sgk])
        if not feats or len(feats) < 5:
            continue
        all_f.append(feats)
        all_t.append(lh[sgk])
        all_id.append(cs[sgk]["id"])
        all_s.append(sgk)
    if len(all_f) < 10:
        return {}, {}

    df = pd.DataFrame(all_f).fillna(0)
    X = df.values
    y = np.array(all_t, dtype=float)
    ids = np.array(all_id)
    uniq = np.unique(ids)

    from sklearn.preprocessing import StandardScaler

    det, confs = {}, {}
    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        y_tr = y[tr]

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=300, max_depth=6,
                                      min_samples_leaf=3, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                          learning_rate=0.03, subsample=0.8,
                                          random_state=42)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)
        elif model_type == "svr":
            from sklearn.svm import SVR
            m = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "bayridge":
            from sklearn.linear_model import BayesianRidge
            m = BayesianRidge()
        elif model_type == "xgb":
            try:
                from xgboost import XGBRegressor
                m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 random_state=42, verbosity=0)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                              learning_rate=0.03, random_state=42)
        elif model_type == "lgbm":
            try:
                from lightgbm import LGBMRegressor
                m = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, verbose=-1)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                              learning_rate=0.03, random_state=42)
        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            m = KNeighborsRegressor(n_neighbors=min(7, tr.sum()),
                                    weights="distance")
        elif model_type == "huber":
            from sklearn.linear_model import HuberRegressor
            m = HuberRegressor(max_iter=500)
        else:
            raise ValueError(model_type)

        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        test_sgks = [all_s[i] for i in np.where(te)[0]]
        for sgk, pred in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, pred))))
            confs[sgk] = 0.5
    return det, confs


def ml_phase_classify_loso(cs, lh):
    """
    Phase classification approach (Yu et al. 2022 inspired):
    Classify each day as follicular(0) vs luteal(1) using ML, then find boundary.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    labeled = [s for s in cs if s in lh]
    sigs_to_use = ["nightly_temperature", "noct_temp", "rhr",
                   "rmssd_mean", "lf_hf_ratio", "noct_hr_mean"]

    all_X, all_y, all_uid, all_sgk, all_dayidx = [], [], [], [], []
    for sgk in labeled:
        data = cs[sgk]
        ov = lh[sgk]
        n = data["cycle_len"]
        for i in range(n):
            row = []
            for sk in sigs_to_use:
                raw = data.get(sk)
                if raw is not None and not np.isnan(raw).all():
                    t = _clean(raw, sigma=1.5)
                    row.append(t[i])
                else:
                    row.append(0)
            row.append(i)
            row.append(data["hist_cycle_len"])
            all_X.append(row)
            all_y.append(1 if i >= ov else 0)
            all_uid.append(data["id"])
            all_sgk.append(sgk)
            all_dayidx.append(i)

    X = np.array(all_X)
    y = np.array(all_y)
    uids = np.array(all_uid)
    sgks = np.array(all_sgk)
    dayidxs = np.array(all_dayidx)
    uniq_uids = np.unique(uids)

    det, conf = {}, {}
    for uid in uniq_uids:
        te = uids == uid
        tr = ~te
        if tr.sum() < 20:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=42)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, 1]

        te_sgks = sgks[te]
        te_days = dayidxs[te]
        for sgk_val in np.unique(te_sgks):
            mask = te_sgks == sgk_val
            days = te_days[mask]
            probs = proba[mask]
            order = np.argsort(days)
            days = days[order]
            probs = probs[order]
            smoothed = gaussian_filter1d(probs, sigma=1.5)
            boundary = None
            for i in range(len(smoothed) - 1):
                if smoothed[i] < 0.5 and smoothed[i + 1] >= 0.5:
                    boundary = int(days[i + 1])
                    break
            if boundary is None:
                boundary = int(round(cs[sgk_val]["hist_cycle_len"] * 0.575))
            det[sgk_val] = boundary
            conf[sgk_val] = float(np.max(probs) - np.min(probs))
    return det, conf


# =====================================================================
# C. 1D-CNN on multi-signal daily time series
# =====================================================================

def cnn_detect_loso(cs, lh, sigs=None, inverts=None, max_len=40):
    """1D-CNN regression on multi-signal daily series."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [SKIP] PyTorch not available for CNN")
        return {}, {}

    if sigs is None:
        sigs = ["nightly_temperature", "noct_temp", "rhr",
                "rmssd_mean", "lf_hf_ratio", "noct_hr_mean"]
    if inverts is None:
        inverts = [False, False, False, True, False, False]

    labeled = [s for s in cs if s in lh]
    all_X, all_y, all_uid, all_sgk = [], [], [], []
    for sgk in labeled:
        data = cs[sgk]
        n = data["cycle_len"]
        channels = []
        valid = True
        for sk, inv in zip(sigs, inverts):
            raw = data.get(sk)
            if raw is None or np.isnan(raw).all():
                valid = False
                break
            t = _clean(raw, sigma=1.5)
            if inv:
                t = -t
            std = np.std(t)
            if std > 1e-8:
                t = (t - np.mean(t)) / std
            padded = np.zeros(max_len)
            padded[:min(n, max_len)] = t[:max_len]
            channels.append(padded)
        if not valid:
            continue
        all_X.append(np.stack(channels))
        all_y.append(lh[sgk])
        all_uid.append(data["id"])
        all_sgk.append(sgk)

    if len(all_X) < 15:
        return {}, {}

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    uids = np.array(all_uid)
    uniq = np.unique(uids)

    class CNN1D(nn.Module):
        def __init__(self, in_ch, seq_len):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc1 = nn.Linear(64 * 8, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.drop(self.relu(self.fc1(x)))
            return self.fc2(x).squeeze(-1)

    det, confs = {}, {}
    for uid in uniq:
        te = uids == uid
        tr = ~te
        if tr.sum() < 10:
            continue
        X_tr = torch.FloatTensor(X[tr])
        y_tr = torch.FloatTensor(y[tr])
        X_te = torch.FloatTensor(X[te])

        model = CNN1D(len(sigs), max_len)
        opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()

        model.train()
        for epoch in range(150):
            opt.zero_grad()
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te).numpy()
        test_sgks = [all_sgk[i] for i in np.where(te)[0]]
        for sgk, p in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, p))))
            confs[sgk] = 0.5
    return det, confs


# =====================================================================
# D. STACKING META-LEARNER
# =====================================================================

def stacking_detect(cs, lh, base_results):
    """Stacking: Ridge on base detector outputs + hist_clen as meta-features."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    labeled = [s for s in cs if s in lh]
    base_names = list(base_results.keys())
    rows, targets, uids, sgk_list = [], [], [], []
    for sgk in labeled:
        feats = [base_results[name][0].get(sgk, cs[sgk]["hist_cycle_len"] * 0.575)
                 for name in base_names]
        feats.append(cs[sgk]["hist_cycle_len"])
        feats.append(cs[sgk]["hist_cycle_len"] * 0.575)
        rows.append(feats)
        targets.append(lh[sgk])
        uids.append(cs[sgk]["id"])
        sgk_list.append(sgk)

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)
    ids = np.array(uids)
    uniq = np.unique(ids)

    det, confs = {}, {}
    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        m = Ridge(alpha=1.0)
        m.fit(X_tr, y[tr])
        preds = m.predict(X_te)
        test_sgks = [sgk_list[i] for i in np.where(te)[0]]
        for sgk, p in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, p))))
            confs[sgk] = 0.6
    return det, confs


# =====================================================================
# E. WEIGHTED ENSEMBLE
# =====================================================================

def weighted_ensemble(results_list, cs, lh, top_n=5):
    """Weighted average of top-N detectors by ±2d accuracy."""
    labeled = set(s for s in cs if s in lh)
    scored = []
    for name, (d, c) in results_list:
        errs = [abs(d[s] - lh[s]) for s in d if s in labeled]
        if errs:
            acc2 = np.mean(np.array(errs) <= 2)
            scored.append((acc2, name, d, c))
    scored.sort(reverse=True)
    top = scored[:top_n]
    if not top:
        return {}, {}

    det, confs = {}, {}
    all_sgks = set()
    for _, _, d, _ in top:
        all_sgks.update(d.keys())
    for sgk in all_sgks:
        vals, ws = [], []
        for acc, _, d, c in top:
            if sgk in d:
                vals.append(d[sgk])
                ws.append(max(acc, 0.01))
        if vals:
            det[sgk] = int(round(np.average(vals, weights=ws)))
            confs[sgk] = np.mean(ws)
    return det, confs


# =====================================================================
# F. MENSTRUAL PREDICTION
# =====================================================================

def predict_menses(cs, det, confs, subj_order, lh, fl=13.0, eval_subset=None, label=""):
    pop_lut = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)
    errs = []
    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0
            ov = det.get(sgk)
            conf = confs.get(sgk, 0.0)
            pred = (ov + lut) if (ov is not None and ov > 3) else acl
            ev = set(eval_subset) if eval_subset else None
            if ev is None or sgk in ev:
                errs.append(pred - actual)
            s_pclen[uid].append(actual)
            if ov is not None:
                el = actual - ov
                if 8 <= el <= 22:
                    s_plut[uid].append(el)
    if not errs:
        return {}
    ae = np.abs(errs)
    return _pr(label, ae, prefix="    ")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"\n{SEP}")
    print(f"  Multi-Signal Ovulation Detection and Menstrual Prediction (new_workspace data)")
    print(f"{SEP}")
    print(f"  Cycle: {CYCLE_OV_CSV}")
    print(f"  Signals: {SIGNALS_DIR}")
    if not CYCLE_OV_CSV.exists():
        raise SystemExit(f"Cycle CSV not found: {CYCLE_OV_CSV}. Run data_clean.py then ovulation_labels.py.")
    if not SIGNALS_DIR.is_dir():
        raise SystemExit(f"Signals dir not found: {SIGNALS_DIR}. Run wearable_signals.py.")
    t0 = time.time()

    lh, cs, quality, so, sig_cols = load_all_signals()
    labeled = set(s for s in cs if s in lh)

    all_results = []

    # ===== SECTION A: Single-signal rule-based =====
    print(f"\n{SEP}\n  A. SINGLE-SIGNAL RULE-BASED (T-test, CUSUM, Bayesian, HMM, SavGol)\n{SEP}")

    configs = [
        ("nightly_temperature", False, "NT"),
        ("noct_temp", False, "NocT"),
        ("noct_hr_mean", False, "NocHR"),
        ("rhr", False, "RHR"),
        ("rmssd_mean", True, "RMSSD"),
        ("hf_mean", True, "HF"),
        ("lf_hf_ratio", False, "LFHF"),
    ]
    methods = [
        ("tt", detect_ttest_optimal),
        ("cusum", detect_cusum),
        ("baybi", detect_bayesian_biphasic),
        ("hmm", detect_hmm_2state),
        ("savgol", detect_savgol_gradient),
    ]

    for sig_key, inv, sname in configs:
        for mname, mfunc in methods:
            for sigma in [1.5, 2.0, 2.5]:
                tag = f"{sname}-{mname}-σ{sigma}"
                d, c = mfunc(cs, sig_key, sigma=sigma, invert=inv)
                errs = [abs(d[s] - lh[s]) for s in d if s in labeled]
                if errs:
                    r = _pr(tag, errs)
                    results_db[tag] = r
                    all_results.append((tag, (d, c)))

    # ===== SECTION B: Multi-signal rule-based =====
    print(f"\n{SEP}\n  B. MULTI-SIGNAL RULE-BASED (Fused T-test, HMM, CUSUM)\n{SEP}")

    multi_combos = [
        ("TempAll", ["nightly_temperature", "noct_temp"], [False, False]),
        ("Temp+HR", ["nightly_temperature", "noct_hr_mean"], [False, False]),
        ("Temp+RMSSD", ["nightly_temperature", "rmssd_mean"], [False, True]),
        ("Temp+HR+RMSSD", ["nightly_temperature", "noct_hr_mean", "rmssd_mean"],
         [False, False, True]),
        ("Temp+HR+HRV", ["nightly_temperature", "noct_hr_mean", "rmssd_mean", "lf_hf_ratio"],
         [False, False, True, False]),
        ("ALL5", ["nightly_temperature", "noct_hr_mean", "rhr", "rmssd_mean", "lf_hf_ratio"],
         [False, False, False, True, False]),
        ("HR+HRV", ["noct_hr_mean", "rmssd_mean", "lf_hf_ratio"], [False, True, False]),
    ]

    for combo_name, sigs, invs in multi_combos:
        for sigma in [1.5, 2.0, 2.5]:
            for method_name, method_fn in [
                ("ftt", lambda cs_, s, sig, inv: detect_multi_signal_fused_ttest(
                    cs_, s, sigma=sig, inverts=inv)),
                ("mhmm", lambda cs_, s, sig, inv: detect_multi_hmm(
                    cs_, s, sigma=sig, inverts=inv)),
                ("mcusum", lambda cs_, s, sig, inv: detect_multi_cusum_fused(
                    cs_, s, sigma=sig, inverts=inv)),
            ]:
                tag = f"{combo_name}-{method_name}-σ{sigma}"
                d, c = method_fn(cs, sigs, sigma, invs)
                errs = [abs(d[s] - lh[s]) for s in d if s in labeled]
                if errs:
                    r = _pr(tag, errs)
                    results_db[tag] = r
                    all_results.append((tag, (d, c)))

    # ===== SECTION C: ML LOSO =====
    print(f"\n{SEP}\n  C. ML MULTI-SIGNAL (LOSO)\n{SEP}")

    for mt in ["ridge", "elastic", "svr", "rf", "gbdt", "bayridge", "knn", "huber", "xgb", "lgbm"]:
        tag = f"ML-{mt}"
        d, c = ml_detect_loso(cs, lh, model_type=mt)
        if d:
            errs = [abs(d[s] - lh[s]) for s in d if s in lh]
            r = _pr(tag, errs)
            results_db[tag] = r
            all_results.append((tag, (d, c)))

    # Phase classification
    print(f"\n  Phase classification approach...")
    tag = "ML-phaseclass"
    d, c = ml_phase_classify_loso(cs, lh)
    if d:
        errs = [abs(d[s] - lh[s]) for s in d if s in lh]
        r = _pr(tag, errs)
        results_db[tag] = r
        all_results.append((tag, (d, c)))

    # 1D-CNN
    print(f"\n  1D-CNN multi-signal...")
    tag = "CNN-multi"
    d, c = cnn_detect_loso(cs, lh)
    if d:
        errs = [abs(d[s] - lh[s]) for s in d if s in lh]
        r = _pr(tag, errs)
        results_db[tag] = r
        all_results.append((tag, (d, c)))

    # ===== SECTION D: STACKING =====
    print(f"\n{SEP}\n  D. STACKING META-LEARNER\n{SEP}")

    scored = [(results_db.get(name, {}).get("acc_2d", 0), name, d, c)
              for name, (d, c) in all_results]
    scored.sort(reverse=True)

    for topN in [5, 10, 15, 20]:
        top = scored[:topN]
        base = {name: (d, c) for _, name, d, c in top}
        if len(base) >= 3:
            tag = f"stack-top{topN}"
            d, c = stacking_detect(cs, lh, base)
            if d:
                errs = [abs(d[s] - lh[s]) for s in d if s in lh]
                r = _pr(tag, errs)
                results_db[tag] = r
                all_results.append((tag, (d, c)))

    # Weighted ensemble
    print(f"\n  Weighted ensembles...")
    for topN in [3, 5, 7, 10, 15]:
        tag = f"wens-top{topN}"
        d, c = weighted_ensemble(all_results, cs, lh, top_n=topN)
        if d:
            errs = [abs(d[s] - lh[s]) for s in d if s in lh]
            r = _pr(tag, errs)
            results_db[tag] = r
            all_results.append((tag, (d, c)))

    # ===== SECTION E: FINAL RANKING =====
    print(f"\n{SEP}\n  E. FINAL RANKING — ALL LABELED (n={len(labeled)})\n{SEP}")
    final = []
    for name, (d, c) in all_results:
        errs = [abs(d[s] - lh[s]) for s in d if s in labeled]
        if errs:
            ae = np.array(errs)
            final.append((name, len(ae), np.mean(ae),
                          (ae <= 1).mean(), (ae <= 2).mean(),
                          (ae <= 3).mean(), (ae <= 5).mean(), d, c))
    final.sort(key=lambda x: x[4], reverse=True)
    print(f"  {'Method':<40s} {'n':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*82}")
    for name, n, mae, a1, a2, a3, a5, _, _ in final[:30]:
        print(f"  {name:<40s} {n:>3} {mae:>5.2f}"
              f" {a1:>5.1%} {a2:>5.1%} {a3:>5.1%} {a5:>5.1%}")

    # Quality subset
    print(f"\n  QUALITY SUBSET (n={len(quality)})")
    print(f"  {'Method':<40s} {'n':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*82}")
    for name, _, _, _, _, _, _, d, c in final[:30]:
        errs_q = [abs(d[s] - lh[s]) for s in d if s in quality and s in lh]
        if errs_q:
            ae = np.array(errs_q)
            print(f"  {name:<40s} {len(ae):>3} {np.mean(ae):>5.2f}"
                  f" {(ae<=1).mean():>5.1%} {(ae<=2).mean():>5.1%}"
                  f" {(ae<=3).mean():>5.1%} {(ae<=5).mean():>5.1%}")

    # ===== SECTION F: MENSTRUAL PREDICTION =====
    print(f"\n{SEP}\n  F. MENSTRUAL PREDICTION — Top 5 detectors\n{SEP}")
    for name, _, _, _, _, _, _, d, c in final[:5]:
        for fl in [12, 13]:
            predict_menses(cs, d, c, so, lh, fl=float(fl),
                           eval_subset=labeled, label=f"{name}+lut{fl}")
            predict_menses(cs, d, c, so, lh, fl=float(fl),
                           eval_subset=quality, label=f"Q-{name}+lut{fl}")

    # Oracle baselines
    print(f"\n  Baselines:")
    oracle = {s: v for s, v in lh.items() if s in cs}
    oconf = {s: 1.0 for s in oracle}
    predict_menses(cs, oracle, oconf, so, lh, fl=13.0,
                   eval_subset=labeled, label="Oracle+lut13")
    predict_menses(cs, {}, {}, so, lh, eval_subset=labeled, label="Calendar")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  DONE ({elapsed:.0f}s) — {len(all_results)} configurations tested\n{SEP}")


if __name__ == "__main__":
    main()

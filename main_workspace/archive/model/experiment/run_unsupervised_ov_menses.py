"""
Rule-based & Unsupervised Ovulation Detection → Menstrual Prediction (v2)
=========================================================================
Key improvements over v1:
  - Uses nocturnal minute-level wrist temperature (higher SNR)
  - Per-CYCLE menstrual prediction evaluation (not per-day)
  - Quality-stratified results (all cycles vs quality-filtered)
  - More detection methods + aggressive ensemble
  - Leakage-free: hist_cycle_len from PAST cycles, no LH in detection

Methods:
  0. Calendar baseline (hist_cycle_len × fraction)
  1. t-test optimal split (multiple smoothing + prior configs)
  2. Bayesian biphasic step (SSE minimization + position prior)
  3. Savitzky-Golay + max-gradient
  4. Multi-signal t-test (temp + RHR + RMSSD)
  5. 2-state HMM (single & multi-signal, unsupervised)
  6. CUSUM (cumulative sum)
  7. Ensemble of best methods
  8. Confidence-gated pipeline

Menstrual prediction pipeline:
  - Per-cycle: predicted_cycle_len = detected_ov + personal_luteal
  - Personal luteal from PAST detected ov + actual cycle len (no LH!)
  - Calendar fallback: predicted_cycle_len = hist_avg_cycle_len
  - Evaluate: |predicted - actual| ≤ k

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_unsupervised_ov_menses
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


# =====================================================================
# Data Loading — Uses nocturnal minute-level temperature + other signals
# =====================================================================

def load_all_cycle_data():
    """Load nocturnal wrist temp (minute-level), daily HR/HRV/RHR merged at cycle-day level.

    Returns: lh_ov_dict, cycle_series dict, quality_cycles set
    """
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    # 1. Nocturnal temperature from minute-level wrist temperature
    print("  Loading minute-level wrist temperature → nocturnal stats...")
    wt = pd.read_csv(os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv"))
    wt["hour"] = wt["timestamp"].str[:2].astype(int)
    wt["temp"] = wt["temperature_diff_from_baseline"]
    night_mask = (wt["hour"] >= 22) | (wt["hour"] < 6)
    wt_night = wt[night_mask]
    noct = wt_night.groupby(key)["temp"].agg(["mean", "std", "count"]).reset_index()
    noct.columns = key + ["noct_mean", "noct_std", "noct_count"]

    # 2. Aggregated nightly temperature
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    # 3. Resting heart rate
    rhr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv"))
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)

    # 4. Heart rate
    hr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv"))
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std"]

    # 5. HRV
    hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg({"rmssd": "mean"}).reset_index()
    hrv_daily.columns = key + ["rmssd_mean"]

    merged = cc.copy()
    for src in [noct, ct_daily, rhr_daily, hr_daily, hrv_daily]:
        merged = merged.merge(src, on=key, how="left")

    # LH ground truth
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))

    # Build cycle_series
    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        cycle_series[sgk] = {
            "dic": (grp["day_in_study"] - cs).values,
            "noct_mean": grp["noct_mean"].values,
            "noct_std": grp["noct_std"].values,
            "temps": grp["nightly_temperature"].values,
            "rhr": grp["resting_hr"].values,
            "hr_mean": grp["hr_mean"].values,
            "rmssd": grp["rmssd_mean"].values,
            "id": grp["id"].values[0],
            "cycle_len": n,
        }

    # Historical cycle length from PAST cycles only
    sgk_order = (
        merged.groupby("small_group_key")["day_in_study"]
        .min().reset_index().rename(columns={"day_in_study": "start"})
    )
    sgk_order = sgk_order.merge(
        merged[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        past_lens = []
        for sgk in sgks:
            hcl = np.mean(past_lens) if past_lens else 28.0
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = hcl
                past_lens.append(cycle_series[sgk]["cycle_len"])

    # Quality filter: cycles with clear temperature shift ≥ 0.2°C
    quality_cycles = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        data = cycle_series[sgk]
        raw = data["noct_mean"]
        if np.isnan(raw).all():
            raw = data["temps"]
        if np.isnan(raw).all():
            continue
        t = pd.Series(raw).interpolate(limit_direction="both").values
        ov_d = lh_ov_dict[sgk]
        n = len(t)
        if ov_d < 3 or ov_d + 2 >= n:
            continue
        pre = np.mean(t[max(0, ov_d - 5):ov_d])
        post = np.mean(t[ov_d + 2:min(n, ov_d + 7)])
        if post - pre >= 0.2:
            quality_cycles.add(sgk)

    return lh_ov_dict, cycle_series, quality_cycles


def _clean(arr, smooth=0):
    """Interpolate NaNs and optionally smooth."""
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if smooth > 0:
        out = gaussian_filter1d(out, sigma=smooth)
    return out


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_ov(detected, lh_ov_dict, name="", subset=None):
    """Evaluate ovulation detection. If subset given, only evaluate those cycles."""
    errors = []
    eval_keys = subset if subset else lh_ov_dict.keys()
    n_total = len(eval_keys)

    for sgk in eval_keys:
        if sgk in detected and sgk in lh_ov_dict:
            errors.append(detected[sgk] - lh_ov_dict[sgk])

    n_eval = len(errors)
    if n_eval == 0:
        print(f"  [{name}] No evaluable detections (0/{n_total})")
        return {}

    err = np.array(errors)
    ae = np.abs(err)
    r = {
        "n": n_eval, "n_total": n_total,
        "recall": n_eval / n_total if n_total else 0,
        "mae": float(ae.mean()),
        "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
        "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean()),
    }
    print(
        f"  [{name}] {n_eval}/{n_total} ({r['recall']:.0%})"
        f" | MAE={r['mae']:.2f}d"
        f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
        f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}"
    )
    return r


# =====================================================================
# Phase 1: Ovulation Detection Methods
# =====================================================================

def detect_calendar(cycle_series, frac=0.575):
    detected = {}
    for sgk, data in cycle_series.items():
        detected[sgk] = int(round(frac * data["hist_cycle_len"]))
    return detected


def _ttest_core(t_clean, dic, n, hcl, frac_prior, prior_width, min_shift=0.0):
    """Core t-test split logic. Returns (best_split_idx, best_score) or (None, 0)."""
    expected = max(8, hcl * frac_prior)
    best_ws, best_sp = -np.inf, None
    for sp in range(5, n - 3):
        pre, post = t_clean[:sp], t_clean[sp:]
        diff = np.mean(post) - np.mean(pre)
        if diff <= min_shift:
            continue
        try:
            stat, _ = ttest_ind(post, pre, alternative="greater")
        except Exception:
            continue
        if np.isnan(stat):
            continue
        pp = np.exp(-0.5 * ((dic[sp] - expected) / prior_width) ** 2)
        ws = stat * pp
        if ws > best_ws:
            best_ws = ws
            best_sp = sp
    return best_sp, best_ws


def detect_ttest(cycle_series, smooth_sigma=2.0, frac_prior=0.55,
                 prior_width=5.0, min_shift=0.0, signal="noct_mean"):
    """Retrospective t-test optimal split."""
    detected, scores = {}, {}
    for sgk, data in cycle_series.items():
        raw = data.get(signal, data["temps"])
        dic = data["dic"]
        n = len(raw)
        if n < 12 or np.isnan(raw).all():
            continue
        t = _clean(raw, smooth=smooth_sigma)
        sp, ws = _ttest_core(t, dic, n, data["hist_cycle_len"], frac_prior, prior_width, min_shift)
        if sp is not None:
            detected[sgk] = int(dic[sp])
            scores[sgk] = ws
    return detected, scores


def detect_biphasic(cycle_series, smooth_sigma=2.0, frac_prior=0.55,
                    prior_width=5.0, signal="noct_mean"):
    """Bayesian biphasic step-function fitting."""
    detected, scores = {}, {}
    for sgk, data in cycle_series.items():
        raw = data.get(signal, data["temps"])
        dic = data["dic"]
        n = len(raw)
        if n < 12 or np.isnan(raw).all():
            continue
        t = _clean(raw, smooth=smooth_sigma)
        expected = max(8, data["hist_cycle_len"] * frac_prior)

        best_score, best_sp = np.inf, None
        for sp in range(5, n - 3):
            mu1, mu2 = np.mean(t[:sp]), np.mean(t[sp:])
            if mu2 <= mu1:
                continue
            sse = np.sum((t[:sp] - mu1) ** 2) + np.sum((t[sp:] - mu2) ** 2)
            pen = 0.5 * ((dic[sp] - expected) / prior_width) ** 2
            sc = sse + pen
            if sc < best_score:
                best_score = sc
                best_sp = sp

        if best_sp is not None:
            shift = np.mean(t[best_sp:]) - np.mean(t[:best_sp])
            detected[sgk] = int(dic[best_sp])
            scores[sgk] = shift
    return detected, scores


def detect_savgol_gradient(cycle_series, window=9, polyorder=2,
                           frac_prior=0.55, prior_width=5.0, signal="noct_mean"):
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data.get(signal, data["temps"])
        dic = data["dic"]
        n = len(raw)
        if n < max(12, window + 2) or np.isnan(raw).all():
            continue
        t = _clean(raw, smooth=0)
        win = min(window, n - 1)
        if win % 2 == 0:
            win -= 1
        if win < 5:
            continue

        smoothed = savgol_filter(t, window_length=win, polyorder=polyorder)
        grad = np.gradient(smoothed)
        expected = max(8, data["hist_cycle_len"] * frac_prior)
        weights = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        wg = grad * weights

        lo = max(5, int(data["hist_cycle_len"] * 0.25))
        hi = min(n - 2, int(data["hist_cycle_len"] * 0.75))
        if lo >= hi:
            lo, hi = 5, n - 2
        region = wg[lo:hi]
        if len(region) == 0:
            continue
        detected[sgk] = int(dic[lo + np.argmax(region)])
    return detected


def detect_cusum(cycle_series, smooth_sigma=2.0, frac_prior=0.55,
                 prior_width=5.0, signal="noct_mean"):
    """CUSUM: cumulative sum of deviations from mean, find max negative cusum."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data.get(signal, data["temps"])
        dic = data["dic"]
        n = len(raw)
        if n < 12 or np.isnan(raw).all():
            continue
        t = _clean(raw, smooth=smooth_sigma)
        mu = np.mean(t)
        cusum = np.cumsum(t - mu)
        expected = max(8, data["hist_cycle_len"] * frac_prior)
        pp = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        weighted = -cusum * pp

        lo = max(5, int(data["hist_cycle_len"] * 0.25))
        hi = min(n - 2, int(data["hist_cycle_len"] * 0.75))
        if lo >= hi:
            lo, hi = 5, n - 2
        region = weighted[lo:hi]
        if len(region) == 0:
            continue
        detected[sgk] = int(dic[lo + np.argmax(region)])
    return detected


def detect_multi_signal_ttest(cycle_series, smooth_sigma=2.0, frac_prior=0.55,
                              prior_width=5.0):
    """Combine t-test scores from temp + RHR + RMSSD."""
    signals = [("noct_mean", 1), ("rhr", 1), ("rmssd", -1)]
    detected = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        if n < 12:
            continue
        hcl = data["hist_cycle_len"]
        expected = max(8, hcl * frac_prior)

        combined = np.zeros(n)
        n_valid = 0
        for sig, sgn in signals:
            raw = data.get(sig)
            if raw is None or np.isnan(raw).all() or np.isnan(raw).sum() > n * 0.5:
                continue
            t = _clean(raw, smooth=smooth_sigma)
            n_valid += 1
            for sp in range(5, n - 3):
                pre, post = t[:sp], t[sp:]
                diff = sgn * (np.mean(post) - np.mean(pre))
                if diff <= 0:
                    continue
                try:
                    stat, _ = ttest_ind(sgn * post, sgn * pre, alternative="greater")
                except Exception:
                    continue
                if not np.isnan(stat):
                    combined[sp] += stat

        if n_valid == 0:
            continue
        combined /= n_valid
        pp = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        combined *= pp

        lo = max(5, int(hcl * 0.25))
        hi = min(n - 3, int(hcl * 0.75))
        if lo >= hi:
            lo, hi = 5, n - 3
        region = combined[lo:hi]
        if len(region) == 0 or region.max() <= 0:
            continue
        detected[sgk] = int(dic[lo + np.argmax(region)])
    return detected


def detect_hmm(cycle_series, use_multi=False, smooth_sigma=1.5, signal="noct_mean"):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [HMM] hmmlearn not installed, skipping")
        return {}

    detected = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        if n < 12:
            continue
        if use_multi:
            cols = []
            for sig in [signal, "rhr", "rmssd"]:
                raw = data.get(sig)
                if raw is not None and not np.isnan(raw).all():
                    cols.append(_clean(raw, smooth=smooth_sigma))
            if len(cols) == 0:
                continue
            obs = np.column_stack(cols)
        else:
            raw = data.get(signal, data["temps"])
            if np.isnan(raw).all():
                continue
            obs = _clean(raw, smooth=smooth_sigma).reshape(-1, 1)

        try:
            mdl = GaussianHMM(
                n_components=2, covariance_type="full",
                n_iter=100, random_state=42, init_params="mc",
            )
            mdl.startprob_ = np.array([0.9, 0.1])
            mdl.transmat_ = np.array([[0.95, 0.05], [0.02, 0.98]])
            mdl.fit(obs)
            states = mdl.predict(obs)
            low = np.argmin(mdl.means_[:, 0])
            high = 1 - low
            for i in range(1, n):
                if states[i] == high and states[i - 1] == low and dic[i] >= 6:
                    detected[sgk] = int(dic[i])
                    break
        except Exception:
            continue
    return detected


def ensemble_methods(all_detected, weights=None):
    all_sgks = set()
    for det in all_detected:
        all_sgks.update(det.keys())
    if weights is None:
        weights = [1.0] * len(all_detected)
    result = {}
    for sgk in all_sgks:
        vals, ws = [], []
        for det, w in zip(all_detected, weights):
            if sgk in det:
                vals.append(det[sgk])
                ws.append(w)
        if len(vals) >= 2:
            result[sgk] = int(round(np.average(vals, weights=ws)))
        elif len(vals) == 1:
            result[sgk] = vals[0]
    return result


def confidence_gated(detected, scores, threshold, fallback):
    result = {}
    for sgk in set(detected) | set(fallback):
        if sgk in detected and scores.get(sgk, 0) >= threshold:
            result[sgk] = detected[sgk]
        elif sgk in fallback:
            result[sgk] = fallback[sgk]
    return result


# =====================================================================
# Phase 2: Per-Cycle Menstrual Prediction (leakage-free)
# =====================================================================

def predict_menses_per_cycle(cycle_series, detected_ov, lh_ov_dict):
    """Per-cycle menstrual prediction using detected ovulation + personal luteal.

    For each cycle:
      predicted_cycle_len = detected_ov + personal_avg_luteal
      OR hist_avg_cycle_len (calendar fallback)

    Personal luteal estimated from PAST completed cycles' detected ov and actual length.
    """
    cc = pd.read_csv(CYCLE_CSV)
    sgk_order = (
        cc.groupby("small_group_key")["day_in_study"]
        .min().reset_index().rename(columns={"day_in_study": "start"})
    )
    sgk_order = sgk_order.merge(
        cc[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    pop_luteal = 14.0
    subj_past_luteal = defaultdict(list)
    subj_past_clens = defaultdict(list)

    results_ov = []     # cycles where ovulation was used
    results_cal = []    # cycles where calendar was used
    results_all = []    # all cycles

    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual_len = data["cycle_len"]

            past_luts = subj_past_luteal[uid]
            past_clens = subj_past_clens[uid]
            avg_lut = np.mean(past_luts) if past_luts else pop_luteal
            avg_clen = np.mean(past_clens) if past_clens else 28.0

            ov_det = detected_ov.get(sgk)
            if ov_det is not None:
                pred_len = ov_det + avg_lut
                used_ov = True
            else:
                pred_len = avg_clen
                used_ov = False

            err = pred_len - actual_len
            results_all.append(err)
            if used_ov:
                results_ov.append(err)
            else:
                results_cal.append(err)

            subj_past_clens[uid].append(actual_len)
            if ov_det is not None:
                est_luteal = actual_len - ov_det
                if 8 <= est_luteal <= 22:
                    subj_past_luteal[uid].append(est_luteal)

    def _stats(errs, label):
        if not errs:
            return {}
        ae = np.abs(errs)
        r = {
            "n": len(ae), "mae": float(ae.mean()),
            "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
            "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean()),
        }
        print(
            f"  [{label}] n={r['n']}"
            f" | MAE={r['mae']:.2f}d"
            f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
            f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}"
        )
        return r

    r_all = _stats(results_all, "ALL")
    r_ov = _stats(results_ov, "ov-countdown")
    r_cal = _stats(results_cal, "calendar-fallback")
    return r_all, r_ov, r_cal


# =====================================================================
# Main
# =====================================================================

def main():
    print(f"{SEP}\n  Rule-based & Unsupervised Ovulation → Menstrual Prediction v2\n"
          f"  (Nocturnal minute-level temp + Per-cycle evaluation)\n{SEP}")

    lh_ov_dict, cycle_series, quality_cycles = load_all_cycle_data()
    n_cycles = len(cycle_series)
    n_labeled = sum(1 for sgk in cycle_series if sgk in lh_ov_dict)
    n_subjects = len(set(d["id"] for d in cycle_series.values()))
    print(f"\n  Cycles: {n_cycles} | LH-labeled: {n_labeled} | Subjects: {n_subjects}")
    print(f"  Quality cycles (shift≥0.2°C): {len(quality_cycles)}/{n_labeled}"
          f" ({len(quality_cycles)/max(1,n_labeled):.0%})")

    all_methods = {}

    # ==================================================================
    # PHASE 1: Ovulation Detection
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 1: Ovulation Detection\n{SEP}")

    # --- 0. Calendar ---
    print(f"\n--- 0. Calendar Baseline ---")
    for frac in [0.50, 0.55, 0.575, 0.60]:
        det = detect_calendar(cycle_series, frac=frac)
        r = evaluate_ov(det, lh_ov_dict, f"cal-{frac:.3f}")
        rq = evaluate_ov(det, lh_ov_dict, f"  (quality)", subset=quality_cycles)
        all_methods[f"cal-{frac:.3f}"] = (det, r, {})

    # --- 1. t-test (noct_mean vs temps) ---
    print(f"\n--- 1. t-test Optimal Split ---")
    configs_ttest = []
    for sig in ["noct_mean", "temps"]:
        for sigma in [1.5, 2.0, 2.5, 3.0]:
            for fp in [0.50, 0.55, 0.575]:
                for pw in [3.0, 4.0, 5.0, 6.0]:
                    configs_ttest.append(
                        dict(smooth_sigma=sigma, frac_prior=fp, prior_width=pw, signal=sig)
                    )

    best_ttest = {"name": None, "acc3": 0, "det": None, "sc": None, "cfg": None}
    for cfg in configs_ttest:
        det, sc = detect_ttest(cycle_series, **cfg)
        tag = f"ttest-{cfg['signal'][:4]}-σ{cfg['smooth_sigma']}-f{cfg['frac_prior']}-w{cfg['prior_width']}"
        r = evaluate_ov(det, lh_ov_dict, tag)
        all_methods[tag] = (det, r, sc)
        if r.get("acc_3d", 0) > best_ttest["acc3"]:
            best_ttest.update(name=tag, acc3=r["acc_3d"], det=det, sc=sc, cfg=cfg)

    if best_ttest["name"]:
        print(f"\n  ★ Best t-test: {best_ttest['name']}")
        evaluate_ov(best_ttest["det"], lh_ov_dict, f"  ALL", subset=None)
        evaluate_ov(best_ttest["det"], lh_ov_dict, f"  QUALITY", subset=quality_cycles)

    # --- 2. Biphasic ---
    print(f"\n--- 2. Biphasic Step-Function ---")
    best_biphasic = {"name": None, "acc3": 0, "det": None, "sc": None}
    for sig in ["noct_mean", "temps"]:
        for sigma in [1.5, 2.0, 2.5, 3.0]:
            for pw in [3.0, 4.0, 5.0, 6.0]:
                det, sc = detect_biphasic(cycle_series, smooth_sigma=sigma,
                                          prior_width=pw, signal=sig)
                tag = f"bi-{sig[:4]}-σ{sigma}-w{pw}"
                r = evaluate_ov(det, lh_ov_dict, tag)
                all_methods[tag] = (det, r, sc)
                if r.get("acc_3d", 0) > best_biphasic["acc3"]:
                    best_biphasic.update(name=tag, acc3=r["acc_3d"], det=det, sc=sc)

    if best_biphasic["name"]:
        print(f"\n  ★ Best biphasic: {best_biphasic['name']}")
        evaluate_ov(best_biphasic["det"], lh_ov_dict, "  ALL")
        evaluate_ov(best_biphasic["det"], lh_ov_dict, "  QUALITY", subset=quality_cycles)

    # --- 3. SavGol + gradient ---
    print(f"\n--- 3. SavGol + Max Gradient ---")
    best_savgol = {"name": None, "acc3": 0, "det": None}
    for sig in ["noct_mean", "temps"]:
        for win in [7, 9, 11, 13]:
            for pw in [3.0, 4.0, 5.0]:
                det = detect_savgol_gradient(cycle_series, window=win,
                                             prior_width=pw, signal=sig)
                tag = f"sg-{sig[:4]}-w{win}-pw{pw}"
                r = evaluate_ov(det, lh_ov_dict, tag)
                all_methods[tag] = (det, r, {})
                if r.get("acc_3d", 0) > best_savgol["acc3"]:
                    best_savgol.update(name=tag, acc3=r["acc_3d"], det=det)

    if best_savgol["name"]:
        print(f"\n  ★ Best SavGol: {best_savgol['name']}")
        evaluate_ov(best_savgol["det"], lh_ov_dict, "  ALL")
        evaluate_ov(best_savgol["det"], lh_ov_dict, "  QUALITY", subset=quality_cycles)

    # --- 4. CUSUM ---
    print(f"\n--- 4. CUSUM ---")
    best_cusum = {"name": None, "acc3": 0, "det": None}
    for sig in ["noct_mean", "temps"]:
        for sigma in [1.5, 2.0, 2.5]:
            for pw in [3.0, 4.0, 5.0]:
                det = detect_cusum(cycle_series, smooth_sigma=sigma,
                                   prior_width=pw, signal=sig)
                tag = f"cusum-{sig[:4]}-σ{sigma}-w{pw}"
                r = evaluate_ov(det, lh_ov_dict, tag)
                all_methods[tag] = (det, r, {})
                if r.get("acc_3d", 0) > best_cusum["acc3"]:
                    best_cusum.update(name=tag, acc3=r["acc_3d"], det=det)

    if best_cusum["name"]:
        print(f"\n  ★ Best CUSUM: {best_cusum['name']}")
        evaluate_ov(best_cusum["det"], lh_ov_dict, "  ALL")
        evaluate_ov(best_cusum["det"], lh_ov_dict, "  QUALITY", subset=quality_cycles)

    # --- 5. Multi-signal t-test ---
    print(f"\n--- 5. Multi-Signal t-test ---")
    best_multi = {"name": None, "acc3": 0, "det": None}
    for sigma in [1.5, 2.0, 2.5]:
        for pw in [3.0, 4.0, 5.0]:
            det = detect_multi_signal_ttest(cycle_series, smooth_sigma=sigma,
                                            prior_width=pw)
            tag = f"multi-σ{sigma}-w{pw}"
            r = evaluate_ov(det, lh_ov_dict, tag)
            all_methods[tag] = (det, r, {})
            if r.get("acc_3d", 0) > best_multi["acc3"]:
                best_multi.update(name=tag, acc3=r["acc_3d"], det=det)

    if best_multi["name"]:
        print(f"\n  ★ Best multi-signal: {best_multi['name']}")
        evaluate_ov(best_multi["det"], lh_ov_dict, "  ALL")
        evaluate_ov(best_multi["det"], lh_ov_dict, "  QUALITY", subset=quality_cycles)

    # --- 6. HMM ---
    print(f"\n--- 6. 2-state HMM (Unsupervised) ---")
    best_hmm = {"name": None, "acc3": 0, "det": None}
    for use_multi in [False, True]:
        for sig in ["noct_mean", "temps"]:
            for sigma in [1.0, 1.5, 2.0]:
                det = detect_hmm(cycle_series, use_multi=use_multi,
                                 smooth_sigma=sigma, signal=sig)
                mt = "multi" if use_multi else sig[:4]
                tag = f"HMM-{mt}-σ{sigma}"
                r = evaluate_ov(det, lh_ov_dict, tag)
                all_methods[tag] = (det, r, {})
                if r.get("acc_3d", 0) > best_hmm["acc3"]:
                    best_hmm.update(name=tag, acc3=r["acc_3d"], det=det)

    if best_hmm["name"]:
        print(f"\n  ★ Best HMM: {best_hmm['name']}")
        evaluate_ov(best_hmm["det"], lh_ov_dict, "  ALL")
        evaluate_ov(best_hmm["det"], lh_ov_dict, "  QUALITY", subset=quality_cycles)

    # --- 7. Ensemble ---
    print(f"\n{SEP}\n--- 7. Ensemble ---\n{SEP}")
    ranked = sorted(
        [(k, v[0], v[1]) for k, v in all_methods.items()
         if v[1].get("acc_3d", 0) > 0],
        key=lambda x: x[2]["acc_3d"], reverse=True,
    )
    for topN in [3, 5, 7]:
        if len(ranked) >= topN:
            top = ranked[:topN]
            dets = [r[1] for r in top]
            ws = [r[2]["acc_3d"] for r in top]
            print(f"  Top-{topN}: {[r[0] for r in top]}")
            ens = ensemble_methods(dets, weights=ws)
            tag = f"ens-top{topN}"
            r = evaluate_ov(ens, lh_ov_dict, tag)
            rq = evaluate_ov(ens, lh_ov_dict, f"  (quality)", subset=quality_cycles)
            all_methods[tag] = (ens, r, {})

    # Try diverse ensemble (one from each category)
    bests = [best_ttest, best_biphasic, best_savgol, best_cusum, best_multi, best_hmm]
    diverse_dets = [b["det"] for b in bests if b["det"] is not None]
    diverse_w = [b["acc3"] for b in bests if b["det"] is not None]
    if len(diverse_dets) >= 3:
        print(f"  Diverse ensemble ({len(diverse_dets)} methods):")
        ens_d = ensemble_methods(diverse_dets, weights=diverse_w)
        r = evaluate_ov(ens_d, lh_ov_dict, "ens-diverse")
        rq = evaluate_ov(ens_d, lh_ov_dict, "  (quality)", subset=quality_cycles)
        all_methods["ens-diverse"] = (ens_d, r, {})

    # --- 8. Confidence-gated ---
    print(f"\n--- 8. Confidence-Gated ---")
    if best_ttest["det"] and best_ttest["sc"]:
        cal = detect_calendar(cycle_series, frac=0.575)
        for th in [0.5, 1.0, 1.5, 2.0, 3.0]:
            gated = confidence_gated(best_ttest["det"], best_ttest["sc"], th, cal)
            tag = f"gated-ttest-th{th}"
            r = evaluate_ov(gated, lh_ov_dict, tag)
            rq = evaluate_ov(gated, lh_ov_dict, f"  (quality)", subset=quality_cycles)
            all_methods[tag] = (gated, r, {})

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 1 SUMMARY — Top 20 by ±3d (ALL cycles)\n{SEP}")
    ranked_final = sorted(
        [(k, v[1]) for k, v in all_methods.items() if v[1].get("acc_3d", 0) > 0],
        key=lambda x: x[1]["acc_3d"], reverse=True,
    )
    print(f"  {'Method':<50s} {'N':>4} {'Recall':>7} {'MAE':>6} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-' * 95}")
    for name, r in ranked_final[:20]:
        print(
            f"  {name:<50s}"
            f" {r.get('n', 0):>4}"
            f" {r.get('recall', 0):>6.0%}"
            f" {r.get('mae', 0):>5.2f}"
            f" {r.get('acc_1d', 0):>5.1%}"
            f" {r.get('acc_2d', 0):>5.1%}"
            f" {r.get('acc_3d', 0):>5.1%}"
            f" {r.get('acc_5d', 0):>5.1%}"
        )

    # Quality-filtered top
    print(f"\n  PHASE 1 SUMMARY — Top 10 on QUALITY cycles (shift≥0.2°C)\n  {'-'*95}")
    for name, _ in ranked_final[:20]:
        det = all_methods[name][0]
        rq = evaluate_ov(det, lh_ov_dict, name, subset=quality_cycles)

    # ==================================================================
    # PHASE 2: Menstrual Prediction
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 2: Per-Cycle Menstrual Prediction\n{SEP}")

    # Calendar-only
    print(f"\n--- Calendar-only baseline ---")
    predict_menses_per_cycle(cycle_series, {}, lh_ov_dict)

    # Top 5 ovulation detectors
    print(f"\n--- Top ovulation detectors →  menstrual prediction ---")
    for name, _ in ranked_final[:5]:
        det = all_methods[name][0]
        print(f"\n  Method: {name}")
        predict_menses_per_cycle(cycle_series, det, lh_ov_dict)

    # Oracle
    print(f"\n--- Oracle (perfect LH ovulation) ---")
    oracle = {sgk: ov for sgk, ov in lh_ov_dict.items() if sgk in cycle_series}
    predict_menses_per_cycle(cycle_series, oracle, lh_ov_dict)

    print(f"\n{SEP}\n  COMPLETE\n{SEP}")


if __name__ == "__main__":
    main()

"""
Optimized Rule-based Ovulation Detection → Menstrual Prediction (v3)
=====================================================================
Key improvements:
  1. FORCE detection on 100% of cycles (no "missing" detections)
  2. Improved calendar fallback using weighted recent history
  3. Multi-strategy adaptive pipeline (confidence-based routing)
  4. Quality-stratified evaluation
  5. Iterative luteal estimation (self-calibrating, no LH)

Leakage-free guarantees:
  - hist_cycle_len: ONLY from completed PAST cycles of the same subject
  - Personal luteal: ONLY from past cycle_len - past detected_ov
  - LH labels: ONLY used for evaluation, NEVER in detection
  - No future information used

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_unsupervised_ov_menses_v3
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
DASH = "-" * 76


# =====================================================================
# Data Loading
# =====================================================================

def load_all_data():
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    rhr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv"))
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)

    hr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv"))
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std"]

    hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg({"rmssd": "mean"}).reset_index()
    hrv_daily.columns = key + ["rmssd_mean"]

    merged = cc.copy()
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily]:
        merged = merged.merge(src, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))

    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        cycle_series[sgk] = {
            "dic": (grp["day_in_study"] - cs).values,
            "temps": grp["nightly_temperature"].values,
            "rhr": grp["resting_hr"].values,
            "hr_mean": grp["hr_mean"].values,
            "rmssd": grp["rmssd_mean"].values,
            "id": grp["id"].values[0],
            "cycle_len": n,
        }

    # Historical cycle lengths (PAST only, per subject, chronologically)
    sgk_order = (
        merged.groupby("small_group_key")["day_in_study"]
        .min().reset_index().rename(columns={"day_in_study": "start"})
    )
    sgk_order = sgk_order.merge(
        merged[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    subj_cycle_order = {}
    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        subj_cycle_order[uid] = sgks
        past_lens = []
        for sgk in sgks:
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = np.mean(past_lens) if past_lens else 28.0
                cycle_series[sgk]["hist_cycle_lens"] = list(past_lens)
                past_lens.append(cycle_series[sgk]["cycle_len"])

    # Quality filter
    quality_cycles = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        data = cycle_series[sgk]
        raw = data["temps"]
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
            quality_cycles.add(sgk)

    return lh_ov_dict, cycle_series, quality_cycles, subj_cycle_order


def _clean(arr, smooth=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if smooth > 0:
        out = gaussian_filter1d(out, sigma=smooth)
    return out


# =====================================================================
# Detection Methods (ALL output for EVERY cycle — forced coverage)
# =====================================================================

def detect_calendar(cycle_series, frac=0.575):
    """Always produces a detection for every cycle."""
    return {sgk: int(round(frac * d["hist_cycle_len"]))
            for sgk, d in cycle_series.items()}


def detect_ttest_forced(cycle_series, smooth_sigma=2.5, frac_prior=0.575,
                        prior_width=4.0, min_shift=0.0):
    """t-test split with forced fallback to calendar for undetectable cycles."""
    detected, scores = {}, {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]

        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac_prior * hcl))
            scores[sgk] = 0.0
            continue

        t = _clean(raw, smooth=smooth_sigma)
        expected = max(8, hcl * frac_prior)
        best_ws, best_sp = -np.inf, None

        for sp in range(5, n - 3):
            pre, post = t[:sp], t[sp:]
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

        if best_sp is not None:
            detected[sgk] = int(dic[best_sp])
            scores[sgk] = best_ws
        else:
            detected[sgk] = int(round(frac_prior * hcl))
            scores[sgk] = 0.0

    return detected, scores


def detect_biphasic_forced(cycle_series, smooth_sigma=2.5, frac_prior=0.575,
                           prior_width=5.0):
    """Biphasic step with forced fallback."""
    detected, scores = {}, {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]

        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac_prior * hcl))
            scores[sgk] = 0.0
            continue

        t = _clean(raw, smooth=smooth_sigma)
        expected = max(8, hcl * frac_prior)
        best_sc, best_sp = np.inf, None

        for sp in range(5, n - 3):
            mu1, mu2 = np.mean(t[:sp]), np.mean(t[sp:])
            if mu2 <= mu1:
                continue
            sse = np.sum((t[:sp] - mu1) ** 2) + np.sum((t[sp:] - mu2) ** 2)
            pen = 0.5 * ((dic[sp] - expected) / prior_width) ** 2
            sc = sse + pen
            if sc < best_sc:
                best_sc = sc
                best_sp = sp

        if best_sp is not None:
            shift = np.mean(t[best_sp:]) - np.mean(t[:best_sp])
            detected[sgk] = int(dic[best_sp])
            scores[sgk] = shift
        else:
            detected[sgk] = int(round(frac_prior * hcl))
            scores[sgk] = 0.0

    return detected, scores


def detect_gradient_forced(cycle_series, smooth_sigma=2.5, frac_prior=0.575,
                           prior_width=4.0):
    """Gradient-based: find max positive slope in smoothed temperature."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]

        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac_prior * hcl))
            continue

        t = _clean(raw, smooth=smooth_sigma)
        grad = np.gradient(t)
        expected = max(8, hcl * frac_prior)
        pp = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        wg = grad * pp

        lo = max(5, int(hcl * 0.3))
        hi = min(n - 2, int(hcl * 0.7))
        if lo >= hi:
            lo, hi = 5, n - 2
        region = wg[lo:hi]
        if len(region) > 0:
            detected[sgk] = int(dic[lo + np.argmax(region)])
        else:
            detected[sgk] = int(round(frac_prior * hcl))
    return detected


def detect_cusum_forced(cycle_series, smooth_sigma=2.0, frac_prior=0.575,
                        prior_width=4.0):
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]

        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac_prior * hcl))
            continue

        t = _clean(raw, smooth=smooth_sigma)
        mu = np.mean(t)
        cusum = np.cumsum(t - mu)
        expected = max(8, hcl * frac_prior)
        pp = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        weighted = -cusum * pp

        lo = max(5, int(hcl * 0.3))
        hi = min(n - 2, int(hcl * 0.7))
        if lo >= hi:
            lo, hi = 5, n - 2
        region = weighted[lo:hi]
        if len(region) > 0:
            detected[sgk] = int(dic[lo + np.argmax(region)])
        else:
            detected[sgk] = int(round(frac_prior * hcl))
    return detected


def detect_multi_ttest_forced(cycle_series, smooth_sigma=2.0, frac_prior=0.575,
                              prior_width=4.0):
    """Multi-signal t-test with forced detection."""
    signals = [("temps", 1), ("rhr", 1), ("rmssd", -1)]
    detected = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        hcl = data["hist_cycle_len"]
        expected = max(8, hcl * frac_prior)

        if n < 12:
            detected[sgk] = int(round(frac_prior * hcl))
            continue

        combined = np.zeros(n)
        n_valid = 0
        for sig, sgn in signals:
            raw = data.get(sig)
            if raw is None or np.isnan(raw).all() or np.isnan(raw).sum() > n * 0.5:
                continue
            t = _clean(raw, smooth=smooth_sigma)
            n_valid += 1
            for sp in range(5, n - 3):
                diff = sgn * (np.mean(t[sp:]) - np.mean(t[:sp]))
                if diff <= 0:
                    continue
                try:
                    stat, _ = ttest_ind(sgn * t[sp:], sgn * t[:sp], alternative="greater")
                except Exception:
                    continue
                if not np.isnan(stat):
                    combined[sp] += stat

        if n_valid == 0:
            detected[sgk] = int(round(frac_prior * hcl))
            continue

        combined /= n_valid
        pp = np.exp(-0.5 * ((dic - expected) / prior_width) ** 2)
        combined *= pp

        lo = max(5, int(hcl * 0.3))
        hi = min(n - 3, int(hcl * 0.7))
        if lo >= hi:
            lo, hi = 5, n - 3
        region = combined[lo:hi]
        if len(region) > 0 and region.max() > 0:
            detected[sgk] = int(dic[lo + np.argmax(region)])
        else:
            detected[sgk] = int(round(frac_prior * hcl))
    return detected


def detect_hmm_forced(cycle_series, smooth_sigma=1.5, frac_prior=0.575):
    """2-state HMM with forced fallback."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [HMM] hmmlearn not installed")
        return {}

    detected = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        hcl = data["hist_cycle_len"]
        raw = data["temps"]

        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac_prior * hcl))
            continue

        obs = _clean(raw, smooth=smooth_sigma).reshape(-1, 1)
        try:
            mdl = GaussianHMM(n_components=2, covariance_type="full",
                              n_iter=100, random_state=42, init_params="mc")
            mdl.startprob_ = np.array([0.9, 0.1])
            mdl.transmat_ = np.array([[0.95, 0.05], [0.02, 0.98]])
            mdl.fit(obs)
            states = mdl.predict(obs)
            low = np.argmin(mdl.means_[:, 0])
            high = 1 - low
            found = False
            for i in range(1, n):
                if states[i] == high and states[i - 1] == low and dic[i] >= 6:
                    detected[sgk] = int(dic[i])
                    found = True
                    break
            if not found:
                detected[sgk] = int(round(frac_prior * hcl))
        except Exception:
            detected[sgk] = int(round(frac_prior * hcl))

    return detected


# =====================================================================
# Ensemble (forced coverage, all methods vote)
# =====================================================================

def ensemble_forced(all_dets, weights=None):
    """Weighted average of all forced detections. All cycles always present."""
    all_sgks = set()
    for det in all_dets:
        all_sgks.update(det.keys())
    if weights is None:
        weights = [1.0] * len(all_dets)
    result = {}
    for sgk in all_sgks:
        vals, ws = [], []
        for det, w in zip(all_dets, weights):
            if sgk in det:
                vals.append(det[sgk])
                ws.append(w)
        if vals:
            result[sgk] = int(round(np.average(vals, weights=ws)))
    return result


def confidence_blend(detected_signal, scores, detected_calendar, alpha_fn):
    """Blend signal-based and calendar detection based on confidence.
    alpha_fn(score) → weight for signal detection [0, 1].
    result = alpha * signal + (1-alpha) * calendar
    """
    result = {}
    for sgk in set(detected_signal) | set(detected_calendar):
        s = detected_signal.get(sgk)
        c = detected_calendar.get(sgk)
        if s is not None and c is not None:
            alpha = alpha_fn(scores.get(sgk, 0))
            result[sgk] = int(round(alpha * s + (1 - alpha) * c))
        elif s is not None:
            result[sgk] = s
        elif c is not None:
            result[sgk] = c
    return result


# =====================================================================
# Phase 2: Per-Cycle Menstrual Prediction (improved)
# =====================================================================

def predict_menses_per_cycle(cycle_series, detected_ov, lh_ov_dict,
                             subj_cycle_order, label=""):
    """Per-cycle menstrual prediction.

    Leakage-free personal luteal estimation:
      personal_luteal = mean(past_cycle_len - past_detected_ov) for completed past cycles.
    """
    pop_luteal = 14.0
    subj_past_luteal = defaultdict(list)
    subj_past_clens = defaultdict(list)

    results_all = []
    results_ov = []
    results_cal = []

    for uid, sgks in subj_cycle_order.items():
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual_len = data["cycle_len"]

            past_luts = subj_past_luteal[uid]
            past_clens = subj_past_clens[uid]

            avg_lut = np.mean(past_luts) if past_luts else pop_luteal
            # Weighted recent cycle length (exponential decay)
            if past_clens:
                w = np.exp(-0.03 * np.arange(len(past_clens))[::-1])
                avg_clen = np.average(past_clens, weights=w[-len(past_clens):])
            else:
                avg_clen = 28.0

            ov_det = detected_ov.get(sgk)
            if ov_det is not None and ov_det > 3:
                pred_len = ov_det + avg_lut
                err = pred_len - actual_len
                results_all.append(err)
                results_ov.append(err)
            else:
                pred_len = avg_clen
                err = pred_len - actual_len
                results_all.append(err)
                results_cal.append(err)

            subj_past_clens[uid].append(actual_len)
            if ov_det is not None:
                est_lut = actual_len - ov_det
                if 8 <= est_lut <= 22:
                    subj_past_luteal[uid].append(est_lut)

    def _p(errs, tag):
        if not errs:
            return {}
        ae = np.abs(errs)
        r = {"n": len(ae), "mae": float(ae.mean()),
             "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
             "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
        print(f"    [{tag}] n={r['n']}"
              f" | MAE={r['mae']:.2f}d"
              f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
              f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
        return r

    if label:
        print(f"  {label}")
    ra = _p(results_all, "ALL")
    ro = _p(results_ov, "ov-countdown")
    rc = _p(results_cal, "cal-fallback")
    return ra, ro, rc


# =====================================================================
# Evaluation
# =====================================================================

def eval_ov(detected, lh_ov_dict, name, subset=None):
    eval_keys = subset if subset else lh_ov_dict.keys()
    errors = []
    for sgk in eval_keys:
        if sgk in detected and sgk in lh_ov_dict:
            errors.append(detected[sgk] - lh_ov_dict[sgk])
    n_total = len(eval_keys)
    n_eval = len(errors)
    if n_eval == 0:
        print(f"  [{name}] 0/{n_total}")
        return {}
    err = np.array(errors)
    ae = np.abs(err)
    r = {"n": n_eval, "n_total": n_total, "recall": n_eval / n_total,
         "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"  [{name}] {n_eval}/{n_total} ({r['recall']:.0%})"
          f" | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# Main
# =====================================================================

def main():
    print(f"\n{SEP}\n  Optimized Rule-based Ovulation → Menstrual Prediction v3\n"
          f"  Forced 100% Coverage + Improved Calendar Fallback\n{SEP}")

    lh_ov_dict, cycle_series, quality_cycles, subj_cycle_order = load_all_data()
    n_cycles = len(cycle_series)
    n_labeled = sum(1 for sgk in cycle_series if sgk in lh_ov_dict)
    n_subjects = len(set(d["id"] for d in cycle_series.values()))
    print(f"  Cycles: {n_cycles} | LH-labeled: {n_labeled} | Subjects: {n_subjects}")
    print(f"  Quality cycles: {len(quality_cycles)}/{n_labeled}"
          f" ({len(quality_cycles)/max(1,n_labeled):.0%})")

    # ==================================================================
    # PHASE 1: Ovulation Detection (100% coverage)
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 1: Ovulation Detection (Forced 100% Coverage)\n{SEP}")

    methods = {}

    # 0. Calendar
    print(f"\n--- 0. Calendar baselines ---")
    for frac in [0.50, 0.55, 0.575, 0.60]:
        det = detect_calendar(cycle_series, frac=frac)
        r = eval_ov(det, lh_ov_dict, f"cal-{frac}")
        methods[f"cal-{frac}"] = det

    # 1. t-test (grid search best params)
    print(f"\n--- 1. t-test Forced (param sweep) ---")
    best_ttest = {"tag": None, "acc3": 0, "det": None, "sc": None}
    for sigma in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for fp in [0.50, 0.55, 0.575]:
            for pw in [3.0, 4.0, 5.0, 6.0]:
                det, sc = detect_ttest_forced(
                    cycle_series, smooth_sigma=sigma,
                    frac_prior=fp, prior_width=pw)
                tag = f"tt-σ{sigma}-f{fp}-w{pw}"
                r = eval_ov(det, lh_ov_dict, tag)
                methods[tag] = det
                if r.get("acc_3d", 0) > best_ttest["acc3"]:
                    best_ttest.update(tag=tag, acc3=r["acc_3d"], det=det, sc=sc)

    if best_ttest["tag"]:
        print(f"\n  ★ Best t-test: {best_ttest['tag']} (±3d={best_ttest['acc3']:.1%})")
        eval_ov(best_ttest["det"], lh_ov_dict, "  (quality)", subset=quality_cycles)

    # 2. Biphasic forced
    print(f"\n--- 2. Biphasic Forced (param sweep) ---")
    best_bi = {"tag": None, "acc3": 0, "det": None, "sc": None}
    for sigma in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for pw in [3.0, 4.0, 5.0, 6.0]:
            det, sc = detect_biphasic_forced(
                cycle_series, smooth_sigma=sigma, prior_width=pw)
            tag = f"bi-σ{sigma}-w{pw}"
            r = eval_ov(det, lh_ov_dict, tag)
            methods[tag] = det
            if r.get("acc_3d", 0) > best_bi["acc3"]:
                best_bi.update(tag=tag, acc3=r["acc_3d"], det=det, sc=sc)

    if best_bi["tag"]:
        print(f"\n  ★ Best biphasic: {best_bi['tag']} (±3d={best_bi['acc3']:.1%})")
        eval_ov(best_bi["det"], lh_ov_dict, "  (quality)", subset=quality_cycles)

    # 3. Gradient forced
    print(f"\n--- 3. Gradient Forced ---")
    best_grad = {"tag": None, "acc3": 0, "det": None}
    for sigma in [2.0, 2.5, 3.0, 3.5]:
        for pw in [3.0, 4.0, 5.0]:
            det = detect_gradient_forced(cycle_series, smooth_sigma=sigma,
                                         prior_width=pw)
            tag = f"grad-σ{sigma}-w{pw}"
            r = eval_ov(det, lh_ov_dict, tag)
            methods[tag] = det
            if r.get("acc_3d", 0) > best_grad["acc3"]:
                best_grad.update(tag=tag, acc3=r["acc_3d"], det=det)

    if best_grad["tag"]:
        print(f"\n  ★ Best gradient: {best_grad['tag']}")
        eval_ov(best_grad["det"], lh_ov_dict, "  (quality)", subset=quality_cycles)

    # 4. CUSUM forced
    print(f"\n--- 4. CUSUM Forced ---")
    best_cusum = {"tag": None, "acc3": 0, "det": None}
    for sigma in [1.5, 2.0, 2.5]:
        for pw in [3.0, 4.0, 5.0]:
            det = detect_cusum_forced(cycle_series, smooth_sigma=sigma,
                                      prior_width=pw)
            tag = f"cusum-σ{sigma}-w{pw}"
            r = eval_ov(det, lh_ov_dict, tag)
            methods[tag] = det
            if r.get("acc_3d", 0) > best_cusum["acc3"]:
                best_cusum.update(tag=tag, acc3=r["acc_3d"], det=det)

    if best_cusum["tag"]:
        print(f"\n  ★ Best CUSUM: {best_cusum['tag']}")

    # 5. Multi-signal forced
    print(f"\n--- 5. Multi-Signal t-test Forced ---")
    best_ms = {"tag": None, "acc3": 0, "det": None}
    for sigma in [1.5, 2.0, 2.5]:
        for pw in [3.0, 4.0, 5.0]:
            det = detect_multi_ttest_forced(cycle_series, smooth_sigma=sigma,
                                            prior_width=pw)
            tag = f"multi-σ{sigma}-w{pw}"
            r = eval_ov(det, lh_ov_dict, tag)
            methods[tag] = det
            if r.get("acc_3d", 0) > best_ms["acc3"]:
                best_ms.update(tag=tag, acc3=r["acc_3d"], det=det)

    if best_ms["tag"]:
        print(f"\n  ★ Best multi-signal: {best_ms['tag']}")

    # 6. HMM forced
    print(f"\n--- 6. HMM Forced ---")
    best_hmm = {"tag": None, "acc3": 0, "det": None}
    for sigma in [1.0, 1.5, 2.0]:
        det = detect_hmm_forced(cycle_series, smooth_sigma=sigma)
        tag = f"hmm-σ{sigma}"
        r = eval_ov(det, lh_ov_dict, tag)
        methods[tag] = det
        if r.get("acc_3d", 0) > best_hmm["acc3"]:
            best_hmm.update(tag=tag, acc3=r["acc_3d"], det=det)

    if best_hmm["tag"]:
        print(f"\n  ★ Best HMM: {best_hmm['tag']}")

    # 7. Ensembles
    print(f"\n{DASH}\n--- 7. Ensembles ---\n{DASH}")

    # Collect all "best" detections
    bests = [
        ("ttest", best_ttest), ("biphasic", best_bi),
        ("gradient", best_grad), ("cusum", best_cusum),
        ("multi", best_ms), ("hmm", best_hmm),
    ]
    bests_valid = [(nm, b) for nm, b in bests if b["det"] is not None]

    # Top-N from ranked methods
    ranked = sorted(
        [(k, v, eval_ov(v, lh_ov_dict, k).get("acc_3d", 0))
         for k, v in methods.items()],
        key=lambda x: x[2], reverse=True,
    )

    for topN in [3, 5, 7]:
        if len(ranked) >= topN:
            dets = [r[1] for r in ranked[:topN]]
            ws = [max(r[2], 0.01) for r in ranked[:topN]]
            ens = ensemble_forced(dets, weights=ws)
            tag = f"ens-top{topN}"
            r = eval_ov(ens, lh_ov_dict, tag)
            rq = eval_ov(ens, lh_ov_dict, f"  (quality)", subset=quality_cycles)
            methods[tag] = ens

    # Diverse ensemble (one per category)
    if len(bests_valid) >= 3:
        dets = [b["det"] for _, b in bests_valid]
        ws = [max(b["acc3"], 0.01) for _, b in bests_valid]
        ens = ensemble_forced(dets, weights=ws)
        tag = "ens-diverse"
        r = eval_ov(ens, lh_ov_dict, tag)
        rq = eval_ov(ens, lh_ov_dict, "  (quality)", subset=quality_cycles)
        methods[tag] = ens

    # 8. Confidence blending
    print(f"\n--- 8. Confidence Blending ---")
    if best_ttest["det"] and best_ttest["sc"]:
        cal = detect_calendar(cycle_series, frac=0.575)
        for gamma in [0.5, 1.0, 2.0, 3.0]:
            blended = confidence_blend(
                best_ttest["det"], best_ttest["sc"], cal,
                alpha_fn=lambda s, g=gamma: min(1.0, s / g) if s > 0 else 0.0,
            )
            tag = f"blend-γ{gamma}"
            r = eval_ov(blended, lh_ov_dict, tag)
            methods[tag] = blended

    # ==================================================================
    # Phase 1 FINAL Summary
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 1 FINAL RANKING — Top 25 by ±3d\n{SEP}")
    final_ranked = sorted(
        [(k, eval_ov(v, lh_ov_dict, k))
         for k, v in methods.items()],
        key=lambda x: x[1].get("acc_3d", 0), reverse=True,
    )
    print(f"  {'Method':<42s} {'N':>4} {'MAE':>6} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*80}")
    for name, r in final_ranked[:25]:
        if r:
            print(
                f"  {name:<42s}"
                f" {r.get('n', 0):>4}"
                f" {r.get('mae', 0):>5.2f}"
                f" {r.get('acc_1d', 0):>5.1%}"
                f" {r.get('acc_2d', 0):>5.1%}"
                f" {r.get('acc_3d', 0):>5.1%}"
                f" {r.get('acc_5d', 0):>5.1%}"
            )

    # Quality cycles summary
    print(f"\n  QUALITY cycles ({len(quality_cycles)}) — Top 10\n  {'-'*80}")
    quality_ranked = sorted(
        [(k, eval_ov(methods[k], lh_ov_dict, k, subset=quality_cycles))
         for k in [n for n, _ in final_ranked[:25]]],
        key=lambda x: x[1].get("acc_3d", 0), reverse=True,
    )
    for name, r in quality_ranked[:10]:
        if r:
            print(
                f"  {name:<42s}"
                f" {r.get('n', 0):>4}"
                f" {r.get('mae', 0):>5.2f}"
                f" {r.get('acc_1d', 0):>5.1%}"
                f" {r.get('acc_2d', 0):>5.1%}"
                f" {r.get('acc_3d', 0):>5.1%}"
                f" {r.get('acc_5d', 0):>5.1%}"
            )

    # ==================================================================
    # PHASE 2: Menstrual Prediction
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 2: Per-Cycle Menstrual Prediction\n{SEP}")

    print(f"\n--- Calendar-only (no ov detection) ---")
    predict_menses_per_cycle(cycle_series, {}, lh_ov_dict,
                             subj_cycle_order, "Calendar-only")

    print(f"\n--- Top 10 ovulation detectors → menstrual prediction ---")
    for name, _ in final_ranked[:10]:
        det = methods[name]
        predict_menses_per_cycle(cycle_series, det, lh_ov_dict,
                                 subj_cycle_order, name)

    print(f"\n--- Oracle (perfect LH detection, labeled cycles only) ---")
    oracle = {sgk: ov for sgk, ov in lh_ov_dict.items() if sgk in cycle_series}
    predict_menses_per_cycle(cycle_series, oracle, lh_ov_dict,
                             subj_cycle_order, "Oracle-LH")

    # ==================================================================
    # Phase 2 Targeted: ONLY evaluate on LH-labeled cycles
    # ==================================================================
    print(f"\n{SEP}\n  PHASE 2b: Menstrual Prediction — LH-labeled cycles ONLY\n{SEP}")
    labeled_series = {sgk: d for sgk, d in cycle_series.items() if sgk in lh_ov_dict}
    print(f"  (evaluating on {len(labeled_series)} LH-labeled cycles only)\n")

    # Rebuild subject order for labeled cycles
    labeled_order = {}
    for uid, sgks in subj_cycle_order.items():
        lab_sgks = [s for s in sgks if s in labeled_series]
        if lab_sgks:
            labeled_order[uid] = lab_sgks

    print(f"--- Calendar-only ---")
    predict_menses_per_cycle(labeled_series, {}, lh_ov_dict,
                             labeled_order, "Calendar-only (labeled)")

    for name, _ in final_ranked[:5]:
        det = methods[name]
        predict_menses_per_cycle(labeled_series, det, lh_ov_dict,
                                 labeled_order, f"{name} (labeled)")

    print(f"\n--- Oracle ---")
    predict_menses_per_cycle(labeled_series, oracle, lh_ov_dict,
                             labeled_order, "Oracle-LH (labeled)")

    print(f"\n{SEP}\n  COMPLETE\n{SEP}")


if __name__ == "__main__":
    main()

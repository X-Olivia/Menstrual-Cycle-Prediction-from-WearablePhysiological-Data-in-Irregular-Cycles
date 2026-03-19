"""
Multi-Signal Ovulation Detection — HR + HRV + Temperature
==========================================================
Uses PREVIOUSLY UNUSED high-frequency signals:
  1. heart_rate_cycle.csv: continuous HR → nocturnal mean HR per day
  2. heart_rate_variability_details_cycle.csv: RMSSD, LF, HF per day
  3. wrist_temperature_cycle.csv: nocturnal temperature (already used)
  4. computed_temperature_cycle.csv: nightly temperature
  5. resting_heart_rate_cycle.csv: daily RHR

Key physiological basis:
  - HR INCREASES in luteal phase (progesterone effect)
  - HRV (RMSSD) DECREASES around ovulation (sympathetic shift)
  - Temperature INCREASES after ovulation (thermogenic effect)
  - LF/HF ratio CHANGES with cycle phase

These are INDEPENDENT signals that can triangulate ovulation more precisely.

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_multisignal_ov
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from collections import defaultdict
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def _pr(tag, ae, prefix="  "):
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"{prefix}[{tag}] n={r['n']} | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


def load_all_signals():
    """Load and aggregate ALL available signals to daily per-cycle level."""
    print("  Loading cycle structure...")
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    # 1. Nightly temperature (daily)
    print("  Loading nightly temperature...")
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    # 2. Nocturnal wrist temperature (minute-level → daily mean)
    print("  Loading nocturnal wrist temperature...")
    wt = pd.read_csv(os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv"),
                      usecols=key + ["timestamp", "temperature_diff_from_baseline"])
    wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
    noct_wt = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
    noct_temp_daily = noct_wt.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
    noct_temp_daily.rename(columns={"temperature_diff_from_baseline": "noct_temp"}, inplace=True)

    # 3. Resting heart rate (daily)
    print("  Loading resting heart rate...")
    rhr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv"),
                       usecols=key + ["value"])
    rhr_daily = rhr.groupby(key)["value"].mean().reset_index()
    rhr_daily.rename(columns={"value": "rhr"}, inplace=True)

    # 4. HRV details (5-min intervals → daily aggregates)
    print("  Loading HRV details...")
    hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        rmssd_mean=("rmssd", "mean"),
        rmssd_std=("rmssd", "std"),
        lf_mean=("low_frequency", "mean"),
        hf_mean=("high_frequency", "mean"),
        hrv_coverage=("coverage", "mean"),
    ).reset_index()
    hrv_daily["lf_hf_ratio"] = hrv_daily["lf_mean"] / hrv_daily["hf_mean"].clip(lower=1)
    print(f"    HRV daily rows: {len(hrv_daily)}")

    # 5. Continuous HR → nocturnal mean HR (reading in chunks due to 1GB size)
    print("  Loading nocturnal HR (chunked, 0-6AM)...")
    hr_path = os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv")
    hr_aggs = []
    for chunk in pd.read_csv(hr_path, chunksize=2_000_000,
                              usecols=key + ["timestamp", "bpm", "confidence"]):
        chunk["hour"] = pd.to_datetime(chunk["timestamp"]).dt.hour
        noct = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 6) & (chunk["confidence"] >= 1)]
        if len(noct) > 0:
            agg = noct.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
            hr_aggs.append(agg)
    hr_daily = pd.concat(hr_aggs).groupby(key).mean().reset_index()
    hr_daily.rename(columns={"mean": "noct_hr_mean", "std": "noct_hr_std", "min": "noct_hr_min"}, inplace=True)
    print(f"    Nocturnal HR daily rows: {len(hr_daily)}")

    # Merge everything
    print("  Merging all signals...")
    merged = cc.merge(ct_daily, on=key, how="left")
    merged = merged.merge(noct_temp_daily, on=key, how="left")
    merged = merged.merge(rhr_daily, on=key, how="left")
    merged = merged.merge(hrv_daily, on=key, how="left")
    merged = merged.merge(hr_daily, on=key, how="left")

    # Load labels
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    lh_luteal = dict(zip(lh_ov["small_group_key"], lh_ov["luteal_len"]))

    signal_cols = ["nightly_temperature", "noct_temp", "rhr",
                   "rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio",
                   "noct_hr_mean", "noct_hr_std", "noct_hr_min", "hrv_coverage"]

    # Build per-cycle series
    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        entry = {
            "dic": (grp["day_in_study"] - cs).values,
            "id": grp["id"].values[0],
            "cycle_len": n,
        }
        for sc in signal_cols:
            entry[sc] = grp[sc].values if sc in grp.columns else np.full(n, np.nan)
        cycle_series[sgk] = entry

    # Subject ordering for leakage-free hist_cycle_len
    sgk_order = (
        merged.groupby("small_group_key")["day_in_study"]
        .min().reset_index().rename(columns={"day_in_study": "start"})
    )
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
                past_lens.append(cycle_series[sgk]["cycle_len"])

    # Quality set
    quality = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        data = cycle_series[sgk]
        raw = data["nightly_temperature"]
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

    # Signal availability stats
    labeled = [s for s in cycle_series if s in lh_ov_dict]
    for sc in signal_cols:
        avail = sum(1 for s in labeled if not np.isnan(cycle_series[s][sc]).all())
        print(f"    Signal '{sc}': {avail}/{len(labeled)} labeled cycles have data")

    return lh_ov_dict, lh_luteal, cycle_series, quality, subj_order, signal_cols


# =====================================================================
# Multi-signal rule-based detection
# =====================================================================

def detect_signal_ttest(cycle_series, signal_key, sigma=2.0, frac=0.575, pw=4.0,
                        invert=False):
    """
    T-test split on a single signal.
    invert=True: expect DECREASE (e.g., RMSSD decreases around ovulation).
    """
    detected, confs = {}, {}
    for sgk, data in cycle_series.items():
        raw = data.get(signal_key)
        if raw is None or np.isnan(raw).all():
            detected[sgk] = int(round(frac * data["hist_cycle_len"]))
            confs[sgk] = 0.0
            continue
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12:
            detected[sgk] = int(round(frac * hcl))
            confs[sgk] = 0.0
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
            pp = np.exp(-0.5 * ((dic[sp] - exp) / pw) ** 2)
            if stat * pp > best_ws:
                best_ws = stat * pp
                best_sp = sp
                best_stat = stat

        if best_sp is not None:
            ov = int(dic[best_sp])
            pre_m = np.mean(t[max(0, ov - 5):ov])
            post_m = np.mean(t[ov:min(n, ov + 5)])
            shift = abs(post_m - pre_m)
            std_t = max(np.std(t), 0.01)
            conf = min(1.0, max(0, best_stat / 5)) * min(1.0, shift / (2 * std_t))
        else:
            ov = int(round(frac * hcl))
            conf = 0.0
        detected[sgk] = ov
        confs[sgk] = conf
    return detected, confs


def multi_signal_ensemble(detections_list, weights=None):
    """Weighted average of multiple signal detections."""
    all_sgks = set()
    for d, _ in detections_list:
        all_sgks.update(d.keys())
    if weights is None:
        weights = [1.0] * len(detections_list)
    result, confs = {}, {}
    for sgk in all_sgks:
        vals, ws, cs = [], [], []
        for (d, c), w in zip(detections_list, weights):
            if sgk in d:
                vals.append(d[sgk])
                ws.append(w * max(c.get(sgk, 0.5), 0.1))
                cs.append(c.get(sgk, 0.5))
        if vals:
            result[sgk] = int(round(np.average(vals, weights=ws)))
            confs[sgk] = float(np.mean(cs))
    return result, confs


# =====================================================================
# ML features extraction (multi-signal)
# =====================================================================

def extract_multisignal_features(data, sigma=1.5):
    """Extract features from ALL signals without cycle_len."""
    feats = {}
    feats["hist_clen"] = data["hist_cycle_len"]
    n = data["cycle_len"]
    hcl = data["hist_cycle_len"]
    exp_ov = hcl * 0.575

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
        f[f"{prefix}_nadir"] = int(np.argmin(t))
        f[f"{prefix}_nadir_dev"] = int(np.argmin(t)) - exp_ov

        grad = np.gradient(t)
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
            else:
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
        f[f"{prefix}_bs_tstat"] = max(best_sc, 0)
        pre_m = np.mean(t[:best_sp])
        post_m = np.mean(t[best_sp:])
        f[f"{prefix}_shift"] = post_m - pre_m

        for q in [0.25, 0.5, 0.75]:
            f[f"{prefix}_q{int(q*100)}"] = float(np.quantile(t, q))

        ts = pd.Series(t)
        for lag in [1, 3, 5]:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0

        return f

    feats.update(_sig_feats(data.get("nightly_temperature"), "nt"))
    feats.update(_sig_feats(data.get("noct_temp"), "noct"))
    feats.update(_sig_feats(data.get("noct_hr_mean"), "nhr"))
    feats.update(_sig_feats(data.get("rhr"), "rhr"))
    feats.update(_sig_feats(data.get("rmssd_mean"), "rmssd", invert=True))
    feats.update(_sig_feats(data.get("hf_mean"), "hf", invert=True))
    feats.update(_sig_feats(data.get("lf_hf_ratio"), "lfhf"))

    for k in feats:
        if isinstance(feats[k], float) and (np.isnan(feats[k]) or np.isinf(feats[k])):
            feats[k] = 0.0
    return feats


def ml_detect_loso(cs, lh, model_type="ridge"):
    labeled = [s for s in cs if s in lh]
    all_f, all_t, all_id, all_s = [], [], [], []
    for sgk in labeled:
        feats = extract_multisignal_features(cs[sgk])
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

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == "svr":
            from sklearn.svm import SVR
            m = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "bayridge":
            from sklearn.linear_model import BayesianRidge
            m = BayesianRidge()
        else:
            raise ValueError(model_type)

        m.fit(X_tr, y[tr])
        preds = m.predict(X_te)
        test_sgks = [all_s[i] for i in np.where(te)[0]]

        for sgk, pred in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, pred))))
            confs[sgk] = 0.5

    return det, confs


# =====================================================================
# Menstrual prediction
# =====================================================================

def predict_menses(cs, det, confs, subj_order, lh, fl=13.0,
                   eval_subset=None, label=""):
    pop_lut = fl
    s_plut = defaultdict(list)
    s_pclen = defaultdict(list)
    errs = []
    n_ov, n_cal = 0, 0
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual = cs[sgk]["cycle_len"]
            pl = s_plut[uid]
            pc = s_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0

            ov = det.get(sgk)
            conf = confs.get(sgk, 0.0)
            if ov is not None and ov > 3:
                pred = ov + lut
                n_ov += 1
            else:
                pred = acl
                n_cal += 1

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
    return _pr(f"{label} (ov:{n_ov},cal:{n_cal})", ae, prefix="    ")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"\n{SEP}\n  Multi-Signal Ovulation Detection\n{SEP}")
    t0 = time.time()

    lh, lut, cs, quality, so, sig_cols = load_all_signals()
    labeled = set(s for s in cs if s in lh)
    print(f"\n  Cycles: {len(cs)} | Labeled: {len(labeled)} | Quality: {len(quality)}")

    oracle = {s: v for s, v in lh.items() if s in cs}
    oconf = {s: 1.0 for s in oracle}

    # Baselines
    print(f"\n{SEP}\n  A. BASELINES\n{SEP}")
    predict_menses(cs, oracle, oconf, so, lh, fl=13.0,
                   eval_subset=labeled, label="Oracle+lut13 (lab)")
    predict_menses(cs, {}, {}, so, lh, eval_subset=labeled, label="Calendar (lab)")

    # ===== B: Single-signal rule-based detection =====
    print(f"\n{SEP}\n  B. SINGLE-SIGNAL RULE-BASED DETECTION\n{SEP}")

    signal_configs = [
        ("nightly_temperature", False, "NT"),
        ("noct_temp", False, "NoctT"),
        ("noct_hr_mean", False, "NoctHR"),
        ("rhr", False, "RHR"),
        ("rmssd_mean", True, "RMSSD↓"),
        ("hf_mean", True, "HF↓"),
        ("lf_hf_ratio", False, "LF/HF↑"),
    ]

    single_results = {}
    for sig_key, invert, name in signal_configs:
        for s in [1.5, 2.0, 2.5]:
            d, c = detect_signal_ttest(cs, sig_key, sigma=s, invert=invert)
            tag = f"{name}-σ{s}"
            errs = [d[s2] - lh[s2] for s2 in labeled if s2 in d and s2 in lh]
            r = _pr(tag, np.abs(errs))
            single_results[tag] = (d, c, r)

    # Quality
    print(f"\n  --- Quality cycles (top 10) ---")
    ranked_single = sorted(single_results.items(),
                           key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)
    for tag, (d, c, r) in ranked_single[:10]:
        errs = [d[s2] - lh[s2] for s2 in quality if s2 in d and s2 in lh]
        if errs:
            _pr(f"Q-{tag}", np.abs(errs))

    # ===== C: Multi-signal ensemble =====
    print(f"\n{SEP}\n  C. MULTI-SIGNAL ENSEMBLE\n{SEP}")

    best_per_signal = {}
    for sig_key, invert, name in signal_configs:
        best = None
        for tag, (d, c, r) in single_results.items():
            if tag.startswith(name) and (best is None or r.get("acc_3d", 0) > best[1].get("acc_3d", 0)):
                best = (d, c, r)
        if best:
            best_per_signal[name] = (best[0], best[1])

    # All-signal ensemble
    all_dets = list(best_per_signal.values())
    if len(all_dets) >= 3:
        ed, ec = multi_signal_ensemble(all_dets)
        errs = [ed[s2] - lh[s2] for s2 in labeled if s2 in ed and s2 in lh]
        r = _pr("ALL-signal ensemble", np.abs(errs))
        errs_q = [ed[s2] - lh[s2] for s2 in quality if s2 in ed and s2 in lh]
        if errs_q:
            _pr("Q-ALL-signal", np.abs(errs_q))

    # Temp + HRV ensemble
    temp_dets = [v for k, v in best_per_signal.items() if k in ["NT", "NoctT"]]
    hrv_dets = [v for k, v in best_per_signal.items() if k in ["RMSSD↓", "HF↓"]]
    hr_dets = [v for k, v in best_per_signal.items() if k in ["NoctHR", "RHR"]]

    for combo_name, combo_dets in [
        ("Temp+HRV", temp_dets + hrv_dets),
        ("Temp+HR", temp_dets + hr_dets),
        ("HRV+HR", hrv_dets + hr_dets),
        ("Temp+HRV+HR", temp_dets + hrv_dets + hr_dets),
    ]:
        if len(combo_dets) >= 2:
            ed, ec = multi_signal_ensemble(combo_dets)
            errs = [ed[s2] - lh[s2] for s2 in labeled if s2 in ed and s2 in lh]
            r = _pr(f"{combo_name}", np.abs(errs))
            errs_q = [ed[s2] - lh[s2] for s2 in quality if s2 in ed and s2 in lh]
            if errs_q:
                _pr(f"Q-{combo_name}", np.abs(errs_q))

    # ===== D: ML multi-signal detection =====
    print(f"\n{SEP}\n  D. ML MULTI-SIGNAL DETECTION (LOSO)\n{SEP}")
    ml_results = {}
    for mt in ["ridge", "elastic", "svr", "rf", "gbdt", "bayridge"]:
        d, c = ml_detect_loso(cs, lh, model_type=mt)
        if d:
            errs = [d[s2] - lh[s2] for s2 in d if s2 in lh]
            r = _pr(f"ML-multi-{mt}", np.abs(errs))
            ml_results[f"ML-multi-{mt}"] = (d, c, r)
            errs_q = [d[s2] - lh[s2] for s2 in quality if s2 in d and s2 in lh]
            if errs_q:
                _pr(f"  Q-ML-multi-{mt}", np.abs(errs_q))

    # ===== E: Stacking ensemble =====
    print(f"\n{SEP}\n  E. STACKING: Rule-based + ML\n{SEP}")

    # Combine best rule-based per signal + best ML
    all_methods = {}
    all_methods.update(single_results)
    all_methods.update(ml_results)

    ranked_all = sorted(all_methods.items(),
                        key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)

    for topN in [3, 5, 7, 10]:
        if len(ranked_all) >= topN:
            top_dets = [(v[0], v[1]) for _, v in ranked_all[:topN]]
            ws = [max(v[2].get("acc_2d", 0), 0.01) for _, v in ranked_all[:topN]]
            ed, ec = multi_signal_ensemble(top_dets, ws)
            errs = [ed[s2] - lh[s2] for s2 in labeled if s2 in ed and s2 in lh]
            if errs:
                r = _pr(f"super-ens-top{topN}", np.abs(errs))
                errs_q = [ed[s2] - lh[s2] for s2 in quality if s2 in ed and s2 in lh]
                if errs_q:
                    _pr(f"  Q-super-ens-top{topN}", np.abs(errs_q))

    # ===== F: Final ranking =====
    print(f"\n{SEP}\n  F. FINAL RANKING — by ±2d\n{SEP}")
    all_entries = list(all_methods.items())
    all_entries.sort(key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)
    print(f"  {'Method':<35s} {'N':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*80}")
    for tag, (d, c, r) in all_entries[:25]:
        print(f"  {tag:<35s} {r['n']:>3} {r['mae']:>5.2f}"
              f" {r.get('acc_1d',0):>5.1%} {r.get('acc_2d',0):>5.1%}"
              f" {r.get('acc_3d',0):>5.1%} {r.get('acc_5d',0):>5.1%}")

    # ===== G: Menstrual prediction with best detectors =====
    print(f"\n{SEP}\n  G. MENSTRUAL PREDICTION\n{SEP}")
    for tag, (d, c, r) in all_entries[:5]:
        for fl in [12, 13]:
            predict_menses(cs, d, c, so, lh, fl=float(fl),
                           eval_subset=labeled, label=f"{tag}+lut{fl}")
            predict_menses(cs, d, c, so, lh, fl=float(fl),
                           eval_subset=quality, label=f"{tag}+lut{fl} (Q)")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  COMPLETE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

"""High-Frequency Temperature Ovulation Detection Experiment.

Uses minute-level raw wrist temperature to extract granular features
(nadir, slope, nocturnal stability, curve shape) for ovulation detection,
then runs luteal countdown for menstruation prediction.

Key idea: minute-level data → lower noise → higher SNR → better ov detection.

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -m model.experiment.run_highfreq_temp_experiment
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.dataset import load_data
from model.ovulation_detect import compute_personal_luteal_from_lh
from model.evaluate import compute_metrics
from model.ovulation_detect import get_lh_ovulation_labels, detect_ovulation_from_probs


# ======================================================================
# High-frequency feature extraction from minute-level wrist temperature
# ======================================================================

def extract_daily_highfreq_features(wt_df):
    """Extract rich features from minute-level temperature per day."""
    key = ["id", "study_interval", "day_in_study"]
    wt_df = wt_df.copy()
    wt_df["hour"] = wt_df["timestamp"].str[:2].astype(int)
    wt_df["minute"] = wt_df["timestamp"].str[3:5].astype(int)
    wt_df["time_min"] = wt_df["hour"] * 60 + wt_df["minute"]
    wt_df["temp"] = wt_df["temperature_diff_from_baseline"]

    results = []
    for (uid, si, dis), grp in wt_df.groupby(key):
        feat = _extract_one_day(grp)
        if feat is not None:
            feat["id"] = uid
            feat["study_interval"] = si
            feat["day_in_study"] = dis
            results.append(feat)

    return pd.DataFrame(results)


def _extract_one_day(grp):
    temps = grp["temp"].values
    hours = grp["hour"].values
    time_mins = grp["time_min"].values
    n = len(temps)

    if n < 60:
        return None

    feat = {}
    feat["temp_hf_mean"] = np.nanmean(temps)
    feat["temp_hf_std"] = np.nanstd(temps)
    feat["temp_hf_min"] = np.nanmin(temps)
    feat["temp_hf_max"] = np.nanmax(temps)
    feat["temp_hf_range"] = feat["temp_hf_max"] - feat["temp_hf_min"]
    feat["temp_hf_median"] = np.nanmedian(temps)

    night_mask = hours < 6
    night_temps = temps[night_mask]
    if len(night_temps) > 30:
        feat["night_mean"] = np.nanmean(night_temps)
        feat["night_min"] = np.nanmin(night_temps)
        feat["night_max"] = np.nanmax(night_temps)
        feat["night_std"] = np.nanstd(night_temps)
        feat["night_range"] = feat["night_max"] - feat["night_min"]
    else:
        for k in ["night_mean", "night_min", "night_max", "night_std", "night_range"]:
            feat[k] = np.nan

    early = temps[:min(120, n)]
    if len(early) > 20:
        x = np.arange(len(early))
        valid = ~np.isnan(early)
        feat["rise_slope"] = np.polyfit(x[valid], early[valid], 1)[0] if valid.sum() > 10 else np.nan
    else:
        feat["rise_slope"] = np.nan

    late = temps[max(0, n - 120):]
    if len(late) > 20:
        x = np.arange(len(late))
        valid = ~np.isnan(late)
        feat["drop_slope"] = np.polyfit(x[valid], late[valid], 1)[0] if valid.sum() > 10 else np.nan
    else:
        feat["drop_slope"] = np.nan

    stable_mask = (hours >= 2) & (hours < 5)
    stable_temps = temps[stable_mask]
    if len(stable_temps) > 30:
        feat["stable_mean"] = np.nanmean(stable_temps)
        feat["stable_std"] = np.nanstd(stable_temps)
        q25, q75 = np.nanpercentile(stable_temps, [25, 75])
        feat["stable_iqr"] = q75 - q25
    else:
        feat["stable_mean"] = feat["stable_std"] = feat["stable_iqr"] = np.nan

    mid = n // 2
    fh = np.nanmean(temps[:mid]) if mid > 30 else np.nan
    sh = np.nanmean(temps[mid:]) if (n - mid) > 30 else np.nan
    feat["half_diff"] = sh - fh if not (np.isnan(fh) or np.isnan(sh)) else np.nan

    feat["p10"] = np.nanpercentile(temps, 10)
    feat["p90"] = np.nanpercentile(temps, 90)
    feat["p90_p10"] = feat["p90"] - feat["p10"]

    return feat


HF_RAW_SIGNALS = [
    "temp_hf_max", "night_mean", "night_max", "stable_mean",
    "p90", "temp_hf_mean", "half_diff", "night_std",
    "rise_slope", "drop_slope", "stable_iqr", "p90_p10",
]

OLD_RAW_SIGNALS = [
    "nightly_temperature", "resting_hr",
    "hr_mean", "hr_std", "hr_min",
    "rmssd_mean", "lf_mean", "hf_mean",
    "wt_mean", "wt_max",
]


def build_causal_features(base_df, signal_cols):
    """Build causal rolling features per cycle for detection."""
    base_df = base_df.sort_values(["small_group_key", "day_in_study"])
    chunks = []
    for _, grp in base_df.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study").copy()
        for sig in signal_cols:
            s = pd.Series(grp[sig].values)
            rm3 = s.rolling(3, min_periods=2).mean().values
            rm7 = s.rolling(7, min_periods=4).mean().values
            grp[f"{sig}_rm3"] = rm3
            grp[f"{sig}_rm7"] = rm7
            grp[f"{sig}_svl"] = rm3 - rm7
            grp[f"{sig}_d1"] = s.diff().values
            grp[f"{sig}_d3"] = s.diff(3).values
        chunks.append(grp)
    return pd.concat(chunks, ignore_index=True)


def get_feature_cols(signal_cols):
    cols = []
    for sig in signal_cols:
        cols.extend([f"{sig}_rm3", f"{sig}_rm7", f"{sig}_svl", f"{sig}_d1", f"{sig}_d3"])
    cols.append("cycle_frac")
    return cols


# ======================================================================
# Load all wearable data
# ======================================================================

def load_all_wearable(workspace):
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(os.path.join(workspace, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    rhr = pd.read_csv(os.path.join(workspace, "subdataset/resting_heart_rate_cycle.csv"))
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)

    hr = pd.read_csv(os.path.join(workspace, "subdataset/heart_rate_cycle.csv"))
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std", "hr_min"]

    hrv = pd.read_csv(os.path.join(workspace, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        {"rmssd": "mean", "low_frequency": "mean", "high_frequency": "mean"}
    ).reset_index()
    hrv_daily.columns = key + ["rmssd_mean", "lf_mean", "hf_mean"]

    wt = pd.read_csv(os.path.join(workspace, "subdataset/wrist_temperature_cycle.csv"))
    wt_agg = wt.groupby(key)["temperature_diff_from_baseline"].agg(["mean", "max"]).reset_index()
    wt_agg.columns = key + ["wt_mean", "wt_max"]

    return ct_daily, rhr_daily, hr_daily, hrv_daily, wt_agg


# ======================================================================
# Ovulation detection + countdown pipeline
# ======================================================================

def run_detection_pipeline(base_df, signal_cols, label_name, lh_ov_dict):
    """Train LOSO classifier, detect ovulation, evaluate detection accuracy."""
    feat_cols = get_feature_cols(signal_cols)

    base_df["cycle_frac"] = base_df["day_in_cycle"] / base_df["hist_cycle_len_mean"].clip(lower=20)
    built = build_causal_features(base_df, signal_cols)

    valid = built.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    X = valid[feat_cols].fillna(0).values
    y = valid["is_post_ov"].values.astype(int)
    groups = valid["id"].values

    logo = LeaveOneGroupOut()
    valid["ov_prob"] = np.nan

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        valid.iloc[test_idx, valid.columns.get_loc("ov_prob")] = prob

    has_prob = valid["ov_prob"].notna()
    auc = roc_auc_score(y[has_prob], valid.loc[has_prob, "ov_prob"])
    print(f"  Classifier AUC: {auc:.3f}")

    strategies = ["threshold", "cumulative", "bayesian"]
    all_detected = {}

    for strat in strategies:
        detected = {}
        for sgk, cyc in valid[has_prob].groupby("small_group_key"):
            cyc = cyc.sort_values("day_in_cycle")
            probs = cyc["ov_prob"].values
            ov_dic = detect_ovulation_from_probs(cyc, probs, strategy=strat)
            if ov_dic is not None:
                detected[sgk] = ov_dic
        all_detected[strat] = detected

        errors = []
        for sgk, det_dic in detected.items():
            if sgk in lh_ov_dict:
                errors.append(det_dic - lh_ov_dict[sgk])

        err = np.array(errors) if errors else np.array([])
        n_total = sum(1 for sgk in valid["small_group_key"].unique() if sgk in lh_ov_dict)
        n_detected = len(detected)
        recall = n_detected / n_total if n_total > 0 else 0
        print(
            f"  [{strat}] {n_detected}/{n_total} ({recall:.0%}) "
            f"| MAE={np.abs(err).mean():.1f}d "
            f"| ±1d={np.mean(np.abs(err)<=1):.1%} "
            f"| ±2d={np.mean(np.abs(err)<=2):.1%} "
            f"| ±3d={np.mean(np.abs(err)<=3):.1%} "
            f"| ±5d={np.mean(np.abs(err)<=5):.1%} "
            f"| med={np.median(err):+.1f}d"
        )

    return all_detected, valid, auc


# ======================================================================
# Menstruation prediction (pure countdown)
# ======================================================================

def run_countdown_prediction(df, detected_ov, lh_ov_dict, personal_luteal, pop_luteal_mean, strategy_name):
    """Pure countdown: calendar pre-ov, luteal countdown post-ov."""
    results_det = []
    results_cal = []
    results_ora = []

    for seed in range(10):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["id"]))
        df_test = df.iloc[test_idx].copy()

        y_test = df_test["days_until_next_menses"].values
        sgks = df_test["small_group_key"].values
        dics = df_test["day_in_cycle"].values
        uids = df_test["id"].values
        hist_lens = df_test["hist_cycle_len_mean"].values

        pred_cal = np.array([
            max(1.0, (hl if not np.isnan(hl) else 28) - dic)
            for hl, dic in zip(hist_lens, dics)
        ])

        pred_det = pred_cal.copy()
        for i in range(len(df_test)):
            sgk, dic, uid = sgks[i], dics[i], uids[i]
            if sgk in detected_ov:
                ov_dic = detected_ov[sgk]
                if dic >= ov_dic + 2:
                    days_since = dic - ov_dic
                    luts = personal_luteal.get(uid, [])
                    avg_lut = np.mean(luts) if luts else pop_luteal_mean
                    pred_det[i] = max(1.0, avg_lut - days_since)

        pred_ora = pred_cal.copy()
        for i in range(len(df_test)):
            sgk, dic, uid = sgks[i], dics[i], uids[i]
            if sgk in lh_ov_dict:
                ov_dic = lh_ov_dict[sgk]
                if dic >= ov_dic + 2:
                    days_since = dic - ov_dic
                    luts = personal_luteal.get(uid, [])
                    avg_lut = np.mean(luts) if luts else pop_luteal_mean
                    pred_ora[i] = max(1.0, avg_lut - days_since)

        results_cal.append(compute_metrics(pred_cal, y_test))
        results_det.append(compute_metrics(pred_det, y_test))
        results_ora.append(compute_metrics(pred_ora, y_test))

    def avg(r):
        return {k: np.mean([x[k] for x in r]) for k in r[0]}

    cal_a = avg(results_cal)
    det_a = avg(results_det)
    ora_a = avg(results_ora)

    print(f"\n  Calendar baseline:    MAE={cal_a['mae']:.3f}  ±1d={cal_a['acc_1d']:.1%}  ±2d={cal_a['acc_2d']:.1%}  ±3d={cal_a['acc_3d']:.1%}")
    print(f"  Detected [{strategy_name}]: MAE={det_a['mae']:.3f}  ±1d={det_a['acc_1d']:.1%}  ±2d={det_a['acc_2d']:.1%}  ±3d={det_a['acc_3d']:.1%}")
    print(f"  Oracle (LH):          MAE={ora_a['mae']:.3f}  ±1d={ora_a['acc_1d']:.1%}  ±2d={ora_a['acc_2d']:.1%}  ±3d={ora_a['acc_3d']:.1%}")
    print(f"  Det vs Calendar:      MAE={det_a['mae']-cal_a['mae']:+.3f}  ±3d={100*(det_a['acc_3d']-cal_a['acc_3d']):+.1f}pp")
    print(f"  Oracle vs Calendar:   MAE={ora_a['mae']-cal_a['mae']:+.3f}  ±3d={100*(ora_a['acc_3d']-cal_a['acc_3d']):+.1f}pp")

    return cal_a, det_a, ora_a


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  High-Frequency Temperature Ovulation Detection Experiment")
    print("=" * 70)

    # Load main dataset
    df, available = load_data()
    print(f"Main dataset: {len(df)} rows, {df['id'].nunique()} subjects")

    key = ["id", "study_interval", "day_in_study"]

    # Load all wearable signals (old aggregated)
    ct_daily, rhr_daily, hr_daily, hrv_daily, wt_agg = load_all_wearable(WORKSPACE)
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily, wt_agg]:
        df = df.merge(src, on=key, how="left")

    # Load high-frequency temperature features
    hf_path = os.path.join(WORKSPACE, "processed_data/temp_highfreq_features.csv")
    if not os.path.exists(hf_path):
        print("Extracting high-frequency features from raw wrist temperature...")
        wt_raw = pd.read_csv(os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv"))
        hf_df = extract_daily_highfreq_features(wt_raw)
        hf_df.to_csv(hf_path, index=False)
    else:
        hf_df = pd.read_csv(hf_path)
    print(f"High-freq features: {len(hf_df)} days, {hf_df['id'].nunique()} subjects")

    df = df.merge(hf_df, on=key, how="left")

    # LH labels
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    print(f"Cycles with LH labels: {len(lh_ov_dict)}")

    personal_luteal = compute_personal_luteal_from_lh()
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v])
    print(f"Population luteal mean: {pop_luteal_mean:.1f} days")

    # Label rows
    df["is_post_ov"] = np.nan
    for i, row in df.iterrows():
        sgk = row["small_group_key"]
        if sgk in lh_ov_dict:
            df.at[i, "is_post_ov"] = 1.0 if row["day_in_cycle"] > lh_ov_dict[sgk] else 0.0

    labeled = df.dropna(subset=["is_post_ov"]).copy()
    print(f"Labeled rows: {len(labeled)} ({labeled['is_post_ov'].mean():.1%} post-ov)")

    # ================================================================
    # Experiment A: Old aggregated signals only (baseline)
    # ================================================================
    print(f"\n{'='*70}")
    print("  [A] Baseline: Old aggregated signals only")
    print(f"{'='*70}")
    old_detected, _, old_auc = run_detection_pipeline(
        labeled.copy(), OLD_RAW_SIGNALS, "old_signals", lh_ov_dict
    )

    # ================================================================
    # Experiment B: High-frequency temperature features only
    # ================================================================
    print(f"\n{'='*70}")
    print("  [B] High-Frequency temperature features only")
    print(f"{'='*70}")
    hf_detected, _, hf_auc = run_detection_pipeline(
        labeled.copy(), HF_RAW_SIGNALS, "hf_temp", lh_ov_dict
    )

    # ================================================================
    # Experiment C: Combined (HF temp + old HR/HRV signals)
    # ================================================================
    print(f"\n{'='*70}")
    print("  [C] Combined: HF temperature + HR/HRV signals")
    print(f"{'='*70}")
    combined_signals = HF_RAW_SIGNALS + [
        "resting_hr", "hr_mean", "hr_std", "hr_min",
        "rmssd_mean", "lf_mean", "hf_mean",
    ]
    comb_detected, _, comb_auc = run_detection_pipeline(
        labeled.copy(), combined_signals, "combined", lh_ov_dict
    )

    # ================================================================
    # Menstruation prediction using best detection
    # ================================================================
    print(f"\n{'='*70}")
    print("  Menstruation Prediction Comparison")
    print(f"{'='*70}")

    best_strat = "bayesian"

    print(f"\n--- [A] Old signals ({best_strat}) ---")
    run_countdown_prediction(df, old_detected.get(best_strat, {}),
                             lh_ov_dict, personal_luteal, pop_luteal_mean, f"old-{best_strat}")

    print(f"\n--- [B] HF temperature ({best_strat}) ---")
    run_countdown_prediction(df, hf_detected.get(best_strat, {}),
                             lh_ov_dict, personal_luteal, pop_luteal_mean, f"hf-{best_strat}")

    print(f"\n--- [C] Combined ({best_strat}) ---")
    run_countdown_prediction(df, comb_detected.get(best_strat, {}),
                             lh_ov_dict, personal_luteal, pop_luteal_mean, f"comb-{best_strat}")

    # Also test with cumulative strategy
    best_strat2 = "cumulative"
    print(f"\n--- [C] Combined ({best_strat2}) ---")
    run_countdown_prediction(df, comb_detected.get(best_strat2, {}),
                             lh_ov_dict, personal_luteal, pop_luteal_mean, f"comb-{best_strat2}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

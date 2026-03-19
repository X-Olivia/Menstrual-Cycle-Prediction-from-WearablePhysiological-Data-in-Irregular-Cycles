"""Oracle + Luteal Countdown Experiment (Menstrual Prediction).

Compares:
  1. LightGBM only (full-cycle regression).
  2. Detected-ovulation hybrid: ovulation after detection → luteal countdown; else LightGBM.
  3. Oracle hybrid: ovulation from LH ground truth → luteal countdown (post-ov); else LightGBM.

The "Oracle" part is: for days after ovulation, use LH true ovulation day + (personal or
population) luteal length to predict days_until_next_menses; no detection error.

All inputs use new_workspace data by default:
  - cycle_cleaned_ov.csv, processed_dataset/signals/*.csv
  - daily_features_v4.csv from new_workspace if present, else main_workspace (fallback).

Usage:
  cd /Users/xujing/FYP/new_workspace
  python record/oracle_luteal_countdown_experiment.py

  Or from repo root:
  python new_workspace/record/oracle_luteal_countdown_experiment.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths: new_workspace data ─────────────────────────────────────────
NEW_WS = Path(__file__).resolve().parent.parent
MAIN_WS = NEW_WS.parent / "main_workspace"
PROCESSED = NEW_WS / "processed_dataset"
SIGNALS_DIR = PROCESSED / "signals"
CYCLE_OV_CSV = PROCESSED / "cycle_cleaned_ov.csv"
FEATURES_V4_NEW = NEW_WS / "processed_dataset" / "daily_features" / "daily_features_v4.csv"

sys.path.insert(0, str(MAIN_WS))
import model.config as config
config.CYCLE_CSV = str(CYCLE_OV_CSV)
config.WORKSPACE = str(NEW_WS)
if FEATURES_V4_NEW.exists():
    config.FEATURES_CSV = str(FEATURES_V4_NEW)
else:
    config.FEATURES_CSV = os.path.join(config.WORKSPACE, "processed_data", "v4", "daily_features_v4.csv")
    if not os.path.isfile(config.FEATURES_CSV):
        config.FEATURES_CSV = os.path.join(MAIN_WS, "processed_data", "v4", "daily_features_v4.csv")

from model.dataset import load_data
from model.config import CYCLE_CSV, WORKSPACE, ALL_FEATURES, LGB_PARAMS_TUNED
from model.ovulation_detect import (
    compute_personal_luteal_from_lh,
    get_lh_ovulation_labels,
    detect_ovulation_from_probs,
)
from model.evaluate import compute_metrics
from model.train_lgb import train_lightgbm


# ======================================================================
# Load wearable from new_workspace processed_dataset/signals
# ======================================================================

def load_wearable_daily(signals_dir: Path):
    """Load and aggregate wearable signals to day level from signals_dir (e.g. processed_dataset/signals)."""
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(signals_dir / "computed_temperature_cycle.csv")
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    rhr = pd.read_csv(signals_dir / "resting_heart_rate_cycle.csv")
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)

    hr = pd.read_csv(signals_dir / "heart_rate_cycle.csv")
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std", "hr_min"]

    hrv = pd.read_csv(signals_dir / "heart_rate_variability_details_cycle.csv")
    hrv_daily = hrv.groupby(key).agg(
        {"rmssd": "mean", "low_frequency": "mean", "high_frequency": "mean"}
    ).reset_index()
    hrv_daily.columns = key + ["rmssd_mean", "lf_mean", "hf_mean"]

    wt = pd.read_csv(signals_dir / "wrist_temperature_cycle.csv")
    wt_daily = wt.groupby(key)["temperature_diff_from_baseline"].agg(["mean", "max"]).reset_index()
    wt_daily.columns = key + ["wt_mean", "wt_max"]

    return ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily


# ======================================================================
# Causal features for ovulation classifier
# ======================================================================

RAW_SIGNALS = [
    "nightly_temperature", "hr_min", "hr_std", "lf_mean",
    "rmssd_mean", "wt_max", "hr_mean", "wt_mean", "hf_mean",
]


def build_ov_detection_features(base_df):
    """Build causal rolling features per cycle for ovulation detection."""
    base_df = base_df.sort_values(["small_group_key", "day_in_study"])
    chunks = []
    for sgk, grp in base_df.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study").copy()
        for sig in RAW_SIGNALS:
            if sig not in grp.columns:
                continue
            vals = grp[sig].values
            s = pd.Series(vals)
            rm3 = s.rolling(3, min_periods=2).mean().values
            rm7 = s.rolling(7, min_periods=4).mean().values
            grp[f"{sig}_rm3"] = rm3
            grp[f"{sig}_rm7"] = rm7
            grp[f"{sig}_svl"] = rm3 - rm7
            grp[f"{sig}_d1"] = s.diff().values
            grp[f"{sig}_d3"] = s.diff(3).values
        chunks.append(grp)
    return pd.concat(chunks, ignore_index=True)


def get_ov_feature_cols():
    cols = []
    for sig in RAW_SIGNALS:
        cols.extend([f"{sig}_rm3", f"{sig}_rm7", f"{sig}_svl", f"{sig}_d1", f"{sig}_d3"])
    cols.append("cycle_frac")
    return cols


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  Oracle + Luteal Countdown Experiment (new_workspace data)")
    print("=" * 70)
    print(f"  Cycle CSV:    {CYCLE_OV_CSV}")
    print(f"  Signals dir:  {SIGNALS_DIR}")
    print(f"  Features CSV: {config.FEATURES_CSV}")

    if not CYCLE_OV_CSV.exists():
        raise SystemExit(f"Cycle CSV not found: {CYCLE_OV_CSV}. Run data_clean.py then ovulation_labels.py.")
    if not SIGNALS_DIR.is_dir():
        raise SystemExit(f"Signals dir not found: {SIGNALS_DIR}. Run wearable_signals.py.")

    df, available = load_data()
    print(f"Main dataset: {len(df)} rows, {df['id'].nunique()} subjects")

    ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily = load_wearable_daily(SIGNALS_DIR)
    key = ["id", "study_interval", "day_in_study"]
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily]:
        df = df.merge(src, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    print(f"Cycles with LH ovulation labels: {len(lh_ov_dict)}")

    personal_luteal = compute_personal_luteal_from_lh(cycle_csv=str(CYCLE_OV_CSV))
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v]) if personal_luteal else 14.0
    print(f"Population luteal mean: {pop_luteal_mean:.1f} days")

    df["is_post_ov"] = np.nan
    for i, row in df.iterrows():
        sgk = row["small_group_key"]
        if sgk in lh_ov_dict:
            df.at[i, "is_post_ov"] = 1.0 if row["day_in_cycle"] > lh_ov_dict[sgk] else 0.0

    labeled = df.dropna(subset=["is_post_ov"])
    print(f"Labeled rows for ov-classifier: {len(labeled)} ({labeled['is_post_ov'].mean():.1%} post-ov)")

    df["cycle_frac"] = df["day_in_cycle"] / df["hist_cycle_len_mean"].clip(lower=20)
    labeled = df.dropna(subset=["is_post_ov"]).copy()
    print("Building causal detection features...")
    labeled = build_ov_detection_features(labeled)
    feat_cols = get_ov_feature_cols()

    valid = labeled.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    print(f"Valid rows after feature filtering: {len(valid)}")

    X_ov = valid[feat_cols].fillna(0).values
    y_ov = valid["is_post_ov"].values.astype(int)
    groups_ov = valid["id"].values

    print("\nTraining ovulation classifier (LOSO)...")
    logo = LeaveOneGroupOut()
    valid["ov_prob"] = np.nan
    for train_idx, test_idx in logo.split(X_ov, y_ov, groups_ov):
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_ov[train_idx], y_ov[train_idx])
        prob = clf.predict_proba(X_ov[test_idx])[:, 1]
        valid.iloc[test_idx, valid.columns.get_loc("ov_prob")] = prob

    has_prob = valid["ov_prob"].notna()
    if has_prob.sum() > 0 and y_ov[has_prob].std() > 0:
        auc = roc_auc_score(y_ov[has_prob], valid.loc[has_prob, "ov_prob"])
        print(f"Ovulation classifier LOSO AUC: {auc:.3f}")

    strategies = ["threshold", "cumulative", "bayesian"]
    all_detected = {}
    for strat in strategies:
        detected_ov = {}
        for sgk, cyc in valid[has_prob].groupby("small_group_key"):
            cyc = cyc.sort_values("day_in_cycle")
            probs = cyc["ov_prob"].values
            ov_dic = detect_ovulation_from_probs(cyc, probs, strategy=strat)
            if ov_dic is not None:
                detected_ov[sgk] = ov_dic
        all_detected[strat] = detected_ov
        errors = [detected_ov[sgk] - lh_ov_dict[sgk] for sgk in detected_ov if sgk in lh_ov_dict]
        err = np.array(errors) if errors else np.array([])
        n_lab = sum(1 for sgk in valid["small_group_key"].unique() if sgk in lh_ov_dict)
        if len(err) > 0:
            print(f"  [{strat}] Detected: {len(detected_ov)}/{n_lab} cycles  | Mean offset: {err.mean():.1f}d  | ±3d: {(np.abs(err) <= 3).mean():.1%}  | ±5d: {(np.abs(err) <= 5).mean():.1%}")
        else:
            print(f"  [{strat}] Detected: {len(detected_ov)}/{n_lab} cycles")

    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"\n{'='*70}")
    print(f"  Hybrid Model: 10-seed evaluation (features: {len(features)})")
    print(f"{'='*70}")

    for strat in strategies:
        detected_ov = all_detected[strat]
        results_hybrid, results_lgb, results_oracle = [], [], []

        for seed in range(10):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            train_idx, test_idx = next(gss.split(df, groups=df["id"]))
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()
            train_uids = set(df_train["id"].unique())

            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + 100)
            tr_idx, val_idx = next(gss2.split(df_train, groups=df_train["id"]))
            X_tr = df_train.iloc[tr_idx][features].values
            y_tr = df_train.iloc[tr_idx]["days_until_next_menses"].values
            X_val = df_train.iloc[val_idx][features].values
            y_val = df_train.iloc[val_idx]["days_until_next_menses"].values
            X_test = df_test[features].values
            y_test = df_test["days_until_next_menses"].values

            model = train_lightgbm(X_tr, y_tr, X_val, y_val, features, params=LGB_PARAMS_TUNED)
            pred_lgb = np.clip(model.predict(X_test), 1.0, None)
            results_lgb.append(compute_metrics(pred_lgb, y_test))

            sgks = df_test["small_group_key"].values
            dics = df_test["day_in_cycle"].values
            uids = df_test["id"].values

            pred_det = pred_lgb.copy()
            for i in range(len(df_test)):
                sgk, dic, uid = sgks[i], dics[i], uids[i]
                if sgk in detected_ov:
                    ov_dic = detected_ov[sgk]
                    if dic >= ov_dic + 2:
                        days_since = dic - ov_dic
                        luts = personal_luteal.get(uid, []) if uid in train_uids else []
                        avg_lut = np.mean(luts) if luts else pop_luteal_mean
                        pred_det[i] = max(1.0, avg_lut - days_since)
            results_hybrid.append(compute_metrics(pred_det, y_test))

            pred_ora = pred_lgb.copy()
            for i in range(len(df_test)):
                sgk, dic, uid = sgks[i], dics[i], uids[i]
                if sgk in lh_ov_dict:
                    ov_dic = lh_ov_dict[sgk]
                    if dic >= ov_dic + 2:
                        days_since = dic - ov_dic
                        luts = personal_luteal.get(uid, []) if uid in train_uids else []
                        avg_lut = np.mean(luts) if luts else pop_luteal_mean
                        pred_ora[i] = max(1.0, avg_lut - days_since)
            results_oracle.append(compute_metrics(pred_ora, y_test))

        def avg(results):
            return {k: np.mean([r[k] for r in results]) for k in results[0]}

        lgb_a, hyb_a, ora_a = avg(results_lgb), avg(results_hybrid), avg(results_oracle)
        print(f"\n--- Strategy: {strat} ---")
        print(f"  LightGBM only:        MAE={lgb_a['mae']:.3f}  ±3d={lgb_a['acc_3d']:.1%}")
        print(f"  Detected-ov hybrid:   MAE={hyb_a['mae']:.3f}  ±3d={hyb_a['acc_3d']:.1%}")
        print(f"  Oracle hybrid:        MAE={ora_a['mae']:.3f}  ±3d={ora_a['acc_3d']:.1%}")
        print(f"  Oracle vs LGB:        MAE={ora_a['mae']-lgb_a['mae']:+.3f}  ±3d={100*(ora_a['acc_3d']-lgb_a['acc_3d']):+.1f}pp")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

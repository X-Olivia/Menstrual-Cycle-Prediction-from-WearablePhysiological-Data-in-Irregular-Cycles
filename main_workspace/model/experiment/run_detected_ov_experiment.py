"""Detected-Ovulation Hybrid Experiment.

Pipeline:
  1. Train a multi-signal classifier (GBDT) to predict P(post-ovulation) per day
     using ONLY causal wearable features + cycle position prior.
  2. Apply a causal detection rule per cycle to estimate ovulation day.
  3. Plug detected ovulation days into the LightGBM + luteal-countdown hybrid model.
  4. Compare: pure LightGBM vs detected-ov hybrid vs Oracle hybrid.

All evaluation is leave-one-subject-out (LOSO) to avoid data leakage.

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -m model.experiment.run_detected_ov_experiment
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from model.dataset import load_data
from model.config import CYCLE_CSV, WORKSPACE, ALL_FEATURES, LGB_PARAMS_TUNED
from model.ovulation_detect import (
    _load_nightly_temp, compute_personal_luteal_from_lh,
    get_lh_ovulation_labels, detect_ovulation_from_probs,
)
from model.evaluate import compute_metrics
from model.train_lgb import train_lightgbm


# ======================================================================
# Step 1: Load all wearable data and build ovulation-aligned dataset
# ======================================================================

def load_wearable_daily(workspace):
    """Load and aggregate all wearable signals to day level."""
    import os
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
    wt_daily = wt.groupby(key)["temperature_diff_from_baseline"].agg(["mean", "max"]).reset_index()
    wt_daily.columns = key + ["wt_mean", "wt_max"]

    return ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily


# ======================================================================
# Step 2: Build causal features for ovulation classifier
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
            vals = grp[sig].values
            s = pd.Series(vals)
            rm3 = s.rolling(3, min_periods=2).mean().values
            rm7 = s.rolling(7, min_periods=4).mean().values
            grp[f"{sig}_rm3"] = rm3
            grp[f"{sig}_rm7"] = rm7
            grp[f"{sig}_svl"] = rm3 - rm7        # short vs long
            grp[f"{sig}_d1"] = s.diff().values    # 1-day diff
            grp[f"{sig}_d3"] = s.diff(3).values   # 3-day diff
        chunks.append(grp)

    return pd.concat(chunks, ignore_index=True)


def get_ov_feature_cols():
    cols = []
    for sig in RAW_SIGNALS:
        cols.extend([f"{sig}_rm3", f"{sig}_rm7", f"{sig}_svl", f"{sig}_d1", f"{sig}_d3"])
    cols.append("cycle_frac")  # cycle position prior
    return cols


# ======================================================================
# Main experiment
# ======================================================================

def main():
    print("=" * 70)
    print("  Detected-Ovulation Hybrid Experiment")
    print("=" * 70)

    # ── Load main features dataset ────────────────────────────────────
    df, available = load_data()
    print(f"Main dataset: {len(df)} rows, {df['id'].nunique()} subjects")

    # ── Load wearable signals ─────────────────────────────────────────
    ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily = load_wearable_daily(WORKSPACE)
    key = ["id", "study_interval", "day_in_study"]
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily]:
        df = df.merge(src, on=key, how="left")

    # ── LH ovulation labels ──────────────────────────────────────────
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    print(f"Cycles with LH ovulation labels: {len(lh_ov_dict)}")

    # ── Personal luteal lengths ───────────────────────────────────────
    personal_luteal = compute_personal_luteal_from_lh()
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v])
    print(f"Population luteal mean: {pop_luteal_mean:.1f} days")

    # ── Label: is_post_ov (for cycles with LH labels) ────────────────
    df["is_post_ov"] = np.nan
    for i, row in df.iterrows():
        sgk = row["small_group_key"]
        if sgk in lh_ov_dict:
            df.at[i, "is_post_ov"] = 1.0 if row["day_in_cycle"] > lh_ov_dict[sgk] else 0.0

    labeled = df.dropna(subset=["is_post_ov"])
    print(f"Labeled rows for ov-classifier: {len(labeled)} ({labeled['is_post_ov'].mean():.1%} post-ov)")

    # ── Cycle position prior feature ──────────────────────────────────
    df["cycle_frac"] = df["day_in_cycle"] / df["hist_cycle_len_mean"].clip(lower=20)
    labeled = df.dropna(subset=["is_post_ov"]).copy()

    # ── Build causal features ─────────────────────────────────────────
    print("Building causal detection features...")
    labeled = build_ov_detection_features(labeled)
    feat_cols = get_ov_feature_cols()

    valid = labeled.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    print(f"Valid rows after feature filtering: {len(valid)}")

    X_ov = valid[feat_cols].fillna(0).values
    y_ov = valid["is_post_ov"].values.astype(int)
    groups_ov = valid["id"].values

    # ==================================================================
    # LOSO ovulation classifier training + per-cycle detection
    # ==================================================================
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
    auc = roc_auc_score(y_ov[has_prob], valid.loc[has_prob, "ov_prob"])
    print(f"Ovulation classifier LOSO AUC: {auc:.3f}")

    # ── Detect ovulation per cycle ────────────────────────────────────
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

        # Evaluate detection accuracy
        errors = []
        for sgk, det_dic in detected_ov.items():
            if sgk in lh_ov_dict:
                errors.append(det_dic - lh_ov_dict[sgk])

        err = np.array(errors) if errors else np.array([])
        n_cycles_with_label = sum(1 for sgk in valid["small_group_key"].unique() if sgk in lh_ov_dict)
        print(
            f"\n  [{strat}] Detected: {len(detected_ov)}/{n_cycles_with_label} cycles"
            f"  | Mean offset: {err.mean():.1f}d"
            f"  | ±3d: {(np.abs(err) <= 3).mean():.1%}"
            f"  | ±5d: {(np.abs(err) <= 5).mean():.1%}"
            f"  | Median: {np.median(err):.1f}d"
        )

    # ==================================================================
    # 10-seed hybrid model evaluation
    # ==================================================================
    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"\n{'='*70}")
    print(f"  Hybrid Model: 10-seed evaluation")
    print(f"  Features for LightGBM: {len(features)}")
    print(f"{'='*70}")

    for strat in strategies:
        detected_ov = all_detected[strat]

        results_hybrid = []
        results_lgb = []
        results_oracle = []

        for seed in range(10):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            train_idx, test_idx = next(gss.split(df, groups=df["id"]))

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()
            train_uids = set(df_train["id"].unique())  # personal only for train users (no leakage)

            # Train LightGBM
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

            # Detected-ov hybrid
            pred_det = pred_lgb.copy()
            sgks = df_test["small_group_key"].values
            dics = df_test["day_in_cycle"].values
            uids = df_test["id"].values

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

            # Oracle hybrid
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

        lgb_a = avg(results_lgb)
        hyb_a = avg(results_hybrid)
        ora_a = avg(results_oracle)

        print(f"\n--- Strategy: {strat} ---")
        print(f"  LightGBM only:        MAE={lgb_a['mae']:.3f}  ±1d={lgb_a['acc_1d']:.1%}  ±2d={lgb_a['acc_2d']:.1%}  ±3d={lgb_a['acc_3d']:.1%}")
        print(f"  Detected-ov hybrid:   MAE={hyb_a['mae']:.3f}  ±1d={hyb_a['acc_1d']:.1%}  ±2d={hyb_a['acc_2d']:.1%}  ±3d={hyb_a['acc_3d']:.1%}")
        print(f"  Oracle hybrid:        MAE={ora_a['mae']:.3f}  ±1d={ora_a['acc_1d']:.1%}  ±2d={ora_a['acc_2d']:.1%}  ±3d={ora_a['acc_3d']:.1%}")
        print(f"  Det vs LGB:           MAE={hyb_a['mae']-lgb_a['mae']:+.3f}  ±3d={100*(hyb_a['acc_3d']-lgb_a['acc_3d']):+.1f}pp")
        print(f"  Oracle vs LGB:        MAE={ora_a['mae']-lgb_a['mae']:+.3f}  ±3d={100*(ora_a['acc_3d']-lgb_a['acc_3d']):+.1f}pp")

    # ==================================================================
    # Soft-blending approach: weight countdown by ov probability
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Soft-Blending Approach: LightGBM weighted by P(post-ov)")
    print(f"{'='*70}")

    # Build ov_prob for ALL rows (not just labeled ones)
    df_all_feat = build_ov_detection_features(df.copy())
    df_all_feat["cycle_frac"] = df_all_feat["day_in_cycle"] / df_all_feat["hist_cycle_len_mean"].clip(lower=20)
    feat_valid = df_all_feat.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    X_all = feat_valid[feat_cols].fillna(0).values

    results_soft = []
    results_lgb2 = []
    results_ora2 = []

    for seed in range(10):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["id"]))

        df_train = df.iloc[train_idx]
        df_test_orig = df.iloc[test_idx].copy()

        # LightGBM
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + 100)
        tr_idx, val_idx = next(gss2.split(df_train, groups=df_train["id"]))
        X_tr = df_train.iloc[tr_idx][features].values
        y_tr = df_train.iloc[tr_idx]["days_until_next_menses"].values
        X_val = df_train.iloc[val_idx][features].values
        y_val = df_train.iloc[val_idx]["days_until_next_menses"].values
        X_test = df_test_orig[features].values
        y_test = df_test_orig["days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features, params=LGB_PARAMS_TUNED)
        pred_lgb = np.clip(model.predict(X_test), 1.0, None)
        results_lgb2.append(compute_metrics(pred_lgb, y_test))

        # Train ov-classifier on train subjects' labeled data
        train_subjects = set(df_train["id"].unique())
        ov_train = valid[valid["id"].isin(train_subjects)].copy()
        if len(ov_train) < 50:
            results_soft.append(compute_metrics(pred_lgb, y_test))
            results_ora2.append(compute_metrics(pred_lgb, y_test))
            continue

        X_ov_tr = ov_train[feat_cols].fillna(0).values
        y_ov_tr = ov_train["is_post_ov"].values.astype(int)
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_ov_tr, y_ov_tr)

        # Get ov probs for test rows
        test_feat = feat_valid[feat_valid.index.isin(df_test_orig.index)].copy()
        if len(test_feat) == 0:
            test_feat = df_all_feat[df_all_feat.index.isin(df_test_orig.index)]
            if len(test_feat) == 0:
                results_soft.append(compute_metrics(pred_lgb, y_test))
                results_ora2.append(compute_metrics(pred_lgb, y_test))
                continue

        # Map test rows
        test_keys = df_test_orig[["id", "study_interval", "day_in_study"]].reset_index()
        feat_keys = df_all_feat[["id", "study_interval", "day_in_study"]].copy()
        feat_keys["feat_idx"] = range(len(feat_keys))

        merged_keys = test_keys.merge(feat_keys, on=["id", "study_interval", "day_in_study"], how="left")
        feat_indices = merged_keys["feat_idx"].dropna().astype(int).values

        X_test_ov = df_all_feat.iloc[feat_indices][feat_cols].fillna(0).values
        ov_probs_test = clf.predict_proba(X_test_ov)[:, 1]

        # Map probs back to test rows
        test_ov_prob = np.full(len(df_test_orig), 0.0)
        valid_test_pos = merged_keys.dropna(subset=["feat_idx"]).index.values
        for j, pos in enumerate(valid_test_pos):
            if j < len(ov_probs_test):
                local_pos = pos - valid_test_pos[0] if pos >= valid_test_pos[0] else pos
                if local_pos < len(test_ov_prob):
                    test_ov_prob[local_pos] = ov_probs_test[j]

        # Soft blend: pred = (1 - w) * lgb + w * countdown
        sgks = df_test_orig["small_group_key"].values
        dics = df_test_orig["day_in_cycle"].values
        uids = df_test_orig["id"].values
        hist_lens = df_test_orig["hist_cycle_len_mean"].values

        pred_soft = pred_lgb.copy()
        pred_ora = pred_lgb.copy()

        for i in range(len(df_test_orig)):
            sgk, dic, uid = sgks[i], dics[i], uids[i]
            hist_len = hist_lens[i] if not np.isnan(hist_lens[i]) else 28
            expected_ov = max(10, hist_len - 14)
            luts = personal_luteal.get(uid, [])
            avg_lut = np.mean(luts) if luts else pop_luteal_mean

            # Soft blend with detected probability
            w = test_ov_prob[i] if i < len(test_ov_prob) else 0
            if dic >= 8 and w > 0.3:
                est_ov_day = expected_ov
                days_since = dic - est_ov_day
                if days_since > 0:
                    countdown = max(1.0, avg_lut - days_since)
                    blend_w = min(1.0, (w - 0.3) / 0.4)
                    pred_soft[i] = (1 - blend_w) * pred_lgb[i] + blend_w * countdown

            # Oracle
            if sgk in lh_ov_dict:
                ov_dic = lh_ov_dict[sgk]
                if dic >= ov_dic + 2:
                    days_since = dic - ov_dic
                    pred_ora[i] = max(1.0, avg_lut - days_since)

        results_soft.append(compute_metrics(pred_soft, y_test))
        results_ora2.append(compute_metrics(pred_ora, y_test))

    lgb2_a = avg(results_lgb2)
    soft_a = avg(results_soft)
    ora2_a = avg(results_ora2)

    print(f"\n  LightGBM only:    MAE={lgb2_a['mae']:.3f}  ±1d={lgb2_a['acc_1d']:.1%}  ±2d={lgb2_a['acc_2d']:.1%}  ±3d={lgb2_a['acc_3d']:.1%}")
    print(f"  Soft-blend:       MAE={soft_a['mae']:.3f}  ±1d={soft_a['acc_1d']:.1%}  ±2d={soft_a['acc_2d']:.1%}  ±3d={soft_a['acc_3d']:.1%}")
    print(f"  Oracle hybrid:    MAE={ora2_a['mae']:.3f}  ±1d={ora2_a['acc_1d']:.1%}  ±2d={ora2_a['acc_2d']:.1%}  ±3d={ora2_a['acc_3d']:.1%}")
    print(f"  Soft vs LGB:      MAE={soft_a['mae']-lgb2_a['mae']:+.3f}  ±3d={100*(soft_a['acc_3d']-lgb2_a['acc_3d']):+.1f}pp")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

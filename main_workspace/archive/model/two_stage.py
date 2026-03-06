"""Two-stage prediction: ovulation detection → conditional menses prediction.

Stage A: Detect whether ovulation has occurred (wearable-only signals)
  Two detection modes:
    - "wt_only": temperature biphasic shift only
    - "hrv_wt":  temperature shift as primary + HRV confirmation (default)

Stage B: Conditional prediction
  - Post-ovulation: full-feature LightGBM + days_since_ovulation
  - Pre-ovulation:  full-feature LightGBM (without ovulation timing)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import (
    ALL_FEATURES, LGB_PARAMS,
    RANDOM_SEED, TEST_SUBJECT_RATIO,
)
from .dataset import load_data, subject_split
from .train_lgb import predict, feature_importance
from .evaluate import compute_metrics, stratified_metrics, print_metrics

OV_MODES = ("wt_only", "hrv_wt")

# ── Stage A: Ovulation detection ─────────────────────────────────────────────

def _compute_ov_signal(df, mode):
    """Compute raw ovulation signal column based on detection mode.

    All modes use wearable-only signals (no LH / hormone data required).
    """
    wt_threshold = 0.15

    if mode == "wt_only":
        df["ov_signal"] = 0
        if "wt_shift_7v3" in df.columns:
            df["ov_signal"] = (df["wt_shift_7v3"] > wt_threshold).astype(int)

    elif mode == "hrv_wt":
        # Primary: WT shift ≥ 0.15°C (progesterone-driven post-ovulation rise)
        ov_wt = (df["wt_shift_7v3"] > wt_threshold).astype(int) if "wt_shift_7v3" in df.columns else 0

        # Auxiliary: HRV sympathetic shift (Hamidovic 2023)
        # HF-HRV drops below personal mean (z < -0.5)
        # AND LF/HF ratio rises above personal mean (z > 0.5)
        hrv_confirm = pd.Series(0, index=df.index)
        if "hf_mean_z" in df.columns and "lf_hf_ratio_z" in df.columns:
            hrv_confirm = (
                (df["hf_mean_z"] < -0.5) & (df["lf_hf_ratio_z"] > 0.5)
            ).astype(int)
        elif "hf_mean_z" in df.columns:
            hrv_confirm = (df["hf_mean_z"] < -0.5).astype(int)

        # WT alone (≥0.15) triggers; weaker WT (≥0.10) + HRV confirmation also triggers
        wt_lower = (df["wt_shift_7v3"] > 0.10).astype(int) if "wt_shift_7v3" in df.columns else 0
        df["ov_signal"] = ((ov_wt == 1) | ((wt_lower == 1) & (hrv_confirm == 1))).astype(int)

    else:
        raise ValueError(f"Unknown ovulation mode: {mode}. Choose from {OV_MODES}")

    return df


def build_ovulation_labels(df, mode="hrv_wt"):
    """Build binary label: has ovulation occurred by this day in the cycle?

    Args:
        mode: "wt_only" | "hrv_wt" (default: "hrv_wt")
    """
    cycle_group = ["id", "study_interval", "small_group_key"]

    df = _compute_ov_signal(df, mode)

    # Once ovulation is detected in a cycle, all subsequent days are post-ovulation
    df = df.sort_values(cycle_group + ["day_in_study"])
    df["is_post_ovulation"] = df.groupby(cycle_group)["ov_signal"].cummax()

    df["days_since_ovulation"] = np.nan
    for _, grp in df.groupby(cycle_group):
        ov_days = grp.loc[grp["ov_signal"] == 1, "day_in_study"]
        if len(ov_days) == 0:
            continue
        first_ov_day = ov_days.iloc[0]
        post_mask = grp["is_post_ovulation"] == 1
        df.loc[grp.index[post_mask], "days_since_ovulation"] = (
            grp.loc[post_mask, "day_in_study"] - first_ov_day
        )

    n_post = df["is_post_ovulation"].sum()
    n_total = len(df)
    print(f"[ov-label][{mode}] Post-ov: {n_post}/{n_total} ({n_post/n_total*100:.1f}%)")

    df = df.drop(columns=["ov_signal"], errors="ignore")
    return df


# ── Stage B: Conditional prediction ──────────────────────────────────────────

def train_two_stage(df, features, train_subj, val_subj):
    """Train two separate LightGBM models for pre- and post-ovulation."""
    params = LGB_PARAMS.copy()
    n_est = params.pop("n_estimators", 2000)
    es_rounds = params.pop("early_stopping_rounds", 80)
    callbacks = [lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(200)]

    # Add days_since_ovulation as extra feature for post-ov model
    post_features = features + ["days_since_ovulation"]
    post_features = [f for f in post_features if f in df.columns]

    train = df[df["id"].isin(train_subj)]
    val = df[df["id"].isin(val_subj)]

    # ── Post-ovulation model ──
    train_post = train[train["is_post_ovulation"] == 1]
    val_post = val[val["is_post_ovulation"] == 1]

    print(f"\n[post-ov] Training on {len(train_post)} rows, val {len(val_post)} rows")
    if len(train_post) > 50 and len(val_post) > 10:
        ds_tr = lgb.Dataset(train_post[post_features], label=train_post["days_until_next_menses"],
                            feature_name=post_features)
        ds_val = lgb.Dataset(val_post[post_features], label=val_post["days_until_next_menses"],
                             feature_name=post_features, reference=ds_tr)
        model_post = lgb.train(params, ds_tr, num_boost_round=n_est,
                               valid_sets=[ds_tr, ds_val], valid_names=["train", "val"],
                               callbacks=callbacks)
        print(f"[post-ov] Best iter: {model_post.best_iteration}, val MAE: {model_post.best_score['val']['l1']:.3f}")
    else:
        model_post = None
        print("[post-ov] Not enough data, skipping")

    # ── Pre-ovulation model ──
    train_pre = train[train["is_post_ovulation"] == 0]
    val_pre = val[val["is_post_ovulation"] == 0]

    print(f"\n[pre-ov] Training on {len(train_pre)} rows, val {len(val_pre)} rows")
    if len(train_pre) > 50 and len(val_pre) > 10:
        ds_tr = lgb.Dataset(train_pre[features], label=train_pre["days_until_next_menses"],
                            feature_name=features)
        ds_val = lgb.Dataset(val_pre[features], label=val_pre["days_until_next_menses"],
                             feature_name=features, reference=ds_tr)
        model_pre = lgb.train(params, ds_tr, num_boost_round=n_est,
                              valid_sets=[ds_tr, ds_val], valid_names=["train", "val"],
                              callbacks=callbacks)
        print(f"[pre-ov] Best iter: {model_pre.best_iteration}, val MAE: {model_pre.best_score['val']['l1']:.3f}")
    else:
        model_pre = None
        print("[pre-ov] Not enough data, skipping")

    return model_pre, model_post, features, post_features


def predict_two_stage(df, model_pre, model_post, features, post_features):
    """Predict using the appropriate model based on ovulation status."""
    pred = np.full(len(df), np.nan)

    post_mask = df["is_post_ovulation"] == 1
    pre_mask = ~post_mask

    if model_post is not None and post_mask.sum() > 0:
        X_post = df.loc[post_mask, post_features].values
        pred[post_mask.values] = np.clip(
            model_post.predict(X_post, num_iteration=model_post.best_iteration), 1.0, None
        )

    if model_pre is not None and pre_mask.sum() > 0:
        X_pre = df.loc[pre_mask, features].values
        pred[pre_mask.values] = np.clip(
            model_pre.predict(X_pre, num_iteration=model_pre.best_iteration), 1.0, None
        )

    # Fallback for rows without a model: use prior
    still_nan = np.isnan(pred)
    if still_nan.sum() > 0:
        prior = df.loc[still_nan, "days_remaining_prior"].values
        pred[still_nan] = np.clip(prior, 1.0, None)

    return pred


# ── Main entry ────────────────────────────────────────────────────────────────

def run_two_stage(mode="hrv_wt"):
    """Run two-stage experiment with specified ovulation detection mode."""
    mode_desc = {"wt_only": "WT only", "hrv_wt": "HRV + WT"}
    print("=" * 60)
    print(f"  Two-Stage: {mode_desc.get(mode, mode)}")
    print("=" * 60)

    df, available = load_data()
    features = [f for f in ALL_FEATURES if f in available]

    df = build_ovulation_labels(df, mode=mode)

    train_subj, val_subj, test_subj = subject_split(
        df, test_ratio=TEST_SUBJECT_RATIO, seed=RANDOM_SEED
    )

    model_pre, model_post, feat_pre, feat_post = train_two_stage(
        df, features, train_subj, val_subj
    )

    # Evaluate
    results = {}
    for set_name, subj_set in [("val", val_subj), ("test", test_subj)]:
        sub_df = df[df["id"].isin(subj_set)].copy()
        sub_df["pred"] = predict_two_stage(sub_df, model_pre, model_post, feat_pre, feat_post)
        metrics = {"pred": sub_df["pred"].values, "true": sub_df["days_until_next_menses"].values}
        overall, hz = print_metrics(metrics, f"{set_name.capitalize()} [{mode}]")

        for ov_name, ov_mask in [("pre_ov", sub_df["is_post_ovulation"] == 0),
                                  ("post_ov", sub_df["is_post_ovulation"] == 1)]:
            s = sub_df[ov_mask]
            if len(s) > 0:
                m = compute_metrics(s["pred"].values, s["days_until_next_menses"].values)
                print(f"  [{ov_name}] n={m['n']}, MAE={m['mae']:.3f}, ±3d={m['acc_3d']:.3f}")
                results[f"{set_name}_{ov_name}"] = m

        results[set_name] = overall
        results[f"{set_name}_hz"] = hz

    n_post = int((df["is_post_ovulation"] == 1).sum())
    results["post_ov_ratio"] = n_post / len(df)

    return results


def run_ovulation_comparison():
    """Run all three ovulation detection modes and print comparison table."""
    print("\n" + "#" * 70)
    print("  OVULATION DETECTION MODE COMPARISON")
    print("#" * 70)

    all_results = {}
    for mode in OV_MODES:
        print(f"\n{'━' * 70}")
        print(f"  MODE: {mode}")
        print(f"{'━' * 70}")
        all_results[mode] = run_two_stage(mode=mode)

    # Summary table
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    header = f"  {'Mode':<12s} {'Post%':>6s} │ {'Val MAE':>8s} {'Val±3d':>7s} │ {'Test MAE':>9s} {'Test±3d':>8s}"
    print(header)
    print("  " + "─" * 62)
    for mode in OV_MODES:
        r = all_results[mode]
        post_pct = r["post_ov_ratio"] * 100
        v = r["val"]
        t = r["test"]
        print(f"  {mode:<12s} {post_pct:5.1f}% │ {v['mae']:8.3f} {v['acc_3d']:7.3f} │ {t['mae']:9.3f} {t['acc_3d']:8.3f}")

    # Per-subset breakdown
    print(f"\n  {'Mode':<12s} │ {'Test Pre-ov MAE':>15s} {'±3d':>6s} │ {'Test Post-ov MAE':>16s} {'±3d':>6s}")
    print("  " + "─" * 62)
    for mode in OV_MODES:
        r = all_results[mode]
        pre = r.get("test_pre_ov", {})
        post = r.get("test_post_ov", {})
        pre_mae = f"{pre['mae']:.3f}" if pre else "  N/A"
        pre_3d = f"{pre['acc_3d']:.3f}" if pre else " N/A"
        post_mae = f"{post['mae']:.3f}" if post else "  N/A"
        post_3d = f"{post['acc_3d']:.3f}" if post else " N/A"
        print(f"  {mode:<12s} │ {pre_mae:>15s} {pre_3d:>6s} │ {post_mae:>16s} {post_3d:>6s}")

    return all_results


if __name__ == "__main__":
    run_ovulation_comparison()

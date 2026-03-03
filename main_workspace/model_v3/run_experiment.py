"""End-to-end experiment: load data, split, train LightGBM, evaluate."""
import numpy as np

from .config import ALL_FEATURES, RANDOM_SEED, TEST_SUBJECT_RATIO
from .dataset import load_data, subject_split
from .train_lgb import train_lightgbm, predict, feature_importance
from .evaluate import print_metrics


def run_experiment(feature_list=None, features_csv=None):
    """Run full experiment with specified feature list and/or data source."""
    features = feature_list or ALL_FEATURES

    # Load
    df, available = load_data(features_csv=features_csv) if features_csv else load_data()
    features = [f for f in features if f in available]
    missing = [f for f in (feature_list or ALL_FEATURES) if f not in available]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    print(f"[exp] Using {len(features)} features")

    # Split
    train_subj, val_subj, test_subj = subject_split(
        df, test_ratio=TEST_SUBJECT_RATIO, seed=RANDOM_SEED
    )

    train_mask = df["id"].isin(train_subj)
    val_mask = df["id"].isin(val_subj)
    test_mask = df["id"].isin(test_subj)

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, "days_until_next_menses"].values
    X_val = df.loc[val_mask, features].values
    y_val = df.loc[val_mask, "days_until_next_menses"].values
    X_test = df.loc[test_mask, features].values
    y_test = df.loc[test_mask, "days_until_next_menses"].values

    print(f"[exp] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} rows")

    # Train
    model = train_lightgbm(X_train, y_train, X_val, y_val, features)

    # Evaluate
    pred_val = predict(model, X_val)
    pred_test = predict(model, X_test)

    val_metrics = {"pred": pred_val, "true": y_val}
    test_metrics = {"pred": pred_test, "true": y_test}

    val_overall, val_hz = print_metrics(val_metrics, "Validation")
    test_overall, test_hz = print_metrics(test_metrics, "Test")

    # Feature importance
    imp = feature_importance(model, features)

    # Sample predictions (first test cycle)
    test_df = df.loc[test_mask].copy()
    test_df["pred"] = pred_test
    first_cycle = test_df.groupby(["id", "small_group_key"]).first().index[0]
    sample = test_df[
        (test_df["id"] == first_cycle[0])
        & (test_df["small_group_key"] == first_cycle[1])
    ].sort_values("day_in_study")
    print(f"\n[sample] First test cycle (id={first_cycle[0]}, cycle={first_cycle[1]}):")
    print(f"  {'day':>4s} {'true':>6s} {'pred':>6s} {'err':>6s}")
    for _, row in sample.head(15).iterrows():
        err = row["days_until_next_menses"] - row["pred"]
        print(f"  {int(row['day_in_cycle']):4d} {row['days_until_next_menses']:6.0f} {row['pred']:6.1f} {err:+6.1f}")

    return {
        "model": model,
        "val": val_overall,
        "test": test_overall,
        "val_by_horizon": val_hz,
        "test_by_horizon": test_hz,
        "importance": imp,
        "features": features,
    }


def run_ablation():
    """Run ablation: add feature groups incrementally."""
    from .config import (
        FEAT_CYCLE_PRIOR, FEAT_WEARABLE_Z, FEAT_SHIFTS,
        FEAT_DELTAS, FEAT_RESPIRATORY_Z, FEAT_SLEEP_Z, FEAT_SYMPTOMS,
    )

    groups = [
        ("Prior only", FEAT_CYCLE_PRIOR),
        ("+ Wearable", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z),
        ("+ Shifts", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS),
        ("+ Deltas", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS),
        ("+ Respiratory", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z),
        ("+ Sleep", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z + FEAT_SLEEP_Z),
        ("+ Symptoms (ALL)", FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z + FEAT_SLEEP_Z + FEAT_SYMPTOMS),
    ]

    results = []
    for name, feats in groups:
        print(f"\n{'#' * 60}")
        print(f"  ABLATION: {name} ({len(feats)} features)")
        print(f"{'#' * 60}")
        r = run_experiment(feature_list=feats)
        results.append((name, len(feats), r["test"]["mae"], r["test"]["acc_3d"]))

    print(f"\n{'=' * 60}")
    print(f"  ABLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Name':30s} {'#feat':>5s} {'MAE':>6s} {'±3d':>6s}")
    print(f"  {'-' * 50}")
    for name, nf, mae, acc3 in results:
        print(f"  {name:30s} {nf:5d} {mae:6.3f} {acc3:6.3f}")


if __name__ == "__main__":
    run_experiment()

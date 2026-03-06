"""Comprehensive comparison: v4 baseline vs LLCO vs v6 features vs LLCO+v6."""
import numpy as np

from .config import (
    ALL_FEATURES, ALL_FEATURES_V6,
    FEATURES_V4_CSV, FEATURES_V6_CSV,
    TEST_SUBJECT_RATIO, LGB_PARAMS,
)
from .dataset import load_data, subject_split, cycle_split
from .train_lgb import train_lightgbm, predict
from .evaluate import compute_metrics, stratified_metrics


def _run_config(df, features, split_fn, n_seeds=10, label=""):
    """Run n_seeds experiments with given features and split function."""
    all_test_mae, all_test_acc3, all_val_mae = [], [], []
    all_hz = {k: [] for k in ["1-5", "6-10", "11-15", "16-20", "21+"]}

    for seed in range(42, 42 + n_seeds):
        result = split_fn(df, seed)

        if isinstance(result, tuple) and len(result) == 3:
            first = result[0]
            if isinstance(first, set):
                train_subj, val_subj, test_subj = result
                train_mask = df["id"].isin(train_subj)
                val_mask = df["id"].isin(val_subj)
                test_mask = df["id"].isin(test_subj)
            else:
                train_mask, val_mask, test_mask = result
        else:
            raise ValueError("split_fn must return 3-tuple")

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, "days_until_next_menses"].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, "days_until_next_menses"].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, "days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features)
        pred_test = predict(model, X_te)
        pred_val = predict(model, X_val)

        test_m = compute_metrics(pred_test, y_te)
        val_m = compute_metrics(pred_val, y_val)
        hz = stratified_metrics(pred_test, y_te)

        all_test_mae.append(test_m["mae"])
        all_test_acc3.append(test_m["acc_3d"])
        all_val_mae.append(val_m["mae"])
        for k in all_hz:
            if k in hz and hz[k]["n"] > 0:
                all_hz[k].append(hz[k]["mae"])

    return {
        "label": label,
        "test_mae": f"{np.mean(all_test_mae):.3f} ± {np.std(all_test_mae):.3f}",
        "test_acc3": f"{np.mean(all_test_acc3):.3f} ± {np.std(all_test_acc3):.3f}",
        "val_mae": f"{np.mean(all_val_mae):.3f} ± {np.std(all_val_mae):.3f}",
        "hz": {k: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "N/A"
               for k, v in all_hz.items()},
        "_mae_raw": all_test_mae,
        "_acc3_raw": all_test_acc3,
    }


def run_comparison(n_seeds=10):
    """Run all 4 configurations and print comparison table."""
    print("=" * 70)
    print("  COMPREHENSIVE COMPARISON (4 configs × {} seeds)".format(n_seeds))
    print("=" * 70)

    # Load datasets
    df_v4, avail_v4 = load_data(features_csv=FEATURES_V4_CSV)
    df_v6, avail_v6 = load_data(features_csv=FEATURES_V6_CSV)

    feats_v4 = [f for f in ALL_FEATURES if f in avail_v4]
    feats_v6 = [f for f in ALL_FEATURES_V6 if f in avail_v6]

    print(f"\n[config] v4 features: {len(feats_v4)}, v6 features: {len(feats_v6)}")

    def subject_split_fn(df, seed):
        return subject_split(df, test_ratio=TEST_SUBJECT_RATIO, seed=seed)

    def cycle_split_fn(df, seed):
        return cycle_split(df, seed=seed)

    configs = [
        ("A: v4 + SubjectSplit (baseline)", df_v4, feats_v4, subject_split_fn),
        ("B: v4 + LLCO", df_v4, feats_v4, cycle_split_fn),
        ("C: v6 + SubjectSplit", df_v6, feats_v6, subject_split_fn),
        ("D: v6 + LLCO", df_v6, feats_v6, cycle_split_fn),
    ]

    results = []
    for label, df, feats, split_fn in configs:
        print(f"\n{'─' * 60}")
        print(f"  Running: {label}")
        print(f"{'─' * 60}")
        r = _run_config(df, feats, split_fn, n_seeds=n_seeds, label=label)
        results.append(r)

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON SUMMARY ({n_seeds} seeds)")
    print(f"{'=' * 70}")
    print(f"  {'Config':<30s} {'Test MAE':<18s} {'Test ±3d':<18s} {'Val MAE':<18s}")
    print(f"  {'─' * 84}")
    for r in results:
        print(f"  {r['label']:<30s} {r['test_mae']:<18s} {r['test_acc3']:<18s} {r['val_mae']:<18s}")

    print(f"\n  Horizon MAE breakdown:")
    print(f"  {'Config':<30s} {'1-5':<14s} {'6-10':<14s} {'11-15':<14s} {'16-20':<14s} {'21+':<14s}")
    print(f"  {'─' * 100}")
    for r in results:
        hz = r["hz"]
        print(f"  {r['label']:<30s} {hz['1-5']:<14s} {hz['6-10']:<14s} "
              f"{hz['11-15']:<14s} {hz['16-20']:<14s} {hz['21+']:<14s}")

    return results


if __name__ == "__main__":
    run_comparison()

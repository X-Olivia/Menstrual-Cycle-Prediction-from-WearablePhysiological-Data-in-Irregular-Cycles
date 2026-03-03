"""Residual modeling: predict (actual - prior) instead of absolute days.

Core idea: days_remaining_prior = hist_cycle_len_mean - day_in_cycle is already
a strong baseline. Let the model learn the *deviation* from this prior so that
wearable features contribute more meaningfully.

Target:  residual = days_until_next_menses - days_remaining_prior
Output:  final_pred = residual_pred + days_remaining_prior  (clipped >= 1)
"""

import numpy as np

from .config import (
    ALL_FEATURES, RANDOM_SEED, TEST_SUBJECT_RATIO, LGB_PARAMS,
)
from .dataset import load_data, subject_split
from .train_lgb import train_lightgbm, feature_importance
from .evaluate import compute_metrics, stratified_metrics, print_metrics

RESIDUAL_FEATURES = [
    f for f in ALL_FEATURES
    if f not in ("days_remaining_prior", "days_remaining_prior_log")
]


def _prepare_residual(df):
    """Add residual column and print statistics."""
    df["residual"] = df["days_until_next_menses"] - df["days_remaining_prior"]
    r = df["residual"]
    print(f"[residual] range: {r.min():.1f} ~ {r.max():.1f}, "
          f"mean: {r.mean():.2f}, std: {r.std():.2f}")
    return df


def run_residual_experiment(feature_list=None, features_csv=None, seed=RANDOM_SEED):
    """Single-run residual experiment with full diagnostics."""
    print("=" * 60)
    print("  Residual Modeling Experiment")
    print("=" * 60)

    features = feature_list or RESIDUAL_FEATURES
    df, available = load_data(features_csv=features_csv) if features_csv else load_data()
    features = [f for f in features if f in available]
    print(f"[residual] Using {len(features)} features (prior-based removed from input)")

    df = _prepare_residual(df)

    train_subj, val_subj, test_subj = subject_split(
        df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
    )

    train_mask = df["id"].isin(train_subj)
    val_mask = df["id"].isin(val_subj)
    test_mask = df["id"].isin(test_subj)

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, "residual"].values
    X_val = df.loc[val_mask, features].values
    y_val = df.loc[val_mask, "residual"].values
    X_test = df.loc[test_mask, features].values

    prior_val = df.loc[val_mask, "days_remaining_prior"].values
    prior_test = df.loc[test_mask, "days_remaining_prior"].values
    y_val_true = df.loc[val_mask, "days_until_next_menses"].values
    y_test_true = df.loc[test_mask, "days_until_next_menses"].values

    print(f"[residual] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} rows")

    model = train_lightgbm(X_train, y_train, X_val, y_val, features)

    res_val = model.predict(X_val, num_iteration=model.best_iteration)
    res_test = model.predict(X_test, num_iteration=model.best_iteration)

    pred_val = np.clip(res_val + prior_val, 1.0, None)
    pred_test = np.clip(res_test + prior_test, 1.0, None)

    val_metrics = {"pred": pred_val, "true": y_val_true}
    test_metrics = {"pred": pred_test, "true": y_test_true}

    val_overall, val_hz = print_metrics(val_metrics, "Validation (Residual)")
    test_overall, test_hz = print_metrics(test_metrics, "Test (Residual)")

    imp = feature_importance(model, features)

    # Sample predictions with prior column
    test_df = df.loc[test_mask].copy()
    test_df["pred"] = pred_test
    first_cycle = test_df.groupby(["id", "small_group_key"]).first().index[0]
    sample = test_df[
        (test_df["id"] == first_cycle[0])
        & (test_df["small_group_key"] == first_cycle[1])
    ].sort_values("day_in_study")
    print(f"\n[sample] First test cycle (id={first_cycle[0]}, cycle={first_cycle[1]}):")
    print(f"  {'day':>4s} {'true':>6s} {'prior':>6s} {'pred':>6s} {'err':>6s}")
    for _, row in sample.head(15).iterrows():
        err = row["days_until_next_menses"] - row["pred"]
        prior = row["days_remaining_prior"]
        print(f"  {int(row['day_in_cycle']):4d} {row['days_until_next_menses']:6.0f} "
              f"{prior:6.1f} {row['pred']:6.1f} {err:+6.1f}")

    return {
        "model": model, "val": val_overall, "test": test_overall,
        "val_by_horizon": val_hz, "test_by_horizon": test_hz,
        "importance": imp, "features": features,
    }


def run_residual_multi_seed(n_seeds=10, features_csv=None, feature_list=None):
    """Multi-seed residual experiment for robust evaluation."""
    print("=" * 60)
    print(f"  Residual Multi-Seed Evaluation ({n_seeds} splits)")
    print("=" * 60)

    df, available = load_data(features_csv=features_csv) if features_csv else load_data()
    features = feature_list or RESIDUAL_FEATURES
    features = [f for f in features if f in available]
    print(f"[residual] Using {len(features)} features")

    df = _prepare_residual(df)

    all_test_mae, all_test_acc3, all_val_mae = [], [], []
    all_hz = {label: [] for label in ["1-5", "6-10", "11-15", "16-20", "21+"]}

    for seed in range(42, 42 + n_seeds):
        train_subj, val_subj, test_subj = subject_split(
            df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        train_mask = df["id"].isin(train_subj)
        val_mask = df["id"].isin(val_subj)
        test_mask = df["id"].isin(test_subj)

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, "residual"].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, "residual"].values
        X_te = df.loc[test_mask, features].values

        prior_val = df.loc[val_mask, "days_remaining_prior"].values
        prior_te = df.loc[test_mask, "days_remaining_prior"].values
        y_val_true = df.loc[val_mask, "days_until_next_menses"].values
        y_te_true = df.loc[test_mask, "days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features)

        res_val = model.predict(X_val, num_iteration=model.best_iteration)
        res_te = model.predict(X_te, num_iteration=model.best_iteration)
        pred_val = np.clip(res_val + prior_val, 1.0, None)
        pred_te = np.clip(res_te + prior_te, 1.0, None)

        test_m = compute_metrics(pred_te, y_te_true)
        val_m = compute_metrics(pred_val, y_val_true)
        hz = stratified_metrics(pred_te, y_te_true)

        all_test_mae.append(test_m["mae"])
        all_test_acc3.append(test_m["acc_3d"])
        all_val_mae.append(val_m["mae"])
        for label in all_hz:
            if label in hz and hz[label]["n"] > 0:
                all_hz[label].append(hz[label]["mae"])

        print(f"  seed={seed}: test MAE={test_m['mae']:.3f}, ±3d={test_m['acc_3d']:.3f}, "
              f"val MAE={val_m['mae']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE ({n_seeds} seeds) — Residual Model")
    print(f"{'=' * 60}")
    print(f"  Test MAE:  {np.mean(all_test_mae):.3f} ± {np.std(all_test_mae):.3f}")
    print(f"  Test ±3d:  {np.mean(all_test_acc3):.3f} ± {np.std(all_test_acc3):.3f}")
    print(f"  Val MAE:   {np.mean(all_val_mae):.3f} ± {np.std(all_val_mae):.3f}")
    print(f"\n  Horizon MAE (mean ± std):")
    for label in ["1-5", "6-10", "11-15", "16-20", "21+"]:
        vals = all_hz[label]
        if vals:
            print(f"    {label:>8s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return {
        "test_mae_mean": np.mean(all_test_mae),
        "test_mae_std": np.std(all_test_mae),
        "test_acc3_mean": np.mean(all_test_acc3),
        "val_mae_mean": np.mean(all_val_mae),
    }


def run_hybrid_multi_seed(n_seeds=10, features_csv=None):
    """Hybrid: keep ALL features but predict residual target."""
    print("=" * 60)
    print(f"  Hybrid Multi-Seed (all feats + residual target, {n_seeds} splits)")
    print("=" * 60)

    df, available = load_data(features_csv=features_csv) if features_csv else load_data()
    features = [f for f in ALL_FEATURES if f in available]
    print(f"[hybrid] Using {len(features)} features (all original)")

    df = _prepare_residual(df)

    all_test_mae, all_test_acc3, all_val_mae = [], [], []
    all_hz = {label: [] for label in ["1-5", "6-10", "11-15", "16-20", "21+"]}

    for seed in range(42, 42 + n_seeds):
        train_subj, val_subj, test_subj = subject_split(
            df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        train_mask = df["id"].isin(train_subj)
        val_mask = df["id"].isin(val_subj)
        test_mask = df["id"].isin(test_subj)

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, "residual"].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, "residual"].values
        X_te = df.loc[test_mask, features].values

        prior_val = df.loc[val_mask, "days_remaining_prior"].values
        prior_te = df.loc[test_mask, "days_remaining_prior"].values
        y_val_true = df.loc[val_mask, "days_until_next_menses"].values
        y_te_true = df.loc[test_mask, "days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features)

        res_val = model.predict(X_val, num_iteration=model.best_iteration)
        res_te = model.predict(X_te, num_iteration=model.best_iteration)
        pred_val = np.clip(res_val + prior_val, 1.0, None)
        pred_te = np.clip(res_te + prior_te, 1.0, None)

        test_m = compute_metrics(pred_te, y_te_true)
        val_m = compute_metrics(pred_val, y_val_true)
        hz = stratified_metrics(pred_te, y_te_true)

        all_test_mae.append(test_m["mae"])
        all_test_acc3.append(test_m["acc_3d"])
        all_val_mae.append(val_m["mae"])
        for label in all_hz:
            if label in hz and hz[label]["n"] > 0:
                all_hz[label].append(hz[label]["mae"])

        print(f"  seed={seed}: test MAE={test_m['mae']:.3f}, ±3d={test_m['acc_3d']:.3f}, "
              f"val MAE={val_m['mae']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE ({n_seeds} seeds) — Hybrid Model")
    print(f"{'=' * 60}")
    print(f"  Test MAE:  {np.mean(all_test_mae):.3f} ± {np.std(all_test_mae):.3f}")
    print(f"  Test ±3d:  {np.mean(all_test_acc3):.3f} ± {np.std(all_test_acc3):.3f}")
    print(f"  Val MAE:   {np.mean(all_val_mae):.3f} ± {np.std(all_val_mae):.3f}")
    print(f"\n  Horizon MAE (mean ± std):")
    for label in ["1-5", "6-10", "11-15", "16-20", "21+"]:
        vals = all_hz[label]
        if vals:
            print(f"    {label:>8s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return {
        "test_mae_mean": np.mean(all_test_mae),
        "test_mae_std": np.std(all_test_mae),
        "test_acc3_mean": np.mean(all_test_acc3),
        "val_mae_mean": np.mean(all_val_mae),
        "all_hz": {k: (np.mean(v), np.std(v)) for k, v in all_hz.items() if v},
    }


def run_comparison(n_seeds=10):
    """Run baseline, residual, and hybrid models — print side-by-side."""
    from .robust_eval import run_multi_seed

    print("#" * 70)
    print("  BASELINE vs RESIDUAL vs HYBRID COMPARISON")
    print("#" * 70)

    print(f"\n{'━' * 70}")
    print("  [A] Baseline (direct prediction, 23 feats)")
    print(f"{'━' * 70}")
    baseline = run_multi_seed(n_seeds)

    print(f"\n{'━' * 70}")
    print("  [B] Residual (21 feats, prior removed from input)")
    print(f"{'━' * 70}")
    residual = run_residual_multi_seed(n_seeds)

    print(f"\n{'━' * 70}")
    print("  [C] Hybrid (all 23 feats + residual target)")
    print(f"{'━' * 70}")
    hybrid = run_hybrid_multi_seed(n_seeds)

    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<30s} {'Test MAE':>12s} {'Test ±3d':>10s} {'Val MAE':>10s}")
    print(f"  {'─' * 62}")
    for name, r in [("Baseline (direct)", baseline),
                     ("Residual (prior removed)", residual),
                     ("Hybrid (all feats+resid)", hybrid)]:
        print(f"  {name:<30s} {r['test_mae_mean']:.3f}±{r['test_mae_std']:.3f}"
              f"   {r['test_acc3_mean']:.3f}"
              f"      {r['val_mae_mean']:.3f}")

    best = min([baseline, residual, hybrid], key=lambda x: x["test_mae_mean"])
    for name, r in [("Baseline", baseline), ("Residual", residual), ("Hybrid", hybrid)]:
        if r is best:
            print(f"\n  ★ Best: {name}")

    return baseline, residual, hybrid

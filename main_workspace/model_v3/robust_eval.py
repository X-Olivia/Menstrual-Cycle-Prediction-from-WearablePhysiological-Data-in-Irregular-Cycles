"""Robust evaluation: multi-seed subject split to reduce variance from small test sets."""
import numpy as np

from .config import ALL_FEATURES, TEST_SUBJECT_RATIO, LGB_PARAMS
from .dataset import load_data, subject_split
from .train_lgb import train_lightgbm, predict
from .evaluate import compute_metrics, stratified_metrics


def run_multi_seed(n_seeds=10, features_csv=None, feature_list=None):
    """Run experiment with different random splits and aggregate results."""
    print("=" * 60)
    print(f"  Multi-Seed Evaluation ({n_seeds} splits)")
    print("=" * 60)

    df, available = load_data(features_csv=features_csv) if features_csv else load_data()
    use_features = feature_list or ALL_FEATURES
    features = [f for f in use_features if f in available]

    all_test_mae = []
    all_test_acc3 = []
    all_val_mae = []
    all_hz = {label: [] for label in ["1-5", "6-10", "11-15", "16-20", "21+"]}

    for seed in range(42, 42 + n_seeds):
        train_subj, val_subj, test_subj = subject_split(
            df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        train_mask = df["id"].isin(train_subj)
        val_mask = df["id"].isin(val_subj)
        test_mask = df["id"].isin(test_subj)

        X_tr, y_tr = df.loc[train_mask, features].values, df.loc[train_mask, "days_until_next_menses"].values
        X_val, y_val = df.loc[val_mask, features].values, df.loc[val_mask, "days_until_next_menses"].values
        X_te, y_te = df.loc[test_mask, features].values, df.loc[test_mask, "days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features)
        pred_test = predict(model, X_te)
        pred_val = predict(model, X_val)

        test_m = compute_metrics(pred_test, y_te)
        val_m = compute_metrics(pred_val, y_val)
        hz = stratified_metrics(pred_test, y_te)

        all_test_mae.append(test_m["mae"])
        all_test_acc3.append(test_m["acc_3d"])
        all_val_mae.append(val_m["mae"])
        for label in all_hz:
            if label in hz and hz[label]["n"] > 0:
                all_hz[label].append(hz[label]["mae"])

        print(f"  seed={seed}: test MAE={test_m['mae']:.3f}, ±3d={test_m['acc_3d']:.3f}, "
              f"val MAE={val_m['mae']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  AGGREGATE ({n_seeds} seeds)")
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


if __name__ == "__main__":
    run_multi_seed()

"""Optuna hyperparameter optimization for LightGBM menstrual cycle prediction."""
import numpy as np
import optuna
import lightgbm as lgb

from model.config import (
    ALL_FEATURES, ALL_FEATURES_V6,
    FEATURES_V4_CSV, FEATURES_V6_CSV,
    TEST_SUBJECT_RATIO,
)
from model.dataset import load_data, subject_split
from model.evaluate import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(trial, df, features, n_seeds=3):
    """Optuna objective: mean validation MAE over n_seeds subject splits."""

    params = {
        "objective": "huber",
        "metric": "mae",
        "huber_delta": trial.suggest_float("huber_delta", 1.0, 5.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 48),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        "verbose": -1,
        "seed": 42,
    }

    val_maes = []
    for seed in range(42, 42 + n_seeds):
        train_subj, val_subj, test_subj = subject_split(
            df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        train_mask = df["id"].isin(train_subj)
        val_mask = df["id"].isin(val_subj)

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, "days_until_next_menses"].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, "days_until_next_menses"].values

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=features, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        pred_val = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 1.0, None)
        val_m = compute_metrics(pred_val, y_val)
        val_maes.append(val_m["mae"])

    return np.mean(val_maes)


def tune(n_trials=80, n_seeds=3, use_v6=False):
    """Run Optuna hyperparameter search.

    Args:
        n_trials: Number of Optuna trials.
        n_seeds: Seeds per trial for validation.
        use_v6: If True, use v6 features; otherwise v4.
    """
    features_csv = FEATURES_V6_CSV if use_v6 else FEATURES_V4_CSV
    feature_list = ALL_FEATURES_V6 if use_v6 else ALL_FEATURES

    df, available = load_data(features_csv=features_csv)
    features = [f for f in feature_list if f in available]

    tag = "v6" if use_v6 else "v4"
    print(f"{'=' * 60}")
    print(f"  Optuna Hyperparameter Tuning ({tag}, {len(features)} features)")
    print(f"  {n_trials} trials × {n_seeds} seeds")
    print(f"{'=' * 60}")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, df, features, n_seeds=n_seeds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  BEST TRIAL")
    print(f"{'=' * 60}")
    best = study.best_trial
    print(f"  Val MAE (mean of {n_seeds} seeds): {best.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.4f}")
        else:
            print(f"    {k:25s}: {v}")

    # Format as LGB_PARAMS dict for easy copy
    print(f"\n  # Copy-paste to config.py:")
    print(f"  LGB_PARAMS_TUNED = {{")
    print(f'      "objective": "huber",')
    print(f'      "metric": "mae",')
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f'      "{k}": {v:.4f},')
        else:
            print(f'      "{k}": {v},')
    print(f'      "verbose": -1,')
    print(f'      "seed": 42,')
    print(f'      "n_estimators": 2000,')
    print(f'      "early_stopping_rounds": 80,')
    print(f"  }}")

    return study


def tune_and_evaluate(n_trials=80, n_seeds_tune=3, n_seeds_eval=10, use_v6=False):
    """Tune hyperparameters, then evaluate best config with 10-seed robust eval."""
    study = tune(n_trials=n_trials, n_seeds=n_seeds_tune, use_v6=use_v6)

    features_csv = FEATURES_V6_CSV if use_v6 else FEATURES_V4_CSV
    feature_list = ALL_FEATURES_V6 if use_v6 else ALL_FEATURES
    tag = "v6" if use_v6 else "v4"

    df, available = load_data(features_csv=features_csv)
    features = [f for f in feature_list if f in available]

    best_params = study.best_params.copy()
    best_params.update({"objective": "huber", "metric": "mae", "verbose": -1, "seed": 42})

    print(f"\n{'=' * 60}")
    print(f"  Robust Evaluation with Tuned Params ({tag})")
    print(f"  {n_seeds_eval} seeds")
    print(f"{'=' * 60}")

    from .train_lgb import predict
    from .evaluate import stratified_metrics

    all_test_mae, all_test_acc3, all_val_mae = [], [], []
    all_hz = {k: [] for k in ["1-5", "6-10", "11-15", "16-20", "21+"]}

    for seed in range(42, 42 + n_seeds_eval):
        train_subj, val_subj, test_subj = subject_split(
            df, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        train_mask = df["id"].isin(train_subj)
        val_mask = df["id"].isin(val_subj)
        test_mask = df["id"].isin(test_subj)

        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, "days_until_next_menses"].values
        X_val = df.loc[val_mask, features].values
        y_val = df.loc[val_mask, "days_until_next_menses"].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, "days_until_next_menses"].values

        train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
        val_set = lgb.Dataset(X_val, label=y_val, feature_name=features, reference=train_set)

        model = lgb.train(
            best_params,
            train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

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

        print(f"  seed={seed}: test MAE={test_m['mae']:.3f}, ±3d={test_m['acc_3d']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  TUNED {tag.upper()} AGGREGATE ({n_seeds_eval} seeds)")
    print(f"{'=' * 60}")
    print(f"  Test MAE:  {np.mean(all_test_mae):.3f} ± {np.std(all_test_mae):.3f}")
    print(f"  Test ±3d:  {np.mean(all_test_acc3):.3f} ± {np.std(all_test_acc3):.3f}")
    print(f"  Val MAE:   {np.mean(all_val_mae):.3f} ± {np.std(all_val_mae):.3f}")
    print(f"\n  Horizon MAE:")
    for k in ["1-5", "6-10", "11-15", "16-20", "21+"]:
        vals = all_hz[k]
        if vals:
            print(f"    {k:>8s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    return {
        "study": study,
        "test_mae_mean": np.mean(all_test_mae),
        "test_mae_std": np.std(all_test_mae),
        "test_acc3_mean": np.mean(all_test_acc3),
    }


if __name__ == "__main__":
    tune_and_evaluate(n_trials=80, use_v6=False)

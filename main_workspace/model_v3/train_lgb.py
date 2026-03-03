"""LightGBM training: day-level regression with Huber loss."""
import numpy as np
import lightgbm as lgb

from .config import LGB_PARAMS


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names, params=None):
    """Train LightGBM with early stopping on validation MAE."""
    params = params or LGB_PARAMS.copy()

    n_estimators = params.pop("n_estimators", 1000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_set)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    print(f"[train] Best iteration: {model.best_iteration}")
    print(f"[train] Best val MAE: {model.best_score['val']['l1']:.4f}")

    return model


def predict(model, X):
    """Predict days_until_next_menses, clamp to ≥ 1."""
    pred = model.predict(X, num_iteration=model.best_iteration)
    return np.clip(pred, 1.0, None)


def feature_importance(model, feature_names, top_k=20):
    """Print top-k feature importances."""
    imp = model.feature_importance(importance_type="gain")
    indices = np.argsort(imp)[::-1]
    print(f"\n[importance] Top-{top_k} features (gain):")
    for i, idx in enumerate(indices[:top_k]):
        print(f"  {i + 1:2d}. {feature_names[idx]:35s} {imp[idx]:10.1f}")
    return {feature_names[i]: float(imp[i]) for i in indices}

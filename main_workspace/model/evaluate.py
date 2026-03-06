"""Evaluation metrics: MAE, ±k-day accuracy, stratified by horizon."""
import numpy as np


HORIZON_BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, None)]


def compute_metrics(pred, true):
    """Compute MAE and ±k-day accuracy."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = np.abs(pred - true)
    return {
        "n": len(pred),
        "mae": float(err.mean()),
        "acc_1d": float((err < 1.5).mean()),
        "acc_2d": float((err < 2.5).mean()),
        "acc_3d": float((err < 3.5).mean()),
    }


def stratified_metrics(pred, true):
    """Metrics stratified by true days_until_next_menses horizon."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    out = {}
    for low, high in HORIZON_BINS:
        if high is None:
            mask = true >= low
            label = f"{low}+"
        else:
            mask = (true >= low) & (true <= high)
            label = f"{low}-{high}"
        if mask.sum() == 0:
            out[label] = {"n": 0, "mae": float("nan"),
                          "acc_1d": float("nan"), "acc_2d": float("nan"), "acc_3d": float("nan")}
        else:
            out[label] = compute_metrics(pred[mask], true[mask])
    return out


def print_metrics(metrics, name=""):
    """Pretty-print overall and stratified metrics."""
    overall = compute_metrics(metrics["pred"], metrics["true"])
    by_hz = stratified_metrics(metrics["pred"], metrics["true"])

    print(f"\n{'=' * 50}")
    print(f"  {name} Results")
    print(f"{'=' * 50}")
    print(f"  MAE: {overall['mae']:.3f}")
    print(f"  ±1d: {overall['acc_1d']:.3f}  ±2d: {overall['acc_2d']:.3f}  ±3d: {overall['acc_3d']:.3f}")
    print(f"\n  {'horizon':>8s} {'n':>5s} {'MAE':>6s} {'±1d':>6s} {'±2d':>6s} {'±3d':>6s}")
    print(f"  {'-' * 40}")
    for label in ("1-5", "6-10", "11-15", "16-20", "21+"):
        s = by_hz.get(label, {})
        n = s.get("n", 0)
        if n == 0:
            print(f"  {label:>8s} {n:5d}      -      -      -      -")
        else:
            print(f"  {label:>8s} {n:5d} {s['mae']:6.2f} {s['acc_1d']:6.3f} {s['acc_2d']:6.3f} {s['acc_3d']:6.3f}")

    # Prediction range
    p = metrics["pred"]
    print(f"\n  Pred range: {p.min():.1f} ~ {p.max():.1f}")
    print(f"  Label range: {metrics['true'].min():.1f} ~ {metrics['true'].max():.1f}")

    return overall, by_hz

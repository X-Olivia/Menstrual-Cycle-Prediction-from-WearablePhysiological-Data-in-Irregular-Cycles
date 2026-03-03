"""Experiment entry: data load, subject split, two-stage training, fixed test evaluation."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    RANDOM_SEED,
    TEST_SUBJECT_RATIO,
    N_FOLDS,
)
from .dataset import prepare_all_sequences, CycleSequenceDataset, collate_cycle_sequences
from .split import split_fixed_test, kfold_trainval
from .net import CycleModel
from .train import run_stage1, run_stage2
from .eval_metrics import evaluate


def get_subject_ids(sequences):
    return [s["id"] for s in sequences]


def run_one_fold(sequences, train_subjects, val_subjects, device, fold_name=""):
    train_ds = CycleSequenceDataset(sequences, train_subjects)
    val_ds = CycleSequenceDataset(sequences, val_subjects)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_cycle_sequences,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_cycle_sequences,
    )
    model = CycleModel().to(device)
    model = run_stage1(model, train_loader, val_loader, device)
    model = run_stage2(model, train_loader, val_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    return model, val_metrics


def run_experiment(
    cycle_csv=None,
    full_csv=None,
    test_ratio=TEST_SUBJECT_RATIO,
    n_folds=N_FOLDS,
    seed=RANDOM_SEED,
):
    from .config import CYCLE_CSV, SLEEP_CSV

    cycle_csv = cycle_csv or CYCLE_CSV
    full_csv = full_csv or SLEEP_CSV   # P2: use sleep window as primary input

    sequences = prepare_all_sequences(cycle_csv, full_csv)
    # Label range check: y should be days until next menses, ~1–35; if max>40 likely label misalignment
    all_y = np.concatenate(
        [s["y_menses"][s["mask_menses"]].flatten() for s in sequences if s["mask_menses"].any()]
    )
    if len(all_y) > 0:
        print("[Labels] days_until_next_menses range:", float(np.nanmin(all_y)), "~", float(np.nanmax(all_y)))
        print("[Labels] percentiles 0/25/50/75/100:", np.nanpercentile(all_y, [0, 25, 50, 75, 100]).tolist())
    ids = get_subject_ids(sequences)
    # One id per sequence for splitting
    subject_ids = list(set(ids))

    test_subjects, trainval_subjects = split_fixed_test(
        subject_ids, test_ratio, seed
    )
    trainval_subjects = list(trainval_subjects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # K-fold for train/val only; train on one fold, then evaluate once on fixed test
    folds = list(kfold_trainval(trainval_subjects, n_folds, seed))
    train_subjects, val_subjects = folds[0]
    model, val_metrics = run_one_fold(
        sequences, train_subjects, val_subjects, device, fold_name="fold0"
    )

    # Evaluate once on fixed test set
    test_ds = CycleSequenceDataset(sequences, test_subjects)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_cycle_sequences,
    )
    test_metrics = evaluate(model, test_loader, device)

    # Stratified evaluation by horizon (true days_until_next_menses): 1-5, 6-10, 11-15, 16-20, 21+
    def _print_by_horizon(by_horizon, name):
        print("[Stratified by horizon] %s" % name)
        print("  horizon   n     MAE   acc_1d  acc_2d  acc_3d")
        for label in ("1-5", "6-10", "11-15", "16-20", "21+"):
            s = by_horizon.get(label, {})
            n = s.get("n", 0)
            mae = s.get("mae", float("nan"))
            a1 = s.get("acc_1d", float("nan"))
            a2 = s.get("acc_2d", float("nan"))
            a3 = s.get("acc_3d", float("nan"))
            if np.isnan(mae):
                print("  %6s  %5d   -      -      -      -" % (label, n))
            else:
                print("  %6s  %5d  %5.2f  %.3f  %.3f  %.3f" % (label, n, mae, a1, a2, a3))
        print()
    if "by_horizon" in test_metrics:
        _print_by_horizon(test_metrics["by_horizon"], "Test")
    if "by_horizon" in val_metrics:
        _print_by_horizon(val_metrics["by_horizon"], "Val")

    # Prediction range check: labels and predictions align by (batch, timestep); same index after mask = same row
    pred_m, true_m, mask_m = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            lengths = batch["lengths"].to(device)
            y_m_pred_log, _ = model(X, lengths)
            pred_m.append(torch.expm1(y_m_pred_log).cpu().numpy())
            true_m.append(batch["y_menses"].numpy())
            mask_m.append(batch["mask_menses"].numpy())
    pred_flat = np.concatenate([p.flatten() for p in pred_m])
    true_flat = np.concatenate([t.flatten() for t in true_m])
    mask_flat = np.concatenate([m.flatten() for m in mask_m])
    p = pred_flat[mask_flat]
    t = true_flat[mask_flat]
    if len(p) > 0:
        print("[Predictions] range:", np.nanmin(p), "~", np.nanmax(p))
        print("[Predictions] percentiles 0/25/50/75/100:", np.nanpercentile(p, [0, 25, 50, 75, 100]))
        print("[Label vs prediction] Model outputs log1p(days), converted to days. Same row = same (day in cycle): idx | label(true) | pred(days) | error")
        for i in range(min(15, len(p))):
            err = float(t[i]) - float(p[i])
            print("  %3d  |  %6.1f    | %6.2f  | %+.1f" % (i, float(t[i]), float(p[i]), err))

    return {
        "val": val_metrics,
        "test": test_metrics,
        "model": model,
    }


if __name__ == "__main__":
    result = run_experiment()
    print("Val:", result["val"])
    print("Test:", result["test"])

"""Evaluation metrics: main task MAE/±k-day accuracy; auxiliary task ovulation metrics."""
from collections import defaultdict

import torch
import numpy as np


# ── Menses (main task) ────────────────────────────────────────────────────────

def menses_mae(pred, target, mask):
    if mask.sum() == 0:
        return float("nan")
    p = pred[mask].cpu().numpy()
    t = target[mask].cpu().numpy()
    return np.abs(p - t).mean()


def menses_accuracy_within_k(pred, target, mask, k):
    if mask.sum() == 0:
        return float("nan")
    p = pred[mask].cpu().numpy()
    t = target[mask].cpu().numpy()
    return (np.abs(p - t) <= k).mean()


HORIZON_BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, None)]


def _stratify_by_horizon(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    out  = {}
    for low, high in HORIZON_BINS:
        if high is None:
            mask  = true >= low
            label = f"{low}+"
        else:
            mask  = (true >= low) & (true <= high)
            label = f"{low}-{high}"
        n = mask.sum()
        if n == 0:
            out[label] = {"n": 0, "mae": float("nan"),
                          "acc_1d": float("nan"), "acc_2d": float("nan"), "acc_3d": float("nan")}
            continue
        p, t = pred[mask], true[mask]
        out[label] = {
            "n":      int(n),
            "mae":    np.abs(p - t).mean(),
            "acc_1d": (np.abs(p - t) <= 1).mean(),
            "acc_2d": (np.abs(p - t) <= 2).mean(),
            "acc_3d": (np.abs(p - t) <= 3).mean(),
        }
    return out


def evaluate_by_horizon(pred, target, mask):
    p = pred[mask].cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)[mask]
    t = target[mask].cpu().numpy() if hasattr(target, "cpu") else np.asarray(target)[mask]
    return _stratify_by_horizon(p, t)


# ── Ovulation (auxiliary task) ────────────────────────────────────────────────

def ovulation_correlation(pred, target, mask):
    """Per-day Pearson correlation between predicted scores and soft labels."""
    if mask.sum() < 2:
        return float("nan")
    p = pred[mask].cpu().numpy()
    t = target[mask].cpu().numpy()
    return float(np.corrcoef(p, t)[0, 1])


def ovulation_cycle_accuracy(
    pred_scores,        # 1-D numpy array, all days in evaluation set (in order)
    cycle_ids,          # list of (subject_id, cycle_key) for each day, same length
    ovulation_probs,    # 1-D numpy array of ovulation_prob_fused for each day
    threshold=0.05,
):
    """Cycle-level ovulation accuracy: for each cycle, top-scored day is in the acceptable set.
    Matches the probe's CycleAcc definition so GRU and LR baselines are directly comparable.
    Returns: (cycle_acc, correct, total, random_baseline)
    """
    cycle_to_indices = defaultdict(list)
    for j, key in enumerate(cycle_ids):
        cycle_to_indices[key].append(j)

    correct = total = 0
    rand_sum = 0.0
    for key, indices in cycle_to_indices.items():
        if not indices:
            continue
        total += 1
        n_acceptable = sum(1 for i in indices if ovulation_probs[i] >= threshold)
        rand_sum    += (n_acceptable / len(indices)) if indices else 0.0
        pred_idx     = indices[int(np.argmax(pred_scores[indices]))]
        if ovulation_probs[pred_idx] >= threshold:
            correct += 1

    acc          = (correct / total)   if total else float("nan")
    rand_base    = (rand_sum  / total) if total else float("nan")
    return acc, correct, total, rand_base


# ── Full evaluation pass ───────────────────────────────────────────────────────

def evaluate(model, loader, device, ov_threshold=0.05):
    """Aggregate main and auxiliary metrics.
    Main task model outputs log1p(days); converts back to days here for MAE/acc.
    Ovulation: both per-day correlation AND cycle-level accuracy.
    """
    model.eval()
    all_pred_m,  all_true_m,  all_mask_m  = [], [], []
    all_pred_ov, all_true_ov, all_mask_ov = [], [], []
    all_cycle_ids = []  # (id, small_group_key) per day — needed for CycleAcc

    with torch.no_grad():
        for batch in loader:
            X        = batch["X"].to(device)
            lengths  = batch["lengths"].to(device)
            y_m_pred_log, p_ov_pred = model(X, lengths)
            pred_days = torch.expm1(y_m_pred_log)
            all_pred_m.append(pred_days)
            all_true_m.append(batch["y_menses"].to(device))
            all_mask_m.append(batch["mask_menses"].to(device))
            all_pred_ov.append(p_ov_pred)
            all_true_ov.append(batch["y_ovulation"].to(device))
            all_mask_ov.append(batch["mask_ovulation"].to(device))
            # Collect cycle ids per day (batch x time)
            if "cycle_ids" in batch:
                all_cycle_ids.extend(batch["cycle_ids"])  # list of lists

    pred_m  = torch.cat([x.flatten() for x in all_pred_m])
    true_m  = torch.cat([x.flatten() for x in all_true_m])
    mask_m  = torch.cat([x.flatten() for x in all_mask_m])
    pred_ov = torch.cat([x.flatten() for x in all_pred_ov])
    true_ov = torch.cat([x.flatten() for x in all_true_ov])
    mask_ov = torch.cat([x.flatten() for x in all_mask_ov])

    mae  = menses_mae(pred_m, true_m, mask_m)
    acc1 = menses_accuracy_within_k(pred_m, true_m, mask_m, 1)
    acc2 = menses_accuracy_within_k(pred_m, true_m, mask_m, 2)
    acc3 = menses_accuracy_within_k(pred_m, true_m, mask_m, 3)

    corr_ov = ovulation_correlation(pred_ov, true_ov, mask_ov)
    by_horizon = evaluate_by_horizon(pred_m, true_m, mask_m)

    # Cycle-level ovulation accuracy (same metric as probe CycleAcc)
    ov_cycle_acc = ov_cycle_correct = ov_cycle_total = ov_rand_base = float("nan")
    if all_cycle_ids:
        flat_cycle_ids = [cid for row in all_cycle_ids for cid in row]
        flat_pred_ov   = pred_ov.cpu().numpy()
        flat_true_ov   = true_ov.cpu().numpy()
        ov_cycle_acc, ov_cycle_correct, ov_cycle_total, ov_rand_base = ovulation_cycle_accuracy(
            flat_pred_ov, flat_cycle_ids, flat_true_ov, threshold=ov_threshold
        )

    return {
        "mae":             mae,
        "acc_1d":          acc1,
        "acc_2d":          acc2,
        "acc_3d":          acc3,
        "ovulation_corr":  corr_ov,
        "ov_cycle_acc":    ov_cycle_acc,
        "ov_cycle_correct": ov_cycle_correct,
        "ov_cycle_total":  ov_cycle_total,
        "ov_rand_base":    ov_rand_base,
        "by_horizon":      by_horizon,
    }

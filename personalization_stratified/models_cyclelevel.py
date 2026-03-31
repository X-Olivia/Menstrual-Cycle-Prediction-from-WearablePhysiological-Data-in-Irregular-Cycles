from __future__ import annotations

from typing import Dict, List

import numpy as np


ANCHORS_PRE = [-7, -3, -1]
ANCHORS_POST = [2, 5, 10]
ANCHORS_ALL = ANCHORS_PRE + ANCHORS_POST


def weighted_cycle_len_mean(cycle_lengths: List[int]) -> float:
    if not cycle_lengths:
        return 28.0
    ws = np.exp(np.linspace(-1, 0, len(cycle_lengths)))
    return float(np.average(np.asarray(cycle_lengths, dtype=float), weights=ws))


def global_cycle_mean(cycle_series: Dict[str, dict]) -> float:
    vals = [int(v["cycle_len"]) for v in cycle_series.values() if "cycle_len" in v]
    if not vals:
        return 28.0
    return float(np.mean(vals))


def predict_menses_start_b1(
    ov_est: int | None,
    anchor_day: int,
    pop_cycle_len: float,
    pop_luteal_len: float,
) -> float:
    # Population-only baseline: never use subject history.
    # If countdown is enabled, use ovulation-based menses start.
    # Otherwise fallback to calendar estimate.
    use_countdown = ov_est is not None and ov_est > 3 and anchor_day >= ov_est + 2
    if use_countdown:
        return float(ov_est + pop_luteal_len)
    return float(pop_cycle_len)


def predict_menses_start_b2(
    ov_est: int | None,
    anchor_day: int,
    hist_cycle_lengths: List[int],
    pop_cycle_len: float,
    pop_luteal_len: float,
) -> float:
    # Personalized cycle-length baseline: personalize calendar fallback only.
    # To avoid LH-noise amplification, luteal countdown uses population luteal length.
    acl = weighted_cycle_len_mean(hist_cycle_lengths) if hist_cycle_lengths else pop_cycle_len
    use_countdown = ov_est is not None and ov_est > 3 and anchor_day >= ov_est + 2
    if use_countdown:
        return float(ov_est + pop_luteal_len)
    return float(acl)


def estimate_b1_history_bias(
    history_sgks: List[str],
    cycle_series: Dict[str, dict],
    lh_dict: Dict[str, int],
    det_dict: Dict[str, int],
    pop_cycle_len: float,
    pop_luteal_len: float,
) -> float:
    errs: List[float] = []
    for sgk in history_sgks:
        if sgk not in cycle_series or sgk not in lh_dict:
            continue
        actual = float(cycle_series[sgk]["cycle_len"])
        ov_true = int(lh_dict[sgk])
        ov_est = det_dict.get(sgk)
        for k in ANCHORS_ALL:
            anchor_day = ov_true + k
            if not (0 <= anchor_day < actual):
                continue
            pred_menses_start = predict_menses_start_b1(
                ov_est=ov_est,
                anchor_day=anchor_day,
                pop_cycle_len=pop_cycle_len,
                pop_luteal_len=pop_luteal_len,
            )
            pred_remaining = pred_menses_start - anchor_day
            true_remaining = actual - anchor_day
            errs.append(pred_remaining - true_remaining)
    if not errs:
        return 0.0
    return float(np.mean(np.asarray(errs)))

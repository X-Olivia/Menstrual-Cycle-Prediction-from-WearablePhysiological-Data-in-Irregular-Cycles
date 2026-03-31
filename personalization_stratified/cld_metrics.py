"""
Consecutive cycle length difference (CLD) metrics, aligned with app-scale literature
(e.g. median |L_{t+1} - L_t| per user).

Not a replacement for CV: use both — CV scales with mean cycle length; CLD is absolute-day jitter.
"""
from __future__ import annotations

from typing import List

import numpy as np


def abs_cld_sequence(lengths: List[int]) -> List[float]:
    """|L_{i+1} - L_i| for consecutive observed cycles (chronological order)."""
    if len(lengths) < 2:
        return []
    arr = np.asarray(lengths, dtype=float)
    return [float(abs(arr[i + 1] - arr[i])) for i in range(len(arr) - 1)]


def median_abs_cld(lengths: List[int]) -> float:
    diffs = abs_cld_sequence(lengths)
    if not diffs:
        return float("nan")
    return float(np.median(np.asarray(diffs, dtype=float)))


def mean_abs_cld(lengths: List[int]) -> float:
    diffs = abs_cld_sequence(lengths)
    if not diffs:
        return float("nan")
    return float(np.mean(np.asarray(diffs, dtype=float)))


def is_high_vol_cld_strict(median_abs: float, threshold_days: float = 9.0) -> bool:
    """Literature-style strict high-vol flag when median |CLD| exceeds threshold."""
    return bool(np.isfinite(median_abs) and median_abs > threshold_days)

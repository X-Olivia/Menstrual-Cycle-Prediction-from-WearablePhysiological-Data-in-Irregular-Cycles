"""
EXPERIMENTAL post-trigger localizer update policies.

freeze / clamp / sticky / soft_sticky / bounded_monotone — kept out of the core
module so the mainline codepath only documents none + score_smooth comparator.
"""
from __future__ import annotations

from protocol import (
    MAX_RIGHT_MARGIN_DAYS,
    MIN_DETECTION_DAY,
    PHASECLS_MONOTONE_BACK_MARGIN,
)


def apply_stabilization_experimental(
    current_ov_est,
    current_score,
    ov_est,
    ov_score,
    day_idx,
    stabilization_policy,
    clamp_radius,
    sticky_radius,
    sticky_improve_margin,
    monotone_back_margin=None,
):
    if monotone_back_margin is None:
        monotone_back_margin = float(PHASECLS_MONOTONE_BACK_MARGIN)

    if stabilization_policy == "freeze":
        if current_ov_est is None and ov_est is not None:
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        return current_ov_est, current_score

    if stabilization_policy == "clamp":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        lo = max(MIN_DETECTION_DAY, int(current_ov_est) - int(clamp_radius))
        hi = min(day_idx, int(current_ov_est) + int(clamp_radius))
        if hi < lo:
            hi = lo
        return int(min(max(int(ov_est), lo), hi)), float(ov_score if ov_score is not None else current_score)

    if stabilization_policy == "sticky":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        ov_est = int(ov_est)
        ov_score = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        if ov_est == int(current_ov_est):
            return int(current_ov_est), max(float(current_score or 0.0), ov_score)
        if abs(ov_est - int(current_ov_est)) <= int(sticky_radius):
            baseline_score = float(current_score or 0.0)
            if ov_score >= baseline_score + float(sticky_improve_margin):
                return ov_est, ov_score
        return int(current_ov_est), float(current_score if current_score is not None else 0.0)

    if stabilization_policy == "soft_sticky":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            ne = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
            return ne, float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        cur = int(current_ov_est)
        new = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
        ns = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        cs = float(current_score if current_score is not None else 0.0)
        if new == cur:
            return cur, max(cs, ns)
        if abs(new - cur) <= int(sticky_radius) and ns >= cs + float(sticky_improve_margin):
            out = cur + (1 if new > cur else -1)
            return out, ns
        return cur, cs

    if stabilization_policy == "bounded_monotone":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            ne = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
            return ne, float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        cur = int(current_ov_est)
        new = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
        ns = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        cs = float(current_score if current_score is not None else 0.0)
        mb = float(monotone_back_margin)
        hi = int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)
        if new == cur:
            return cur, max(cs, ns)
        if new < cur:
            out = max(int(new), cur - 1)
            out = max(MIN_DETECTION_DAY, min(out, hi))
            return out, max(ns, cs)
        if ns < cs + mb:
            return cur, cs
        out = min(cur + 1, int(new))
        out = max(MIN_DETECTION_DAY, min(out, hi))
        return out, ns

    raise ValueError(f"Unknown experimental stabilization_policy: {stabilization_policy}")

"""
MAINLINE post-trigger localizer state: `none` only at this layer.

`score_smooth` is implemented by smoothing candidates upstream, then applying `none`.
All other policies dispatch to experimental.stabilization_extra (same numerics as before).
"""
from __future__ import annotations

_EXPERIMENTAL_POLICIES = frozenset({"freeze", "clamp", "sticky", "soft_sticky", "bounded_monotone"})


def apply_stabilization(
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
    if stabilization_policy == "none":
        if ov_est is None:
            return current_ov_est, current_score
        return int(ov_est), float(ov_score if ov_score is not None else 0.0)

    if stabilization_policy in _EXPERIMENTAL_POLICIES:
        from experimental.stabilization_extra import apply_stabilization_experimental

        return apply_stabilization_experimental(
            current_ov_est,
            current_score,
            ov_est,
            ov_score,
            day_idx,
            stabilization_policy,
            clamp_radius,
            sticky_radius,
            sticky_improve_margin,
            monotone_back_margin=monotone_back_margin,
        )

    raise ValueError(f"Unknown stabilization_policy: {stabilization_policy}")

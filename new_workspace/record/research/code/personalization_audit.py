"""
Active-effect audit for L1/L2/L3 personalization layers.

Quantifies how often each layer changes daily ovulation tracks, in what direction,
and pairs per-cycle PostTrigger MAE (same definition as menses.evaluate_prefix_post_trigger)
for base wearable vs personalized sequences.

PostTrigger cohort state (ACL/LUT via s_pclen/s_plut) is advanced by one full pass over
``subj_order`` per detection track: one pass for base ``det_by_day`` and one for personalized.
When multiple L2 variants share the same base wearable, the base pass may be computed once
and reused (see ``audit_l2_active_effect(..., cached_base_post_trigger_df=...)``).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paths import PIPELINE_DIR

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from menses import (  # type: ignore  # noqa: E402
    DEFAULT_HISTORY_CYCLE_LEN,
    DEFAULT_POPULATION_LUTEAL_LENGTH,
    MENSES_LUTEAL_UPDATE_MAX,
    MENSES_LUTEAL_UPDATE_MIN,
    _predict_menses_logic_core,
    _predict_menses_static_baseline_day,
)


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (int, float)):
        return x
    return str(x)


def _first_nonnull_1based(seq: list[Any]) -> float:
    for i, v in enumerate(seq):
        if v is not None:
            return float(i + 1)
    return float("nan")


def _post_trigger_signed_errors_for_cycle(
    cycle_row: dict[str, Any],
    det_seq: list[int | None],
    conf_seq: list[float],
    actual: int,
    acl: float,
    lut: float,
    hist_fols: list[float],
    baseline_mode: str,
    use_stability_gate: bool = False,
) -> list[float]:
    """Mirror evaluate_prefix_post_trigger inner loop for a single cycle; signed errors only."""
    stable_days_required = 2
    stable_tol_days = 1
    errs: list[float] = []
    countdown_started = False
    last_non_none_ov: int | None = None
    consecutive_stable_non_none = 0

    for day_idx in range(actual):
        cycle_day = day_idx + 1
        ov_est_today = det_seq[day_idx] if day_idx < len(det_seq) else None
        conf_today = conf_seq[day_idx] if day_idx < len(conf_seq) else 0.5

        if ov_est_today is not None and last_non_none_ov is not None:
            if abs(int(ov_est_today) - int(last_non_none_ov)) <= stable_tol_days:
                consecutive_stable_non_none += 1
            else:
                consecutive_stable_non_none = 1
        elif ov_est_today is not None:
            consecutive_stable_non_none = 1
        else:
            consecutive_stable_non_none = 0

        if ov_est_today is not None:
            last_non_none_ov = int(ov_est_today)

        use_countdown = ov_est_today is not None
        if use_stability_gate:
            use_countdown = ov_est_today is not None and consecutive_stable_non_none >= stable_days_required

        if use_countdown and not countdown_started:
            countdown_started = True

        if not countdown_started:
            continue

        if baseline_mode == "static" and not use_countdown:
            pred_menses_day = _predict_menses_static_baseline_day(acl)
        else:
            pred_menses_day = _predict_menses_logic_core(
                cycle_row,
                day_idx,
                ov_est_today,
                conf_today,
                hist_fols,
                acl,
                lut,
                use_countdown,
            )
        pred_remaining = pred_menses_day - cycle_day
        true_remaining = actual - cycle_day
        errs.append(float(pred_remaining - true_remaining))

    return errs


def _mae_from_signed(errs: list[float]) -> float:
    if not errs:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(errs, dtype=float))))


def _iter_cycles(
    subj_order: dict[Any, list[str]],
    cycle_series: dict[str, dict[str, Any]],
) -> list[tuple[Any, str]]:
    rows: list[tuple[Any, str]] = []
    for uid, sgks in subj_order.items():
        if isinstance(sgks, (int, str)):
            sgks = [str(sgks)]
        for sgk in sgks:
            if sgk in cycle_series:
                rows.append((uid, sgk))
    return rows


def _acl_lut_for_uid(
    uid: Any,
    sgk: str,
    cs: dict[str, dict[str, Any]],
    s_plut: dict[Any, list],
    s_pclen: dict[Any, list],
    pop_luteal_len: float,
    custom_cycle_priors: dict[str, float] | None,
    use_population_only_prior: bool,
) -> tuple[float, float]:
    actual = int(cs[sgk]["cycle_len"])
    pl, pc = s_plut[uid], s_pclen[uid]
    lut = float(np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len)
    acl = (
        float(np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))))
        if pc
        else float(DEFAULT_HISTORY_CYCLE_LEN)
    )
    if use_population_only_prior:
        lut = float(pop_luteal_len)
        acl = float(DEFAULT_HISTORY_CYCLE_LEN)
    if custom_cycle_priors and sgk in custom_cycle_priors:
        acl = float(custom_cycle_priors[sgk])
    return acl, lut


def _update_state_after_cycle(
    uid: Any,
    sgk: str,
    cs: dict[str, dict[str, Any]],
    det_seq: list[int | None],
    s_plut: dict[Any, list],
    s_pclen: dict[Any, list],
) -> None:
    actual = int(cs[sgk]["cycle_len"])
    s_pclen[uid].append(actual)
    final_ov_est = next((v for v in reversed(det_seq) if v is not None), None)
    if final_ov_est is not None:
        el = actual - int(final_ov_est)
        if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
            s_plut[uid].append(el)


def audit_l1_active_effect(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    profile_df: pd.DataFrame,
    base_det: dict[str, list[int | None]],
    pers_det: dict[str, list[int | None]],
) -> dict[str, Any]:
    prof = profile_df.set_index("small_group_key")
    rows: list[dict[str, Any]] = []
    suppress_days: list[int] = []
    n_cycles = 0
    n_ready = 0
    n_cycles_any_suppress = 0
    n_suppress_days_total = 0
    n_recovered = 0

    for uid, sgk in _iter_cycles(subj_order, cycle_series):
        if sgk not in lh_ov_dict:
            continue
        n_cycles += 1
        actual = int(cycle_series[sgk]["cycle_len"])
        bseq = base_det.get(sgk, [None] * actual)[:actual]
        pseq = pers_det.get(sgk, [None] * actual)[:actual]
        if len(bseq) < actual:
            bseq = list(bseq) + [None] * (actual - len(bseq))
        if len(pseq) < actual:
            pseq = list(pseq) + [None] * (actual - len(pseq))

        ready = bool(prof.loc[sgk]["profile_ready"]) if sgk in prof.index else False
        if ready:
            n_ready += 1

        supp_idx = [i for i in range(actual) if bseq[i] is not None and pseq[i] is None]
        n_sup = len(supp_idx)
        if n_sup > 0:
            n_cycles_any_suppress += 1
            n_suppress_days_total += n_sup
            suppress_days.extend(supp_idx)
            last_s = max(supp_idx)
            recovered = any(pseq[j] is not None for j in range(last_s + 1, actual))
            if recovered:
                n_recovered += 1

        rows.append(
            {
                "user_id": uid,
                "small_group_key": sgk,
                "profile_ready": ready,
                "n_suppress_days": n_sup,
                "suppress_first_day_1based": float(supp_idx[0] + 1) if supp_idx else float("nan"),
                "suppress_last_day_1based": float(supp_idx[-1] + 1) if supp_idx else float("nan"),
                "recovered_nonnull_after_last_suppress": bool(n_sup and any(pseq[j] is not None for j in range(max(supp_idx) + 1, actual))),
                "true_ov_day_1based": float(lh_ov_dict[sgk] + 1),
            }
        )

    hist: dict[str, int] = {}
    for d in suppress_days:
        hist[str(d)] = hist.get(str(d), 0) + 1

    summary = {
        "audit_version": "active_effect_v1",
        "level": "L1",
        "n_labeled_cycles": n_cycles,
        "n_cycles_profile_ready": n_ready,
        "n_cycles_with_any_suppress": n_cycles_any_suppress,
        "n_suppress_days_total": n_suppress_days_total,
        "fraction_ready_cycles": float(n_ready / n_cycles) if n_cycles else 0.0,
        "fraction_cycles_with_suppress": float(n_cycles_any_suppress / n_cycles) if n_cycles else 0.0,
        "suppress_day_idx_histogram_0based": hist,
        "n_cycles_suppress_then_recover_nonnull": n_recovered,
    }
    return {"summary": _json_safe(summary), "cycle_df": pd.DataFrame(rows)}


def _pad_seq(seq: list[Any] | None, actual: int, fill: Any = None) -> list[Any]:
    s = list(seq or [])
    if len(s) < actual:
        s.extend([fill] * (actual - len(s)))
    return s[:actual]


def _n_adjust_days_for_cycle(
    sgk: str,
    base_det: dict[str, list[int | None]],
    pers_det: dict[str, list[int | None]],
    cycle_series: dict[str, dict[str, Any]],
) -> int:
    actual = int(cycle_series[sgk]["cycle_len"])
    bseq = _pad_seq(base_det.get(sgk), actual, None)
    pseq = _pad_seq(pers_det.get(sgk), actual, None)
    return sum(
        1
        for i in range(actual)
        if bseq[i] is not None and pseq[i] is not None and int(bseq[i]) != int(pseq[i])
    )


_POST_TRIGGER_TRACK_COLS = frozenset({"user_id", "small_group_key", "n_post_trigger_days", "post_trigger_mae"})


def compute_post_trigger_mae_per_cycle_track(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    det_by_day: dict[str, list[int | None]],
    conf_by_day: dict[str, list[float]],
    *,
    baseline_mode: str = "dynamic",
    use_population_only_prior: bool = False,
    custom_cycle_priors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    One cohort pass over ``subj_order``: ACL/LUT state updates match evaluate_prefix_post_trigger
    for the given det/conf track.
    """
    pop_luteal_len = float(DEFAULT_POPULATION_LUTEAL_LENGTH)
    s_plut: dict[Any, list] = defaultdict(list)
    s_pclen: dict[Any, list] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for uid, sgk in _iter_cycles(subj_order, cycle_series):
        if sgk not in lh_ov_dict:
            continue
        cs_row = cycle_series[sgk]
        actual = int(cs_row["cycle_len"])
        acl, lut = _acl_lut_for_uid(
            uid,
            sgk,
            cycle_series,
            s_plut,
            s_pclen,
            pop_luteal_len,
            custom_cycle_priors,
            use_population_only_prior,
        )

        det_seq = _pad_seq(det_by_day.get(sgk), actual, None)
        conf_seq = _pad_seq(conf_by_day.get(sgk), actual, 0.5)
        conf_seq = [float(c) for c in conf_seq]

        hist_fols: list[float] = []
        signed = _post_trigger_signed_errors_for_cycle(
            cs_row, det_seq, conf_seq, actual, acl, lut, hist_fols, baseline_mode, False
        )
        mae = _mae_from_signed(signed)

        rows.append(
            {
                "user_id": uid,
                "small_group_key": sgk,
                "n_post_trigger_days": len(signed),
                "post_trigger_mae": mae,
            }
        )
        _update_state_after_cycle(uid, sgk, cycle_series, det_seq, s_plut, s_pclen)

    return pd.DataFrame(rows)


def _assert_cached_base_post_trigger_df(df: pd.DataFrame) -> None:
    missing = _POST_TRIGGER_TRACK_COLS - set(df.columns)
    if missing:
        raise ValueError(f"cached_base_post_trigger_df missing columns: {sorted(missing)}")


def _merge_impute_and_post_trigger(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    base_det: dict[str, list[int | None]],
    pers_det: dict[str, list[int | None]],
    base_pt: pd.DataFrame,
    pers_pt: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    b_mae = base_pt.set_index("small_group_key")["post_trigger_mae"].to_dict()
    b_n = base_pt.set_index("small_group_key")["n_post_trigger_days"].to_dict()
    p_mae = pers_pt.set_index("small_group_key")["post_trigger_mae"].to_dict()
    p_n = pers_pt.set_index("small_group_key")["n_post_trigger_days"].to_dict()

    for uid, sgk in _iter_cycles(subj_order, cycle_series):
        if sgk not in lh_ov_dict:
            continue
        actual = int(cycle_series[sgk]["cycle_len"])
        bseq = _pad_seq(base_det.get(sgk), actual, None)
        pseq = _pad_seq(pers_det.get(sgk), actual, None)
        first_b = _first_nonnull_1based(bseq)
        first_p = _first_nonnull_1based(pseq)
        impute_idx = [i for i in range(actual) if bseq[i] is None and pseq[i] is not None]
        mae_b = float(b_mae.get(sgk, float("nan")))
        mae_p = float(p_mae.get(sgk, float("nan")))
        rows.append(
            {
                "user_id": uid,
                "small_group_key": sgk,
                "n_impute_days": len(impute_idx),
                "impute_first_day_1based": float(impute_idx[0] + 1) if impute_idx else float("nan"),
                "first_nonnull_base_1based": first_b,
                "first_nonnull_pers_1based": first_p,
                "delta_first_detection_1based": float(first_b - first_p)
                if np.isfinite(first_b) and np.isfinite(first_p)
                else float("nan"),
                "n_post_trigger_days_base": int(b_n.get(sgk, 0)),
                "n_post_trigger_days_pers": int(p_n.get(sgk, 0)),
                "post_trigger_mae_base": mae_b,
                "post_trigger_mae_pers": mae_p,
                "post_trigger_mae_delta_pers_minus_base": float(mae_p - mae_b)
                if np.isfinite(mae_b) and np.isfinite(mae_p)
                else float("nan"),
                "true_ov_day_1based": float(lh_ov_dict[sgk] + 1),
            }
        )
    return pd.DataFrame(rows)


def audit_l2_active_effect(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    profile_df: pd.DataFrame,
    base_det: dict[str, list[int | None]],
    base_conf: dict[str, list[float]],
    pers_det: dict[str, list[int | None]],
    pers_conf: dict[str, list[float]],
    *,
    variant_label: str,
    baseline_mode: str = "dynamic",
    custom_cycle_priors: dict[str, float] | None = None,
    cached_base_post_trigger_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prof = profile_df.set_index("small_group_key")
    if cached_base_post_trigger_df is not None:
        _assert_cached_base_post_trigger_df(cached_base_post_trigger_df)
        base_pt = cached_base_post_trigger_df
    else:
        base_pt = compute_post_trigger_mae_per_cycle_track(
            cycle_series,
            lh_ov_dict,
            subj_order,
            base_det,
            base_conf,
            baseline_mode=baseline_mode,
            use_population_only_prior=False,
            custom_cycle_priors=custom_cycle_priors,
        )
    pers_pt = compute_post_trigger_mae_per_cycle_track(
        cycle_series,
        lh_ov_dict,
        subj_order,
        pers_det,
        pers_conf,
        baseline_mode=baseline_mode,
        use_population_only_prior=False,
        custom_cycle_priors=custom_cycle_priors,
    )
    cycle_df = _merge_impute_and_post_trigger(
        cycle_series,
        lh_ov_dict,
        subj_order,
        base_det,
        pers_det,
        base_pt,
        pers_pt,
    )
    cycle_df.insert(0, "variant", variant_label)
    ready_flags = [bool(prof.loc[k]["profile_ready"]) if k in prof.index else False for k in cycle_df["small_group_key"]]
    cycle_df["profile_ready"] = ready_flags

    n_lab = len(cycle_df)
    n_ready = int(sum(ready_flags))
    impute_mask = cycle_df["n_impute_days"] > 0
    n_cycles_impute = int(impute_mask.sum())
    n_impute_days = int(cycle_df["n_impute_days"].sum())

    delta_pt = cycle_df.loc[impute_mask, "post_trigger_mae_delta_pers_minus_base"].dropna()
    better = int((delta_pt < -1e-9).sum())
    worse = int((delta_pt > 1e-9).sum())
    same = int(delta_pt.size - better - worse)

    summary = {
        "audit_version": "active_effect_v1",
        "level": "L2",
        "variant": variant_label,
        "n_labeled_cycles": n_lab,
        "n_cycles_profile_ready": n_ready,
        "n_cycles_with_any_impute": n_cycles_impute,
        "n_impute_days_total": n_impute_days,
        "fraction_ready_cycles": float(n_ready / n_lab) if n_lab else 0.0,
        "among_impute_cycles_post_trigger_mae_pers_vs_base": {
            "n_cycles_with_finite_delta": int(delta_pt.size),
            "n_better": better,
            "n_worse": worse,
            "n_same": same,
            "mean_delta_pers_minus_base": float(delta_pt.mean()) if len(delta_pt) else None,
        },
    }
    return {"summary": _json_safe(summary), "cycle_df": cycle_df}


def audit_l3_active_effect(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    profile_df: pd.DataFrame,
    base_det: dict[str, list[int | None]],
    base_conf: dict[str, list[float]],
    pers_det: dict[str, list[int | None]],
    pers_conf: dict[str, list[float]],
    *,
    baseline_mode: str = "dynamic",
    custom_cycle_priors: dict[str, float] | None = None,
) -> dict[str, Any]:
    prof = profile_df.set_index("small_group_key")
    base_pt = compute_post_trigger_mae_per_cycle_track(
        cycle_series,
        lh_ov_dict,
        subj_order,
        base_det,
        base_conf,
        baseline_mode=baseline_mode,
        use_population_only_prior=False,
        custom_cycle_priors=custom_cycle_priors,
    )
    pers_pt = compute_post_trigger_mae_per_cycle_track(
        cycle_series,
        lh_ov_dict,
        subj_order,
        pers_det,
        pers_conf,
        baseline_mode=baseline_mode,
        use_population_only_prior=False,
        custom_cycle_priors=custom_cycle_priors,
    )
    cycle_df = _merge_impute_and_post_trigger(
        cycle_series,
        lh_ov_dict,
        subj_order,
        base_det,
        pers_det,
        base_pt,
        pers_pt,
    )

    adjust_rows: list[dict[str, Any]] = []
    for _, r in cycle_df.iterrows():
        sgk = r["small_group_key"]
        ready = bool(prof.loc[sgk]["profile_ready"]) if sgk in prof.index else False
        actual = int(cycle_series[sgk]["cycle_len"])
        bseq = base_det.get(sgk, [None] * actual)[:actual]
        pseq = pers_det.get(sgk, [None] * actual)[:actual]
        if len(bseq) < actual:
            bseq = list(bseq) + [None] * (actual - len(bseq))
        if len(pseq) < actual:
            pseq = list(pseq) + [None] * (actual - len(pseq))
        tv = float(lh_ov_dict[sgk])
        for day_idx in range(actual):
            bv, pv = bseq[day_idx], pseq[day_idx]
            if bv is not None and pv is not None and int(bv) != int(pv):
                ab = abs(int(bv) - tv)
                ap = abs(int(pv) - tv)
                adjust_rows.append(
                    {
                        "small_group_key": sgk,
                        "profile_ready": ready,
                        "day_idx_0based": day_idx,
                        "base_ov_0based": int(bv),
                        "pers_ov_0based": int(pv),
                        "delta_pers_minus_base": int(pv) - int(bv),
                        "abs_err_base": ab,
                        "abs_err_pers": ap,
                        "delta_abs_err_pers_minus_base": ap - ab,
                    }
                )

    adjust_df = pd.DataFrame(adjust_rows)
    cycle_df["profile_ready"] = [bool(prof.loc[k]["profile_ready"]) if k in prof.index else False for k in cycle_df["small_group_key"]]
    cycle_df["n_adjust_days"] = cycle_df["small_group_key"].map(
        lambda s: _n_adjust_days_for_cycle(str(s), base_det, pers_det, cycle_series)
    )

    n_lab = len(cycle_df)
    n_ready = int(cycle_df["profile_ready"].sum())
    n_cycles_any_adjust = int((cycle_df["n_adjust_days"] > 0).sum())
    n_adjust_days_total = int(cycle_df["n_adjust_days"].sum())

    if not adjust_df.empty:
        d = adjust_df["delta_abs_err_pers_minus_base"].dropna()
        better = int((d < -1e-9).sum())
        worse = int((d > 1e-9).sum())
        same = int(d.size - better - worse)
        mean_abs_step = float(adjust_df["delta_pers_minus_base"].abs().mean())
    else:
        better = worse = same = 0
        mean_abs_step = float("nan")

    summary = {
        "audit_version": "active_effect_v1",
        "level": "L3",
        "n_labeled_cycles": n_lab,
        "n_cycles_profile_ready": n_ready,
        "n_cycles_with_any_adjust": n_cycles_any_adjust,
        "n_adjust_days_total": n_adjust_days_total,
        "n_impute_days_total": int(cycle_df["n_impute_days"].sum()),
        "fraction_ready_cycles": float(n_ready / n_lab) if n_lab else 0.0,
        "among_adjust_days_vs_true_ov": {
            "n_adjust_days": int(len(adjust_df)),
            "n_abs_err_better": better,
            "n_abs_err_worse": worse,
            "n_abs_err_same": same,
            "mean_abs_ov_day_change": mean_abs_step if np.isfinite(mean_abs_step) else None,
        },
    }
    return {
        "summary": _json_safe(summary),
        "cycle_df": cycle_df,
        "adjust_day_df": adjust_df,
    }


def summarize_matrix_detect_post_trigger_mae(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    base_det: dict[str, list[int | None]],
    base_conf: dict[str, list[float]],
    pers_det: dict[str, list[int | None]],
    pers_conf: dict[str, list[float]],
    history_acl: dict[str, float],
) -> dict[str, Any]:
    """
    Compare per-cycle PostTrigger MAE (same definition as main benchmark) for Base vs PersDet
    under population vs history ACL countdown priors.
    """
    by_mode: dict[str, Any] = {}
    for key, custom in (("population", None), ("history_acl", history_acl)):
        base_pt = compute_post_trigger_mae_per_cycle_track(
            cycle_series,
            lh_ov_dict,
            subj_order,
            base_det,
            base_conf,
            custom_cycle_priors=custom,
        )
        pers_pt = compute_post_trigger_mae_per_cycle_track(
            cycle_series,
            lh_ov_dict,
            subj_order,
            pers_det,
            pers_conf,
            custom_cycle_priors=custom,
        )
        cycle_df = _merge_impute_and_post_trigger(
            cycle_series,
            lh_ov_dict,
            subj_order,
            base_det,
            pers_det,
            base_pt,
            pers_pt,
        )
        d = cycle_df["post_trigger_mae_delta_pers_minus_base"].dropna()
        by_mode[key] = {
            "n_labeled_cycles": int(len(cycle_df)),
            "n_impute_days_total": int(cycle_df["n_impute_days"].sum()),
            "mean_post_trigger_mae_base": float(cycle_df["post_trigger_mae_base"].mean()),
            "mean_post_trigger_mae_pers": float(cycle_df["post_trigger_mae_pers"].mean()),
            "mean_delta_pers_minus_base_over_cycles": float(d.mean()) if len(d) else None,
            "median_delta_pers_minus_base_over_cycles": float(d.median()) if len(d) else None,
        }
    return _json_safe(
        {
            "audit_version": "matrix_detect_v1",
            "detect_variant": "HistPhysRefine_vs_Base",
            "by_countdown_prior": by_mode,
        }
    )


def write_audit_artifacts(
    summary: dict[str, Any],
    cycle_df: pd.DataFrame,
    json_path: Path,
    csv_path: Path,
    adjust_df: pd.DataFrame | None = None,
    adjust_csv_path: Path | None = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    cycle_df.to_csv(csv_path, index=False)
    if adjust_df is not None and adjust_csv_path is not None:
        adjust_df.to_csv(adjust_csv_path, index=False)


def write_l2_active_effect_bundle(
    by_variant: dict[str, dict[str, Any]],
    cycle_df: pd.DataFrame,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Write L2 multi-variant audit JSON + concatenated cycle CSV in one place."""
    payload = _json_safe({"audit_version": "active_effect_v1_l2_bundle", "by_variant": by_variant})
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    cycle_df.to_csv(csv_path, index=False)

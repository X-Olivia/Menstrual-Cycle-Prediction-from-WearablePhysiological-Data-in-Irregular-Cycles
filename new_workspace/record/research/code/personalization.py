from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from protocol import EXPECTED_OVULATION_FRACTION


@dataclass(frozen=True)
class L1Config:
    prior_min_cycles: int = 1
    prior_std_floor: float = 0.08
    trigger_bias_scale: float = 0.50
    trigger_bias_clip: float = 0.06
    profile_version: str = "l1_v1"


@dataclass(frozen=True)
class L2Config:
    prior_std_floor: float = 0.08
    trigger_bias_scale: float = 0.50
    trigger_bias_clip: float = 0.06
    temp_shift_window_pre: int = 5
    temp_shift_window_post: int = 5
    temp_shift_post_offset: int = 2
    hr_baseline_window_pre: int = 7
    temp_evidence_scale: float = 0.50
    temp_evidence_floor: float = 0.05
    hr_evidence_margin: float = 0.0
    localizer_score_min: float = 1.50
    profile_version: str = "l2_v1"


@dataclass(frozen=True)
class L3Config:
    max_prior_cycles: int = 3
    prior_std_floor: float = 0.08
    trigger_bias_scale: float = 0.50
    trigger_bias_clip: float = 0.06
    temp_shift_window_pre: int = 5
    temp_shift_window_post: int = 5
    temp_shift_post_offset: int = 2
    hr_baseline_window_pre: int = 7
    temp_evidence_scale: float = 0.50
    temp_evidence_floor: float = 0.05
    hr_evidence_margin: float = 0.0
    localizer_score_min: float = 1.50
    refine_radius_min: int = 1
    refine_radius_max: int = 3
    refine_weight_low: float = 0.35
    refine_weight_high: float = 0.65
    profile_version: str = "l3_v1"


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=0))


def build_zero_shot_personalization_profile_table(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    cfg: L1Config | None = None,
) -> pd.DataFrame:
    cfg = cfg or L1Config()
    rows: list[dict[str, Any]] = []

    for user_id, sgks in subj_order.items():
        prior_ov_fracs: list[float] = []

        for idx, sgk in enumerate(sgks):
            if sgk not in cycle_series:
                continue
            prior_mean = _safe_mean(prior_ov_fracs)
            prior_std = _safe_std(prior_ov_fracs)
            has_prior = len(prior_ov_fracs) >= cfg.prior_min_cycles
            if has_prior:
                trigger_bias = float(
                    np.clip(
                        (prior_mean - EXPECTED_OVULATION_FRACTION) * cfg.trigger_bias_scale,
                        -cfg.trigger_bias_clip,
                        cfg.trigger_bias_clip,
                    )
                )
                ov_frac_prior_std = (
                    prior_std if np.isfinite(prior_std) else cfg.prior_std_floor
                )
                ov_frac_prior_std = float(max(ov_frac_prior_std, cfg.prior_std_floor))
                earliest_trigger_frac = float(
                    np.clip(prior_mean - ov_frac_prior_std + trigger_bias, 0.0, 1.0)
                )
            else:
                trigger_bias = 0.0
                ov_frac_prior_std = float("nan")
                earliest_trigger_frac = float("nan")

            rows.append(
                {
                    "user_id": user_id,
                    "small_group_key": sgk,
                    "cycle_order_index": idx,
                    "n_prior_ov_cycles": len(prior_ov_fracs),
                    "ov_frac_prior_mean": prior_mean,
                    "ov_frac_prior_std": ov_frac_prior_std,
                    "trigger_bias": trigger_bias,
                    "earliest_trigger_frac": earliest_trigger_frac,
                    "profile_ready": has_prior,
                    "profile_version": cfg.profile_version,
                }
            )

            if sgk in lh_ov_dict:
                cycle_len = float(cycle_series[sgk]["cycle_len"])
                if cycle_len > 0:
                    prior_ov_fracs.append(float(lh_ov_dict[sgk]) / cycle_len)

    return pd.DataFrame(rows).sort_values(["user_id", "cycle_order_index"]).reset_index(drop=True)


def apply_l1_zero_shot_personalization(
    cs: dict[str, dict[str, Any]],
    det_by_day: dict[str, list[int | None]],
    conf_by_day: dict[str, list[float]],
    profile_df: pd.DataFrame,
) -> tuple[dict[str, list[int | None]], dict[str, list[float]]]:
    profile_lookup = profile_df.set_index("small_group_key").to_dict(orient="index")
    out_det: dict[str, list[int | None]] = {}
    out_conf: dict[str, list[float]] = {}

    for sgk, seq in det_by_day.items():
        conf_seq = conf_by_day.get(sgk, [0.0] * len(seq))
        profile = profile_lookup.get(sgk)
        if not profile or not bool(profile.get("profile_ready")):
            out_det[sgk] = list(seq)
            out_conf[sgk] = list(conf_seq)
            continue

        earliest_trigger_frac = float(profile["earliest_trigger_frac"])
        cycle_len = float(cs[sgk]["cycle_len"])
        new_seq: list[int | None] = []
        new_conf: list[float] = []
        for day_idx, ov_est in enumerate(seq):
            cycle_frac = float(day_idx / cycle_len) if cycle_len > 0 else 0.0
            if ov_est is not None and cycle_frac < earliest_trigger_frac:
                new_seq.append(None)
                new_conf.append(0.0)
            else:
                new_seq.append(ov_est)
                new_conf.append(float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0)
        out_det[sgk] = new_seq
        out_conf[sgk] = new_conf

    return out_det, out_conf


def apply_history_prior_menses_prediction(
    cs: dict[str, dict[str, Any]],
    subgroup_df: pd.DataFrame,
) -> dict[str, float]:
    """
    TASK 1: History-only personalization baseline.
    Explicitly extracts the personalized history-based cycle length mean (ACL) 
    for each small_group_key from the subgroup table.
    """
    if subgroup_df.empty:
        return {}
    
    # Map each cycle to its prefix-safe hist_cycle_len_mean
    # This value was computed in build_user_history_table and is strictly prefix-safe.
    priors = subgroup_df.set_index("small_group_key")["hist_cycle_len_mean"].to_dict()
    
    # Filter to only return valid finite values
    return {k: float(v) for k, v in priors.items() if np.isfinite(v)}


def l1_manifest(profile_df: pd.DataFrame, cfg: L1Config | None = None) -> dict[str, Any]:
    cfg = cfg or L1Config()
    ready = profile_df[profile_df["profile_ready"]]
    return {
        "profile_version": cfg.profile_version,
        "config": asdict(cfg),
        "n_rows": int(len(profile_df)),
        "n_ready": int(len(ready)),
        "n_users_ready": int(ready["user_id"].nunique()) if not ready.empty else 0,
    }


def _compute_temp_shift_scale(data: dict[str, Any], ov_day: int, cfg: L2Config) -> float:
    raw = np.asarray(data.get("nightly_temperature"), dtype=float)
    if raw.size == 0 or np.isnan(raw).all():
        return float("nan")
    pre_lo = max(0, ov_day - cfg.temp_shift_window_pre)
    pre_hi = max(pre_lo + 1, ov_day)
    post_lo = ov_day + cfg.temp_shift_post_offset
    post_hi = min(len(raw), post_lo + cfg.temp_shift_window_post)
    if post_lo >= len(raw) or post_hi <= post_lo:
        return float("nan")
    pre = raw[pre_lo:pre_hi]
    post = raw[post_lo:post_hi]
    if np.isnan(pre).all() or np.isnan(post).all():
        return float("nan")
    return float(np.nanmean(post) - np.nanmean(pre))


def _compute_hr_baseline(data: dict[str, Any], ov_day: int, cfg: L2Config) -> float:
    raw = np.asarray(data.get("rhr"), dtype=float)
    if raw.size == 0 or np.isnan(raw).all():
        return float("nan")
    lo = max(0, ov_day - cfg.hr_baseline_window_pre)
    hi = max(lo + 1, ov_day)
    seg = raw[lo:hi]
    if np.isnan(seg).all():
        return float("nan")
    return float(np.nanmean(seg))


def build_one_shot_personalization_profile_table(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    cfg: L2Config | None = None,
) -> pd.DataFrame:
    cfg = cfg or L2Config()
    rows: list[dict[str, Any]] = []

    for user_id, sgks in subj_order.items():
        last_labeled_cycle_key = None
        last_ov_frac = float("nan")
        last_temp_shift_scale = float("nan")
        last_hr_baseline = float("nan")

        for idx, sgk in enumerate(sgks):
            if sgk not in cycle_series:
                continue
            has_prior = last_labeled_cycle_key is not None
            if has_prior:
                trigger_bias = float(
                    np.clip(
                        (last_ov_frac - EXPECTED_OVULATION_FRACTION) * cfg.trigger_bias_scale,
                        -cfg.trigger_bias_clip,
                        cfg.trigger_bias_clip,
                    )
                )
                ov_frac_prior_std = cfg.prior_std_floor
                earliest_trigger_frac = float(
                    np.clip(last_ov_frac - ov_frac_prior_std + trigger_bias, 0.0, 1.0)
                )
            else:
                trigger_bias = 0.0
                ov_frac_prior_std = float("nan")
                earliest_trigger_frac = float("nan")

            rows.append(
                {
                    "user_id": user_id,
                    "small_group_key": sgk,
                    "cycle_order_index": idx,
                    "one_shot_source_cycle": last_labeled_cycle_key,
                    "ov_frac_prior_mean": last_ov_frac,
                    "ov_frac_prior_std": ov_frac_prior_std,
                    "trigger_bias": trigger_bias,
                    "earliest_trigger_frac": earliest_trigger_frac,
                    "temp_shift_scale": last_temp_shift_scale,
                    "hr_baseline": last_hr_baseline,
                    "profile_ready": has_prior,
                    "profile_version": cfg.profile_version,
                }
            )

            if sgk in lh_ov_dict:
                cycle_len = float(cycle_series[sgk]["cycle_len"])
                ov_day = int(lh_ov_dict[sgk])
                if cycle_len > 0:
                    last_labeled_cycle_key = sgk
                    last_ov_frac = float(ov_day / cycle_len)
                    last_temp_shift_scale = _compute_temp_shift_scale(cycle_series[sgk], ov_day, cfg)
                    last_hr_baseline = _compute_hr_baseline(cycle_series[sgk], ov_day, cfg)

    return pd.DataFrame(rows).sort_values(["user_id", "cycle_order_index"]).reset_index(drop=True)


def apply_l2_one_shot_personalization(
    cs: dict[str, dict[str, Any]],
    det_by_day: dict[str, list[int | None]],
    conf_by_day: dict[str, list[float]],
    profile_df: pd.DataFrame,
    localizer_table: dict[str, list[int | None]],
    score_table: dict[str, list[float | None]],
    cfg: L2Config | None = None,
    variant: str | None = None,
) -> tuple[dict[str, list[int | None]], dict[str, list[float]]]:
    """
    TASK 2: Add diagnostic variants L2a and L2b.
    variant=None: Original L2 (All gates)
    variant="L2a": Prior-trigger + Localizer only (No signal gating)
    variant="L2b": Prior-trigger + Temp evidence only (No HR gating, No Localizer refinement)
    """
    cfg = cfg or L2Config()
    profile_lookup = profile_df.set_index("small_group_key").to_dict(orient="index")
    out_det: dict[str, list[int | None]] = {}
    out_conf: dict[str, list[float]] = {}

    for sgk, seq in det_by_day.items():
        conf_seq = conf_by_day.get(sgk, [0.0] * len(seq))
        profile = profile_lookup.get(sgk)
        data = cs[sgk]
        loc_seq = localizer_table.get(sgk, [None] * len(seq))
        sc_seq = score_table.get(sgk, [None] * len(seq))
        nt = np.asarray(data.get("nightly_temperature"), dtype=float)
        rhr = np.asarray(data.get("rhr"), dtype=float)
        new_seq: list[int | None] = []
        new_conf: list[float] = []

        for day_idx, ov_est in enumerate(seq):
            if ov_est is not None:
                new_seq.append(ov_est)
                new_conf.append(float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0)
                continue

            if not profile or not bool(profile.get("profile_ready")):
                new_seq.append(None)
                new_conf.append(0.0)
                continue

            cycle_len = float(data["cycle_len"])
            cycle_frac = float(day_idx / cycle_len) if cycle_len > 0 else 0.0
            earliest_trigger_frac = float(profile["earliest_trigger_frac"])
            if cycle_frac < earliest_trigger_frac:
                new_seq.append(None)
                new_conf.append(0.0)
                continue

            loc_est = loc_seq[day_idx] if day_idx < len(loc_seq) else None
            loc_score = sc_seq[day_idx] if day_idx < len(sc_seq) else None

            # Localizer Gate (L2 and L2a)
            if variant != "L2b":
                if loc_est is None or loc_score is None or float(loc_score) < cfg.localizer_score_min:
                    new_seq.append(None)
                    new_conf.append(0.0)
                    continue

            # Physiological Evidence Gates
            temp_ok = True
            hr_ok = True

            if variant is None or variant == "L2b":
                temp_shift_scale = float(profile.get("temp_shift_scale", np.nan))
                temp_recent = np.nanmean(nt[max(0, day_idx - 2) : day_idx + 1]) if nt.size else np.nan
                temp_prefix_min = np.nanmin(nt[: day_idx + 1]) if nt.size else np.nan
                temp_evidence = (
                    temp_recent - temp_prefix_min
                    if np.isfinite(temp_recent) and np.isfinite(temp_prefix_min)
                    else np.nan
                )
                temp_threshold = max(
                    cfg.temp_evidence_floor,
                    (temp_shift_scale * cfg.temp_evidence_scale)
                    if np.isfinite(temp_shift_scale)
                    else cfg.temp_evidence_floor,
                )
                temp_ok = bool(np.isfinite(temp_evidence) and temp_evidence >= temp_threshold)

            if variant is None:
                hr_baseline = float(profile.get("hr_baseline", np.nan))
                if np.isfinite(hr_baseline) and rhr.size:
                    hr_recent = np.nanmean(rhr[max(0, day_idx - 2) : day_idx + 1])
                    hr_ok = bool(np.isfinite(hr_recent) and hr_recent >= (hr_baseline + cfg.hr_evidence_margin))
                else:
                    hr_ok = True

            if temp_ok and hr_ok:
                # TASK 2 FIX: Improved L2b trigger logic
                # If variant L2b and loc_est is None, use personal history prior for ovulation day
                # instead of the current day_idx (which is biased).
                if loc_est is not None:
                    trigger_day = loc_est
                else:
                    # Minimal dependency fallback: use historical average ovulation fraction
                    ov_frac_prior = float(profile.get("ov_frac_prior_mean", 0.50))
                    trigger_day = int(round(ov_frac_prior * cycle_len))
                
                trigger_day = min(int(trigger_day), day_idx)
                new_seq.append(trigger_day)
                
                if loc_score is not None:
                    conf = float(max(0.0, min(1.0, (float(loc_score) - cfg.localizer_score_min) / 2.0 + 0.5)))
                else:
                    conf = 0.5
                new_conf.append(conf)
            else:
                new_seq.append(None)
                new_conf.append(0.0)

        out_det[sgk] = new_seq
        out_conf[sgk] = new_conf

    return out_det, out_conf


def l2_manifest(profile_df: pd.DataFrame, cfg: L2Config | None = None) -> dict[str, Any]:
    cfg = cfg or L2Config()
    ready = profile_df[profile_df["profile_ready"]]
    return {
        "profile_version": cfg.profile_version,
        "config": asdict(cfg),
        "n_rows": int(len(profile_df)),
        "n_ready": int(len(ready)),
        "n_users_ready": int(ready["user_id"].nunique()) if not ready.empty else 0,
    }


def build_few_shot_personalization_profile_table(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    cfg: L3Config | None = None,
) -> pd.DataFrame:
    cfg = cfg or L3Config()
    rows: list[dict[str, Any]] = []

    for user_id, sgks in subj_order.items():
        prior_cycles: list[dict[str, float | str]] = []
        for idx, sgk in enumerate(sgks):
            if sgk not in cycle_series:
                continue
            keep = prior_cycles[-cfg.max_prior_cycles :]
            has_prior = len(keep) >= 2
            if has_prior:
                ov_fracs = [float(p["ov_frac"]) for p in keep]
                temp_scales = [float(p["temp_shift_scale"]) for p in keep if np.isfinite(float(p["temp_shift_scale"]))]
                hr_bases = [float(p["hr_baseline"]) for p in keep if np.isfinite(float(p["hr_baseline"]))]
                ov_frac_prior_mean = float(np.mean(ov_fracs))
                ov_frac_prior_std = float(max(np.std(ov_fracs, ddof=0), cfg.prior_std_floor))
                trigger_bias = float(
                    np.clip(
                        (ov_frac_prior_mean - EXPECTED_OVULATION_FRACTION) * cfg.trigger_bias_scale,
                        -cfg.trigger_bias_clip,
                        cfg.trigger_bias_clip,
                    )
                )
                earliest_trigger_frac = float(
                    np.clip(ov_frac_prior_mean - ov_frac_prior_std + trigger_bias, 0.0, 1.0)
                )
                temp_shift_scale = float(np.mean(temp_scales)) if temp_scales else float("nan")
                hr_baseline = float(np.mean(hr_bases)) if hr_bases else float("nan")
                ov_day_sd_days = float(np.std([float(p["ov_day"]) for p in keep], ddof=0))
                localizer_refine_radius = int(
                    np.clip(round(max(cfg.refine_radius_min, ov_day_sd_days)), cfg.refine_radius_min, cfg.refine_radius_max)
                )
                localizer_refine_weight = (
                    cfg.refine_weight_high if ov_frac_prior_std <= 0.10 else cfg.refine_weight_low
                )
            else:
                ov_frac_prior_mean = float("nan")
                ov_frac_prior_std = float("nan")
                trigger_bias = 0.0
                earliest_trigger_frac = float("nan")
                temp_shift_scale = float("nan")
                hr_baseline = float("nan")
                localizer_refine_radius = cfg.refine_radius_max
                localizer_refine_weight = cfg.refine_weight_low

            rows.append(
                {
                    "user_id": user_id,
                    "small_group_key": sgk,
                    "cycle_order_index": idx,
                    "n_fewshot_cycles": len(keep),
                    "ov_frac_prior_mean": ov_frac_prior_mean,
                    "ov_frac_prior_std": ov_frac_prior_std,
                    "trigger_bias": trigger_bias,
                    "earliest_trigger_frac": earliest_trigger_frac,
                    "temp_shift_scale": temp_shift_scale,
                    "hr_baseline": hr_baseline,
                    "localizer_refine_radius": localizer_refine_radius,
                    "localizer_refine_weight": localizer_refine_weight,
                    "profile_ready": has_prior,
                    "profile_version": cfg.profile_version,
                }
            )

            if sgk in lh_ov_dict:
                cycle_len = float(cycle_series[sgk]["cycle_len"])
                ov_day = int(lh_ov_dict[sgk])
                if cycle_len > 0:
                    prior_cycles.append(
                        {
                            "cycle_key": sgk,
                            "ov_frac": float(ov_day / cycle_len),
                            "ov_day": float(ov_day),
                            "temp_shift_scale": _compute_temp_shift_scale(cycle_series[sgk], ov_day, cfg),
                            "hr_baseline": _compute_hr_baseline(cycle_series[sgk], ov_day, cfg),
                        }
                    )

    return pd.DataFrame(rows).sort_values(["user_id", "cycle_order_index"]).reset_index(drop=True)


def apply_l3_few_shot_personalization(
    cs: dict[str, dict[str, Any]],
    det_by_day: dict[str, list[int | None]],
    conf_by_day: dict[str, list[float]],
    profile_df: pd.DataFrame,
    localizer_table: dict[str, list[int | None]],
    score_table: dict[str, list[float | None]],
    cfg: L3Config | None = None,
    *,
    allow_imputation: bool = True,
) -> tuple[dict[str, list[int | None]], dict[str, list[float]]]:
    """
    Few-shot personalization over daily ovulation tracks.

    When allow_imputation is True (default, legacy): empty base days may be filled using
    earliest_trigger_frac + localizer + temp/HR gates.

    When allow_imputation is False (matrix PersDet): only bounded localizer refinement
    applies when base_est is non-None; base None days are left unchanged (same as base).
    """
    cfg = cfg or L3Config()
    profile_lookup = profile_df.set_index("small_group_key").to_dict(orient="index")
    out_det: dict[str, list[int | None]] = {}
    out_conf: dict[str, list[float]] = {}

    for sgk, seq in det_by_day.items():
        conf_seq = conf_by_day.get(sgk, [0.0] * len(seq))
        profile = profile_lookup.get(sgk)
        data = cs[sgk]
        loc_seq = localizer_table.get(sgk, [None] * len(seq))
        sc_seq = score_table.get(sgk, [None] * len(seq))
        nt = np.asarray(data.get("nightly_temperature"), dtype=float)
        rhr = np.asarray(data.get("rhr"), dtype=float)
        new_seq: list[int | None] = []
        new_conf: list[float] = []

        for day_idx, base_est in enumerate(seq):
            loc_est = loc_seq[day_idx] if day_idx < len(loc_seq) else None
            loc_score = sc_seq[day_idx] if day_idx < len(sc_seq) else None

            if not profile or not bool(profile.get("profile_ready")):
                new_seq.append(base_est)
                new_conf.append(float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0)
                continue

            if base_est is None and not allow_imputation:
                new_seq.append(None)
                new_conf.append(float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0)
                continue

            cycle_len = float(data["cycle_len"])
            cycle_frac = float(day_idx / cycle_len) if cycle_len > 0 else 0.0
            earliest_trigger_frac = float(profile["earliest_trigger_frac"])
            temp_shift_scale = float(profile.get("temp_shift_scale", np.nan))
            hr_baseline = float(profile.get("hr_baseline", np.nan))
            temp_recent = np.nanmean(nt[max(0, day_idx - 2) : day_idx + 1]) if nt.size else np.nan
            temp_prefix_min = np.nanmin(nt[: day_idx + 1]) if nt.size else np.nan
            temp_evidence = (
                temp_recent - temp_prefix_min
                if np.isfinite(temp_recent) and np.isfinite(temp_prefix_min)
                else np.nan
            )
            temp_threshold = max(
                cfg.temp_evidence_floor,
                (temp_shift_scale * cfg.temp_evidence_scale)
                if np.isfinite(temp_shift_scale)
                else cfg.temp_evidence_floor,
            )
            temp_ok = bool(np.isfinite(temp_evidence) and temp_evidence >= temp_threshold)
            if np.isfinite(hr_baseline) and rhr.size:
                hr_recent = np.nanmean(rhr[max(0, day_idx - 2) : day_idx + 1])
                hr_ok = bool(np.isfinite(hr_recent) and hr_recent >= (hr_baseline + cfg.hr_evidence_margin))
            else:
                hr_ok = True

            if base_est is None:
                if (
                    cycle_frac >= earliest_trigger_frac
                    and loc_est is not None
                    and loc_score is not None
                    and float(loc_score) >= cfg.localizer_score_min
                    and temp_ok
                    and hr_ok
                ):
                    assert loc_est <= day_idx
                    new_seq.append(int(loc_est))
                    conf = float(max(0.0, min(1.0, (float(loc_score) - cfg.localizer_score_min) / 2.0 + 0.5)))
                    new_conf.append(conf)
                else:
                    new_seq.append(None)
                    new_conf.append(0.0)
                continue

            if (
                loc_est is not None
                and loc_score is not None
                and float(loc_score) >= cfg.localizer_score_min
            ):
                radius = int(profile.get("localizer_refine_radius", cfg.refine_radius_max))
                weight = float(profile.get("localizer_refine_weight", cfg.refine_weight_low))
                if abs(int(loc_est) - int(base_est)) <= radius:
                    refined = int(round((1.0 - weight) * float(base_est) + weight * float(loc_est)))
                    refined = min(refined, day_idx)
                    new_seq.append(refined)
                    base_conf = float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0
                    new_conf.append(float(max(base_conf, 0.5)))
                    continue

            new_seq.append(base_est)
            new_conf.append(float(conf_seq[day_idx]) if day_idx < len(conf_seq) else 0.0)

        out_det[sgk] = new_seq
        out_conf[sgk] = new_conf

    return out_det, out_conf


def l3_manifest(profile_df: pd.DataFrame, cfg: L3Config | None = None) -> dict[str, Any]:
    cfg = cfg or L3Config()
    ready = profile_df[profile_df["profile_ready"]]
    return {
        "profile_version": cfg.profile_version,
        "config": asdict(cfg),
        "n_rows": int(len(profile_df)),
        "n_ready": int(len(ready)),
        "n_users_ready": int(ready["user_id"].nunique()) if not ready.empty else 0,
    }

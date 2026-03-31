from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


TYPICAL_CYCLE_MIN = 23.0
TYPICAL_CYCLE_MAX = 35.0

VARIABILITY_LOW_SD_MAX = 4.0
VARIABILITY_HIGH_SD_MIN = 7.0

OVULATORY_RATE_CONSISTENT_MIN = 0.80
OVULATORY_RATE_ANOV_MAX = 0.20


@dataclass(frozen=True)
class SubgroupConfig:
    typical_cycle_min: float = TYPICAL_CYCLE_MIN
    typical_cycle_max: float = TYPICAL_CYCLE_MAX
    variability_low_sd_max: float = VARIABILITY_LOW_SD_MAX
    variability_high_sd_min: float = VARIABILITY_HIGH_SD_MIN
    ovulatory_rate_consistent_min: float = OVULATORY_RATE_CONSISTENT_MIN
    ovulatory_rate_anov_max: float = OVULATORY_RATE_ANOV_MAX
    cycle_length_min_history: int = 1
    variability_min_history: int = 2
    ovulatory_status_min_history: int = 2
    subgroup_version: str = "v1"


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=0))


def _safe_cv(mean: float, std: float) -> float:
    if not np.isfinite(mean) or mean <= 0 or not np.isfinite(std):
        return float("nan")
    return float(std / mean)


def compute_cycle_length_level_group(hist_cycle_len_mean: float, cfg: SubgroupConfig) -> str | None:
    if not np.isfinite(hist_cycle_len_mean):
        return None
    if hist_cycle_len_mean < cfg.typical_cycle_min:
        return "short"
    if hist_cycle_len_mean > cfg.typical_cycle_max:
        return "long"
    return "typical"


def compute_cycle_variability_group(hist_cycle_len_std: float, cfg: SubgroupConfig) -> str | None:
    if not np.isfinite(hist_cycle_len_std):
        return None
    if hist_cycle_len_std < cfg.variability_low_sd_max:
        return "low-variability"
    if hist_cycle_len_std >= cfg.variability_high_sd_min:
        return "high-variability"
    return "medium-variability"


def compute_ovulatory_status_group(hist_ovulatory_rate: float, cfg: SubgroupConfig) -> str | None:
    if not np.isfinite(hist_ovulatory_rate):
        return None
    if hist_ovulatory_rate >= cfg.ovulatory_rate_consistent_min:
        return "consistently-ovulatory"
    if hist_ovulatory_rate <= cfg.ovulatory_rate_anov_max:
        return "frequently-anovulatory"
    return "mixed-ovulatory"


def derive_stable_length_profile(
    cycle_length_level_group: str | None, cycle_variability_group: str | None
) -> str | None:
    if cycle_length_level_group is None or cycle_variability_group != "low-variability":
        return None
    return f"{cycle_length_level_group}-stable"


def build_user_history_table(
    cycle_series: dict[str, dict[str, Any]],
    lh_ov_dict: dict[str, int],
    subj_order: dict[Any, list[str]],
    cfg: SubgroupConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or SubgroupConfig()
    rows: list[dict[str, Any]] = []

    for user_id, sgks in subj_order.items():
        prior_cycle_lens: list[float] = []
        prior_ovulatory_flags: list[int] = []

        for idx, sgk in enumerate(sgks):
            if sgk not in cycle_series:
                continue

            hist_cycle_len_mean = _safe_mean(prior_cycle_lens)
            hist_cycle_len_std = _safe_std(prior_cycle_lens)
            hist_cycle_len_cv = _safe_cv(hist_cycle_len_mean, hist_cycle_len_std)
            hist_ovulatory_rate = _safe_mean(prior_ovulatory_flags)

            cycle_length_level_group = None
            if len(prior_cycle_lens) >= cfg.cycle_length_min_history:
                cycle_length_level_group = compute_cycle_length_level_group(hist_cycle_len_mean, cfg)

            cycle_variability_group = None
            if len(prior_cycle_lens) >= cfg.variability_min_history:
                cycle_variability_group = compute_cycle_variability_group(hist_cycle_len_std, cfg)

            ovulatory_status_group = None
            if len(prior_ovulatory_flags) >= cfg.ovulatory_status_min_history:
                ovulatory_status_group = compute_ovulatory_status_group(hist_ovulatory_rate, cfg)

            rows.append(
                {
                    "user_id": user_id,
                    "small_group_key": sgk,
                    "cycle_order_index": idx,
                    "target_cycle_len": float(cycle_series[sgk]["cycle_len"]),
                    "n_history_cycles": len(prior_cycle_lens),
                    "n_history_ovulatory_observed": len(prior_ovulatory_flags),
                    "hist_cycle_len_mean": hist_cycle_len_mean,
                    "hist_cycle_len_std": hist_cycle_len_std,
                    "hist_cycle_len_cv": hist_cycle_len_cv,
                    "hist_ovulatory_rate": hist_ovulatory_rate,
                    "cycle_length_level_group": cycle_length_level_group,
                    "cycle_variability_group": cycle_variability_group,
                    "ovulatory_status_group": ovulatory_status_group,
                    "stable_length_profile": derive_stable_length_profile(
                        cycle_length_level_group, cycle_variability_group
                    ),
                    "history_insufficient_for_cycle_length": len(prior_cycle_lens)
                    < cfg.cycle_length_min_history,
                    "history_insufficient_for_variability": len(prior_cycle_lens)
                    < cfg.variability_min_history,
                    "history_insufficient_for_ovulatory_status": len(prior_ovulatory_flags)
                    < cfg.ovulatory_status_min_history,
                    "target_cycle_has_lh_ovulation_label": sgk in lh_ov_dict,
                    "subgroup_version": cfg.subgroup_version,
                }
            )

            prior_cycle_lens.append(float(cycle_series[sgk]["cycle_len"]))
            prior_ovulatory_flags.append(1 if sgk in lh_ov_dict else 0)

    return pd.DataFrame(rows).sort_values(["user_id", "cycle_order_index"]).reset_index(drop=True)


def build_subgroup_summary(
    user_cycle_df: pd.DataFrame,
    cfg: SubgroupConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or SubgroupConfig()
    rows: list[dict[str, Any]] = []

    subgroup_cols = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "ovulatory_status_group",
        "stable_length_profile",
    ]
    for subgroup_col in subgroup_cols:
        sub = user_cycle_df[user_cycle_df[subgroup_col].notna()].copy()
        if sub.empty:
            continue
        grp = (
            sub.groupby(subgroup_col)
            .agg(
                n_rows=("small_group_key", "size"),
                n_users=("user_id", "nunique"),
                mean_history_cycles=("n_history_cycles", "mean"),
                labeled_cycle_rate=("target_cycle_has_lh_ovulation_label", "mean"),
                mean_target_cycle_len=("target_cycle_len", "mean"),
            )
            .reset_index()
            .rename(columns={subgroup_col: "subgroup_name"})
        )
        grp.insert(0, "subgroup_family", subgroup_col)
        rows.extend(grp.to_dict(orient="records"))

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["subgroup_version"] = cfg.subgroup_version
    return summary.sort_values(["subgroup_family", "subgroup_name"]).reset_index(drop=True)


def subgroup_manifest(
    user_cycle_df: pd.DataFrame,
    cfg: SubgroupConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or SubgroupConfig()
    return {
        "subgroup_version": cfg.subgroup_version,
        "config": asdict(cfg),
        "n_rows": int(len(user_cycle_df)),
        "n_users": int(user_cycle_df["user_id"].nunique()) if not user_cycle_df.empty else 0,
        "n_cycles": int(user_cycle_df["small_group_key"].nunique()) if not user_cycle_df.empty else 0,
        "n_cycle_length_assigned": int(user_cycle_df["cycle_length_level_group"].notna().sum()),
        "n_variability_assigned": int(user_cycle_df["cycle_variability_group"].notna().sum()),
        "n_ovulatory_status_assigned": int(user_cycle_df["ovulatory_status_group"].notna().sum()),
    }

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from paths import (
    BASELINE_MANIFEST_JSON,
    BASELINE_RESULTS_CSV,
    BASELINE_RESULTS_MD,
    L1_AUDIT_CYCLE_CSV,
    L1_AUDIT_JSON,
    L1_PROFILE_CSV,
    L1_MANIFEST_JSON,
    L1_RESULTS_CSV,
    L1_RESULTS_MD,
    L2_AUDIT_CYCLE_CSV,
    L2_AUDIT_JSON,
    L2_PROFILE_CSV,
    L2_MANIFEST_JSON,
    L2_RESULTS_CSV,
    L2_RESULTS_MD,
    L3_AUDIT_ADJUST_DAY_CSV,
    L3_AUDIT_CYCLE_CSV,
    L3_AUDIT_JSON,
    L3_PROFILE_CSV,
    L3_MANIFEST_JSON,
    L3_RESULTS_CSV,
    L3_RESULTS_MD,
    MATRIX_AUDIT_JSON,
    MATRIX_MANIFEST_JSON,
    MATRIX_PROFILE_CSV,
    MATRIX_RESULTS_CSV,
    MATRIX_RESULTS_MD,
    PIPELINE_DIR,
    SUBGROUP_TABLE_CSV,
    SUBGROUP_SUMMARY_CSV,
    SUBGROUP_SUMMARY_MD,
    MANIFEST_JSON,
    ensure_research_dirs,
)

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data import load_all_signals  # type: ignore  # noqa: E402
from benchmark_main import (  # type: ignore  # noqa: E402
    _phase_candidate,
    _collapse_daily_to_cycle_estimate,
    _detected_cycle_bundle,
    _ovulation_accuracy_summary_subset,
    _silent_call,
)
from core.localizer import _precompute_prefix_localizer_payload  # type: ignore  # noqa: E402
from menses import (  # type: ignore  # noqa: E402
    evaluate_prefix_current_day,
    evaluate_prefix_post_trigger,
    predict_menses_by_anchors,
)
from protocol import (  # type: ignore  # noqa: E402
    DEFAULT_POPULATION_LUTEAL_LENGTH,
    PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
    PHASECLS_PHASE_ENSEMBLE_MODELS,
)

from exports import (  # noqa: E402
    export_baseline_markdown,
    export_csv,
    export_json,
    export_matrix_markdown,
    export_summary_markdown,
)
from personalization_audit import (  # noqa: E402
    audit_l1_active_effect,
    audit_l2_active_effect,
    audit_l3_active_effect,
    compute_post_trigger_mae_per_cycle_track,
    summarize_matrix_detect_post_trigger_mae,
    write_audit_artifacts,
    write_l2_active_effect_bundle,
)
from personalization import (  # noqa: E402
    L1Config,
    L2Config,
    L3Config,
    apply_history_prior_menses_prediction,
    apply_l1_zero_shot_personalization,
    apply_l2_one_shot_personalization,
    apply_l3_few_shot_personalization,
    build_few_shot_personalization_profile_table,
    build_one_shot_personalization_profile_table,
    build_zero_shot_personalization_profile_table,
    l1_manifest,
    l2_manifest,
    l3_manifest,
)
from subgrouping import (  # noqa: E402
    SubgroupConfig,
    build_subgroup_summary,
    build_user_history_table,
    subgroup_manifest,
)
from method_spec import MethodSpec, coerce_method_specs  # noqa: E402


BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_CI_LEVEL = 0.95
DEFAULT_WEARABLE_REFERENCE_KEY = "champion"


@dataclass(frozen=True)
class WearableReferenceSpec:
    key: str
    method_name: str
    group_name: str
    phase_kwargs: dict[str, Any]
    localizer_sigs: tuple[str, ...]
    lookback_localize: int
    localizer_cache_tag: str


CHAMPION_WEARABLE_SPEC = WearableReferenceSpec(
    key="champion",
    method_name="PhaseCls-ENS-Temp+HR[Champion]",
    group_name="Temp+HR",
    phase_kwargs={
        "model_type": "rf",
        "phase_ensemble_models": PHASECLS_PHASE_ENSEMBLE_MODELS,
        "stabilization_policy": "score_smooth",
        "localizer_smooth_window_m": int(PHASECLS_LOCALIZER_SCORE_SMOOTH_M),
        "name_suffix": "[Champion]",
    },
    localizer_sigs=("nightly_temperature", "rhr", "noct_hr_min"),
    lookback_localize=10,
    localizer_cache_tag="nightly_temperature_rhr_noct_hr_min",
)

WEARABLE_REFERENCE_SPECS = {
    CHAMPION_WEARABLE_SPEC.key: CHAMPION_WEARABLE_SPEC,
}


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def _bootstrap_mean_ci(
    abs_errs: list[float] | None,
    seed_key: str,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> dict[str, float | int | None]:
    arr = np.asarray(abs_errs or [], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mae_ci_low": None,
            "mae_ci_high": None,
        }

    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return {
            "n": int(arr.size),
            "mae_ci_low": mean_val,
            "mae_ci_high": mean_val,
        }

    alpha = (1.0 - ci_level) / 2.0
    rng = np.random.default_rng(_stable_seed(seed_key, arr.size, n_resamples))
    sample_idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[sample_idx].mean(axis=1)
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "n": int(arr.size),
        "mae_ci_low": float(low),
        "mae_ci_high": float(high),
    }


def _resolve_wearable_reference_spec(reference_key: str | None = None) -> WearableReferenceSpec:
    key = (reference_key or DEFAULT_WEARABLE_REFERENCE_KEY).strip().lower()
    aliases = {
        "champion": "champion",
    }
    canonical_key = aliases.get(key, key)
    if canonical_key not in WEARABLE_REFERENCE_SPECS:
        raise ValueError(
            f"Unknown wearable reference '{reference_key}'. "
            f"Expected one of: {', '.join(sorted(WEARABLE_REFERENCE_SPECS))}"
        )
    return WEARABLE_REFERENCE_SPECS[canonical_key]


def _wearable_reference_manifest(spec: WearableReferenceSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "method_name": spec.method_name,
        "group_name": spec.group_name,
        "phase_kwargs": dict(spec.phase_kwargs),
        "localizer_sigs": list(spec.localizer_sigs),
        "lookback_localize": spec.lookback_localize,
        "localizer_cache_tag": spec.localizer_cache_tag,
    }


def _personalized_method_name(base_spec: WearableReferenceSpec, level_tag: str) -> str:
    if "[" in base_spec.method_name and base_spec.method_name.endswith("]"):
        prefix, bracket_suffix = base_spec.method_name.rsplit("[", 1)
        base_label = bracket_suffix[:-1]
        return f"{prefix}[{base_label}-{level_tag}]"
    return f"{base_spec.method_name}-{level_tag}"


def _anchor_day_summary(anchor_summary: dict[str, Any], day_offset: int) -> dict[str, Any]:
    each_anchor = anchor_summary.get("each_anchor", {}) if anchor_summary else {}
    return each_anchor.get(day_offset, {}) or {}


def _load_all_signals_silent():
    return _silent_call(load_all_signals)


def run_subgroup_build(cfg: SubgroupConfig | None = None) -> dict[str, object]:
    cfg = cfg or SubgroupConfig()
    ensure_research_dirs()

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    user_cycle_df = build_user_history_table(cycle_series, lh_ov_dict, subj_order, cfg=cfg)
    summary_df = build_subgroup_summary(user_cycle_df, cfg=cfg)
    manifest = subgroup_manifest(user_cycle_df, cfg=cfg)
    manifest.update(
        {
            "quality_cycles": len(quality),
            "signal_cols": signal_cols,
            "artifacts": {
                "user_cycle_subgroups_csv": str(SUBGROUP_TABLE_CSV),
                "subgroup_summary_csv": str(SUBGROUP_SUMMARY_CSV),
                "subgroup_summary_md": str(SUBGROUP_SUMMARY_MD),
                "manifest_json": str(MANIFEST_JSON),
            },
        }
    )

    export_csv(user_cycle_df, SUBGROUP_TABLE_CSV)
    export_csv(summary_df, SUBGROUP_SUMMARY_CSV)
    export_summary_markdown(summary_df, SUBGROUP_SUMMARY_MD)
    export_json(manifest, MANIFEST_JSON)

    return {
        "user_cycle_df": user_cycle_df,
        "summary_df": summary_df,
        "manifest": manifest,
    }


def _print_subgroup_build_summary(result: dict[str, object]) -> None:
    summary_df: pd.DataFrame = result["summary_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research Subgroup Build")
    print("============================================================================")
    if summary_df.empty:
        print("\n  No subgroup rows generated.")
        return
    print("\n  Subgroup families:")
    for family, fam_df in summary_df.groupby("subgroup_family"):
        print(f"  - {family}:")
        for _, row in fam_df.iterrows():
            print(
                f"      {row['subgroup_name']}: rows={int(row['n_rows'])} "
                f"users={int(row['n_users'])} "
                f"mean_hist_cycles={row['mean_history_cycles']:.2f}"
            )


def _evaluate_method_for_subset(
    cs,
    lh,
    subj_order,
    det_by_day,
    conf_by_day,
    subset,
    label,
    use_pop_prior=False,
    custom_acl=None,
    baseline_mode="dynamic",
):
    det, confs = _collapse_daily_to_cycle_estimate(det_by_day or {}, conf_by_day or {})
    summary = _silent_call(
        evaluate_prefix_current_day,
        cs,
        det_by_day or {},
        conf_by_day or {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=label,
        use_stability_gate=False,
        use_population_only_prior=use_pop_prior,
        custom_cycle_priors=custom_acl,
        baseline_mode=baseline_mode,
    )
    post_trigger = _silent_call(
        evaluate_prefix_post_trigger,
        cs,
        det_by_day or {},
        conf_by_day or {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=label,
        use_stability_gate=False,
        use_population_only_prior=use_pop_prior,
        custom_cycle_priors=custom_acl,
        baseline_mode=baseline_mode,
    )
    anchor = _silent_call(
        predict_menses_by_anchors,
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=label,
        use_population_only_prior=use_pop_prior,
        custom_cycle_priors=custom_acl,
        baseline_mode=baseline_mode,
        anchor_days=[3, 5, 10],
    )
    anchor_triggered = _silent_call(
        predict_menses_by_anchors,
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=f"{label} Triggered",
        use_population_only_prior=use_pop_prior,
        custom_cycle_priors=custom_acl,
        baseline_mode=baseline_mode,
        anchor_mode="triggered",
        det_by_day=det_by_day or {},
        confs_by_day=conf_by_day or {},
        anchor_days=[3, 5, 10],
        use_stability_gate=False,
    )
    ov_summary = _silent_call(_ovulation_accuracy_summary_subset, det_by_day or {}, lh, label, subset)
    detected = _silent_call(
        _detected_cycle_bundle,
        cs,
        lh,
        subj_order,
        det_by_day or {},
        conf_by_day or {},
        subset,
        label,
        False,
    )
    return {
        "summary": summary,
        "post_trigger": post_trigger,
        "anchor": anchor,
        "anchor_triggered": anchor_triggered,
        "ov_summary": ov_summary,
        "detected": detected,
    }


def _build_rows_for_subgroups(
    subgroup_df: pd.DataFrame,
    subgroup_families: list[str],
    methods: Sequence[MethodSpec | tuple[Any, ...]],
    cycle_series,
    lh_ov_dict,
    subj_order,
    subgroup_version: str,
    custom_acl=None,
) -> pd.DataFrame:
    labeled = set(s for s in cycle_series if s in lh_ov_dict)
    rows: list[dict[str, object]] = []

    def _append_rows(
        subgroup_family: str,
        subgroup_name: str,
        subset: set[str],
        n_users: int,
        n_cycles: int,
    ) -> None:
        evaluated_methods: list[tuple[str, dict[str, object]]] = []
        for spec in coerce_method_specs(methods):
            det_by_day = spec.det_by_day or {}
            conf_by_day = spec.conf_by_day or {}
            current_acl = custom_acl if spec.countdown_prior_mode == "history_acl" else None

            bundle = _evaluate_method_for_subset(
                cycle_series,
                lh_ov_dict,
                subj_order,
                det_by_day,
                conf_by_day,
                subset,
                f"{spec.name} {subgroup_family}={subgroup_name}",
                use_pop_prior=spec.use_population_only_prior,
                custom_acl=current_acl,
                baseline_mode=spec.baseline_mode,
            )
            evaluated_methods.append((spec.name, bundle))

        for method_name, bundle in evaluated_methods:
            summary = bundle["summary"]
            post_trigger = bundle["post_trigger"]
            anchor = bundle["anchor"]
            anchor_triggered = bundle.get("anchor_triggered", {}) or {}
            ov_summary = bundle["ov_summary"]
            detected = bundle["detected"]["cycles"]
            ov_plus_5 = _anchor_day_summary(anchor, 5)
            ov_plus_10 = _anchor_day_summary(anchor, 10)
            ov_plus_3 = _anchor_day_summary(anchor, 3)
            ov_plus_3_triggered = _anchor_day_summary(anchor_triggered, 3)
            ov_plus_5_triggered = _anchor_day_summary(anchor_triggered, 5)
            ov_plus_10_triggered = _anchor_day_summary(anchor_triggered, 10)
            post_ov_ci = _bootstrap_mean_ci(
                summary.get("_raw_abs_errs", {}).get("post_ov_days"),  # type: ignore[union-attr]
                f"{subgroup_family}:{subgroup_name}:{method_name}:post_ov",
            )
            ov_plus_5_ci = _bootstrap_mean_ci(
                ov_plus_5.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_5",
            )
            ov_plus_10_ci = _bootstrap_mean_ci(
                ov_plus_10.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_10",
            )
            ov_plus_3_ci = _bootstrap_mean_ci(
                ov_plus_3.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_3",
            )
            ov_plus_3_triggered_ci = _bootstrap_mean_ci(
                ov_plus_3_triggered.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_3_triggered",
            )
            ov_plus_5_triggered_ci = _bootstrap_mean_ci(
                ov_plus_5_triggered.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_5_triggered",
            )
            ov_plus_10_triggered_ci = _bootstrap_mean_ci(
                ov_plus_10_triggered.get("_raw_abs_errs"),
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_plus_10_triggered",
            )
            post_trigger_ci = _bootstrap_mean_ci(
                post_trigger.get("_raw_abs_errs"),  # type: ignore[union-attr]
                f"{subgroup_family}:{subgroup_name}:{method_name}:post_trigger",
            )
            ov_first_ci = _bootstrap_mean_ci(
                ov_summary.get("first", {}).get("_raw_abs_errs"),  # type: ignore[union-attr]
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_first",
            )
            ov_final_ci = _bootstrap_mean_ci(
                ov_summary.get("final", {}).get("_raw_abs_errs"),  # type: ignore[union-attr]
                f"{subgroup_family}:{subgroup_name}:{method_name}:ov_final",
            )
            rows.append(
                {
                    "subgroup_family": subgroup_family,
                    "subgroup_name": subgroup_name,
                    "method": method_name,
                    "n_cycles": n_cycles,
                    "n_users": n_users,
                    "post_ov_mae": summary.get("post_ov_days", {}).get("mae"),
                    "post_ov_mae_ci_low": post_ov_ci["mae_ci_low"],
                    "post_ov_mae_ci_high": post_ov_ci["mae_ci_high"],
                    "post_ov_acc_2d": summary.get("post_ov_days", {}).get("acc_2d"),
                    "post_ov_acc_3d": summary.get("post_ov_days", {}).get("acc_3d"),
                    "ov_plus_5_mae": ov_plus_5.get("mae"),
                    "ov_plus_5_mae_ci_low": ov_plus_5_ci["mae_ci_low"],
                    "ov_plus_5_mae_ci_high": ov_plus_5_ci["mae_ci_high"],
                    "ov_plus_5_acc_2d": ov_plus_5.get("acc_2d"),
                    "ov_plus_5_acc_3d": ov_plus_5.get("acc_3d"),
                    "ov_plus_10_mae": ov_plus_10.get("mae"),
                    "ov_plus_10_mae_ci_low": ov_plus_10_ci["mae_ci_low"],
                    "ov_plus_10_mae_ci_high": ov_plus_10_ci["mae_ci_high"],
                    "ov_plus_10_acc_2d": ov_plus_10.get("acc_2d"),
                    "ov_plus_10_acc_3d": ov_plus_10.get("acc_3d"),
                    "ov_plus_3_mae": ov_plus_3.get("mae"),
                    "ov_plus_3_mae_ci_low": ov_plus_3_ci["mae_ci_low"],
                    "ov_plus_3_mae_ci_high": ov_plus_3_ci["mae_ci_high"],
                    "ov_plus_3_acc_2d": ov_plus_3.get("acc_2d"),
                    "ov_plus_3_acc_3d": ov_plus_3.get("acc_3d"),
                    "ov_plus_3_triggered_mae": ov_plus_3_triggered.get("mae"),
                    "ov_plus_3_triggered_mae_ci_low": ov_plus_3_triggered_ci["mae_ci_low"],
                    "ov_plus_3_triggered_mae_ci_high": ov_plus_3_triggered_ci["mae_ci_high"],
                    "ov_plus_3_triggered_acc_2d": ov_plus_3_triggered.get("acc_2d"),
                    "ov_plus_3_triggered_acc_3d": ov_plus_3_triggered.get("acc_3d"),
                    "ov_plus_5_triggered_mae": ov_plus_5_triggered.get("mae"),
                    "ov_plus_5_triggered_mae_ci_low": ov_plus_5_triggered_ci["mae_ci_low"],
                    "ov_plus_5_triggered_mae_ci_high": ov_plus_5_triggered_ci["mae_ci_high"],
                    "ov_plus_5_triggered_acc_2d": ov_plus_5_triggered.get("acc_2d"),
                    "ov_plus_5_triggered_acc_3d": ov_plus_5_triggered.get("acc_3d"),
                    "ov_plus_10_triggered_mae": ov_plus_10_triggered.get("mae"),
                    "ov_plus_10_triggered_mae_ci_low": ov_plus_10_triggered_ci["mae_ci_low"],
                    "ov_plus_10_triggered_mae_ci_high": ov_plus_10_triggered_ci["mae_ci_high"],
                    "ov_plus_10_triggered_acc_2d": ov_plus_10_triggered.get("acc_2d"),
                    "ov_plus_10_triggered_acc_3d": ov_plus_10_triggered.get("acc_3d"),
                    "post_trigger_mae": post_trigger.get("mae"),
                    "post_trigger_mae_ci_low": post_trigger_ci["mae_ci_low"],
                    "post_trigger_mae_ci_high": post_trigger_ci["mae_ci_high"],
                    "post_trigger_acc_2d": post_trigger.get("acc_2d"),
                    "post_trigger_acc_3d": post_trigger.get("acc_3d"),
                    "all_days_mae": summary.get("all_days", {}).get("mae"),
                    "ov_first_mae": ov_summary.get("first", {}).get("mae"),
                    "ov_first_mae_ci_low": ov_first_ci["mae_ci_low"],
                    "ov_first_mae_ci_high": ov_first_ci["mae_ci_high"],
                    "ov_final_mae": ov_summary.get("final", {}).get("mae"),
                    "ov_final_mae_ci_low": ov_final_ci["mae_ci_low"],
                    "ov_final_mae_ci_high": ov_final_ci["mae_ci_high"],
                    "detected_cycle_rate": detected.get("detected_cycle_rate"),
                    "first_detection_cycle_day_mean": detected.get("first_detection_cycle_day_mean"),
                    "latency_days_mean": detected.get("latency_days_mean"),
                    "anchor_post_all_mae": anchor.get("post_all", {}).get("mae"),
                    "subgroup_version": subgroup_version,
                }
            )

    overall_subset = set(subgroup_df["small_group_key"]) & labeled
    if overall_subset:
        overall_users = int(
            subgroup_df[subgroup_df["small_group_key"].isin(overall_subset)]["user_id"].nunique()
        )
        _append_rows(
            "overall",
            "all-labeled",
            overall_subset,
            overall_users,
            int(len(overall_subset)),
        )

    for family in subgroup_families:
        fam_df = subgroup_df[subgroup_df[family].notna()].copy()
        for subgroup_name, sg_df in fam_df.groupby(family):
            subset = set(sg_df["small_group_key"]) & labeled
            if not subset:
                continue
            n_users = int(sg_df["user_id"].nunique())
            n_cycles = int(len(subset))
            _append_rows(family, subgroup_name, subset, n_users, n_cycles)
    return pd.DataFrame(rows).sort_values(
        ["subgroup_family", "subgroup_name", "method"]
    ).reset_index(drop=True)


def _build_population_wearable_predictions(
    cs,
    lh,
    reference_key: str | None = None,
) -> tuple[WearableReferenceSpec, str, dict[str, list[int | None]], dict[str, list[float]]]:
    ref_spec = _resolve_wearable_reference_spec(reference_key)
    candidate = _phase_candidate(ref_spec.group_name, **ref_spec.phase_kwargs)
    det_by_day, conf_by_day = _silent_call(candidate["fn"], cs, lh)
    return ref_spec, candidate["name"], det_by_day, conf_by_day


def run_subgroup_baseline_analysis(
    cfg: SubgroupConfig | None = None,
    wearable_reference: str | None = None,
) -> dict[str, object]:
    cfg = cfg or SubgroupConfig()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    wearable_spec, wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series,
        lh_ov_dict,
        reference_key=wearable_reference,
    )
    history_acl = apply_history_prior_menses_prediction(cycle_series, subgroup_df)

    methods = [
        MethodSpec("Calendar", {}, {}, True, "static", "population"),
        MethodSpec("HistoryPrior-Menses", None, None, False, "static", "history_acl"),
        MethodSpec(wearable_name, wearable_det, wearable_conf, True, "dynamic", "population"),
    ]

    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "stable_length_profile",  # TASK 3
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg.subgroup_version,
        custom_acl=history_acl,
    )
    manifest = {
        "analysis_version": "baseline_subgroup_v1",
        "subgroup_version": cfg.subgroup_version,
        "methods": [m.name for m in methods],
        "wearable_reference": _wearable_reference_manifest(wearable_spec),
        "subgroup_families": subgroup_families,
        "includes_overall_row": True,
        "statistics": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "metrics_with_ci": [
                "ov_plus_3_triggered_mae",
                "ov_plus_5_triggered_mae",
                "ov_plus_10_triggered_mae",
                "ov_plus_3_mae",
                "ov_plus_5_mae",
                "ov_plus_10_mae",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
            ],
        },
        "n_rows": int(len(results_df)),
        "artifacts": {
            "baseline_results_csv": str(BASELINE_RESULTS_CSV),
            "baseline_results_md": str(BASELINE_RESULTS_MD),
            "baseline_manifest_json": str(BASELINE_MANIFEST_JSON),
        },
    }
    export_csv(results_df, BASELINE_RESULTS_CSV)
    export_baseline_markdown(results_df, BASELINE_RESULTS_MD)
    export_json(manifest, BASELINE_MANIFEST_JSON)

    return {
        "results_df": results_df,
        "manifest": manifest,
    }


def run_l1_zero_shot_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l1_cfg: L1Config | None = None,
    wearable_reference: str | None = None,
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l1_cfg = l1_cfg or L1Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    wearable_spec, wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series,
        lh_ov_dict,
        reference_key=wearable_reference,
    )
    history_acl = apply_history_prior_menses_prediction(cycle_series, subgroup_df)
    
    l1_profile_df = build_zero_shot_personalization_profile_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l1_cfg,
    )
    l1_det, l1_conf = apply_l1_zero_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l1_profile_df,
    )
    l1_audit = audit_l1_active_effect(
        cycle_series,
        lh_ov_dict,
        subj_order,
        l1_profile_df,
        wearable_det,
        l1_det,
    )
    write_audit_artifacts(
        l1_audit["summary"],
        l1_audit["cycle_df"],
        L1_AUDIT_JSON,
        L1_AUDIT_CYCLE_CSV,
    )

    methods = [
        MethodSpec("Calendar", {}, {}, True, "static", "population"),
        MethodSpec("HistoryPrior-Menses", None, None, False, "static", "history_acl"),
        MethodSpec(wearable_name, wearable_det, wearable_conf, True, "dynamic", "population"),
        MethodSpec(
            _personalized_method_name(wearable_spec, "L1-zero-shot"),
            l1_det,
            l1_conf,
            True,
            "dynamic",
            "population",
        ),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "stable_length_profile",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
        custom_acl=history_acl,
    )
    manifest = {
        "analysis_version": "l1_zero_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "profile_version": l1_cfg.profile_version,
        "methods": [m.name for m in methods],
        "wearable_reference": _wearable_reference_manifest(wearable_spec),
        "subgroup_families": subgroup_families,
        "includes_overall_row": True,
        "statistics": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "metrics_with_ci": [
                "ov_plus_3_triggered_mae",
                "ov_plus_5_triggered_mae",
                "ov_plus_10_triggered_mae",
                "ov_plus_3_mae",
                "ov_plus_5_mae",
                "ov_plus_10_mae",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
            ],
        },
        "n_rows": int(len(results_df)),
        "l1_manifest": l1_manifest(l1_profile_df, cfg=l1_cfg),
        "artifacts": {
            "l1_profile_csv": str(L1_PROFILE_CSV),
            "l1_results_csv": str(L1_RESULTS_CSV),
            "l1_results_md": str(L1_RESULTS_MD),
            "l1_manifest_json": str(L1_MANIFEST_JSON),
            "l1_active_effect_audit_json": str(L1_AUDIT_JSON),
            "l1_active_effect_cycles_csv": str(L1_AUDIT_CYCLE_CSV),
        },
    }
    export_csv(l1_profile_df, L1_PROFILE_CSV)
    export_csv(results_df, L1_RESULTS_CSV)
    export_baseline_markdown(results_df, L1_RESULTS_MD)
    export_json(manifest, L1_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "profile_df": l1_profile_df,
        "manifest": manifest,
    }


def run_l2_one_shot_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l2_cfg: L2Config | None = None,
    wearable_reference: str | None = None,
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l2_cfg = l2_cfg or L2Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    wearable_spec, wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series,
        lh_ov_dict,
        reference_key=wearable_reference,
    )
    history_acl = apply_history_prior_menses_prediction(cycle_series, subgroup_df)

    l2_profile_df = build_one_shot_personalization_profile_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l2_cfg,
    )
    localizer_payload = _silent_call(
        _precompute_prefix_localizer_payload,
        cycle_series,
        list(wearable_spec.localizer_sigs),
        wearable_spec.lookback_localize,
        wearable_spec.localizer_cache_tag,
    )
    l2_det, l2_conf = apply_l2_one_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l2_profile_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l2_cfg,
    )
    l2a_det, l2a_conf = apply_l2_one_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l2_profile_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l2_cfg,
        variant="L2a",
    )
    l2b_det, l2b_conf = apply_l2_one_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l2_profile_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l2_cfg,
        variant="L2b",
    )
    l2_base_pt = compute_post_trigger_mae_per_cycle_track(
        cycle_series,
        lh_ov_dict,
        subj_order,
        wearable_det,
        wearable_conf,
        baseline_mode="dynamic",
        use_population_only_prior=False,
        custom_cycle_priors=history_acl,
    )
    l2_audit_by_variant: dict[str, Any] = {}
    l2_cycle_parts: list[pd.DataFrame] = []
    for label, det, conf in (
        ("L2", l2_det, l2_conf),
        ("L2a", l2a_det, l2a_conf),
        ("L2b", l2b_det, l2b_conf),
    ):
        l2_variant_audit = audit_l2_active_effect(
            cycle_series,
            lh_ov_dict,
            subj_order,
            l2_profile_df,
            wearable_det,
            wearable_conf,
            det,
            conf,
            variant_label=label,
            custom_cycle_priors=history_acl,
            cached_base_post_trigger_df=l2_base_pt,
        )
        l2_audit_by_variant[label] = l2_variant_audit["summary"]
        l2_cycle_parts.append(l2_variant_audit["cycle_df"])
    write_l2_active_effect_bundle(
        l2_audit_by_variant,
        pd.concat(l2_cycle_parts, ignore_index=True),
        L2_AUDIT_JSON,
        L2_AUDIT_CYCLE_CSV,
    )

    methods = [
        MethodSpec("Calendar", {}, {}, True, "static", "population"),
        MethodSpec("HistoryPrior-Menses", None, None, False, "static", "history_acl"),
        MethodSpec(wearable_name, wearable_det, wearable_conf, True, "dynamic", "population"),
        MethodSpec(
            _personalized_method_name(wearable_spec, "L2-one-shot"),
            l2_det,
            l2_conf,
            True,
            "dynamic",
            "population",
        ),
        MethodSpec(
            _personalized_method_name(wearable_spec, "L2a-LocalizerOnly"),
            l2a_det,
            l2a_conf,
            True,
            "dynamic",
            "population",
        ),
        MethodSpec(
            _personalized_method_name(wearable_spec, "L2b-TempEvidenceOnly"),
            l2b_det,
            l2b_conf,
            True,
            "dynamic",
            "population",
        ),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "stable_length_profile",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
        custom_acl=history_acl,
    )
    manifest = {
        "analysis_version": "l2_one_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "profile_version": l2_cfg.profile_version,
        "methods": [m.name for m in methods],
        "wearable_reference": _wearable_reference_manifest(wearable_spec),
        "subgroup_families": subgroup_families,
        "includes_overall_row": True,
        "statistics": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "metrics_with_ci": [
                "ov_plus_3_triggered_mae",
                "ov_plus_5_triggered_mae",
                "ov_plus_10_triggered_mae",
                "ov_plus_3_mae",
                "ov_plus_5_mae",
                "ov_plus_10_mae",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
            ],
        },
        "n_rows": int(len(results_df)),
        "l2_manifest": l2_manifest(l2_profile_df, cfg=l2_cfg),
        "artifacts": {
            "l2_profile_csv": str(L2_PROFILE_CSV),
            "l2_results_csv": str(L2_RESULTS_CSV),
            "l2_results_md": str(L2_RESULTS_MD),
            "l2_manifest_json": str(L2_MANIFEST_JSON),
            "l2_active_effect_audit_json": str(L2_AUDIT_JSON),
            "l2_active_effect_cycles_csv": str(L2_AUDIT_CYCLE_CSV),
        },
    }
    export_csv(l2_profile_df, L2_PROFILE_CSV)
    export_csv(results_df, L2_RESULTS_CSV)
    export_baseline_markdown(results_df, L2_RESULTS_MD)
    export_json(manifest, L2_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "profile_df": l2_profile_df,
        "manifest": manifest,
    }


def run_l3_few_shot_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l3_cfg: L3Config | None = None,
    wearable_reference: str | None = None,
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l3_cfg = l3_cfg or L3Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    wearable_spec, wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series,
        lh_ov_dict,
        reference_key=wearable_reference,
    )
    history_acl = apply_history_prior_menses_prediction(cycle_series, subgroup_df)

    l3_profile_df = build_few_shot_personalization_profile_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l3_cfg,
    )
    localizer_payload = _silent_call(
        _precompute_prefix_localizer_payload,
        cycle_series,
        list(wearable_spec.localizer_sigs),
        wearable_spec.lookback_localize,
        wearable_spec.localizer_cache_tag,
    )
    l3_det, l3_conf = apply_l3_few_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l3_profile_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l3_cfg,
    )
    l3_audit = audit_l3_active_effect(
        cycle_series,
        lh_ov_dict,
        subj_order,
        l3_profile_df,
        wearable_det,
        wearable_conf,
        l3_det,
        l3_conf,
        custom_cycle_priors=history_acl,
    )
    write_audit_artifacts(
        l3_audit["summary"],
        l3_audit["cycle_df"],
        L3_AUDIT_JSON,
        L3_AUDIT_CYCLE_CSV,
        adjust_df=l3_audit["adjust_day_df"],
        adjust_csv_path=L3_AUDIT_ADJUST_DAY_CSV,
    )

    methods = [
        MethodSpec("Calendar", {}, {}, True, "static", "population"),
        MethodSpec("HistoryPrior-Menses", None, None, False, "static", "history_acl"),
        MethodSpec(wearable_name, wearable_det, wearable_conf, True, "dynamic", "population"),
        MethodSpec(
            _personalized_method_name(wearable_spec, "L3-few-shot"),
            l3_det,
            l3_conf,
            True,
            "dynamic",
            "population",
        ),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "stable_length_profile",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
        custom_acl=history_acl,
    )
    manifest = {
        "analysis_version": "l3_few_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "profile_version": l3_cfg.profile_version,
        "methods": [m.name for m in methods],
        "wearable_reference": _wearable_reference_manifest(wearable_spec),
        "subgroup_families": subgroup_families,
        "includes_overall_row": True,
        "statistics": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "metrics_with_ci": [
                "ov_plus_3_triggered_mae",
                "ov_plus_5_triggered_mae",
                "ov_plus_10_triggered_mae",
                "ov_plus_3_mae",
                "ov_plus_5_mae",
                "ov_plus_10_mae",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
            ],
        },
        "n_rows": int(len(results_df)),
        "l3_manifest": l3_manifest(l3_profile_df, cfg=l3_cfg),
        "artifacts": {
            "l3_profile_csv": str(L3_PROFILE_CSV),
            "l3_results_csv": str(L3_RESULTS_CSV),
            "l3_results_md": str(L3_RESULTS_MD),
            "l3_manifest_json": str(L3_MANIFEST_JSON),
            "l3_active_effect_audit_json": str(L3_AUDIT_JSON),
            "l3_active_effect_cycles_csv": str(L3_AUDIT_CYCLE_CSV),
            "l3_active_effect_adjust_days_csv": str(L3_AUDIT_ADJUST_DAY_CSV),
        },
    }
    export_csv(l3_profile_df, L3_PROFILE_CSV)
    export_csv(results_df, L3_RESULTS_CSV)
    export_baseline_markdown(results_df, L3_RESULTS_MD)
    export_json(manifest, L3_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "profile_df": l3_profile_df,
        "manifest": manifest,
    }


def run_personalization_matrix_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l3_cfg: L3Config | None = None,
    wearable_reference: str | None = None,
) -> dict[str, object]:
    """
    2×2-style matrix: BaseDet (population wearable) vs PersDet (histophys bounded refine-only,
    no imputation) × population vs history ACL countdown priors. Legacy L1/L2/L3 paths unchanged.
    """
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l3_cfg = l3_cfg or L3Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = _load_all_signals_silent()
    wearable_spec, wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series,
        lh_ov_dict,
        reference_key=wearable_reference,
    )
    history_acl = apply_history_prior_menses_prediction(cycle_series, subgroup_df)

    l3_profile_df = build_few_shot_personalization_profile_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l3_cfg,
    )
    localizer_payload = _silent_call(
        _precompute_prefix_localizer_payload,
        cycle_series,
        list(wearable_spec.localizer_sigs),
        wearable_spec.lookback_localize,
        wearable_spec.localizer_cache_tag,
    )
    matrix_det, matrix_conf = apply_l3_few_shot_personalization(
        cycle_series,
        wearable_det,
        wearable_conf,
        l3_profile_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l3_cfg,
        allow_imputation=False,
    )

    matrix_audit = summarize_matrix_detect_post_trigger_mae(
        cycle_series,
        lh_ov_dict,
        subj_order,
        wearable_det,
        wearable_conf,
        matrix_det,
        matrix_conf,
        history_acl,
    )
    export_json(matrix_audit, MATRIX_AUDIT_JSON)

    methods = [
        MethodSpec("RefCalendar[Matrix]", {}, {}, True, "static", "population"),
        MethodSpec("RefHistoryACL[Matrix]", None, None, False, "static", "history_acl"),
        MethodSpec("BaseDet+PopCount[Matrix]", wearable_det, wearable_conf, True, "dynamic", "population"),
        MethodSpec(
            "BaseDet+HistACLCount[Matrix]",
            wearable_det,
            wearable_conf,
            True,
            "dynamic",
            "history_acl",
        ),
        MethodSpec("PersDet+PopCount[Matrix]", matrix_det, matrix_conf, True, "dynamic", "population"),
        MethodSpec(
            "PersDet+HistACLCount[Matrix]",
            matrix_det,
            matrix_conf,
            True,
            "dynamic",
            "history_acl",
        ),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
        "stable_length_profile",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
        custom_acl=history_acl,
    )
    manifest: dict[str, object] = {
        "analysis_version": "personalization_matrix_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "profile_version": l3_cfg.profile_version,
        "methods": [m.name for m in methods],
        "wearable_reference": _wearable_reference_manifest(wearable_spec),
        "subgroup_families": subgroup_families,
        "includes_overall_row": True,
        "l3_allow_imputation": False,
        "detect_variant": "HistPhysRefine_vs_Base",
        "countdown_prior_modes": ["population", "history_acl"],
        "statistics": {
            "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "metrics_with_ci": [
                "ov_plus_3_triggered_mae",
                "ov_plus_5_triggered_mae",
                "ov_plus_10_triggered_mae",
                "ov_plus_3_mae",
                "ov_plus_5_mae",
                "ov_plus_10_mae",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
            ],
        },
        "n_rows": int(len(results_df)),
        "histphys_refine_profile_manifest": l3_manifest(l3_profile_df, cfg=l3_cfg),
        "matrix_detect_audit": matrix_audit,
        "artifacts": {
            "matrix_histphys_refine_profile_csv": str(MATRIX_PROFILE_CSV),
            "matrix_results_csv": str(MATRIX_RESULTS_CSV),
            "matrix_results_md": str(MATRIX_RESULTS_MD),
            "matrix_manifest_json": str(MATRIX_MANIFEST_JSON),
            "matrix_detect_audit_json": str(MATRIX_AUDIT_JSON),
        },
    }
    export_csv(l3_profile_df, MATRIX_PROFILE_CSV)
    export_csv(results_df, MATRIX_RESULTS_CSV)
    export_matrix_markdown(results_df, MATRIX_RESULTS_MD)
    export_json(manifest, MATRIX_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "profile_df": l3_profile_df,
        "manifest": manifest,
        "matrix_audit": matrix_audit,
    }


def _print_baseline_subgroup_summary(result: dict[str, object]) -> None:
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research Baseline Subgroup Analysis")
    print("============================================================================")
    if results_df.empty:
        print("\n  No subgroup baseline rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                ov_plus_5 = "nan" if pd.isna(row["ov_plus_5_mae"]) else f"{row['ov_plus_5_mae']:.2f}"
                ov_plus_10 = "nan" if pd.isna(row["ov_plus_10_mae"]) else f"{row['ov_plus_10_mae']:.2f}"
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                post_ov = "nan" if pd.isna(row["post_ov_mae"]) else f"{row['post_ov_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"Ov+5={ov_plus_5} "
                    f"Ov+10={ov_plus_10} "
                    f"PostTrig={post_trig} "
                    f"PostOvAll={post_ov} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l1_summary(result: dict[str, object]) -> None:
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L1 Zero-Shot Analysis")
    print("============================================================================")
    if results_df.empty:
        print("\n  No L1 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                ov_plus_5 = "nan" if pd.isna(row["ov_plus_5_mae"]) else f"{row['ov_plus_5_mae']:.2f}"
                ov_plus_10 = "nan" if pd.isna(row["ov_plus_10_mae"]) else f"{row['ov_plus_10_mae']:.2f}"
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                post_ov = "nan" if pd.isna(row["post_ov_mae"]) else f"{row['post_ov_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"Ov+5={ov_plus_5} "
                    f"Ov+10={ov_plus_10} "
                    f"PostTrig={post_trig} "
                    f"PostOvAll={post_ov} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l2_summary(result: dict[str, object]) -> None:
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L2 One-Shot Analysis")
    print("============================================================================")
    if results_df.empty:
        print("\n  No L2 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                ov_plus_5 = "nan" if pd.isna(row["ov_plus_5_mae"]) else f"{row['ov_plus_5_mae']:.2f}"
                ov_plus_10 = "nan" if pd.isna(row["ov_plus_10_mae"]) else f"{row['ov_plus_10_mae']:.2f}"
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                post_ov = "nan" if pd.isna(row["post_ov_mae"]) else f"{row['post_ov_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"Ov+5={ov_plus_5} "
                    f"Ov+10={ov_plus_10} "
                    f"PostTrig={post_trig} "
                    f"PostOvAll={post_ov} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l3_summary(result: dict[str, object]) -> None:
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L3 Few-Shot Analysis")
    print("============================================================================")
    if results_df.empty:
        print("\n  No L3 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                ov_plus_5 = "nan" if pd.isna(row["ov_plus_5_mae"]) else f"{row['ov_plus_5_mae']:.2f}"
                ov_plus_10 = "nan" if pd.isna(row["ov_plus_10_mae"]) else f"{row['ov_plus_10_mae']:.2f}"
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                post_ov = "nan" if pd.isna(row["post_ov_mae"]) else f"{row['post_ov_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"Ov+5={ov_plus_5} "
                    f"Ov+10={ov_plus_10} "
                    f"PostTrig={post_trig} "
                    f"PostOvAll={post_ov} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_matrix_summary(result: dict[str, object]) -> None:
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    audit: dict[str, object] = result.get("matrix_audit", {})  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research Personalization Matrix (Detect × Countdown)")
    print("============================================================================")
    if audit:
        by_cd = audit.get("by_countdown_prior", {})
        print("\n  PostTrigger MAE audit (Pers minus Base, per labeled cycle):")
        for mode, stats in by_cd.items():
            print(f"    {mode}: mean_delta={stats.get('mean_delta_pers_minus_base_over_cycles')} "
                  f"n_impute_days_total={stats.get('n_impute_days_total')}")
    if results_df.empty:
        print("\n  No matrix rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                ov_plus_5 = "nan" if pd.isna(row["ov_plus_5_mae"]) else f"{row['ov_plus_5_mae']:.2f}"
                ov_plus_10 = "nan" if pd.isna(row["ov_plus_10_mae"]) else f"{row['ov_plus_10_mae']:.2f}"
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                post_ov = "nan" if pd.isna(row["post_ov_mae"]) else f"{row['post_ov_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"Ov+5={ov_plus_5} "
                    f"Ov+10={ov_plus_10} "
                    f"PostTrig={post_trig} "
                    f"PostOvAll={post_ov} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research experiment runner")
    parser.add_argument(
        "--mode",
        choices=[
            "build-subgroups",
            "subgroup-baseline",
            "personalize-l1",
            "personalize-l2",
            "personalize-l3",
            "personalize-matrix",
        ],
        default="build-subgroups",
        help="Research mode to execute.",
    )
    parser.add_argument(
        "--wearable-reference",
        choices=sorted(WEARABLE_REFERENCE_SPECS),
        default=DEFAULT_WEARABLE_REFERENCE_KEY,
        help="Population wearable reference model used for baseline/personalization analyses.",
    )
    args = parser.parse_args()

    if args.mode == "build-subgroups":
        result = run_subgroup_build()
        _print_subgroup_build_summary(result)
    elif args.mode == "subgroup-baseline":
        result = run_subgroup_baseline_analysis(wearable_reference=args.wearable_reference)
        _print_baseline_subgroup_summary(result)
    elif args.mode == "personalize-l1":
        result = run_l1_zero_shot_analysis(wearable_reference=args.wearable_reference)
        _print_l1_summary(result)
    elif args.mode == "personalize-l2":
        result = run_l2_one_shot_analysis(wearable_reference=args.wearable_reference)
        _print_l2_summary(result)
    elif args.mode == "personalize-l3":
        result = run_l3_few_shot_analysis(wearable_reference=args.wearable_reference)
        _print_l3_summary(result)
    elif args.mode == "personalize-matrix":
        result = run_personalization_matrix_analysis(wearable_reference=args.wearable_reference)
        _print_matrix_summary(result)


if __name__ == "__main__":
    main()

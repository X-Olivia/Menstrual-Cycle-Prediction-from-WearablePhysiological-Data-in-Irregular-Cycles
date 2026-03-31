from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from paths import (
    BASELINE_MANIFEST_JSON,
    BASELINE_RESULTS_CSV,
    BASELINE_RESULTS_MD,
    L1_CALIBRATION_CSV,
    L1_MANIFEST_JSON,
    L1_RESULTS_CSV,
    L1_RESULTS_MD,
    L2_CALIBRATION_CSV,
    L2_MANIFEST_JSON,
    L2_RESULTS_CSV,
    L2_RESULTS_MD,
    L3_CALIBRATION_CSV,
    L3_MANIFEST_JSON,
    L3_RESULTS_CSV,
    L3_RESULTS_MD,
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
from protocol import DEFAULT_POPULATION_LUTEAL_LENGTH  # type: ignore  # noqa: E402

from exports import (  # noqa: E402
    export_baseline_markdown,
    export_csv,
    export_json,
    export_summary_markdown,
)
from personalization import (  # noqa: E402
    L1Config,
    L2Config,
    L3Config,
    apply_l1_zero_shot_calibration,
    apply_l2_one_shot_calibration,
    apply_l3_few_shot_calibration,
    build_few_shot_calibration_table,
    build_one_shot_calibration_table,
    build_zero_shot_calibration_table,
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


def run_subgroup_build(cfg: SubgroupConfig | None = None) -> dict[str, object]:
    cfg = cfg or SubgroupConfig()
    ensure_research_dirs()

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = load_all_signals()
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
    manifest = result["manifest"]
    summary_df: pd.DataFrame = result["summary_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research Subgroup Build")
    print("============================================================================")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
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


def _evaluate_method_for_subset(cs, lh, subj_order, det_by_day, conf_by_day, subset, label):
    det, confs = _collapse_daily_to_cycle_estimate(det_by_day, conf_by_day)
    summary = _silent_call(
        evaluate_prefix_current_day,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=label,
        use_stability_gate=False,
    )
    post_trigger = _silent_call(
        evaluate_prefix_post_trigger,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=subset,
        label=label,
        use_stability_gate=False,
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
    )
    ov_summary = _silent_call(_ovulation_accuracy_summary_subset, det_by_day, lh, label, subset)
    detected = _silent_call(
        _detected_cycle_bundle,
        cs,
        lh,
        subj_order,
        det_by_day,
        conf_by_day,
        subset,
        label,
        False,
    )
    return {
        "summary": summary,
        "post_trigger": post_trigger,
        "anchor": anchor,
        "ov_summary": ov_summary,
        "detected": detected,
    }


def _build_rows_for_subgroups(
    subgroup_df: pd.DataFrame,
    subgroup_families: list[str],
    methods: list[tuple[str, dict, dict]],
    cycle_series,
    lh_ov_dict,
    subj_order,
    subgroup_version: str,
) -> pd.DataFrame:
    labeled = set(s for s in cycle_series if s in lh_ov_dict)
    rows: list[dict[str, object]] = []
    for family in subgroup_families:
        fam_df = subgroup_df[subgroup_df[family].notna()].copy()
        for subgroup_name, sg_df in fam_df.groupby(family):
            subset = set(sg_df["small_group_key"]) & labeled
            if not subset:
                continue
            n_users = int(sg_df["user_id"].nunique())
            n_cycles = int(len(subset))
            for method_name, det_by_day, conf_by_day in methods:
                bundle = _evaluate_method_for_subset(
                    cycle_series,
                    lh_ov_dict,
                    subj_order,
                    det_by_day,
                    conf_by_day,
                    subset,
                    f"{method_name} {family}={subgroup_name}",
                )
                summary = bundle["summary"]
                post_trigger = bundle["post_trigger"]
                anchor = bundle["anchor"]
                ov_summary = bundle["ov_summary"]
                detected = bundle["detected"]["cycles"]
                rows.append(
                    {
                        "subgroup_family": family,
                        "subgroup_name": subgroup_name,
                        "method": method_name,
                        "n_cycles": n_cycles,
                        "n_users": n_users,
                        "post_ov_mae": summary.get("post_ov_days", {}).get("mae"),
                        "post_ov_acc_2d": summary.get("post_ov_days", {}).get("acc_2d"),
                        "post_ov_acc_3d": summary.get("post_ov_days", {}).get("acc_3d"),
                        "post_trigger_mae": post_trigger.get("mae"),
                        "post_trigger_acc_2d": post_trigger.get("acc_2d"),
                        "post_trigger_acc_3d": post_trigger.get("acc_3d"),
                        "all_days_mae": summary.get("all_days", {}).get("mae"),
                        "ov_first_mae": ov_summary.get("first", {}).get("mae"),
                        "ov_final_mae": ov_summary.get("final", {}).get("mae"),
                        "detected_cycle_rate": detected.get("detected_cycle_rate"),
                        "first_detection_cycle_day_mean": detected.get("first_detection_cycle_day_mean"),
                        "latency_days_mean": detected.get("latency_days_mean"),
                        "anchor_post_all_mae": anchor.get("post_all", {}).get("mae"),
                        "subgroup_version": subgroup_version,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["subgroup_family", "subgroup_name", "method"]
    ).reset_index(drop=True)


def _build_population_wearable_predictions(cs, lh):
    spec = _phase_candidate("Temp+HR", model_type="rf", name_suffix="[RF-baseline]")
    det_by_day, conf_by_day = spec["fn"](cs, lh)
    return spec["name"], det_by_day, conf_by_day


def run_subgroup_baseline_analysis(cfg: SubgroupConfig | None = None) -> dict[str, object]:
    cfg = cfg or SubgroupConfig()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = load_all_signals()
    wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series, lh_ov_dict
    )
    methods = [
        ("Calendar", {}, {}),
        (wearable_name, wearable_det, wearable_conf),
    ]

    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg.subgroup_version,
    )
    manifest = {
        "analysis_version": "baseline_subgroup_v1",
        "subgroup_version": cfg.subgroup_version,
        "methods": [m[0] for m in methods],
        "subgroup_families": subgroup_families,
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
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l1_cfg = l1_cfg or L1Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = load_all_signals()
    wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series, lh_ov_dict
    )
    l1_calibration_df = build_zero_shot_calibration_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l1_cfg,
    )
    l1_det, l1_conf = apply_l1_zero_shot_calibration(
        cycle_series,
        wearable_det,
        wearable_conf,
        l1_calibration_df,
    )

    methods = [
        ("Calendar", {}, {}),
        (wearable_name, wearable_det, wearable_conf),
        ("PhaseCls-Temp+HR[L1-zero-shot]", l1_det, l1_conf),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
    )
    manifest = {
        "analysis_version": "l1_zero_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "calibration_version": l1_cfg.calibration_version,
        "methods": [m[0] for m in methods],
        "subgroup_families": subgroup_families,
        "n_rows": int(len(results_df)),
        "l1_manifest": l1_manifest(l1_calibration_df, cfg=l1_cfg),
        "artifacts": {
            "l1_calibration_csv": str(L1_CALIBRATION_CSV),
            "l1_results_csv": str(L1_RESULTS_CSV),
            "l1_results_md": str(L1_RESULTS_MD),
            "l1_manifest_json": str(L1_MANIFEST_JSON),
        },
    }
    export_csv(l1_calibration_df, L1_CALIBRATION_CSV)
    export_csv(results_df, L1_RESULTS_CSV)
    export_baseline_markdown(results_df, L1_RESULTS_MD)
    export_json(manifest, L1_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "calibration_df": l1_calibration_df,
        "manifest": manifest,
    }


def run_l2_one_shot_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l2_cfg: L2Config | None = None,
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l2_cfg = l2_cfg or L2Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = load_all_signals()
    wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series, lh_ov_dict
    )
    l2_calibration_df = build_one_shot_calibration_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l2_cfg,
    )
    localizer_payload = _precompute_prefix_localizer_payload(
        cycle_series,
        ["nightly_temperature", "rhr", "noct_hr_min"],
        10,
        "nightly_temperature_rhr_noct_hr_min",
    )
    l2_det, l2_conf = apply_l2_one_shot_calibration(
        cycle_series,
        wearable_det,
        wearable_conf,
        l2_calibration_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l2_cfg,
    )

    methods = [
        ("Calendar", {}, {}),
        (wearable_name, wearable_det, wearable_conf),
        ("PhaseCls-Temp+HR[L2-one-shot]", l2_det, l2_conf),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
    )
    manifest = {
        "analysis_version": "l2_one_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "calibration_version": l2_cfg.calibration_version,
        "methods": [m[0] for m in methods],
        "subgroup_families": subgroup_families,
        "n_rows": int(len(results_df)),
        "l2_manifest": l2_manifest(l2_calibration_df, cfg=l2_cfg),
        "artifacts": {
            "l2_calibration_csv": str(L2_CALIBRATION_CSV),
            "l2_results_csv": str(L2_RESULTS_CSV),
            "l2_results_md": str(L2_RESULTS_MD),
            "l2_manifest_json": str(L2_MANIFEST_JSON),
        },
    }
    export_csv(l2_calibration_df, L2_CALIBRATION_CSV)
    export_csv(results_df, L2_RESULTS_CSV)
    export_baseline_markdown(results_df, L2_RESULTS_MD)
    export_json(manifest, L2_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "calibration_df": l2_calibration_df,
        "manifest": manifest,
    }


def run_l3_few_shot_analysis(
    subgroup_cfg: SubgroupConfig | None = None,
    l3_cfg: L3Config | None = None,
) -> dict[str, object]:
    subgroup_cfg = subgroup_cfg or SubgroupConfig()
    l3_cfg = l3_cfg or L3Config()
    ensure_research_dirs()

    subgroup_result = run_subgroup_build(cfg=subgroup_cfg)
    subgroup_df: pd.DataFrame = subgroup_result["user_cycle_df"]  # type: ignore[assignment]

    lh_ov_dict, cycle_series, quality, subj_order, signal_cols = load_all_signals()
    wearable_name, wearable_det, wearable_conf = _build_population_wearable_predictions(
        cycle_series, lh_ov_dict
    )
    l3_calibration_df = build_few_shot_calibration_table(
        cycle_series,
        lh_ov_dict,
        subj_order,
        cfg=l3_cfg,
    )
    localizer_payload = _precompute_prefix_localizer_payload(
        cycle_series,
        ["nightly_temperature", "rhr", "noct_hr_min"],
        10,
        "nightly_temperature_rhr_noct_hr_min",
    )
    l3_det, l3_conf = apply_l3_few_shot_calibration(
        cycle_series,
        wearable_det,
        wearable_conf,
        l3_calibration_df,
        localizer_payload["localizer_table"],
        localizer_payload["score_table"],
        cfg=l3_cfg,
    )

    methods = [
        ("Calendar", {}, {}),
        (wearable_name, wearable_det, wearable_conf),
        ("PhaseCls-Temp+HR[L3-few-shot]", l3_det, l3_conf),
    ]
    subgroup_families = [
        "cycle_length_level_group",
        "cycle_variability_group",
    ]
    results_df = _build_rows_for_subgroups(
        subgroup_df,
        subgroup_families,
        methods,
        cycle_series,
        lh_ov_dict,
        subj_order,
        subgroup_cfg.subgroup_version,
    )
    manifest = {
        "analysis_version": "l3_few_shot_v1",
        "subgroup_version": subgroup_cfg.subgroup_version,
        "calibration_version": l3_cfg.calibration_version,
        "methods": [m[0] for m in methods],
        "subgroup_families": subgroup_families,
        "n_rows": int(len(results_df)),
        "l3_manifest": l3_manifest(l3_calibration_df, cfg=l3_cfg),
        "artifacts": {
            "l3_calibration_csv": str(L3_CALIBRATION_CSV),
            "l3_results_csv": str(L3_RESULTS_CSV),
            "l3_results_md": str(L3_RESULTS_MD),
            "l3_manifest_json": str(L3_MANIFEST_JSON),
        },
    }
    export_csv(l3_calibration_df, L3_CALIBRATION_CSV)
    export_csv(results_df, L3_RESULTS_CSV)
    export_baseline_markdown(results_df, L3_RESULTS_MD)
    export_json(manifest, L3_MANIFEST_JSON)
    return {
        "results_df": results_df,
        "calibration_df": l3_calibration_df,
        "manifest": manifest,
    }


def _print_baseline_subgroup_summary(result: dict[str, object]) -> None:
    manifest = result["manifest"]
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research Baseline Subgroup Analysis")
    print("============================================================================")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if results_df.empty:
        print("\n  No subgroup baseline rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                ov_first = "nan" if pd.isna(row["ov_first_mae"]) else f"{row['ov_first_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"PostOv={row['post_ov_mae']:.2f} "
                    f"PostTrig={post_trig} "
                    f"OvFirst={ov_first} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l1_summary(result: dict[str, object]) -> None:
    manifest = result["manifest"]
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L1 Zero-Shot Analysis")
    print("============================================================================")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if results_df.empty:
        print("\n  No L1 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                ov_first = "nan" if pd.isna(row["ov_first_mae"]) else f"{row['ov_first_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"PostOv={row['post_ov_mae']:.2f} "
                    f"PostTrig={post_trig} "
                    f"OvFirst={ov_first} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l2_summary(result: dict[str, object]) -> None:
    manifest = result["manifest"]
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L2 One-Shot Analysis")
    print("============================================================================")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if results_df.empty:
        print("\n  No L2 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                ov_first = "nan" if pd.isna(row["ov_first_mae"]) else f"{row['ov_first_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"PostOv={row['post_ov_mae']:.2f} "
                    f"PostTrig={post_trig} "
                    f"OvFirst={ov_first} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def _print_l3_summary(result: dict[str, object]) -> None:
    manifest = result["manifest"]
    results_df: pd.DataFrame = result["results_df"]  # type: ignore[assignment]
    print("\n============================================================================")
    print("  Research L3 Few-Shot Analysis")
    print("============================================================================")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if results_df.empty:
        print("\n  No L3 rows generated.")
        return
    for family, fam_df in results_df.groupby("subgroup_family"):
        print(f"\n  {family}:")
        for subgroup_name, sg_df in fam_df.groupby("subgroup_name"):
            print(f"    {subgroup_name}:")
            for _, row in sg_df.iterrows():
                post_trig = "nan" if pd.isna(row["post_trigger_mae"]) else f"{row['post_trigger_mae']:.2f}"
                ov_first = "nan" if pd.isna(row["ov_first_mae"]) else f"{row['ov_first_mae']:.2f}"
                print(
                    f"      {row['method']}: "
                    f"PostOv={row['post_ov_mae']:.2f} "
                    f"PostTrig={post_trig} "
                    f"OvFirst={ov_first} "
                    f"DetectRate={row['detected_cycle_rate']:.1%}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research experiment runner")
    parser.add_argument(
        "--mode",
        choices=["build-subgroups", "subgroup-baseline", "personalize-l1", "personalize-l2", "personalize-l3"],
        default="build-subgroups",
        help="Research mode to execute.",
    )
    args = parser.parse_args()

    if args.mode == "build-subgroups":
        result = run_subgroup_build()
        _print_subgroup_build_summary(result)
    elif args.mode == "subgroup-baseline":
        result = run_subgroup_baseline_analysis()
        _print_baseline_subgroup_summary(result)
    elif args.mode == "personalize-l1":
        result = run_l1_zero_shot_analysis()
        _print_l1_summary(result)
    elif args.mode == "personalize-l2":
        result = run_l2_one_shot_analysis()
        _print_l2_summary(result)
    elif args.mode == "personalize-l3":
        result = run_l3_few_shot_analysis()
        _print_l3_summary(result)


if __name__ == "__main__":
    main()

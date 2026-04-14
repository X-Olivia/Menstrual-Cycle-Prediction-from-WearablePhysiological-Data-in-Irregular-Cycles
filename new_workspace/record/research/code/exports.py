from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _format_mae_ci(value, ci_low, ci_high) -> str:
    if pd.isna(value):
        return "NA"
    if pd.notna(ci_low) and pd.notna(ci_high):
        return f"{float(value):.3f} [{float(ci_low):.3f}, {float(ci_high):.3f}]"
    return f"{float(value):.3f}"


def _attach_ci_display_cols(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        display_col = f"{metric}_95ci"
        if metric not in out.columns:
            continue
        ci_low = out[low_col] if low_col in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
        ci_high = out[high_col] if high_col in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
        out[display_col] = [
            _format_mae_ci(value, low, high)
            for value, low, high in zip(out[metric], ci_low, ci_high)
        ]
    return out


def export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def export_summary_markdown(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        text = "# Subgroup Summary\n\nNo subgroup rows were generated.\n"
    else:
        text = "# Subgroup Summary\n\n"
        for family, fam_df in summary_df.groupby("subgroup_family"):
            text += f"## {family}\n\n"
            text += fam_df.to_markdown(index=False)
            text += "\n\n"
    path.write_text(text, encoding="utf-8")


def export_baseline_markdown(results_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if results_df.empty:
        path.write_text("# Baseline Subgroup Analysis\n\nNo rows generated.\n", encoding="utf-8")
        return

    results_df = _attach_ci_display_cols(
        results_df,
        [
            "ov_plus_5_mae",
            "ov_plus_10_mae",
            "post_ov_mae",
            "post_trigger_mae",
            "ov_first_mae",
            "ov_final_mae",
        ],
    )

    text = "# Baseline Subgroup Analysis\n\n"
    text += "Primary post-ovulation metrics are shown as fixed anchors (`Ov+5`, `Ov+10`) plus `PostTrigger`.\n"
    text += "MAE metrics are shown as `mean [95% bootstrap CI]`.\n\n"
    for family, fam_df in results_df.groupby("subgroup_family"):
        text += f"## {family}\n\n"
        for method, method_df in fam_df.groupby("method"):
            text += f"### {method}\n\n"
            cols = [
                ("subgroup_name", "Subgroup"),
                ("n_cycles", "Cycles"),
                ("n_users", "Users"),
                ("ov_plus_5_mae_95ci", "Ov+5 MAE [95% CI]"),
                ("ov_plus_5_acc_2d", "Ov+5 ±2d"),
                ("ov_plus_5_acc_3d", "Ov+5 ±3d"),
                ("ov_plus_10_mae_95ci", "Ov+10 MAE [95% CI]"),
                ("ov_plus_10_acc_2d", "Ov+10 ±2d"),
                ("ov_plus_10_acc_3d", "Ov+10 ±3d"),
                ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                ("post_trigger_acc_2d", "PostTrigger ±2d"),
                ("post_trigger_acc_3d", "PostTrigger ±3d"),
                ("post_ov_mae_95ci", "PostOv pooled MAE [95% CI]"),
                ("post_ov_acc_2d", "PostOv pooled ±2d"),
                ("post_ov_acc_3d", "PostOv pooled ±3d"),
                ("ov_first_mae_95ci", "OvFirst MAE [95% CI]"),
                ("ov_final_mae_95ci", "OvFinal MAE [95% CI]"),
                ("detected_cycle_rate", "Detected Rate"),
                ("latency_days_mean", "Latency"),
            ]
            keep = [src for src, _ in cols if src in method_df.columns]
            show = method_df[keep].rename(columns={src: dst for src, dst in cols if src in method_df.columns})
            text += show.to_markdown(index=False)
            text += "\n\n"
    path.write_text(text, encoding="utf-8")


def export_matrix_markdown(results_df: pd.DataFrame, path: Path) -> None:
    """Subgroup-style table for detect × countdown matrix runs (not baseline analysis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if results_df.empty:
        path.write_text("# Personalization Matrix Analysis\n\nNo rows generated.\n", encoding="utf-8")
        return

    results_df = _attach_ci_display_cols(
        results_df,
        [
            "ov_plus_5_mae",
            "ov_plus_10_mae",
            "post_ov_mae",
            "post_trigger_mae",
            "ov_first_mae",
            "ov_final_mae",
        ],
    )

    text = "# Personalization Matrix Analysis\n\n"
    text += "Detect axis: population wearable (BaseDet) vs bounded localizer refinement on non-empty base days only (PersDet, `allow_imputation=False`). "
    text += "Countdown axis: population cycle prior vs per-user history ACL (`history_acl`). "
    text += "Metrics use the same subgroup layout as other research exports; MAE columns are `mean [95% bootstrap CI]`.\n\n"
    for family, fam_df in results_df.groupby("subgroup_family"):
        text += f"## {family}\n\n"
        for method, method_df in fam_df.groupby("method"):
            text += f"### {method}\n\n"
            cols = [
                ("subgroup_name", "Subgroup"),
                ("n_cycles", "Cycles"),
                ("n_users", "Users"),
                ("ov_plus_5_mae_95ci", "Ov+5 MAE [95% CI]"),
                ("ov_plus_5_acc_2d", "Ov+5 ±2d"),
                ("ov_plus_5_acc_3d", "Ov+5 ±3d"),
                ("ov_plus_10_mae_95ci", "Ov+10 MAE [95% CI]"),
                ("ov_plus_10_acc_2d", "Ov+10 ±2d"),
                ("ov_plus_10_acc_3d", "Ov+10 ±3d"),
                ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                ("post_trigger_acc_2d", "PostTrigger ±2d"),
                ("post_trigger_acc_3d", "PostTrigger ±3d"),
                ("post_ov_mae_95ci", "PostOv pooled MAE [95% CI]"),
                ("post_ov_acc_2d", "PostOv pooled ±2d"),
                ("post_ov_acc_3d", "PostOv pooled ±3d"),
                ("ov_first_mae_95ci", "OvFirst MAE [95% CI]"),
                ("ov_final_mae_95ci", "OvFinal MAE [95% CI]"),
                ("detected_cycle_rate", "Detected Rate"),
                ("latency_days_mean", "Latency"),
            ]
            keep = [src for src, _ in cols if src in method_df.columns]
            show = method_df[keep].rename(columns={src: dst for src, dst in cols if src in method_df.columns})
            text += show.to_markdown(index=False)
            text += "\n\n"
    path.write_text(text, encoding="utf-8")

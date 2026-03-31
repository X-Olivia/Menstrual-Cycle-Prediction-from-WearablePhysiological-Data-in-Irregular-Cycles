from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


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

    text = "# Baseline Subgroup Analysis\n\n"
    for family, fam_df in results_df.groupby("subgroup_family"):
        text += f"## {family}\n\n"
        for method, method_df in fam_df.groupby("method"):
            text += f"### {method}\n\n"
            cols = [
                "subgroup_name",
                "n_cycles",
                "n_users",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
                "detected_cycle_rate",
                "latency_days_mean",
            ]
            keep = [c for c in cols if c in method_df.columns]
            text += method_df[keep].to_markdown(index=False)
            text += "\n\n"
    path.write_text(text, encoding="utf-8")

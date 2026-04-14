from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import (
    BASELINE_RESULTS_CSV,
    L1_RESULTS_CSV,
    L2_RESULTS_CSV,
    L3_RESULTS_CSV,
    MATRIX_RESULTS_CSV,
    REPORTS_DIR,
    ensure_research_dirs,
)


REPORT_PATH = REPORTS_DIR / "BASELINE_AND_PERSONALIZATION_REPORT_v1.md"


def _load_results(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["experiment_tag"] = tag
    return df


def _try_load_results(path: Path, tag: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _load_results(path, tag)


def _method_short_name(name: str) -> str:
    if name == "Calendar":
        return "Calendar (Static)"
    if name == "HistoryPrior-Menses":
        return "History-only (Static)"
    if name == "PhaseCls-ENS-Temp+HR[Champion]":
        return "L0 Population"
    if "L2a-LocalizerOnly" in name:
        return "L2a LocalizerOnly"
    if "L2b-TempEvidenceOnly" in name:
        return "L2b TempOnly"
    if "L1-zero-shot" in name:
        return "L1 Zero-shot"
    if "L2-one-shot" in name:
        return "L2 One-shot"
    if "L3-few-shot" in name:
        return "L3 Few-shot"
    if name in ("Calendar[Matrix]", "RefCalendar[Matrix]"):
        return "Matrix: Ref Calendar"
    if name in ("HistoryPrior-Menses[Matrix]", "RefHistoryACL[Matrix]"):
        return "Matrix: Ref HistACL"
    if name in ("Champion[Matrix]", "BaseDet+PopCount[Matrix]"):
        return "Matrix: Base+Pop"
    if name in ("Champion+HistACL[Matrix]", "BaseDet+HistACLCount[Matrix]"):
        return "Matrix: Base+HistACL"
    if name in ("HistPhysRefine+Pop[Matrix]", "PersDet+PopCount[Matrix]"):
        return "Matrix: Pers+Pop"
    if name in ("HistPhysRefine+HistACL[Matrix]", "PersDet+HistACLCount[Matrix]"):
        return "Matrix: Pers+HistACL"
    return name


def _round_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
    return out


def _keep_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[col for col in cols if col in df.columns]].copy()


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


def _present_table(df: pd.DataFrame, column_map: list[tuple[str, str]]) -> str:
    cols = [src for src, _ in column_map if src in df.columns]
    show = df[cols].copy()
    show = show.rename(columns={src: dst for src, dst in column_map if src in show.columns})
    return show.to_markdown(index=False)


def _method_order(name: str) -> int:
    order = {
        "Calendar (Static)": 0,
        "History-only (Static)": 1,
        "L0 Population": 2,
        "L1 Zero-shot": 3,
        "L2 One-shot": 4,
        "L2a LocalizerOnly": 5,
        "L2b TempOnly": 6,
        "L3 Few-shot": 7,
        "Matrix: Ref Calendar": 18,
        "Matrix: Ref HistACL": 19,
        "Matrix: Base+Pop": 20,
        "Matrix: Base+HistACL": 21,
        "Matrix: Pers+Pop": 22,
        "Matrix: Pers+HistACL": 23,
    }
    return order.get(name, 99)


def build_report() -> str:
    result_specs = [
        ("baseline", BASELINE_RESULTS_CSV),
        ("l1", L1_RESULTS_CSV),
        ("l2", L2_RESULTS_CSV),
        ("l3", L3_RESULTS_CSV),
        ("matrix", MATRIX_RESULTS_CSV),
    ]
    available_frames: list[pd.DataFrame] = []
    missing_tags: list[str] = []
    for tag, path in result_specs:
        maybe_df = _try_load_results(path, tag)
        if maybe_df is None:
            missing_tags.append(tag)
        else:
            available_frames.append(maybe_df)

    if not available_frames:
        missing_desc = ", ".join(f"{tag}={path}" for tag, path in result_specs)
        raise FileNotFoundError(
            f"No research result CSVs were found. Expected one or more of: {missing_desc}"
        )

    df = pd.concat(available_frames, ignore_index=True)
    df["method_short"] = df["method"].map(_method_short_name)
    df = _round_cols(
        df,
        [
            "ov_plus_5_mae",
            "ov_plus_5_mae_ci_low",
            "ov_plus_5_mae_ci_high",
            "ov_plus_5_acc_2d",
            "ov_plus_5_acc_3d",
            "ov_plus_10_mae",
            "ov_plus_10_mae_ci_low",
            "ov_plus_10_mae_ci_high",
            "ov_plus_10_acc_2d",
            "ov_plus_10_acc_3d",
            "ov_plus_3_mae",
            "ov_plus_3_mae_ci_low",
            "ov_plus_3_mae_ci_high",
            "ov_plus_3_acc_2d",
            "ov_plus_3_acc_3d",
            "ov_plus_3_triggered_mae",
            "ov_plus_3_triggered_mae_ci_low",
            "ov_plus_3_triggered_mae_ci_high",
            "ov_plus_3_triggered_acc_2d",
            "ov_plus_3_triggered_acc_3d",
            "ov_plus_5_triggered_mae",
            "ov_plus_5_triggered_mae_ci_low",
            "ov_plus_5_triggered_mae_ci_high",
            "ov_plus_5_triggered_acc_2d",
            "ov_plus_5_triggered_acc_3d",
            "ov_plus_10_triggered_mae",
            "ov_plus_10_triggered_mae_ci_low",
            "ov_plus_10_triggered_mae_ci_high",
            "ov_plus_10_triggered_acc_2d",
            "ov_plus_10_triggered_acc_3d",
            "post_ov_mae",
            "post_ov_mae_ci_low",
            "post_ov_mae_ci_high",
            "post_ov_acc_2d",
            "post_ov_acc_3d",
            "post_trigger_mae",
            "post_trigger_mae_ci_low",
            "post_trigger_mae_ci_high",
            "post_trigger_acc_2d",
            "post_trigger_acc_3d",
            "ov_first_mae",
            "ov_first_mae_ci_low",
            "ov_first_mae_ci_high",
            "ov_final_mae",
            "ov_final_mae_ci_low",
            "ov_final_mae_ci_high",
            "detected_cycle_rate",
            "latency_days_mean",
            "anchor_post_all_mae",
            "first_detection_cycle_day_mean",
        ],
    )
    df = _attach_ci_display_cols(
        df,
        [
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
    )

    overall_all = (
        df[df["subgroup_family"] == "overall"]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    matrix_overall = overall_all[overall_all["experiment_tag"] == "matrix"].copy()
    if not matrix_overall.empty:
        matrix_overall["method_order"] = matrix_overall["method_short"].map(_method_order)
        matrix_overall = matrix_overall.sort_values(["method_order", "method_short"]).reset_index(drop=True)

    overall_df = overall_all[overall_all["experiment_tag"] != "matrix"].copy()
    if not overall_df.empty:
        overall_df["method_order"] = overall_df["method_short"].map(_method_order)
        overall_df = overall_df.sort_values(["method_order", "method_short"]).reset_index(drop=True)

    subgroup_only = df[df["subgroup_family"] != "overall"].copy()

    # 1. Wearable Gain Table
    # Compare static Calendar (pop-only) vs static History-only vs L0 wearable
    baseline_only = (
        subgroup_only[
            subgroup_only["method_short"].isin(
                ["Calendar (Static)", "History-only (Static)", "L0 Population"]
            )
        ]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    gain_rows = []
    for (family, subgroup_name), grp in baseline_only.groupby(["subgroup_family", "subgroup_name"]):
        cal = grp[grp["method_short"] == "Calendar (Static)"]
        hist = grp[grp["method_short"] == "History-only (Static)"]
        l0 = grp[grp["method_short"] == "L0 Population"]
        if cal.empty or l0.empty:
            continue
        cal_row = cal.iloc[0]
        l0_row = l0.iloc[0]
        gain_rows.append(
            {
                "subgroup_family": family,
                "subgroup_name": subgroup_name,
                "calendar_ov_plus_3_all": cal_row["ov_plus_3_mae"],
                "calendar_ov_plus_10_all": cal_row["ov_plus_10_mae"],
                "history_only_ov_plus_3_all": hist.iloc[0]["ov_plus_3_mae"] if not hist.empty else float("nan"),
                "history_only_ov_plus_10_all": hist.iloc[0]["ov_plus_10_mae"] if not hist.empty else float("nan"),
                "l0_ov_plus_3_all": l0_row["ov_plus_3_mae"],
                "l0_ov_plus_10_all": l0_row["ov_plus_10_mae"],
                "wearable_gain_vs_cal_ov_plus_3_all": cal_row["ov_plus_3_mae"] - l0_row["ov_plus_3_mae"],
                "wearable_gain_vs_cal_all": cal_row["ov_plus_10_mae"] - l0_row["ov_plus_10_mae"],
                "wearable_gain_vs_hist_ov_plus_3_all": (
                    hist.iloc[0]["ov_plus_3_mae"] - l0_row["ov_plus_3_mae"]
                ) if not hist.empty and pd.notna(hist.iloc[0]["ov_plus_3_mae"]) else float("nan"),
                "wearable_gain_vs_hist_all": (
                    hist.iloc[0]["ov_plus_10_mae"] - l0_row["ov_plus_10_mae"]
                ) if not hist.empty and pd.notna(hist.iloc[0]["ov_plus_10_mae"]) else float("nan"),
            }
        )
    gain_df = pd.DataFrame(gain_rows)
    if not gain_df.empty:
        gain_df = _round_cols(
            gain_df,
            [
                "calendar_ov_plus_10",
                "history_only_ov_plus_10",
                "l0_ov_plus_10",
                "wearable_gain_vs_cal",
                "wearable_gain_vs_hist",
                "calendar_ov_plus_10_all",
                "history_only_ov_plus_10_all",
                "l0_ov_plus_10_all",
                "wearable_gain_vs_cal_all",
                "wearable_gain_vs_hist_all",
                "calendar_ov_plus_3_all",
                "history_only_ov_plus_3_all",
                "l0_ov_plus_3_all",
                "wearable_gain_vs_cal_ov_plus_3_all",
                "wearable_gain_vs_hist_ov_plus_3_all",
            ],
        )

    # 2. Personalization Table
    pers_methods = ["L0 Population", "L1 Zero-shot", "L2 One-shot", "L3 Few-shot"]
    pers_df = (
        subgroup_only[subgroup_only["method_short"].isin(pers_methods)]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    if not pers_df.empty:
        pers_df["method_order"] = pers_df["method_short"].map(_method_order)
        pers_df = pers_df.sort_values(
            ["subgroup_family", "subgroup_name", "method_order", "method_short"]
        ).reset_index(drop=True)

    # 3. Diagnostic Table
    diag_methods = ["L2 One-shot", "L2a LocalizerOnly", "L2b TempOnly"]
    diag_df = (
        subgroup_only[subgroup_only["method_short"].isin(diag_methods)]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    if not diag_df.empty:
        diag_df["method_order"] = diag_df["method_short"].map(_method_order)
        diag_df = diag_df.sort_values(
            ["subgroup_family", "subgroup_name", "method_order", "method_short"]
        ).reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Baseline and Personalization Research Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(
        "This report summarizes research-stage experiments with an explicit **detect × countdown** matrix "
        "(`personalize-matrix`), alongside legacy baseline and L1–L3 personalization exports."
    )
    lines.append("")
    lines.append("Key structure:")
    lines.append(
        "- **Matrix layout**: BaseDet vs PersDet (bounded refine, no imputation) × population vs `history_acl` "
        "countdown priors (`MATRIX_RESULTS_CSV`)"
    )
    lines.append("- **Legacy baselines**: static `Calendar` and `History-only` (fixed cycle-length priors)")
    lines.append(
        "- **Legacy diagnostics**: `L2a/L2b` ablations (Section 7) predate the matrix and are kept as appendix-style context"
    )
    lines.append("- **Subgroup axes**: `stable_length_profile` joins cycle length and variability groups")
    lines.append("")
    lines.append("Primary subgroup axes:")
    lines.append("- `cycle_length_level_group` (short, typical, long)")
    lines.append("- `cycle_variability_group` (low, medium, high)")
    lines.append("- `stable_length_profile` (shifted-but-stable analysis)")
    lines.append("")
    lines.append("Subgroup labels are retrospective user-level phenotype labels used only for post hoc analysis.")
    lines.append("Cold-start cycles remain cold-start operationally: when no prior user history exists, detector personalization falls back to `L0 Population` and countdown priors fall back to the population default.")
    lines.append("Primary metrics below use trigger-gated post-ovulation anchors (`Triggered Ov+3`, `Triggered Ov+5`, `Triggered Ov+10`) and `PostTrigger`.")
    lines.append("Fixed anchors (`AllAnchors Ov+5`, `AllAnchors Ov+10`) are retained as secondary physiology-aligned comparators.")
    lines.append("MAE metrics below are shown as `mean [95% bootstrap CI]` using 2000 resamples within each reported cell.")
    if missing_tags:
        lines.append(
            "This report was built from a partial result set. Missing experiment files: "
            + ", ".join(f"`{tag}`" for tag in missing_tags)
            + "."
        )
    lines.append("")
    lines.append("## 2. Main Findings")
    lines.append("")
    lines.append("### 2.1 Does wearable physiology provide gain beyond history?")
    lines.append("")
    lines.append("Wearable physiology (`L0 Population`) is compared against a static population-only `Calendar` and a static `History-only` personalized baseline.")
    lines.append("")
    lines.append("### 2.2 Detect × countdown matrix (headline design)")
    lines.append("")
    lines.append(
        "The `personalize-matrix` pipeline evaluates **BaseDet+PopCount** vs **BaseDet+HistACLCount** and "
        "**PersDet+PopCount** vs **PersDet+HistACLCount**, plus static reference rows aligned to Calendar / "
        "history-only countdowns. Section 3 shows the **overall** slice; per-subgroup tables are in "
        "`MATRIX_RESULTS_CSV` / `MATRIX_RESULTS_MD`."
    )
    lines.append("")
    lines.append("### 2.3 Legacy L2 diagnostic ablations (appendix context)")
    lines.append("")
    lines.append(
        "The `L2a` (localizer-only) and `L2b` (temperature-evidence-only) variants predate the matrix; "
        "they decompose one-shot personalization and are summarized in Section 7."
    )
    lines.append("")
    lines.append("## 3. Matrix: overall (`all-labeled`)")
    lines.append("")
    if matrix_overall.empty:
        lines.append(
            "No matrix rows were found. Generate them with `python experiment_runner.py --mode personalize-matrix` "
            "(writes `MATRIX_RESULTS_CSV`)."
        )
    else:
        lines.append(
            _present_table(
                matrix_overall,
                [
                    ("method_short", "Method"),
                    ("n_cycles", "Cycles"),
                    ("n_users", "Users"),
                    ("ov_plus_3_triggered_mae_95ci", "Triggered Ov+3 MAE [95% CI]"),
                    ("ov_plus_3_triggered_acc_3d", "Triggered Ov+3 ±3d"),
                    ("ov_plus_5_triggered_mae_95ci", "Triggered Ov+5 MAE [95% CI]"),
                    ("ov_plus_5_triggered_acc_3d", "Triggered Ov+5 ±3d"),
                    ("ov_plus_10_triggered_mae_95ci", "Triggered Ov+10 MAE [95% CI]"),
                    ("ov_plus_10_triggered_acc_3d", "Triggered Ov+10 ±3d"),
                    ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                    ("post_trigger_acc_3d", "PostTrigger ±3d"),
                    ("ov_plus_3_mae_95ci", "AllAnchors Ov+3 MAE [95% CI]"),
                    ("ov_plus_10_mae_95ci", "AllAnchors Ov+10 MAE [95% CI]"),
                    ("detected_cycle_rate", "Detected Rate"),
                    ("latency_days_mean", "Latency"),
                ],
            )
        )
    lines.append("")
    lines.append("## 4. Overall comparison (non-matrix experiments)")
    lines.append("")
    if overall_df.empty:
        lines.append("No non-matrix overall rows were found in the current result files.")
    else:
        lines.append(
            _present_table(
                overall_df,
                [
                    ("method_short", "Method"),
                    ("n_cycles", "Cycles"),
                    ("n_users", "Users"),
                    ("ov_plus_3_triggered_mae_95ci", "Triggered Ov+3 MAE [95% CI]"),
                    ("ov_plus_3_triggered_acc_3d", "Triggered Ov+3 ±3d"),
                    ("ov_plus_5_triggered_mae_95ci", "Triggered Ov+5 MAE [95% CI]"),
                    ("ov_plus_5_triggered_acc_3d", "Triggered Ov+5 ±3d"),
                    ("ov_plus_10_triggered_mae_95ci", "Triggered Ov+10 MAE [95% CI]"),
                    ("ov_plus_10_triggered_acc_3d", "Triggered Ov+10 ±3d"),
                    ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                    ("post_trigger_acc_3d", "PostTrigger ±3d"),
                    ("ov_plus_3_mae_95ci", "AllAnchors Ov+3 MAE [95% CI]"),
                    ("ov_plus_10_mae_95ci", "AllAnchors Ov+10 MAE [95% CI]"),
                    ("detected_cycle_rate", "Detected Rate"),
                    ("latency_days_mean", "Latency"),
                ],
            )
        )
    lines.append("")
    lines.append("## 5. Baseline Comparison: Wearable vs. History vs. Calendar")
    lines.append("")
    if gain_df.empty:
        lines.append("No gain table could be generated.")
    else:
        lines.append(
            _present_table(
                gain_df,
                [
                    ("subgroup_family", "Family"),
                    ("subgroup_name", "Subgroup"),
                    ("calendar_ov_plus_3_all", "Calendar AllAnchors Ov+3 MAE"),
                    ("calendar_ov_plus_10_all", "Calendar AllAnchors Ov+10 MAE"),
                    ("history_only_ov_plus_3_all", "History-only AllAnchors Ov+3 MAE"),
                    ("history_only_ov_plus_10_all", "History-only AllAnchors Ov+10 MAE"),
                    ("l0_ov_plus_3_all", "L0 AllAnchors Ov+3 MAE"),
                    ("l0_ov_plus_10_all", "L0 AllAnchors Ov+10 MAE"),
                    ("wearable_gain_vs_cal_ov_plus_3_all", "Gain vs Calendar Ov+3 (AllAnchors)"),
                    ("wearable_gain_vs_cal_all", "Gain vs Calendar (AllAnchors)"),
                    ("wearable_gain_vs_hist_ov_plus_3_all", "Gain vs History Ov+3 (AllAnchors)"),
                    ("wearable_gain_vs_hist_all", "Gain vs History (AllAnchors)"),
                ],
            )
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `wearable_gain_vs_hist_all > 0` indicates wearable signal adds value beyond simple history-only prediction on the late fixed anchor comparator.")
    lines.append("")
    lines.append("## 6. Personalization Comparison by Subgroup")
    lines.append("")
    for family, fam_df in pers_df.groupby("subgroup_family"):
        lines.append(f"### Subgroup: {family}")
        lines.append("")
        lines.append(
            _present_table(
                fam_df,
                [
                    ("subgroup_name", "Subgroup"),
                    ("method_short", "Method"),
                    ("ov_plus_3_triggered_mae_95ci", "Triggered Ov+3 MAE [95% CI]"),
                    ("ov_plus_3_triggered_acc_3d", "Triggered Ov+3 ±3d"),
                    ("ov_plus_5_triggered_mae_95ci", "Triggered Ov+5 MAE [95% CI]"),
                    ("ov_plus_5_triggered_acc_3d", "Triggered Ov+5 ±3d"),
                    ("ov_plus_10_triggered_mae_95ci", "Triggered Ov+10 MAE [95% CI]"),
                    ("ov_plus_10_triggered_acc_3d", "Triggered Ov+10 ±3d"),
                    ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                    ("post_trigger_acc_3d", "PostTrigger ±3d"),
                    ("ov_plus_3_mae_95ci", "AllAnchors Ov+3 MAE [95% CI]"),
                    ("ov_plus_10_mae_95ci", "AllAnchors Ov+10 MAE [95% CI]"),
                    ("detected_cycle_rate", "Detected Rate"),
                    ("latency_days_mean", "Latency"),
                ],
            )
        )
        lines.append("")
    lines.append("## 7. Legacy diagnostic ablations (L2 variants)")
    lines.append("")
    for family, fam_df in diag_df.groupby("subgroup_family"):
        lines.append(f"### Diagnostic: {family}")
        lines.append("")
        lines.append(
            _present_table(
                fam_df,
                [
                    ("subgroup_name", "Subgroup"),
                    ("method_short", "Method"),
                    ("ov_plus_3_triggered_mae_95ci", "Triggered Ov+3 MAE [95% CI]"),
                    ("ov_plus_3_triggered_acc_3d", "Triggered Ov+3 ±3d"),
                    ("ov_plus_5_triggered_mae_95ci", "Triggered Ov+5 MAE [95% CI]"),
                    ("ov_plus_5_triggered_acc_3d", "Triggered Ov+5 ±3d"),
                    ("ov_plus_10_triggered_mae_95ci", "Triggered Ov+10 MAE [95% CI]"),
                    ("ov_plus_10_triggered_acc_3d", "Triggered Ov+10 ±3d"),
                    ("post_trigger_mae_95ci", "PostTrigger MAE [95% CI]"),
                    ("post_trigger_acc_3d", "PostTrigger ±3d"),
                    ("ov_plus_3_mae_95ci", "AllAnchors Ov+3 MAE [95% CI]"),
                    ("ov_plus_10_mae_95ci", "AllAnchors Ov+10 MAE [95% CI]"),
                    ("detected_cycle_rate", "Detected Rate"),
                    ("latency_days_mean", "Latency"),
                ],
            )
        )
        lines.append("")
    lines.append("## 8. Interpretation by Research Question")
    lines.append("")
    lines.append("### Q1. Where do wearable signals help most?")
    lines.append("Check trigger-gated `Ov+3/Ov+5/Ov+10` as primary metrics in `shifted-but-stable` profiles; use all-anchor Ov+10 only as secondary comparator.")
    lines.append("")
    lines.append("### Q2. Does simple history-only personalization help?")
    lines.append("Compare `History-only (Static)` vs `Calendar (Static)` in the baseline table.")
    lines.append("")
    lines.append("### Q3. How does detector-side refinement interact with countdown customization?")
    lines.append(
        "In Section 3, compare `Matrix: Pers+Pop` vs `Matrix: Base+Pop`, and `Matrix: Pers+HistACL` vs `Matrix: Base+HistACL`, "
        "to read detect-side effects at fixed countdown settings."
    )
    lines.append(
        "For historical context on one-shot degradation, still compare `L2` vs `L2a` vs `L2b` in Section 7."
    )
    lines.append("")
    lines.append("## 9. Methodological Cautions")
    lines.append("- Some subgroup cells (e.g., `high-variability`) have very small sample sizes.")
    lines.append("- `PostOv` pooled metrics are still exported, but main interpretation should rely on fixed anchors rather than averaging all post-ovulation days together.")
    lines.append("- Detector inference and countdown evaluation remain strict-prefix; subgroup labels are retrospective analysis-only phenotypes.")
    lines.append("")

    return "\n".join(lines)



def main() -> None:
    ensure_research_dirs()
    text = build_report()
    REPORT_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

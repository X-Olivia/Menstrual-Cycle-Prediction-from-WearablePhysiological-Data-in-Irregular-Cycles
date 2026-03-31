from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import (
    BASELINE_RESULTS_CSV,
    L1_RESULTS_CSV,
    L2_RESULTS_CSV,
    L3_RESULTS_CSV,
    REPORTS_DIR,
    ensure_research_dirs,
)


REPORT_PATH = REPORTS_DIR / "BASELINE_AND_PERSONALIZATION_REPORT_v1.md"


def _load_results(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["experiment_tag"] = tag
    return df


def _method_short_name(name: str) -> str:
    mapping = {
        "Calendar": "Calendar",
        "PhaseCls-Temp+HR[RF-baseline]": "L0 Population",
        "PhaseCls-Temp+HR[L1-zero-shot]": "L1 Zero-shot",
        "PhaseCls-Temp+HR[L2-one-shot]": "L2 One-shot",
        "PhaseCls-Temp+HR[L3-few-shot]": "L3 Few-shot",
    }
    return mapping.get(name, name)


def _round_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: round(float(x), 3) if pd.notna(x) else x)
    return out


def build_report() -> str:
    baseline = _load_results(BASELINE_RESULTS_CSV, "baseline")
    l1 = _load_results(L1_RESULTS_CSV, "l1")
    l2 = _load_results(L2_RESULTS_CSV, "l2")
    l3 = _load_results(L3_RESULTS_CSV, "l3")
    df = pd.concat([baseline, l1, l2, l3], ignore_index=True)
    df["method_short"] = df["method"].map(_method_short_name)

    baseline_only = (
        df[df["method_short"].isin(["Calendar", "L0 Population"])]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    gain_rows = []
    for (family, subgroup_name), grp in baseline_only.groupby(["subgroup_family", "subgroup_name"]):
        cal = grp[grp["method_short"] == "Calendar"]
        l0 = grp[grp["method_short"] == "L0 Population"]
        if cal.empty or l0.empty:
            continue
        cal_row = cal.iloc[0]
        l0_row = l0.iloc[0]
        gain_rows.append(
            {
                "subgroup_family": family,
                "subgroup_name": subgroup_name,
                "calendar_post_ov": cal_row["post_ov_mae"],
                "l0_post_ov": l0_row["post_ov_mae"],
                "wearable_gain_post_ov": cal_row["post_ov_mae"] - l0_row["post_ov_mae"],
                "l0_post_trigger": l0_row["post_trigger_mae"],
                "l0_ov_first": l0_row["ov_first_mae"],
                "l0_detect_rate": l0_row["detected_cycle_rate"],
            }
        )
    gain_df = pd.DataFrame(gain_rows)
    if not gain_df.empty:
        gain_df = _round_cols(
            gain_df,
            [
                "calendar_post_ov",
                "l0_post_ov",
                "wearable_gain_post_ov",
                "l0_post_trigger",
                "l0_ov_first",
                "l0_detect_rate",
            ],
        )

    pers_df = (
        df[df["method_short"].isin(["L0 Population", "L1 Zero-shot", "L2 One-shot", "L3 Few-shot"])]
        .drop_duplicates(["subgroup_family", "subgroup_name", "method_short"])
        .copy()
    )
    pers_df = _round_cols(
        pers_df,
        [
            "post_ov_mae",
            "post_trigger_mae",
            "ov_first_mae",
            "ov_final_mae",
            "detected_cycle_rate",
            "latency_days_mean",
        ],
    )

    lines: list[str] = []
    lines.append("# Baseline and Personalization Research Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("This report summarizes the first completed research-stage experiments under the heterogeneity-sensitive methodology.")
    lines.append("")
    lines.append("Completed stages:")
    lines.append("")
    lines.append("- history-only subgroup construction")
    lines.append("- subgroup baseline analysis")
    lines.append("- `L1` zero-shot detector calibration")
    lines.append("- `L2` one-shot detector calibration")
    lines.append("- `L3` few-shot detector calibration")
    lines.append("")
    lines.append("Primary subgroup axes used here:")
    lines.append("")
    lines.append("- `cycle_length_level_group`")
    lines.append("- `cycle_variability_group`")
    lines.append("")
    lines.append("## 2. Main Findings")
    lines.append("")
    lines.append("### 2.1 Wearable physiology is not uniformly useful")
    lines.append("")
    lines.append("The current strict-prefix wearable baseline (`L0 Population`) helps some subgroups substantially more than others.")
    lines.append("")
    lines.append("Most visible gains appear in:")
    lines.append("")
    lines.append("- shifted cycle-length groups (`short`, `long`)")
    lines.append("- `medium-variability` users")
    lines.append("- the small `high-variability` subgroup")
    lines.append("")
    lines.append("The weakest gain appears in:")
    lines.append("")
    lines.append("- `low-variability` users, where wearable performance is nearly identical to Calendar on `PostOvDays`")
    lines.append("")
    lines.append("### 2.2 Current detector personalization does not improve results")
    lines.append("")
    lines.append("At this stage:")
    lines.append("")
    lines.append("- `L1` makes no effective change")
    lines.append("- `L2` generally worsens performance")
    lines.append("- `L3` also generally worsens performance")
    lines.append("")
    lines.append("This means detector personalization should not be assumed to be beneficial. In the current implementation, personalization is either neutral (`L1`) or harmful (`L2/L3`).")
    lines.append("")
    lines.append("### 2.3 Current evidence supports the wearable subgroup claim, but not a positive personalization claim")
    lines.append("")
    lines.append("The current results support the paper's primary wearable-benefit claim much more clearly than any positive personalization claim.")
    lines.append("")
    lines.append("- primary claim supported: wearable physiology helps selectively across irregularity profiles")
    lines.append("- secondary personalization claim not supported: current personalization does not improve harder subgroups and is not yet a positive finding")
    lines.append("")
    lines.append("## 3. Wearable Gain over Calendar by Subgroup")
    lines.append("")
    if gain_df.empty:
        lines.append("No gain table could be generated.")
    else:
        lines.append(gain_df.to_markdown(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `wearable_gain_post_ov > 0` means the wearable baseline reduces `PostOvDays MAE` relative to Calendar")
    lines.append("- the largest gains are currently observed in `short`, `long`, and `medium-variability` groups")
    lines.append("- `low-variability` shows almost no gain, suggesting Calendar remains competitive there")
    lines.append("")
    lines.append("## 4. Personalization Comparison by Subgroup")
    lines.append("")
    for family, fam_df in pers_df.groupby("subgroup_family"):
        lines.append(f"### {family}")
        lines.append("")
        show = fam_df[
            [
                "subgroup_name",
                "method_short",
                "n_cycles",
                "n_users",
                "post_ov_mae",
                "post_trigger_mae",
                "ov_first_mae",
                "ov_final_mae",
                "detected_cycle_rate",
                "latency_days_mean",
            ]
        ].copy()
        lines.append(show.to_markdown(index=False))
        lines.append("")
    lines.append("## 5. Interpretation by Research Question")
    lines.append("")
    lines.append("### Q1. Which irregularity profiles are hardest?")
    lines.append("")
    lines.append("Under Calendar, the hardest currently observed groups are:")
    lines.append("")
    lines.append("- `short` cycle-length level")
    lines.append("- `long` cycle-length level")
    lines.append("- `medium-variability` and `high-variability`")
    lines.append("")
    lines.append("### Q2. Where do wearable signals help?")
    lines.append("")
    lines.append("They help most where Calendar assumptions are structurally weakest:")
    lines.append("")
    lines.append("- users with shifted but stable cycle length")
    lines.append("- users with non-trivial between-cycle variability")
    lines.append("")
    lines.append("### Q3. Where does personalization help?")
    lines.append("")
    lines.append("At the current implementation stage, personalization does not help. This is a meaningful negative result rather than a missing result.")
    lines.append("")
    lines.append("- `L1` is effectively neutral")
    lines.append("- `L2` and `L3` degrade the current baseline")
    lines.append("")
    lines.append("### Q4. What does this imply for the paper?")
    lines.append("")
    lines.append("The paper can already support a strong claim that wearable physiology provides selective benefit across irregularity profiles.")
    lines.append("")
    lines.append("However, the detector-personalization claim must remain secondary and provisional. The current code does not yet support a positive personalization result.")
    lines.append("")
    lines.append("## 6. Methodological Cautions")
    lines.append("")
    lines.append("- subgroup sizes are still small in several cells, especially `high-variability`")
    lines.append("- current subgroup tables are useful for directional evidence, not strong final subgroup claims")
    lines.append("- `ovulatory-status` has not yet been incorporated into the main report and should remain secondary")
    lines.append("")
    lines.append("## 7. Immediate Next Step")
    lines.append("")
    lines.append("The next research step should not be to claim personalization works.")
    lines.append("")
    lines.append("Instead, it should be to:")
    lines.append("")
    lines.append("1. freeze subgroup reporting thresholds (`U_min`, `C_min`)")
    lines.append("2. keep the current wearable subgroup-baseline result as the main positive finding")
    lines.append("3. redesign detector personalization before making any stronger personalization claims")
    lines.append("")
    lines.append("In practical terms, the current evidence says:")
    lines.append("")
    lines.append("**wearable physiology already shows selective value; current detector personalization is currently a negative or null finding rather than a positive contribution.**")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_research_dirs()
    text = build_report()
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()

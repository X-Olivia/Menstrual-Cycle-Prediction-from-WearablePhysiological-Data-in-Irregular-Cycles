"""
Ovulation labeling: daily ovulation probabilities and LH gold-standard labels.

Pipeline: run data_clean.py first → cycle_cleaned.csv; then run this script → cycle_cleaned_ov.csv.

1. add_ovulation_probabilities(df): For a DataFrame with small_group_key, lh, phase, day_in_study,
   adds ovulation_day_method1, ovulation_day_method2, ovulation_prob_fused (surge detection + τ marginalization).
2. get_lh_ovulation_labels(cycle_csv): From a cycle CSV that already has ovulation_prob_fused,
   returns one integer ovulation day per cycle (ov_prob > 0.5 + luteal [8,20]), as a DataFrame with ov_dic for evaluation/Oracle.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import truncnorm

# Pipeline: input = output of data_clean.py; output = same dir, basename_ov.csv
PROCESSED_DIR = "processed_dataset"
CLEANED_CYCLE_FILENAME = "cycle_cleaned.csv"
OV_LABELED_FILENAME = "cycle_cleaned_ov.csv"

# ---------------------------------------------------------------------------
# Ovulation probability model constants (aligned with cycle_clean)
# τ~U(7,24); Onset->ovulation TruncNorm(μ=36h,σ=6h,[22,47]h); Peak->ovulation TruncNorm(μ=12h,σ=4h,[8,20]h)
# ---------------------------------------------------------------------------

_TAU_LO, _TAU_HI = 7.0, 24.0
_TAU_WEIGHT = 1.0 / (_TAU_HI - _TAU_LO)
_OVULATION_ONSET_MU, _OVULATION_ONSET_SIGMA = 36.0, 6.0
_OVULATION_ONSET_LOW, _OVULATION_ONSET_HIGH = 22.0, 47.0
_onset_a = (_OVULATION_ONSET_LOW - _OVULATION_ONSET_MU) / _OVULATION_ONSET_SIGMA
_onset_b = (_OVULATION_ONSET_HIGH - _OVULATION_ONSET_MU) / _OVULATION_ONSET_SIGMA
_dist_onset = truncnorm(_onset_a, _onset_b, loc=_OVULATION_ONSET_MU, scale=_OVULATION_ONSET_SIGMA)
_OVULATION_PEAK_MU, _OVULATION_PEAK_SIGMA = 12.0, 4.0
_OVULATION_PEAK_LOW, _OVULATION_PEAK_HIGH = 8.0, 20.0
_peak_a = (_OVULATION_PEAK_LOW - _OVULATION_PEAK_MU) / _OVULATION_PEAK_SIGMA
_peak_b = (_OVULATION_PEAK_HIGH - _OVULATION_PEAK_MU) / _OVULATION_PEAK_SIGMA
_dist_peak = truncnorm(_peak_a, _peak_b, loc=_OVULATION_PEAK_MU, scale=_OVULATION_PEAK_SIGMA)

SURGE_RATIO_THRESHOLD = 2.5  # LH/baseline >= 2.5 treated as surge


# ---------------------------------------------------------------------------
# Menstruation end day, baseline, surge segments
# ---------------------------------------------------------------------------


def find_menstruation_end_day(cycle_df: pd.DataFrame) -> float | None:
    """Last day_in_study of Menstrual phase within the cycle."""
    menstrual_days = cycle_df[cycle_df["phase"] == "Menstrual"]
    if len(menstrual_days) == 0:
        return None
    return menstrual_days["day_in_study"].max()


def calculate_baseline(cycle_df: pd.DataFrame, menstruation_end_day: float | None) -> float | None:
    """Mean of LH over days 1–4 after menstruation end, after removing min/max (if ≥3 values)."""
    if menstruation_end_day is None:
        baseline_days = cycle_df.head(4)
    else:
        start_day = menstruation_end_day + 1
        end_day = menstruation_end_day + 4
        baseline_days = cycle_df[
            (cycle_df["day_in_study"] >= start_day) & (cycle_df["day_in_study"] <= end_day)
        ]
    if baseline_days.empty:
        return None
    lh_values = baseline_days["lh"].dropna().values
    if lh_values.size == 0:
        return None
    if lh_values.size < 3:
        return float(np.mean(lh_values))
    lh_values = np.delete(lh_values, [np.argmax(lh_values), np.argmin(lh_values)])
    return float(np.mean(lh_values))


def find_surge_segments(
    cycle_df: pd.DataFrame, baseline: float | None, menstruation_end_day: float | None
) -> list[tuple[int | float, int | float]]:
    """Continuous segments (start_day, end_day) where LH/baseline >= SURGE_RATIO_THRESHOLD; gaps <=2 days are merged."""
    if baseline is None or baseline == 0 or menstruation_end_day is None:
        return []
    start_check_day = menstruation_end_day + 5
    check_days = cycle_df[cycle_df["day_in_study"] >= start_check_day].copy()
    if len(check_days) == 0:
        return []
    check_days = check_days[check_days["lh"].notna()].copy()
    check_days["ratio"] = check_days["lh"] / baseline
    check_days = check_days.sort_values("day_in_study").reset_index(drop=True)
    surge_mask = check_days["ratio"] >= SURGE_RATIO_THRESHOLD
    if surge_mask.sum() == 0:
        return []
    surge_segments = []
    in_surge, surge_start, prev_day = False, None, None
    for idx in range(len(check_days)):
        day = check_days.iloc[idx]["day_in_study"]
        is_surge = surge_mask.iloc[idx]
        if is_surge:
            if not in_surge:
                surge_start = day
                in_surge = True
            elif prev_day is not None and day != prev_day + 1:
                surge_segments.append((surge_start, prev_day))
                surge_start = day
        else:
            if in_surge:
                surge_segments.append((surge_start, prev_day))
                in_surge = False
        prev_day = day
    if in_surge:
        surge_segments.append((surge_start, prev_day))
    merged = []
    for seg in surge_segments:
        if merged and seg[0] - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return merged


# ---------------------------------------------------------------------------
# P(ov|τ) marginalization
# ---------------------------------------------------------------------------


def _p_day_given_tau(tau: float, k: int, dist, low_support: float, high_support: float) -> float:
    """P(ovulation falls on day k | τ) integrated over [low_support, high_support]."""
    start_h = max(0.0, 24.0 * k - tau)
    end_h = 24.0 * (k + 1) - tau
    if start_h >= end_h:
        return 0.0
    start_clip = max(low_support, start_h)
    end_clip = min(high_support, end_h)
    if start_clip >= end_clip:
        return 0.0
    return float(dist.cdf(end_clip) - dist.cdf(start_clip))


def prob_ovulation_given_onset(onset_day: int | float, day_in_study: int | float) -> float:
    """P(ovulation | onset_day) marginalized over τ~U(7,24)."""
    k = day_in_study - onset_day
    if k < 0:
        return 0.0
    integral, _ = quad(
        lambda tau: _p_day_given_tau(
            tau, int(k), _dist_onset, _OVULATION_ONSET_LOW, _OVULATION_ONSET_HIGH
        ),
        _TAU_LO,
        _TAU_HI,
        limit=50,
    )
    return float(integral * _TAU_WEIGHT)


def prob_ovulation_given_peak(peak_day: int | float, day_in_study: int | float) -> float:
    """P(ovulation | peak_day) marginalized over τ~U(7,24)."""
    k = day_in_study - peak_day
    if k < 0:
        return 0.0
    integral, _ = quad(
        lambda tau: _p_day_given_tau(
            tau, int(k), _dist_peak, _OVULATION_PEAK_LOW, _OVULATION_PEAK_HIGH
        ),
        _TAU_LO,
        _TAU_HI,
        limit=50,
    )
    return float(integral * _TAU_WEIGHT)


# ---------------------------------------------------------------------------
# Add ovulation probability columns to DataFrame (run ovulation labeling)
# ---------------------------------------------------------------------------


def add_ovulation_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Add ovulation_day_method1, ovulation_day_method2, ovulation_prob_fused to df; computed per small_group_key."""
    df = df.copy()
    df["ovulation_day_method1"] = 0.0
    df["ovulation_day_method2"] = 0.0
    df["ovulation_prob_fused"] = 0.0

    for sk, cycle_df in df.groupby("small_group_key", sort=False):
        cycle_df = cycle_df.sort_values("day_in_study").copy()
        menstruation_end_day = find_menstruation_end_day(cycle_df)
        baseline = calculate_baseline(cycle_df, menstruation_end_day)
        if baseline is None or baseline == 0:
            continue
        surge_segments = find_surge_segments(cycle_df, baseline, menstruation_end_day)
        if len(surge_segments) == 0:
            continue
        cycle_lh = cycle_df["lh"].dropna()
        max_lh_val = cycle_lh.max()
        max_lh_days = cycle_df.loc[cycle_df["lh"] == max_lh_val, "day_in_study"].values
        max_lh_day = max_lh_days[0] if len(max_lh_days) > 0 else None
        selected_surge = None
        for seg in surge_segments:
            if max_lh_day is not None and seg[0] <= max_lh_day <= seg[1]:
                selected_surge = seg
                break
        if selected_surge is None:
            selected_surge = surge_segments[-1]
        surge_start_day, surge_end_day = selected_surge
        surge_data = cycle_df[
            (cycle_df["day_in_study"] >= surge_start_day)
            & (cycle_df["day_in_study"] <= surge_end_day)
        ].copy()
        surge_data = surge_data[surge_data["lh"].notna()]
        if len(surge_data) == 0:
            continue
        onset_day = surge_start_day
        peak_days_global = surge_data[surge_data["lh"] == surge_data["lh"].max()]["day_in_study"].values
        peak_day = int(min(peak_days_global))

        for _, row in cycle_df.iterrows():
            d = row["day_in_study"]
            p1 = prob_ovulation_given_onset(onset_day, d)
            p2 = prob_ovulation_given_peak(peak_day, d)
            mask = (df["small_group_key"] == sk) & (df["day_in_study"] == d)
            df.loc[mask, "ovulation_day_method1"] = p1
            df.loc[mask, "ovulation_day_method2"] = p2
            df.loc[mask, "ovulation_prob_fused"] = 0.5 * p1 + 0.5 * p2

        # Normalize fused within cycle
        fused_sum = df.loc[df["small_group_key"] == sk, "ovulation_prob_fused"].sum()
        if fused_sum > 0:
            df.loc[df["small_group_key"] == sk, "ovulation_prob_fused"] /= fused_sum

    df["ovulation_day_method1"] = df["ovulation_day_method1"].fillna(0.0).astype(float)
    df["ovulation_day_method2"] = df["ovulation_day_method2"].fillna(0.0).astype(float)
    df["ovulation_prob_fused"] = df["ovulation_prob_fused"].fillna(0.0).astype(float)
    return df


# ---------------------------------------------------------------------------
# LH labels: one integer ovulation day per cycle (for evaluation / Oracle)
# ---------------------------------------------------------------------------


def get_lh_ovulation_labels(cycle_csv: str | None = None) -> pd.DataFrame:
    """Get LH-based ovulation labels with reasonable luteal lengths.

    ovulation_prob_fused > 0.5 + luteal length in [8, 20] days.

    Args:
        cycle_csv: Path to cycle CSV with columns small_group_key, day_in_study, ovulation_prob_fused.
                   If None, uses {workspace}/processed_dataset/cycle_cleaned_ov.csv.
    Returns:
        DataFrame with columns: small_group_key, id, study_interval, ov_day_in_study, cs, ce, luteal_len, ov_dic.
    """
    if cycle_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.dirname(script_dir)
        cycle_csv = os.path.join(workspace, PROCESSED_DIR, OV_LABELED_FILENAME)
    cc = pd.read_csv(cycle_csv)
    ov = cc[cc["ovulation_prob_fused"] > 0.5]
    lh_ov = (
        ov.groupby("small_group_key")
        .apply(lambda g: g.loc[g["ovulation_prob_fused"].idxmax()], include_groups=False)
        [["id", "study_interval", "day_in_study"]]
        .reset_index()
        .rename(columns={"day_in_study": "ov_day_in_study"})
    )
    cs = cc.groupby("small_group_key")["day_in_study"].min().reset_index().rename(columns={"day_in_study": "cs"})
    ce = cc.groupby("small_group_key")["day_in_study"].max().reset_index().rename(columns={"day_in_study": "ce"})
    lh_ov = lh_ov.merge(cs, on="small_group_key").merge(ce, on="small_group_key")
    lh_ov["luteal_len"] = lh_ov["ce"] - lh_ov["ov_day_in_study"]
    lh_ov = lh_ov[(lh_ov["luteal_len"] >= 8) & (lh_ov["luteal_len"] <= 20)]
    lh_ov["ov_dic"] = lh_ov["ov_day_in_study"] - lh_ov["cs"]
    return lh_ov


# ---------------------------------------------------------------------------
# run standalone for ovulation labeling
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: read cleaned cycle CSV (from data_clean) → add ovulation columns → write; optional LH labels CSV."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Add ovulation probability columns to cleaned cycle data. Run data_clean.py first."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=f"Cleaned cycle CSV (output of data_clean). Default: {PROCESSED_DIR}/{CLEANED_CYCLE_FILENAME}",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help=f"Output CSV with ovulation columns. Default: {PROCESSED_DIR}/<input_basename>_ov.csv",
    )
    parser.add_argument(
        "--labels",
        default=None,
        metavar="PATH",
        help="If set, save per-cycle LH labels (ov_dic etc.) to this CSV.",
    )
    parser.add_argument(
        "--no-labels-summary",
        action="store_true",
        help="Do not print LH label count.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(script_dir)
    out_dir = os.path.join(workspace, PROCESSED_DIR)

    input_path = args.input or os.path.join(workspace, PROCESSED_DIR, CLEANED_CYCLE_FILENAME)
    if not os.path.isfile(input_path):
        raise SystemExit(
            f"Input not found: {input_path}. Run data_clean.py first to produce {PROCESSED_DIR}/{CLEANED_CYCLE_FILENAME}"
        )
    args.input = input_path

    if args.output is None:
        os.makedirs(out_dir, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(args.input))
        args.output = os.path.join(out_dir, base + "_ov" + (ext or ".csv"))

    # add ovulation probabilities → write
    df = pd.read_csv(args.input)
    required = ["id", "study_interval", "day_in_study", "phase", "lh", "small_group_key"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV missing columns: {missing}")
    df = add_ovulation_probabilities(df)
    df.to_csv(args.output, index=False)
    print(f"Wrote cycle table with ovulation probabilities: {args.output}")

    # LH labels: from the file just written
    lh_ov = get_lh_ovulation_labels(cycle_csv=args.output)
    if not args.no_labels_summary:
        print(f"LH-labeled cycles (ov_prob>0.5 and luteal 8–20 days): {len(lh_ov)}")
    if args.labels:
        lh_ov.to_csv(args.labels, index=False)
        print(f"Wrote LH labels table: {args.labels}")


if __name__ == "__main__":
    main()

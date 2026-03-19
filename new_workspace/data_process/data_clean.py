"""
Cycle Clean — data cleaning (no ovulation labeling).

From hormones_and_selfreport.csv:
1. Column extraction (id, study_interval, day_in_study, phase, lh)
2. Big-group / cycle grouping (big_group = id+study_interval; first Menstrual starts cycle1; study start to first Menstrual = cycle0)
3. LH missing fill (within-cycle neighbor mean); drop whole cycle if consecutive missing >5 days
4. Cleaning rules (only for cycles with no LHmissing): drop cycle0, cycle length <6 days, >1 missing day in cycle
5. Boundary cycles: remove each subject’s last cycle per study_interval (may be truncated by study end; labels unreliable)

Output: id, study_interval, day_in_study, phase, lh,  big_group, small_group_key only.
Next step: run ovulation_labels.py on the output to add ovulation columns (e.g. cycle_cleaned_ov.csv).

Usage:
    cd /path/to/new_workspace
    python -m data_process.data_clean
    or
    python data_process/data_clean.py --data-dir . --output-dir subdataset
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants 
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0"
DEFAULT_OUTPUT_DIR = "processed_dataset"
INPUT_FILENAME = "hormones_and_selfreport.csv"
OUTPUT_FILENAME = "cycle_cleaned.csv"

COLUMNS_TO_EXTRACT = ["id", "study_interval", "day_in_study", "phase", "lh"]
MAX_CONSECUTIVE_MISSING_DAYS = 5  # Drop whole cycle if consecutive missing >5 days


# ---------------------------------------------------------------------------
# Grouping: big group and cycle
# ---------------------------------------------------------------------------


def assign_cycle_ids(group_df: pd.DataFrame) -> list[int]:
    """Assign cycle id per big group from phase sequence: first Menstrual starts cycle1; before that is cycle0."""
    phases = group_df["phase"].values
    n = len(phases)
    cycle_ids = []
    current_cycle = 0
    left_menstrual = False
    for i in range(n):
        phase = phases[i]
        if i == 0:
            if phase == "Menstrual":
                current_cycle = 1
                left_menstrual = False
            else:
                current_cycle = 0
                left_menstrual = True
        else:
            if phase != "Menstrual":
                left_menstrual = True
            elif phase == "Menstrual" and left_menstrual:
                current_cycle += 1
                left_menstrual = False
        cycle_ids.append(current_cycle)
    return cycle_ids


def add_cycle_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add big_group, small_group, small_group_key."""
    df = df.copy()
    df["big_group"] = df["id"].astype(str) + "_" + df["study_interval"].astype(str)
    df["small_group"] = -1
    for big_group_name, big_group_df in df.groupby("big_group", sort=False):
        cycle_ids = assign_cycle_ids(big_group_df)
        df.loc[big_group_df.index, "small_group"] = cycle_ids
    df["small_group_key"] = df["big_group"] + "_cycle" + df["small_group"].astype(str)
    return df


# ---------------------------------------------------------------------------
# LH missing fill and whole-cycle drop
# ---------------------------------------------------------------------------


def fill_lh_and_drop_bad_cycles(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Fill lh and within cycle using neighbor mean; drop whole cycle if consecutive missing > MAX_CONSECUTIVE_MISSING_DAYS.
    Returns (df after fill and bad-cycle removal, list of dropped small_group_key).
    """
    cycles_skip_fill: list[str] = []
    df_filled = df.copy()

    for sk, g in df.groupby("small_group_key", sort=False):
        g = g.sort_values("day_in_study")
        missing = g["lh"].isna()
        if not missing.any():
            continue
        run, max_run = 0, 0
        for m in missing:
            if m:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run > MAX_CONSECUTIVE_MISSING_DAYS:
            cycles_skip_fill.append(sk)
            continue
        idx = g.index.tolist()
        n = len(idx)
        for i in range(n):
            if pd.isna(g["lh"].iloc[i]):
                prev_lh = df_filled.loc[idx[i - 1], "lh"] if i > 0 else np.nan
                next_lh = df_filled.loc[idx[i + 1], "lh"] if i < n - 1 else np.nan
                if not pd.isna(prev_lh) and not pd.isna(next_lh):
                    df_filled.loc[idx[i], "lh"] = (float(prev_lh) + float(next_lh)) / 2
                elif not pd.isna(prev_lh):
                    df_filled.loc[idx[i], "lh"] = float(prev_lh)
                elif not pd.isna(next_lh):
                    df_filled.loc[idx[i], "lh"] = float(next_lh)
                df_out = df_filled[~df_filled["small_group_key"].isin(cycles_skip_fill)].copy().reset_index(drop=True)
    return df_out, cycles_skip_fill


# ---------------------------------------------------------------------------
# Cleaning rules: for cycles with no LH missing, drop cycle0, too short, >1 missing day
# ---------------------------------------------------------------------------


def build_cycle_info(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cycle: small_group_key, big_group, cycle_num, n_rows, has_lh_missing, n_missing_days."""
    cycle_info = []
    for sk, g in df.groupby("small_group_key", sort=False):
        parts = sk.rsplit("_cycle", 1)
        big_group, cycle_num = parts[0], int(parts[1])
        n_rows = len(g)
        has_missing = g["lh"].isna().any()
        days = sorted(g["day_in_study"].dropna().astype(int).tolist())
        if len(days) >= 2:
            expected = set(range(days[0], days[-1] + 1))
            actual = set(days)
            n_missing_days = len(expected - actual)
        else:
            n_missing_days = 0
        cycle_info.append({
            "small_group_key": sk,
            "big_group": big_group,
            "cycle_num": cycle_num,
            "n_rows": n_rows,
            "has_lh_missing": has_missing,
            "n_missing_days": n_missing_days,
        })
    return pd.DataFrame(cycle_info)


def should_keep_cycle(row: pd.Series) -> bool:
    """Cycles with LH missing are kept without filtering; for others: drop cycle0, n_rows<6, n_missing_days>1."""
    if row["has_lh_missing"]:
        return True
    if row["cycle_num"] == 0:
        return False
    if row["n_rows"] < 6:
        return False
    if row["n_missing_days"] > 1:
        return False
    return True


def apply_cleaning(df: pd.DataFrame, cycle_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows belonging to cycles with keep=True in cycle_df."""
    cycle_df = cycle_df.copy()
    cycle_df["keep"] = cycle_df.apply(should_keep_cycle, axis=1)
    kept_keys = set(cycle_df[cycle_df["keep"]]["small_group_key"])
    df_clean = df[df["small_group_key"].isin(kept_keys)].copy()
    return df_clean.sort_values(["id", "study_interval", "day_in_study"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Boundary cycles: remove each subject’s last cycle
# ---------------------------------------------------------------------------


def remove_boundary_cycles(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Remove the last cycle per (id, study_interval) (boundary cycle may be truncated by study end; labels like days_until_next_menses unreliable).
    """
    cycle_num = df["small_group_key"].str.extract(r"_cycle(\d+)$", expand=False).astype(int)
    df = df.copy()
    df["_cycle_num"] = cycle_num
    last_per_group = (
        df.groupby(["id", "study_interval"])["_cycle_num"]
        .max()
        .reset_index()
        .rename(columns={"_cycle_num": "_max_cycle"})
    )
    # Last cycle per (id, study_interval) is where cycle_num == max
    df = df.merge(last_per_group, on=["id", "study_interval"], how="left")
    boundary_mask = df["_cycle_num"] == df["_max_cycle"]
    n_cycles_before = df["small_group_key"].nunique()
    n_rows_removed = boundary_mask.sum()
    df = df[~boundary_mask].drop(columns=["_cycle_num", "_max_cycle"], errors="ignore").reset_index(drop=True)
    n_cycles_after = df["small_group_key"].nunique()
    if verbose:
        print(
            f"[boundary] Removed {n_rows_removed} rows from {n_cycles_before - n_cycles_after} last cycles "
            f"→ {len(df)} rows, {n_cycles_after} cycles remain"
        )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_path: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run full cleaning pipeline and write to output_path.
    Returns the final DataFrame (same as the output CSV).
    """
    if verbose:
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")

    df_original = pd.read_csv(input_path)
    df = df_original[COLUMNS_TO_EXTRACT].copy()
    if verbose:
        print(f"Original shape: {df_original.shape} -> extracted: {df.shape}")

    df = add_cycle_groups(df)
    if verbose:
        print(f"Big groups: {df['big_group'].nunique()}, cycles: {df['small_group_key'].nunique()}")

    df, cycles_skip_fill = fill_lh_and_drop_bad_cycles(df)
    if verbose:
        print(f"Cycles removed (consecutive missing >{MAX_CONSECUTIVE_MISSING_DAYS} days): {len(cycles_skip_fill)}")
        if cycles_skip_fill:
            for sk in cycles_skip_fill:
                print(f"  {sk}")
        print(f"Rows after removal: {len(df)}")

    cycle_df = build_cycle_info(df)
    cycle_df["keep"] = cycle_df.apply(should_keep_cycle, axis=1)
    kept_keys = set(cycle_df[cycle_df["keep"]]["small_group_key"])
    n_dropped = (~cycle_df["keep"]).sum()
    if verbose:
        print(f"Cycles kept: {cycle_df['keep'].sum()}, cycles dropped: {n_dropped}")

    df_clean = apply_cleaning(df, cycle_df)
    if verbose:
        print(f"Rows after cleaning: {len(df_clean)}, cycles: {df_clean['small_group_key'].nunique()}")

    df_clean = remove_boundary_cycles(df_clean, verbose=verbose)

    out_cols = [
        "id", "study_interval", "day_in_study", "phase", "lh",
        "big_group", "small_group_key",
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_clean[out_cols].to_csv(output_path, index=False)
    if verbose:
        print(f"Saved: {output_path} (rows={len(df_clean)}, cycles={df_clean['small_group_key'].nunique()})")

    return df_clean[out_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle clean: data cleaning (no ovulation labeling) -> cycle_cleaned.csv")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="mcPHASES data directory (contains hormones_and_selfreport.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--output-name",
        default=OUTPUT_FILENAME,
        help="Output filename",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    # Use parent of script dir as workspace so it runs from new_workspace or main_workspace
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(script_dir)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(workspace, args.data_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(workspace, args.output_dir)
    input_path = os.path.join(data_dir, INPUT_FILENAME)
    output_path = os.path.join(output_dir, args.output_name)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    run_pipeline(input_path, output_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()

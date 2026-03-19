"""
Wearable signals: filter to cycle-anchor days only.

Run after data_clean.py and ovulation_labels.py. Uses the cleaned cycle table
(processed_dataset/cycle_cleaned.csv) as anchor; keeps only rows whose
(id, study_interval, day_in_study) appear in that table. Writes to processed_dataset/signals/.

- resting_heart_rate, heart_rate_variability_details, wrist_temperature, heart_rate:
  filter by (id, study_interval, day_in_study).
- computed_temperature: filter by (id, study_interval, sleep_end_day_in_study), then add
  day_in_study = sleep_end_day_in_study for alignment with daily tables.

Pipeline: data_clean → ovulation_labels → wearable_signals.

Usage:
    python -m data_process.wearable_signals
    python data_process/wearable_signals.py --cycle-csv path/to/cycle_cleaned.csv --data-dir path/to/mcphases
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Align with data_clean: same default data dir and processed output
DEFAULT_DATA_DIR = "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0"
PROCESSED_DIR = "processed_dataset"
CLEANED_CYCLE_FILENAME = "cycle_cleaned.csv"
SIGNALS_SUBDIR = "signals"

CHUNK_SIZE_WRIST = 500_000
CHUNK_SIZE_HEART_RATE = 2_000_000


def _paths(workspace: Path, data_dir: str, cycle_csv: str | None, out_dir: str | None) -> tuple[Path, Path, Path]:
    """Resolve mcPHASES dir, cycle anchor path, output dir."""
    mcphases = workspace / data_dir if not os.path.isabs(data_dir) else Path(data_dir)
    cycle_path = Path(cycle_csv) if cycle_csv else workspace / PROCESSED_DIR / CLEANED_CYCLE_FILENAME
    out = workspace / out_dir if out_dir else workspace / PROCESSED_DIR / SIGNALS_SUBDIR
    return mcphases, cycle_path, out


def run_pipeline(
    cycle_csv: str | Path,
    mcphases_dir: str | Path,
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """Filter wearable tables to (id, study_interval, day_in_study) in cycle anchor; write to out_dir."""
    cycle = pd.read_csv(cycle_csv)
    cycle_days = cycle[["id", "study_interval", "day_in_study"]].drop_duplicates()
    n_anchor = len(cycle_days)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Cycle anchor: {len(cycle)} rows, {n_anchor} unique (id, study_interval, day_in_study)")

    # 1. resting_heart_rate: daily; inner join, aggregate by key (mean if multiple per day)
    rhr = pd.read_csv(Path(mcphases_dir) / "resting_heart_rate.csv")
    rhr_cycle = rhr.merge(cycle_days, on=["id", "study_interval", "day_in_study"], how="inner")
    rhr_cycle = rhr_cycle.groupby(["id", "study_interval", "day_in_study"], as_index=False)["value"].mean()
    rhr_cycle.to_csv(Path(out_dir) / "resting_heart_rate_cycle.csv", index=False)
    if verbose:
        print(f"resting_heart_rate: raw {len(rhr)} -> in-cycle (daily agg) {len(rhr_cycle)}")

    # 2. heart_rate_variability_details: inner join
    hrv = pd.read_csv(Path(mcphases_dir) / "heart_rate_variability_details.csv")
    hrv_cycle = hrv.merge(cycle_days, on=["id", "study_interval", "day_in_study"], how="inner")
    hrv_cycle.to_csv(Path(out_dir) / "heart_rate_variability_details_cycle.csv", index=False)
    if verbose:
        print(f"heart_rate_variability_details: raw {len(hrv)} -> in-cycle {len(hrv_cycle)}")

    # 3. computed_temperature: align by sleep_end_day_in_study; add day_in_study
    ct = pd.read_csv(Path(mcphases_dir) / "computed_temperature.csv")
    ct_cycle = ct.merge(
        cycle_days.rename(columns={"day_in_study": "sleep_end_day_in_study"}),
        on=["id", "study_interval", "sleep_end_day_in_study"],
        how="inner",
    )
    ct_cycle["day_in_study"] = ct_cycle["sleep_end_day_in_study"]
    ct_cycle.to_csv(Path(out_dir) / "computed_temperature_cycle.csv", index=False)
    if verbose:
        print(f"computed_temperature: raw {len(ct)} -> in-cycle {len(ct_cycle)}")

    # 4. wrist_temperature: chunked read and merge
    wt_path = Path(mcphases_dir) / "wrist_temperature.csv"
    out_wt = Path(out_dir) / "wrist_temperature_cycle.csv"
    first = True
    total_in, total_out = 0, 0
    for chunk in pd.read_csv(wt_path, chunksize=CHUNK_SIZE_WRIST):
        total_in += len(chunk)
        c = chunk.merge(cycle_days, on=["id", "study_interval", "day_in_study"], how="inner")
        total_out += len(c)
        c.to_csv(out_wt, mode="w" if first else "a", header=first, index=False)
        first = False
    if verbose:
        print(f"wrist_temperature: raw {total_in} -> in-cycle {total_out} -> {out_wt.name}")

    # 5. heart_rate: chunked read and merge
    hr_path = Path(mcphases_dir) / "heart_rate.csv"
    out_hr = Path(out_dir) / "heart_rate_cycle.csv"
    first = True
    total_in, total_out = 0, 0
    for chunk in pd.read_csv(hr_path, chunksize=CHUNK_SIZE_HEART_RATE):
        total_in += len(chunk)
        c = chunk.merge(cycle_days, on=["id", "study_interval", "day_in_study"], how="inner")
        total_out += len(c)
        c.to_csv(out_hr, mode="w" if first else "a", header=first, index=False)
        first = False
    if verbose:
        print(f"heart_rate: raw {total_in} -> in-cycle {total_out} -> {out_hr.name}")

    if verbose:
        print("\nOutput files:")
        for name in [
            "resting_heart_rate_cycle.csv",
            "heart_rate_variability_details_cycle.csv",
            "computed_temperature_cycle.csv",
            "wrist_temperature_cycle.csv",
            "heart_rate_cycle.csv",
        ]:
            p = Path(out_dir) / name
            if p.exists():
                print(f"  {name} -> {p.stat().st_size / 1024 / 1024:.2f} MB")
            else:
                print(f"  {name} -> not generated")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter wearable tables to cycle-anchor days. Run data_clean.py first."
    )
    parser.add_argument(
        "--cycle-csv",
        default=None,
        help=f"Cycle anchor CSV (id, study_interval, day_in_study). Default: {PROCESSED_DIR}/{CLEANED_CYCLE_FILENAME}",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="mcPHASES data directory (contains resting_heart_rate.csv, heart_rate.csv, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory. Default: {PROCESSED_DIR}/{SIGNALS_SUBDIR}",
    )
    parser.add_argument("--quiet", action="store_true", help="Less verbose output.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    mcphases_dir, cycle_path, out_dir = _paths(
        workspace, args.data_dir, args.cycle_csv, args.output_dir
    )

    if not cycle_path.exists():
        raise SystemExit(
            f"Cycle anchor not found: {cycle_path}. Run data_clean.py first."
        )
    if not mcphases_dir.is_dir():
        raise SystemExit(f"mcPHASES directory not found: {mcphases_dir}")

    run_pipeline(cycle_path, mcphases_dir, out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()

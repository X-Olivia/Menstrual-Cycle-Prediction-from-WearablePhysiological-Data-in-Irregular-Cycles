from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

# File location: new_workspace/record/experiment/calendar_baseline_error_report.py
# parents[2] -> new_workspace
NEW_WS = Path(__file__).resolve().parents[2]
DEFAULT_CYCLE_CSV = NEW_WS / "processed_dataset" / "cycle_cleaned_ov.csv"


def load_cycle_structure(cycle_csv: Path):
    """
    Load only cycle-level structure for calendar baseline:
      - user id
      - cycle key (small_group_key)
      - cycle length
      - per-user chronological order
    """
    usecols = ["id", "small_group_key", "day_in_study"]
    df = pd.read_csv(cycle_csv, usecols=usecols)

    # One row per cycle with start day and cycle length.
    cyc = (
        df.groupby("small_group_key", as_index=False)
        .agg(
            id=("id", "first"),
            start_day=("day_in_study", "min"),
            cycle_len=("day_in_study", "nunique"),
        )
        .sort_values(["id", "start_day"])
    )

    cs = {}
    subj_order = defaultdict(list)
    for row in cyc.itertuples(index=False):
        sgk = row.small_group_key
        uid = row.id
        clen = int(row.cycle_len)
        cs[sgk] = {"id": uid, "cycle_len": clen}
        subj_order[uid].append(sgk)
    return cs, dict(subj_order)


def build_calendar_predictions(cs, subj_order):
    """
    Reproduce the Calendar baseline logic:
      - prediction = subject historical cycle-length weighted average
      - fallback = 28.0 when no history
    """
    rows = []
    by_user = defaultdict(list)

    for uid, sgks in subj_order.items():
        past_cycle_lens = []
        for sgk in sgks:
            if sgk not in cs:
                continue

            actual = float(cs[sgk]["cycle_len"])
            if past_cycle_lens:
                # Same weighting used in multisignal_menses.py
                weights = np.exp(np.linspace(-1, 0, len(past_cycle_lens)))
                pred = float(np.average(past_cycle_lens, weights=weights))
            else:
                pred = 28.0

            record = {
                "uid": uid,
                "sgk": sgk,
                "actual": actual,
                "pred": pred,
                "err": pred - actual,
                "abs_err": abs(pred - actual),
                "history_before": [float(x) for x in past_cycle_lens],
            }

            rows.append(record)
            by_user[uid].append(record)

            # No leakage: update after scoring this cycle
            past_cycle_lens.append(actual)

    return rows, by_user


def print_report(rows, by_user, cs, subj_order, top_n=10, min_user_mae=0.0):
    if not rows:
        print("没有可评估样本。")
        return

    user_stats = []
    for uid, rs in by_user.items():
        if not rs:
            continue
        mae = mean(r["abs_err"] for r in rs)
        user_stats.append((uid, mae, len(rs), rs))
    user_stats.sort(key=lambda x: x[1], reverse=True)

    if min_user_mae > 0:
        user_stats = [x for x in user_stats if x[1] >= min_user_mae]
    if top_n > 0:
        user_stats = user_stats[:top_n]

    overall_mae = mean(r["abs_err"] for r in rows)
    print("=" * 88)
    print("Calendar Baseline User Error Report")
    print("=" * 88)
    print(f"Evaluated cycles: {len(rows)} | Users: {len(by_user)} | Overall MAE: {overall_mae:.2f}")
    print(f"Showing users: top_n={top_n}, min_user_mae={min_user_mae:.2f}")
    print("-" * 88)

    if not user_stats:
        print("没有满足筛选条件的高误差用户。")
        return

    for idx, (uid, mae, n_cycles, rs) in enumerate(user_stats, start=1):
        all_lens = [int(cs[sgk]["cycle_len"]) for sgk in subj_order.get(uid, []) if sgk in cs]
        print(f"[{idx}] user={uid} | user_MAE={mae:.2f} | evaluated_cycles={n_cycles}")
        print(f"    all_cycle_lengths={all_lens}")
        print("    cycle_details (sgk, pred, actual, abs_err, history_before):")
        for r in rs:
            h = [int(x) for x in r["history_before"]]
            print(
                f"      - {r['sgk']} | pred={r['pred']:.2f} | actual={r['actual']:.0f} "
                f"| abs_err={r['abs_err']:.2f} | history_before={h}"
            )
        print("-" * 88)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run calendar baseline and print high-error users with full cycle-length history."
    )
    p.add_argument("--cycle-csv", type=Path, default=DEFAULT_CYCLE_CSV, help="Path to cycle CSV.")
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of high-error users to print (default: 10).",
    )
    p.add_argument(
        "--min-user-mae",
        type=float,
        default=0.0,
        help="Only print users with MAE >= this threshold.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.cycle_csv.exists():
        raise SystemExit(f"Cycle CSV not found: {args.cycle_csv}")
    print(f"Loading cycle structure from: {args.cycle_csv}")
    cs, subj_order = load_cycle_structure(args.cycle_csv)
    rows, by_user = build_calendar_predictions(cs, subj_order)
    print_report(
        rows=rows,
        by_user=by_user,
        cs=cs,
        subj_order=subj_order,
        top_n=args.top_n,
        min_user_mae=args.min_user_mae,
    )


if __name__ == "__main__":
    main()

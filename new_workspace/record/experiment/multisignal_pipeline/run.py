"""Thin CLI entry for benchmark execution only."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmark_main import run_prefix_benchmark
from candidate_registry import (
    candidate_defs_for_pool,
    slow_candidate_families,
)
from data import load_all_signals
from protocol import FAST_PREFIX_BENCHMARK
from report_utils import (
    SEP,
    _print_auxiliary_bests,
    _print_detected_cycle_tables,
    _print_header,
    _print_operational_tables,
    _print_ranking_rule,
    _print_summary_table,
    _rank_candidates,
    _report_rows,
)


class _Tee:
    """Duplicate writes to multiple text streams (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _pool_summary(mode, include_slow=True, family=None):
    full_total = len(candidate_defs_for_pool("full", include_slow=True))
    slow_total = len(candidate_defs_for_pool("slow"))
    selected = len(candidate_defs_for_pool(mode, include_slow=include_slow, family=family))

    if mode == "fast":
        return f"fast subset of full total pool ({selected}/{full_total} candidates)"
    if mode == "slow":
        if family is not None:
            return (
                f"slow legacy subset filtered to {family} "
                f"({selected}/{slow_total} slow candidates; part of full total pool of {full_total})"
            )
        return f"slow legacy subset of full total pool ({selected}/{full_total} candidates)"
    if include_slow:
        return f"full total pool ({selected} candidates; includes {slow_total} slow legacy ML candidates)"
    return f"full total pool excluding slow legacy ML ({selected}/{full_total} candidates)"


def main(mode=None, include_slow=True, family=None, return_results=False):
    mode = mode or ("fast" if FAST_PREFIX_BENCHMARK else "full")
    _print_header(mode, pool_summary=_pool_summary(mode, include_slow=include_slow, family=family))
    t0 = time.time()

    lh, cs, quality, subj_order, _signal_cols = load_all_signals()
    labeled = set(s for s in cs if s in lh)
    quality_subset = set(quality) & labeled

    candidate_rows = []
    ranked_rows = []
    report_rows = []
    oracle_baseline = {}
    calendar_baseline = {}

    print(f"\n{SEP}\n  A. PREFIX CANDIDATES\n{SEP}")
    bench = run_prefix_benchmark(
        cs,
        lh,
        subj_order,
        labeled,
        quality_subset,
        mode=mode,
        include_slow=include_slow,
        family=family,
    )
    candidate_rows = bench["candidate_rows"]
    oracle_baseline = bench["oracle_baseline"]
    calendar_baseline = bench["calendar_baseline"]
    oracle_summary = oracle_baseline["summary"]
    calendar_summary = calendar_baseline["summary"]

    _print_ranking_rule()
    ranked_rows = _rank_candidates(candidate_rows, calendar_summary)
    _print_summary_table(ranked_rows)
    _print_auxiliary_bests(ranked_rows)
    report_rows = _report_rows(oracle_baseline, calendar_baseline, ranked_rows)
    _print_operational_tables(report_rows)
    _print_detected_cycle_tables(report_rows)

    if ranked_rows:
        best = ranked_rows[0]
        winner_label = {
            "fast": "Best fast-benchmark method",
            "full": "Best full-benchmark method",
            "slow": "Best slow-family method",
        }[mode]
        print(f"\n  {winner_label}: {best['name']}")
        print(
            "  "
            f"Calendar PostOvDays MAE={calendar_summary['post_ov_days']['mae']:.2f} | "
            f"Oracle PostOvDays MAE={oracle_summary['post_ov_days']['mae']:.2f} | "
            f"Best PostOvDays MAE={best['summary']['post_ov_days']['mae']:.2f}"
        )

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  DONE ({elapsed:.0f}s) — prefix benchmark complete\n{SEP}")
    if return_results:
        return {
            "candidate_rows": candidate_rows,
            "ranked_rows": ranked_rows,
            "report_rows": report_rows,
            "oracle_baseline": oracle_baseline,
            "calendar_baseline": calendar_baseline,
            "elapsed_sec": elapsed,
            "mode": mode,
            "include_slow": include_slow,
            "family": family,
        }


def _parse_cli(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Multisignal prefix benchmark. Fast is the curated subset; "
            "full is the total pool; slow legacy ML candidates can be run separately."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast",
        action="store_const",
        const="fast",
        dest="bench_mode",
        help="Run the curated fast benchmark subset.",
    )
    mode.add_argument(
        "--full",
        action="store_const",
        const="full",
        dest="bench_mode",
        help="Run the full benchmark pool, including slow legacy ML candidates.",
    )
    mode.add_argument(
        "--slow-only",
        action="store_const",
        const="slow",
        dest="bench_mode",
        help="Run only the slow legacy ML candidates (still part of the full pool).",
    )
    p.add_argument(
        "--full-skip-slow",
        action="store_true",
        help="Run the full benchmark pool except the slow legacy ML candidates.",
    )
    p.add_argument(
        "--family",
        choices=slow_candidate_families(),
        help="With --slow-only: restrict to one slow family.",
    )
    p.add_argument(
        "--log",
        type=Path,
        metavar="FILE",
        help="Also write stdout to FILE (UTF-8). Parent dirs are created as needed.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli()
    log_fp = None
    old_stdout = sys.stdout
    try:
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_fp = args.log.open("w", encoding="utf-8")
            sys.stdout = _Tee(old_stdout, log_fp)

        if args.family is not None and args.bench_mode != "slow":
            print("error: --family can only be used with --slow-only", file=sys.stderr)
            sys.exit(2)
        if args.full_skip_slow and args.bench_mode == "slow":
            print("error: --full-skip-slow cannot be combined with --slow-only", file=sys.stderr)
            sys.exit(2)
        if args.full_skip_slow and args.bench_mode == "fast":
            print("error: --full-skip-slow cannot be combined with --fast", file=sys.stderr)
            sys.exit(2)

        bench_mode = args.bench_mode
        include_slow = True
        if args.full_skip_slow:
            bench_mode = "full"
            include_slow = False
        elif bench_mode is None:
            bench_mode = "fast" if FAST_PREFIX_BENCHMARK else "full"

        main(mode=bench_mode, include_slow=include_slow, family=args.family)
    finally:
        sys.stdout = old_stdout
        if log_fp is not None:
            log_fp.close()

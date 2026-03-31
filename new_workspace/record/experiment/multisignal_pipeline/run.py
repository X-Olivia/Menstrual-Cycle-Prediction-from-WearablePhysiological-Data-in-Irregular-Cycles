"""Thin CLI entry: loads data, optional ablations, main benchmark, reporting."""
from __future__ import annotations

import sys
import time

from benchmark_main import run_prefix_benchmark
from data import load_all_signals
from experimental.ablation_phase import (
    _run_localizer_refinement_ablation,
    _run_phase_policy_search,
    _run_stateful_localizer_ablation,
    _run_trigger_mechanism_ablation,
)
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


def main(
    mode=None,
    search_phase_policy=False,
    compare_freeze_vs_clamp=False,
    compare_trigger_families=False,
    compare_stateful_localizer=False,
    compare_localizer_refinement=False,
    return_results=False,
):
    mode = mode or ("fast" if FAST_PREFIX_BENCHMARK else "full")
    _print_header(mode)
    t0 = time.time()

    lh, cs, quality, subj_order, _signal_cols = load_all_signals()
    labeled = set(s for s in cs if s in lh)
    quality_subset = set(quality) & labeled

    phase_search_rows = []
    best_phase_by_policy = {}
    if search_phase_policy:
        phase_search_rows, best_phase_by_policy = _run_phase_policy_search(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
            compare_freeze_vs_clamp,
        )

    trigger_family_results = {}
    trigger_family_best_all = {}
    trigger_family_best_quality = {}
    if compare_trigger_families:
        trigger_family_results, trigger_family_best_all, trigger_family_best_quality = _run_trigger_mechanism_ablation(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
            model_types=("rf", "hgb"),
        )

    stateful_localizer_results = {}
    stateful_localizer_best_all = {}
    stateful_localizer_best_quality = {}
    if compare_stateful_localizer:
        stateful_localizer_results, stateful_localizer_best_all, stateful_localizer_best_quality = _run_stateful_localizer_ablation(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
            model_types=("rf", "hgb"),
        )

    localizer_refinement_rows = []
    if compare_localizer_refinement:
        localizer_refinement_rows = _run_localizer_refinement_ablation(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
        )

    print(f"\n{SEP}\n  A. PREFIX CANDIDATES\n{SEP}")
    bench = run_prefix_benchmark(cs, lh, subj_order, labeled, quality_subset, mode)
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
        print(f"\n  Best valid prefix method: {best['name']}")
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
            "phase_search_rows": phase_search_rows,
            "best_phase_by_policy": best_phase_by_policy,
            "trigger_family_results": trigger_family_results,
            "trigger_family_best_all": trigger_family_best_all,
            "trigger_family_best_quality": trigger_family_best_quality,
            "stateful_localizer_results": stateful_localizer_results,
            "stateful_localizer_best_all": stateful_localizer_best_all,
            "stateful_localizer_best_quality": stateful_localizer_best_quality,
            "localizer_refinement_rows": localizer_refinement_rows,
            "elapsed_sec": elapsed,
        }


if __name__ == "__main__":
    main(compare_localizer_refinement="--localizer-refinement" in sys.argv)

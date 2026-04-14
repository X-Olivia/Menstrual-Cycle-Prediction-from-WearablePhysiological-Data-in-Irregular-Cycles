"""CLI entry for ablation and design-validation experiments."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from data import load_all_signals
from experimental.ablation_phase import (
    _run_localizer_refinement_ablation,
    _run_phase_policy_search,
    _run_stateful_localizer_ablation,
    _run_trigger_mechanism_ablation,
)
from report_utils import SEP, _print_header


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


def main(
    search_phase_policy=False,
    compare_freeze_vs_clamp=False,
    compare_trigger_families=False,
    compare_stateful_localizer=False,
    compare_localizer_refinement=False,
    return_results=False,
):
    _print_header(
        "ablation",
        pool_summary="design-validation harness only; ablation winners do not enter fast/full benchmark ranking automatically",
    )
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

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  DONE ({elapsed:.0f}s) — ablation harness complete\n{SEP}")
    if return_results:
        return {
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


def _parse_cli(argv=None):
    p = argparse.ArgumentParser(
        description="Multisignal ablation harness. These runs are design validation, not benchmark ranking.",
    )
    p.add_argument(
        "--search-phase-policy",
        action="store_true",
        help="Run S: phase policy grid search (trigger/confirm/lookback/stabilization).",
    )
    p.add_argument(
        "--freeze-clamp",
        action="store_true",
        help="With --search-phase-policy: include freeze/clamp policies in the sweep.",
    )
    p.add_argument(
        "--trigger-families",
        action="store_true",
        help="Run T: trigger mechanism ablation (baseline vs hysteresis vs hybrid).",
    )
    p.add_argument(
        "--stateful-localizer",
        action="store_true",
        help="Run U: stateful localizer ablation.",
    )
    p.add_argument(
        "--localizer-refinement",
        action="store_true",
        help="Run V: localizer refinement sweep.",
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

        any_ablation = (
            args.search_phase_policy
            or args.trigger_families
            or args.stateful_localizer
            or args.localizer_refinement
        )
        if not any_ablation:
            print(
                "error: choose at least one of --search-phase-policy, --trigger-families, "
                "--stateful-localizer, --localizer-refinement",
                file=sys.stderr,
            )
            sys.exit(2)

        main(
            search_phase_policy=args.search_phase_policy,
            compare_freeze_vs_clamp=args.freeze_clamp,
            compare_trigger_families=args.trigger_families,
            compare_stateful_localizer=args.stateful_localizer,
            compare_localizer_refinement=args.localizer_refinement,
        )
    finally:
        sys.stdout = old_stdout
        if log_fp is not None:
            log_fp.close()

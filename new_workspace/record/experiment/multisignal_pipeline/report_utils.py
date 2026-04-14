"""Printing, formatting, and ranking for prefix benchmark reports."""
from __future__ import annotations

import math

from data import CYCLE_OV_CSV, SIGNALS_DIR
from protocol import PREFIX_BENCHMARK_ML_SIGMA, PREFIX_BENCHMARK_RULE_SIGMA

SEP = "=" * 76

def _rank_candidates(candidate_rows, calendar_summary):
    cal_post_mae = calendar_summary["post_ov_days"]["mae"]
    cal_all_mae = calendar_summary["all_days"]["mae"]

    ranked = []
    for row in candidate_rows:
        summary = row["summary"]
        ranked.append(
            {
                **row,
                "postov_delta_vs_calendar": summary["post_ov_days"]["mae"] - cal_post_mae,
                "alldays_delta_vs_calendar": summary["all_days"]["mae"] - cal_all_mae,
            }
        )

    def _sort_key(row):
        summary = row["summary"]
        return (
            row["postov_delta_vs_calendar"],
            row["alldays_delta_vs_calendar"],
            summary.get("first_detection_day_mean", math.inf),
            summary.get("first_detection_ov_mae", math.inf),
            -summary.get("availability_rate", 0.0),
        )

    ranked.sort(key=_sort_key)
    return ranked


def _print_ranking_rule():
    print("  Ranking rule:")
    print("    1. Lower PostOvDays MAE relative to Calendar")
    print("    2. Lower AllDays MAE relative to Calendar")
    print("    3. Earlier first_detection_day_mean")
    print("    4. Lower first_detection_ov_mae")
    print("    5. Higher availability_rate")


def _print_summary_table(ranked_rows):
    print(f"\n{SEP}\n  D. PREFIX BENCHMARK SUMMARY\n{SEP}")
    print(
        "  "
        f"{'Rank':<4} {'Method':<28} {'Group':<12} {'AllMAE':>7} {'PostMAE':>8}"
        f" {'PostΔCal':>9} {'AllΔCal':>8} {'FirstDet':>9} {'OvMAE':>7}"
        f" {'Avail':>7} {'TimeSec':>8}"
    )
    print(f"  {'-' * 124}")
    for idx, row in enumerate(ranked_rows, start=1):
        s = row["summary"]
        print(
            "  "
            f"{idx:<4} {row['name']:<28} {row['signal_group']:<12}"
            f" {s['all_days']['mae']:>7.2f}"
            f" {s['post_ov_days']['mae']:>8.2f}"
            f" {row['postov_delta_vs_calendar']:>9.2f}"
            f" {row['alldays_delta_vs_calendar']:>8.2f}"
            f" {s.get('first_detection_day_mean', float('nan')):>9.2f}"
            f" {s.get('first_detection_ov_mae', float('nan')):>7.2f}"
            f" {s.get('availability_rate', 0.0):>6.1%}"
            f" {row['elapsed_sec']:>8.2f}"
        )


def _print_auxiliary_bests(ranked_rows):
    valid_post_trigger = [
        row for row in ranked_rows
        if row["post_trigger_summary"].get("post_trigger_days", 0) > 0
        and "mae" in row["post_trigger_summary"]
    ]
    if valid_post_trigger:
        best_post_trigger = min(
            valid_post_trigger,
            key=lambda row: row["post_trigger_summary"]["mae"],
        )
        print(
            "\n  Auxiliary best post-trigger:"
            f" {best_post_trigger['name']} "
            f"MAE={best_post_trigger['post_trigger_summary']['mae']:.2f}"
            f" n={best_post_trigger['post_trigger_summary']['post_trigger_days']}"
        )

    valid_anchor_post = [
        row for row in ranked_rows
        if row["anchor_summary"].get("post_all", {}).get("mae") is not None
    ]
    if valid_anchor_post:
        best_anchor_post = min(
            valid_anchor_post,
            key=lambda row: row["anchor_summary"]["post_all"]["mae"],
        )
        print(
            "  Auxiliary best anchor-post aggregate:"
            f" {best_anchor_post['name']} "
            f"MAE={best_anchor_post['anchor_summary']['post_all']['mae']:.2f}"
        )


def _fmt_num(v, digits=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{digits}f}"


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.1%}"


def _report_rows(oracle_baseline, calendar_baseline, ranked_rows):
    return [oracle_baseline, calendar_baseline] + list(ranked_rows)


def _print_operational_tables(report_rows):
    print(f"\n{SEP}\n  E. OPERATIONAL REPORTING\n{SEP}")
    print("  Main focus = PostOvDays / PostTrigger / anchor-post; AllDays remains secondary.")
    for title, summary_key, post_trigger_key, anchor_key in [
        ("All labeled", "summary", "post_trigger_summary", "anchor_summary"),
        ("Quality group", "quality_summary", "quality_post_trigger_summary", "quality_anchor_summary"),
    ]:
        print(f"\n  {title}")
        print(
            "  "
            f"{'Method':<28} {'PostOv':>7} {'±2d':>6} {'±3d':>6}"
            f" {'PostTrig':>9} {'±2d':>6} {'±3d':>6}"
            f" {'AnchorPost':>10} {'Avail':>7} {'FirstDet':>9} {'Ov1st':>7} {'Time':>7}"
        )
        print(f"  {'-' * 122}")
        for row in report_rows:
            s = row[summary_key]
            pt = row[post_trigger_key]
            anchor = row[anchor_key]
            ov_first = row["ov_summary"].get("first", {})
            print(
                "  "
                f"{row['name']:<28}"
                f" {_fmt_num(s.get('post_ov_days', {}).get('mae')):>7}"
                f" {_fmt_pct(s.get('post_ov_days', {}).get('acc_2d')):>6}"
                f" {_fmt_pct(s.get('post_ov_days', {}).get('acc_3d')):>6}"
                f" {_fmt_num(pt.get('mae')):>9}"
                f" {_fmt_pct(pt.get('acc_2d')):>6}"
                f" {_fmt_pct(pt.get('acc_3d')):>6}"
                f" {_fmt_num(anchor.get('post_all', {}).get('mae')):>10}"
                f" {_fmt_pct(s.get('availability_rate')):>7}"
                f" {_fmt_num(s.get('first_detection_day_mean')):>9}"
                f" {_fmt_num(ov_first.get('mae')):>7}"
                f" {_fmt_num(row.get('elapsed_sec', 0.0)):>7}"
            )


def _print_detected_cycle_tables(report_rows):
    print(f"\n{SEP}\n  F. DETECTED-CYCLE / APPLE-ALIGNED REPORTING\n{SEP}")
    print("  Detected-cycle tables restrict evaluation to cycles that produced at least one ovulation estimate.")
    for title, detected_key in [
        ("All labeled detected cycles", "detected_cycle_summary"),
        ("Quality detected cycles", "quality_detected_cycle_summary"),
    ]:
        print(f"\n  {title}")
        print(
            "  "
            f"{'Method':<28} {'DetectRate':>10} {'n_det':>7} {'Latency':>8}"
            f" {'Ov1st':>7} {'±2d':>6} {'±3d':>6}"
            f" {'PostTrig':>9} {'±2d':>6} {'±3d':>6} {'AnchorPost':>10}"
        )
        print(f"  {'-' * 120}")
        for row in report_rows:
            det_bundle = row[detected_key]
            cycles = det_bundle["cycles"]
            ov_first = det_bundle["ov_summary"].get("first", {})
            pt = det_bundle["post_trigger_summary"]
            anchor = det_bundle["anchor_summary"]
            print(
                "  "
                f"{row['name']:<28}"
                f" {_fmt_pct(cycles.get('detected_cycle_rate')):>10}"
                f" {cycles.get('detected_cycles', 0):>7d}"
                f" {_fmt_num(cycles.get('latency_days_mean')):>8}"
                f" {_fmt_num(ov_first.get('mae')):>7}"
                f" {_fmt_pct(ov_first.get('acc_2d')):>6}"
                f" {_fmt_pct(ov_first.get('acc_3d')):>6}"
                f" {_fmt_num(pt.get('mae')):>9}"
                f" {_fmt_pct(pt.get('acc_2d')):>6}"
                f" {_fmt_pct(pt.get('acc_3d')):>6}"
                f" {_fmt_num(anchor.get('post_all', {}).get('mae')):>10}"
            )


def _print_header(mode, pool_summary=None):
    print(f"\n{SEP}")
    print("  Multi-Signal Prefix Benchmark Selector")
    print(f"{SEP}")
    print(f"  Mode: {mode}")
    print(f"  Cycle: {CYCLE_OV_CSV}")
    print(f"  Signals: {SIGNALS_DIR}")
    print(f"  Rule σ={PREFIX_BENCHMARK_RULE_SIGMA} | ML σ={PREFIX_BENCHMARK_ML_SIGMA}")
    if pool_summary:
        print(f"  Candidate pool: {pool_summary}")

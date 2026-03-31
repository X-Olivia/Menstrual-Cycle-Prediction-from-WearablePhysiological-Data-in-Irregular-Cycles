"""Search, sweeps, and ablations (research harness; not the default benchmark path)."""
from __future__ import annotations

import itertools
import math
import time

from protocol import (
    PHASECLS_CLAMP_RADIUS,
    PHASECLS_CONFIRM_DAYS,
    PHASECLS_DEFAULT_GROUPS,
    PHASECLS_LOCALIZER_OVERRIDES,
    PHASECLS_LOCALIZER_SCORE_MIN,
    PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
    PHASECLS_LOCALIZER_SHIFT_MIN,
    PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    PHASECLS_LOCALIZER_AGREEMENT_TOL,
    PHASECLS_LOOKBACK_LOCALIZE,
    PHASECLS_MONOTONE_BACK_MARGIN,
    PHASECLS_SOFT_STICKY_MARGIN,
    PHASECLS_SOFT_STICKY_RADIUS,
    PHASECLS_STICKY_IMPROVE_MARGIN,
    PHASECLS_STICKY_RADIUS,
    PHASECLS_TRIGGER_ALPHA,
    PHASECLS_TRIGGER_PROB,
    PHASE_POLICY_SWEEP_CONFIRM_DAYS,
    PHASE_POLICY_SWEEP_LOOKBACKS,
    PHASE_POLICY_SWEEP_TRIGGER_ALPHAS,
    PHASE_POLICY_SWEEP_TRIGGER_PROBS,
    PREFIX_BENCHMARK_ML_SIGMA,
)
from report_utils import SEP, _fmt_num, _fmt_pct
from detectors_ml import prefix_phase_classify_loso
import benchmark_main as bm

_phase_candidate = bm._phase_candidate
_evaluate_bundle_silent = bm._evaluate_bundle_silent
_group_lookup = bm._group_lookup
_phase_row_sort_key = bm._phase_row_sort_key

def _phase_search_specs(compare_freeze_vs_clamp):
    policies = ["none"]
    if compare_freeze_vs_clamp:
        policies.extend(["freeze", "clamp"])
    for group_name in PHASECLS_DEFAULT_GROUPS:
        for trigger_alpha, trigger_prob, confirm_days, lookback_localize in itertools.product(
            PHASE_POLICY_SWEEP_TRIGGER_ALPHAS,
            PHASE_POLICY_SWEEP_TRIGGER_PROBS,
            PHASE_POLICY_SWEEP_CONFIRM_DAYS,
            PHASE_POLICY_SWEEP_LOOKBACKS,
        ):
            for stabilization_policy in policies:
                suffix = (
                    f"[a{trigger_alpha:.2f}|p{trigger_prob:.2f}|c{confirm_days}"
                    f"|lb{lookback_localize}|{stabilization_policy}]"
                )
                yield _phase_candidate(
                    group_name,
                    model_type="rf",
                    trigger_alpha=trigger_alpha,
                    trigger_prob=trigger_prob,
                    confirm_days=confirm_days,
                    lookback_localize=lookback_localize,
                    stabilization_policy=stabilization_policy,
                    clamp_radius=PHASECLS_CLAMP_RADIUS,
                    name_suffix=suffix,
                )


def _run_phase_policy_search(cs, lh, subj_order, labeled, quality_subset, compare_freeze_vs_clamp):
    print(f"\n{SEP}\n  S. PHASE POLICY SEARCH\n{SEP}")
    print(
        "  Sweep grid: "
        f"alpha={PHASE_POLICY_SWEEP_TRIGGER_ALPHAS}, "
        f"prob={PHASE_POLICY_SWEEP_TRIGGER_PROBS}, "
        f"confirm={PHASE_POLICY_SWEEP_CONFIRM_DAYS}, "
        f"lookback={PHASE_POLICY_SWEEP_LOOKBACKS}, "
        f"policy={['none', 'freeze', 'clamp'] if compare_freeze_vs_clamp else ['none']}"
    )
    rows = []
    for spec in _phase_search_specs(compare_freeze_vs_clamp):
        t0 = time.time()
        det_by_day, conf_by_day = spec["fn"](cs, lh)
        elapsed = time.time() - t0
        row = _evaluate_bundle_silent(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
            spec,
            det_by_day,
            conf_by_day,
        )
        row.update(
            {
                "elapsed_sec": elapsed,
                "trigger_alpha": spec["trigger_alpha"],
                "trigger_prob": spec["trigger_prob"],
                "confirm_days": spec["confirm_days"],
                "lookback_localize": spec["lookback_localize"],
                "stabilization_policy": spec["stabilization_policy"],
                "clamp_radius": spec["clamp_radius"],
            }
        )
        rows.append(row)

    rows.sort(key=_phase_row_sort_key)
    print(
        "  "
        f"{'Rank':<4} {'Method':<18} {'Policy':<7} {'α':>4} {'p':>4} {'c':>2} {'lb':>3}"
        f" {'QPostTrig':>10} {'QPostOv':>8} {'AllPostTrig':>11} {'AllPostOv':>9}"
        f" {'Ov1st':>7} {'Avail':>7}"
    )
    print(f"  {'-' * 112}")
    for idx, row in enumerate(rows[:12], start=1):
        print(
            "  "
            f"{idx:<4} {row['signal_group']:<18} {row['stabilization_policy']:<7}"
            f" {row['trigger_alpha']:>4.2f} {row['trigger_prob']:>4.2f}"
            f" {row['confirm_days']:>2d} {row['lookback_localize']:>3d}"
            f" {row['quality_post_trigger_summary'].get('mae', float('nan')):>10.2f}"
            f" {row['quality_summary'].get('post_ov_days', {}).get('mae', float('nan')):>8.2f}"
            f" {row['post_trigger_summary'].get('mae', float('nan')):>11.2f}"
            f" {row['summary'].get('post_ov_days', {}).get('mae', float('nan')):>9.2f}"
            f" {row['ov_summary'].get('first', {}).get('mae', float('nan')):>7.2f}"
            f" {row['summary'].get('availability_rate', 0.0):>6.1%}"
        )

    best_by_policy = {}
    for row in rows:
        best_by_policy.setdefault(row["stabilization_policy"], row)
    print("\n  Best by stabilization policy:")
    for policy in ["none", "freeze", "clamp"]:
        row = best_by_policy.get(policy)
        if row is None:
            continue
        print(
            "  "
            f"{policy:<7} {row['signal_group']:<10}"
            f" α={row['trigger_alpha']:.2f}"
            f" p={row['trigger_prob']:.2f}"
            f" c={row['confirm_days']}"
            f" lb={row['lookback_localize']}"
            f" | QPostTrig={row['quality_post_trigger_summary'].get('mae', float('nan')):.2f}"
            f" | AllPostTrig={row['post_trigger_summary'].get('mae', float('nan')):.2f}"
            f" | OvFirstMAE={row['ov_summary'].get('first', {}).get('mae', float('nan')):.2f}"
        )
    return rows, best_by_policy


def _fixed_phase_policy_kwargs():
    return {
        "trigger_alpha": 0.20,
        "trigger_prob": 0.60,
        "confirm_days": 2,
        "lookback_localize": 10,
        "stabilization_policy": "none",
        "clamp_radius": PHASECLS_CLAMP_RADIUS,
    }


def _phase_family_specs():
    fixed = _fixed_phase_policy_kwargs()
    for model_type in ["rf", "hgb", "lgbm"]:
        for group_name in PHASECLS_DEFAULT_GROUPS:
            yield _phase_candidate(group_name, model_type=model_type, **fixed)


def _temp_hr_localizer_basis_specs(model_type):
    fixed = _fixed_phase_policy_kwargs()
    basis_defs = [
        ("NT-only", ["nightly_temperature"]),
        ("NT+NocT", ["nightly_temperature", "noct_temp"]),
        ("NT+RHR", ["nightly_temperature", "rhr"]),
        ("NT+RHR+NHRmin", ["nightly_temperature", "rhr", "noct_hr_min"]),
        ("NT+NocT+RHR", ["nightly_temperature", "noct_temp", "rhr"]),
    ]
    specs = []
    for label, localizer_sigs in basis_defs:
        specs.append(
            _phase_candidate(
                "Temp+HR",
                model_type=model_type,
                localizer_sigs_override=localizer_sigs,
                localizer_label_override=label,
                name_suffix=f"[Loc={label}]",
                **fixed,
            )
        )
    return specs


def _trigger_temp_hr_phase_kwargs(model_type):
    return {
        "group_name": "Temp+HR",
        "model_type": model_type,
        "localizer_sigs_override": PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"],
        "localizer_label_override": "+".join(PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"]),
    }


def _trigger_baseline_spec(model_type="rf"):
    fixed = _fixed_phase_policy_kwargs()
    return _phase_candidate(
        name_suffix="[Trigger=baseline]",
        **_trigger_temp_hr_phase_kwargs(model_type),
        **fixed,
    )


def _trigger_hysteresis_specs(model_type="rf"):
    fixed = _fixed_phase_policy_kwargs()
    specs = []
    for enter_threshold in [0.55, 0.60]:
        for stay_threshold in [0.35, 0.40, 0.45]:
            if stay_threshold > enter_threshold:
                continue
            for confirm_days in [1, 2]:
                specs.append(
                    _phase_candidate(
                        trigger_mode="hysteresis",
                        enter_threshold=enter_threshold,
                        stay_threshold=stay_threshold,
                        confirm_days=confirm_days,
                        name_suffix=(
                            f"[Trigger=hys|enter{enter_threshold:.2f}"
                            f"|stay{stay_threshold:.2f}|c{confirm_days}]"
                        ),
                        **_trigger_temp_hr_phase_kwargs(model_type),
                        **{k: v for k, v in fixed.items() if k != "confirm_days"},
                    )
                )
    return specs


def _trigger_hybrid_specs(model_type="rf"):
    fixed = _fixed_phase_policy_kwargs()
    specs = []
    for k in [2, 3]:
        for tol in [1, 2]:
            for lower_prob in [0.40, 0.45, 0.50]:
                specs.append(
                    _phase_candidate(
                        trigger_mode="hybrid",
                        hybrid_k=k,
                        hybrid_tol=tol,
                        hybrid_lower_prob=lower_prob,
                        name_suffix=(
                            f"[Trigger=hyb|k{k}|tol{tol}|lp{lower_prob:.2f}]"
                        ),
                        **_trigger_temp_hr_phase_kwargs(model_type),
                        **fixed,
                    )
                )
    return specs


def _trigger_variant_sort_key(row):
    s = row["summary"]
    q = row["quality_summary"]
    pt = row["post_trigger_summary"]
    qpt = row["quality_post_trigger_summary"]
    ov = row["ov_summary"]
    return (
        pt.get("mae", math.inf),
        s.get("post_ov_days", {}).get("mae", math.inf),
        s.get("first_detection_day_mean", math.inf),
        s.get("first_detection_ov_mae", math.inf),
        ov.get("first", {}).get("mae", math.inf),
        qpt.get("mae", math.inf),
        q.get("post_ov_days", {}).get("mae", math.inf),
        -s.get("availability_rate", 0.0),
    )


def _quality_trigger_variant_sort_key(row):
    s = row["summary"]
    q = row["quality_summary"]
    pt = row["post_trigger_summary"]
    qpt = row["quality_post_trigger_summary"]
    ov = row["ov_summary"]
    return (
        qpt.get("mae", math.inf),
        q.get("post_ov_days", {}).get("mae", math.inf),
        qpt.get("acc_3d", -math.inf) * -1.0 if qpt.get("acc_3d") is not None else math.inf,
        q.get("first_detection_day_mean", math.inf),
        q.get("first_detection_ov_mae", math.inf),
        ov.get("first", {}).get("mae", math.inf),
        pt.get("mae", math.inf),
        s.get("post_ov_days", {}).get("mae", math.inf),
    )


def _run_trigger_mechanism_ablation(cs, lh, subj_order, labeled, quality_subset, model_types=("rf",)):
    print(f"\n{SEP}\n  T. TRIGGER MECHANISM ABLATION\n{SEP}")
    results = {model_type: {"baseline": [], "hysteresis": [], "hybrid": []} for model_type in model_types}
    for model_type in model_types:
        spec_groups = {
            "baseline": [_trigger_baseline_spec(model_type=model_type)],
            "hysteresis": _trigger_hysteresis_specs(model_type=model_type),
            "hybrid": _trigger_hybrid_specs(model_type=model_type),
        }
        print(f"  Model family: {model_type}")
        for mech, specs in spec_groups.items():
            print(f"    Running {mech}: {len(specs)} variants")
            for spec in specs:
                t0 = time.time()
                det_by_day, conf_by_day = spec["fn"](cs, lh)
                row = _evaluate_bundle_silent(
                    cs,
                    lh,
                    subj_order,
                    labeled,
                    quality_subset,
                    spec,
                    det_by_day,
                    conf_by_day,
                )
                row["elapsed_sec"] = time.time() - t0
                row["trigger_mode"] = mech
                row["model_family"] = model_type
                results[model_type][mech].append(row)

    best_all = {model_type: {} for model_type in model_types}
    best_quality = {model_type: {} for model_type in model_types}
    for model_type in model_types:
        for mech, rows in results[model_type].items():
            best_all[model_type][mech] = min(rows, key=_trigger_variant_sort_key)
            best_quality[model_type][mech] = min(rows, key=_quality_trigger_variant_sort_key)

    print("\n  Best trigger variants by family:")
    for model_type in model_types:
        for mech in ["baseline", "hysteresis", "hybrid"]:
            row = best_all[model_type][mech]
            print(
                "  "
                f"{model_type.upper():<4} {mech:<10} {row['name']}"
                f" | PostTrigger={row['post_trigger_summary'].get('mae', float('nan')):.2f}"
                f" | PostOv={row['summary']['post_ov_days']['mae']:.2f}"
                f" | FirstDet={row['summary'].get('first_detection_day_mean', float('nan')):.2f}"
                f" | OvFirst={row['ov_summary']['first'].get('mae', float('nan')):.2f}"
            )
    return results, best_all, best_quality


def _stateful_temp_hr_specs(model_type):
    localizer_sigs = PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"]
    common = {
        "group_name": "Temp+HR",
        "model_type": model_type,
        "localizer_sigs_override": localizer_sigs,
        "localizer_label_override": "+".join(localizer_sigs),
        "trigger_alpha": PHASECLS_TRIGGER_ALPHA,
        "trigger_prob": PHASECLS_TRIGGER_PROB,
        "confirm_days": PHASECLS_CONFIRM_DAYS,
        "lookback_localize": PHASECLS_LOOKBACK_LOCALIZE,
        "trigger_mode": "baseline",
    }
    return [
        _phase_candidate(
            name_suffix="[State=none]",
            stabilization_policy="none",
            **common,
        ),
        _phase_candidate(
            name_suffix="[State=sticky-r1-m010]",
            stabilization_policy="sticky",
            sticky_radius=1,
            sticky_improve_margin=0.10,
            **common,
        ),
        _phase_candidate(
            name_suffix="[State=sticky-r2-m010]",
            stabilization_policy="sticky",
            sticky_radius=2,
            sticky_improve_margin=0.10,
            **common,
        ),
        _phase_candidate(
            name_suffix="[State=sticky-r2-m025]",
            stabilization_policy="sticky",
            sticky_radius=2,
            sticky_improve_margin=0.25,
            **common,
        ),
    ]


def _stateful_localizer_sort_key(row):
    s = row["summary"]
    pt = row["post_trigger_summary"]
    ov = row["ov_summary"]
    return (
        pt.get("mae", math.inf),
        s.get("post_ov_days", {}).get("mae", math.inf),
        ov.get("final", {}).get("mae", math.inf),
        s.get("first_detection_day_mean", math.inf),
        s.get("first_detection_ov_mae", math.inf),
        -s.get("availability_rate", 0.0),
    )


def _stateful_quality_sort_key(row):
    q = row["quality_summary"]
    qpt = row["quality_post_trigger_summary"]
    ov = row["ov_summary"]
    return (
        qpt.get("mae", math.inf),
        q.get("post_ov_days", {}).get("mae", math.inf),
        ov.get("final", {}).get("mae", math.inf),
        q.get("first_detection_day_mean", math.inf),
        q.get("first_detection_ov_mae", math.inf),
        -q.get("availability_rate", 0.0),
    )


def _run_stateful_localizer_ablation(cs, lh, subj_order, labeled, quality_subset, model_types=("rf", "hgb")):
    print(f"\n{SEP}\n  U. STATEFUL LOCALIZER ABLATION\n{SEP}")
    results = {model_type: [] for model_type in model_types}
    for model_type in model_types:
        print(f"  Model family: {model_type}")
        for spec in _stateful_temp_hr_specs(model_type):
            t0 = time.time()
            det_by_day, conf_by_day = spec["fn"](cs, lh)
            row = _evaluate_bundle_silent(
                cs,
                lh,
                subj_order,
                labeled,
                quality_subset,
                spec,
                det_by_day,
                conf_by_day,
            )
            row["elapsed_sec"] = time.time() - t0
            row["model_family"] = model_type
            results[model_type].append(row)

    best_all = {model_type: min(rows, key=_stateful_localizer_sort_key) for model_type, rows in results.items()}
    best_quality = {model_type: min(rows, key=_stateful_quality_sort_key) for model_type, rows in results.items()}

    print("\n  Best stateful variant by family:")
    for model_type in model_types:
        row = best_all[model_type]
        print(
            "  "
            f"{model_type.upper():<4} {row['name']}"
            f" | PostTrigger={row['post_trigger_summary'].get('mae', float('nan')):.2f}"
            f" | PostOv={row['summary']['post_ov_days']['mae']:.2f}"
            f" | OvFinal={row['ov_summary']['final'].get('mae', float('nan')):.2f}"
            f" | FirstDet={row['summary'].get('first_detection_day_mean', float('nan')):.2f}"
        )
    return results, best_all, best_quality


def _phasecls_rf_temp_hr_refinement_spec(localizer_variant, name_suffix, stabilization_policy, **pcall_extra):
    sigs = _group_lookup()["Temp+HR"]
    loc = PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"]
    lbl = "+".join(loc)

    def _fn(cs, lh):
        return prefix_phase_classify_loso(
            cs,
            lh,
            sigs=sigs,
            localizer_sigs=loc,
            model_type="rf",
            sigma=PREFIX_BENCHMARK_ML_SIGMA,
            trigger_prob=PHASECLS_TRIGGER_PROB,
            trigger_alpha=PHASECLS_TRIGGER_ALPHA,
            confirm_days=PHASECLS_CONFIRM_DAYS,
            lookback_localize=PHASECLS_LOOKBACK_LOCALIZE,
            stabilization_policy=stabilization_policy,
            clamp_radius=PHASECLS_CLAMP_RADIUS,
            trigger_mode="baseline",
            sticky_radius=pcall_extra.get("sticky_radius", PHASECLS_STICKY_RADIUS),
            sticky_improve_margin=pcall_extra.get("sticky_improve_margin", PHASECLS_STICKY_IMPROVE_MARGIN),
            monotone_back_margin=pcall_extra.get("monotone_back_margin", PHASECLS_MONOTONE_BACK_MARGIN),
            localizer_smooth_window_m=pcall_extra.get("localizer_smooth_window_m", 0),
        )

    return {
        "localizer_variant": localizer_variant,
        "name": f"PhaseCls-Temp+HR{name_suffix}",
        "family": "phasecls-rf",
        "signal_group": "Temp+HR",
        "localizer_group": lbl,
        "use_stability_gate": False,
        "fn": _fn,
    }


def _localizer_refinement_ablation_specs():
    return [
        _phasecls_rf_temp_hr_refinement_spec(
            "overwrite",
            "[Loc=overwrite]",
            "none",
        ),
        _phasecls_rf_temp_hr_refinement_spec(
            "soft_sticky",
            "[Loc=soft_sticky]",
            "soft_sticky",
            sticky_radius=PHASECLS_SOFT_STICKY_RADIUS,
            sticky_improve_margin=PHASECLS_SOFT_STICKY_MARGIN,
        ),
        _phasecls_rf_temp_hr_refinement_spec(
            "bounded_monotone",
            "[Loc=bounded_monotone]",
            "bounded_monotone",
        ),
        _phasecls_rf_temp_hr_refinement_spec(
            "score_smooth_m3",
            "[Loc=score_smooth_m3]",
            "score_smooth",
            localizer_smooth_window_m=PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
        ),
    ]


def _localizer_refinement_sort_key(row):
    s = row["summary"]
    pt = row["post_trigger_summary"]
    ov = row["ov_summary"]
    return (
        s.get("post_ov_days", {}).get("mae", math.inf),
        pt.get("mae", math.inf),
        ov.get("final", {}).get("mae", math.inf),
        s.get("first_detection_day_mean", math.inf),
        -s.get("availability_rate", 0.0),
    )


def _print_localizer_refinement_table(title, rows, summary_key, post_key, det_key):
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print(
        "  "
        f"{'Variant':<22}"
        f" {'AllMAE':>7} {'PostOv':>7} {'PO±2':>6} {'PO±3':>6}"
        f" {'PT_MAE':>7} {'PT±2':>6} {'PT±3':>6}"
        f" {'1stDet':>7} {'OvMAE':>7} {'Avail':>7}"
    )
    print(f"  {'-' * 100}")
    for row in rows:
        s = row[summary_key]
        pt = row[post_key]
        po = s.get("post_ov_days", {})
        print(
            "  "
            f"{row['localizer_variant']:<22}"
            f" {_fmt_num(s.get('all_days', {}).get('mae')):>7}"
            f" {_fmt_num(po.get('mae')):>7}"
            f" {_fmt_pct(po.get('acc_2d')):>6}"
            f" {_fmt_pct(po.get('acc_3d')):>6}"
            f" {_fmt_num(pt.get('mae')):>7}"
            f" {_fmt_pct(pt.get('acc_2d')):>6}"
            f" {_fmt_pct(pt.get('acc_3d')):>6}"
            f" {_fmt_num(s.get('first_detection_day_mean')):>7}"
            f" {_fmt_num(s.get('first_detection_ov_mae')):>7}"
            f" {_fmt_pct(s.get('availability_rate')):>7}"
        )


def _print_ov_stability_table(title, rows):
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print(
        "  "
        f"{'Variant':<22}"
        f" {'Ov1_MAE':>8} {'O1±2':>6} {'O1±3':>6}"
        f" {'OvF_MAE':>8} {'OF±2':>6} {'OF±3':>6}"
    )
    print(f"  {'-' * 80}")
    for row in rows:
        o = row["ov_summary"]
        f = o.get("first", {})
        fn = o.get("final", {})
        print(
            "  "
            f"{row['localizer_variant']:<22}"
            f" {_fmt_num(f.get('mae')):>8}"
            f" {_fmt_pct(f.get('acc_2d')):>6}"
            f" {_fmt_pct(f.get('acc_3d')):>6}"
            f" {_fmt_num(fn.get('mae')):>8}"
            f" {_fmt_pct(fn.get('acc_2d')):>6}"
            f" {_fmt_pct(fn.get('acc_3d')):>6}"
        )


def _print_detected_cycle_refinement_table(title, rows, det_key):
    print(f"\n{SEP}\n  {title}\n{SEP}")
    print(
        "  "
        f"{'Variant':<22}"
        f" {'Det%':>7} {'DetDay':>7} {'Lat':>7}"
        f" {'PT_MAE':>8} {'AnchPost':>9}"
    )
    print(f"  {'-' * 80}")
    for row in rows:
        b = row[det_key]
        cyc = b["cycles"]
        pt = b["post_trigger_summary"]
        an = b["anchor_summary"]
        print(
            "  "
            f"{row['localizer_variant']:<22}"
            f" {_fmt_pct(cyc.get('detected_cycle_rate')):>7}"
            f" {_fmt_num(cyc.get('first_detection_cycle_day_mean')):>7}"
            f" {_fmt_num(cyc.get('latency_days_mean')):>7}"
            f" {_fmt_num(pt.get('mae')):>8}"
            f" {_fmt_num(an.get('post_all', {}).get('mae')):>9}"
        )


def _run_localizer_refinement_ablation(cs, lh, subj_order, labeled, quality_subset):
    print(f"\n{SEP}\n  V. LOCALIZER REFINEMENT (fixed baseline trigger)\n{SEP}")
    rows = []
    for spec in _localizer_refinement_ablation_specs():
        t0 = time.time()
        det_by_day, conf_by_day = spec["fn"](cs, lh)
        bundle = _evaluate_bundle_silent(
            cs,
            lh,
            subj_order,
            labeled,
            quality_subset,
            spec,
            det_by_day,
            conf_by_day,
        )
        bundle["localizer_variant"] = spec["localizer_variant"]
        bundle["elapsed_sec"] = time.time() - t0
        rows.append(bundle)

    best = min(rows, key=_localizer_refinement_sort_key)
    print("\n  Best by (PostOv MAE, PostTrigger MAE, OvFinal MAE, ...):")
    print(
        "  "
        f"{best['localizer_variant']}"
        f" PostOv={best['summary']['post_ov_days']['mae']:.3f}"
        f" PostTrig={best['post_trigger_summary'].get('mae', float('nan')):.3f}"
        f" OvFin={best['ov_summary'].get('final', {}).get('mae', float('nan')):.3f}"
    )
    _print_localizer_refinement_table(
        "TABLE 1 — All labeled (main operational metrics)",
        rows,
        "summary",
        "post_trigger_summary",
        "detected_cycle_summary",
    )
    _print_localizer_refinement_table(
        "TABLE 2 — Quality group (offline stratification)",
        rows,
        "quality_summary",
        "quality_post_trigger_summary",
        "quality_detected_cycle_summary",
    )
    _print_ov_stability_table("TABLE 3 — Ovulation estimate stability (all labeled)", rows)
    print(f"\n{SEP}\n  TABLE 4 — Detected-cycle analysis\n{SEP}")
    _print_detected_cycle_refinement_table(
        "  Block A — All labeled (cycles with ≥1 detection)",
        rows,
        "detected_cycle_summary",
    )
    _print_detected_cycle_refinement_table(
        "  Block B — Quality group (same restriction)",
        rows,
        "quality_detected_cycle_summary",
    )
    return rows


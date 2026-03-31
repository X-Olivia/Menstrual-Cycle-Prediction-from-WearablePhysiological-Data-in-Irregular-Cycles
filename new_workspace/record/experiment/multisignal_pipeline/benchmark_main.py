from __future__ import annotations

import contextlib
import io
import math
import time
import numpy as np

from detectors_ml import (
    prefix_ml_detect_loso,
    prefix_phase_classify_loso,
    prefix_rule_state_detect,
)
from detectors_rule import (
    detect_cusum_prefix_daily,
    detect_multi_cusum_fused_prefix_daily,
    detect_multi_signal_fused_ttest_prefix_daily,
    detect_ttest_prefix_daily,
)
from menses import (
    evaluate_prefix_current_day,
    evaluate_prefix_post_trigger,
    predict_menses_by_anchors,
)
from protocol import (
    DEFAULT_POPULATION_LUTEAL_LENGTH,
    FAST_PREFIX_BENCHMARK,
    PHASECLS_CONFIRM_DAYS,
    PHASECLS_DEFAULT_GROUPS,
    PHASECLS_CLAMP_RADIUS,
    PHASECLS_LOOKBACK_LOCALIZE,
    PHASECLS_LOCALIZER_OVERRIDES,
    PHASECLS_LOCALIZER_SCORE_MIN,
    PHASECLS_LOCALIZER_SHIFT_MIN,
    PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    PHASECLS_LOCALIZER_AGREEMENT_TOL,
    PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
    PHASECLS_MODEL_TYPES,
    PHASECLS_PHASE_ENSEMBLE_MODELS,
    PHASECLS_MONOTONE_BACK_MARGIN,
    PHASECLS_SOFT_STICKY_MARGIN,
    PHASECLS_SOFT_STICKY_RADIUS,
    PHASECLS_STABILIZATION_POLICY,
    PHASECLS_STICKY_IMPROVE_MARGIN,
    PHASECLS_STICKY_RADIUS,
    PHASECLS_TRIGGER_ALPHA,
    PHASECLS_TRIGGER_PROB,
    PREFIX_BENCHMARK_ML_SIGMA,
    PREFIX_BENCHMARK_RULE_SIGMA,
    PREFIX_ML_MODELS,
    PREFIX_ML_SIGNAL_GROUPS,
    PREFIX_RULE_SIGNAL_GROUPS,
    PREFIX_SINGLE_SIGNAL_SPECS,
)

from report_utils import SEP


def _group_lookup():
    return {name: sigs for name, sigs in PREFIX_ML_SIGNAL_GROUPS}


def _rule_group_lookup():
    return {name: (sigs, inverts) for name, sigs, inverts in PREFIX_RULE_SIGNAL_GROUPS}


def _build_oracle_prefix_daily(cs, lh):
    det_by_day, conf_by_day = {}, {}
    for sgk, data in cs.items():
        n = data["cycle_len"]
        det_seq = [None] * n
        conf_seq = [0.0] * n
        ov_true = lh.get(sgk)
        if ov_true is not None:
            for day_idx in range(int(ov_true), n):
                det_seq[day_idx] = int(ov_true)
                conf_seq[day_idx] = 1.0
        det_by_day[sgk] = det_seq
        conf_by_day[sgk] = conf_seq
    return det_by_day, conf_by_day


def _collapse_daily_to_cycle_estimate(det_by_day, conf_by_day):
    det, confs = {}, {}
    all_sgks = set(det_by_day) | set(conf_by_day)
    for sgk in all_sgks:
        det_seq = det_by_day.get(sgk, [])
        conf_seq = conf_by_day.get(sgk, [])
        final_idx = next((idx for idx in range(len(det_seq) - 1, -1, -1) if det_seq[idx] is not None), None)
        if final_idx is None:
            continue
        det[sgk] = det_seq[final_idx]
        confs[sgk] = conf_seq[final_idx] if final_idx < len(conf_seq) else 0.0
    return det, confs


def _ovulation_accuracy_summary(det_by_day, lh, label):
    first_errs = []
    final_errs = []
    for sgk, ov_true in lh.items():
        seq = det_by_day.get(sgk, [])
        first_est = next((v for v in seq if v is not None), None)
        final_est = next((v for v in reversed(seq) if v is not None), None)
        if first_est is not None:
            first_errs.append(abs(first_est - ov_true))
        if final_est is not None:
            final_errs.append(abs(final_est - ov_true))

    def _summ(errs, suffix):
        if not errs:
            print(f"    [{label} {suffix}] n=0")
            return {"n": 0}
        ae = np.asarray(errs, dtype=float)
        res = {
            "n": int(len(ae)),
            "mae": float(np.mean(ae)),
            "acc_1d": float(np.mean(ae <= 1)),
            "acc_2d": float(np.mean(ae <= 2)),
            "acc_3d": float(np.mean(ae <= 3)),
        }
        print(
            f"    [{label} {suffix}] n={res['n']}"
            f" MAE={res['mae']:.2f}"
            f" ±1d={res['acc_1d']:.1%}"
            f" ±2d={res['acc_2d']:.1%}"
            f" ±3d={res['acc_3d']:.1%}"
        )
        return res

    return {
        "first": _summ(first_errs, "OvFirst"),
        "final": _summ(final_errs, "OvFinal"),
    }


def _ovulation_accuracy_summary_subset(det_by_day, lh, label, eval_subset=None):
    if eval_subset is None:
        subset_lh = lh
    else:
        ev = set(eval_subset)
        subset_lh = {sgk: ov_true for sgk, ov_true in lh.items() if sgk in ev}
    return _ovulation_accuracy_summary(det_by_day, subset_lh, label)


def _detected_cycle_summary(
    cs,
    det_by_day,
    lh,
    label,
    eval_subset=None,
    prefix="    ",
):
    ev = set(eval_subset) if eval_subset else set(lh)
    labeled_sgks = [sgk for sgk in cs if sgk in lh and sgk in ev]
    detected_sgks = []
    first_cycle_days = []
    latency_days = []
    for sgk in labeled_sgks:
        seq = det_by_day.get(sgk, [])
        first_idx = next((idx for idx, v in enumerate(seq) if v is not None), None)
        if first_idx is None:
            continue
        detected_sgks.append(sgk)
        first_cycle_days.append(first_idx + 1)
        latency_days.append(first_idx - int(lh[sgk]))

    total_cycles = len(labeled_sgks)
    detected_cycles = len(detected_sgks)
    summary = {
        "total_cycles": total_cycles,
        "detected_cycles": detected_cycles,
        "detected_cycle_rate": (detected_cycles / total_cycles) if total_cycles else 0.0,
        "detected_sgks": set(detected_sgks),
    }
    if first_cycle_days:
        summary["first_detection_cycle_day_mean"] = float(np.mean(first_cycle_days))
    if latency_days:
        summary["latency_days_mean"] = float(np.mean(latency_days))

    print(
        f"{prefix}[{label} DetectedCycles] detected={detected_cycles}/{total_cycles}"
        f" rate={summary['detected_cycle_rate']:.1%}"
        + (
            f" mean_day={summary['first_detection_cycle_day_mean']:.2f}"
            if "first_detection_cycle_day_mean" in summary else ""
        )
        + (
            f" mean_latency={summary['latency_days_mean']:.2f}"
            if "latency_days_mean" in summary else ""
        )
    )
    return summary


def _detected_cycle_bundle(
    cs,
    lh,
    subj_order,
    det_by_day,
    conf_by_day,
    eval_subset,
    label,
    use_stability_gate,
):
    cycle_summary = _detected_cycle_summary(
        cs,
        det_by_day,
        lh,
        label,
        eval_subset=eval_subset,
    )
    detected_sgks = cycle_summary["detected_sgks"]
    if not detected_sgks:
        return {
            "cycles": cycle_summary,
            "ov_summary": {"first": {"n": 0}, "final": {"n": 0}},
            "post_trigger_summary": {"post_trigger_days": 0},
            "anchor_summary": {"pre_all": {}, "post_all": {}},
        }

    det, confs = _collapse_daily_to_cycle_estimate(det_by_day, conf_by_day)
    ov_summary = _silent_call(
        _ovulation_accuracy_summary_subset,
        det_by_day,
        lh,
        label,
        detected_sgks,
    )
    post_trigger_summary = _silent_call(
        evaluate_prefix_post_trigger,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=detected_sgks,
        label=label,
        use_stability_gate=use_stability_gate,
    )
    anchor_summary = _silent_call(
        predict_menses_by_anchors,
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=detected_sgks,
        label=label,
    )
    return {
        "cycles": cycle_summary,
        "ov_summary": ov_summary,
        "post_trigger_summary": post_trigger_summary,
        "anchor_summary": anchor_summary,
    }


def _silent_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _phase_row_sort_key(row):
    summary = row["summary"]
    quality_summary = row["quality_summary"]
    post_trigger = row["post_trigger_summary"]
    quality_post_trigger = row["quality_post_trigger_summary"]
    anchor_summary = row["anchor_summary"]
    return (
        quality_post_trigger.get("mae", math.inf),
        quality_summary.get("post_ov_days", {}).get("mae", math.inf),
        post_trigger.get("mae", math.inf),
        summary.get("post_ov_days", {}).get("mae", math.inf),
        anchor_summary.get("post_all", {}).get("mae", math.inf),
        summary.get("first_detection_ov_mae", math.inf),
        summary.get("first_detection_day_mean", math.inf),
        -summary.get("availability_rate", 0.0),
    )


def _evaluate_bundle_silent(cs, lh, subj_order, labeled, quality_subset, spec, det_by_day, conf_by_day):
    det, confs = _collapse_daily_to_cycle_estimate(det_by_day, conf_by_day)
    summary = _silent_call(
        evaluate_prefix_current_day,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
        use_stability_gate=spec["use_stability_gate"],
    )
    quality_summary = _silent_call(
        evaluate_prefix_current_day,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality_subset,
        label=f"{spec['name']} Quality",
        use_stability_gate=spec["use_stability_gate"],
    )
    post_trigger_summary = _silent_call(
        evaluate_prefix_post_trigger,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
        use_stability_gate=spec["use_stability_gate"],
    )
    quality_post_trigger_summary = _silent_call(
        evaluate_prefix_post_trigger,
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality_subset,
        label=f"{spec['name']} Quality",
        use_stability_gate=spec["use_stability_gate"],
    )
    anchor_summary = _silent_call(
        predict_menses_by_anchors,
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
    )
    quality_anchor_summary = _silent_call(
        predict_menses_by_anchors,
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality_subset,
        label=f"{spec['name']} Quality",
    )
    ov_summary = _silent_call(_ovulation_accuracy_summary, det_by_day, lh, spec["name"])
    detected_cycle_summary = _silent_call(
        _detected_cycle_bundle,
        cs,
        lh,
        subj_order,
        det_by_day,
        conf_by_day,
        labeled,
        spec["name"],
        spec["use_stability_gate"],
    )
    quality_detected_cycle_summary = _silent_call(
        _detected_cycle_bundle,
        cs,
        lh,
        subj_order,
        det_by_day,
        conf_by_day,
        quality_subset,
        f"{spec['name']} Quality",
        spec["use_stability_gate"],
    )
    return {
        "name": spec["name"],
        "family": spec["family"],
        "signal_group": spec["signal_group"],
        "localizer_group": spec.get("localizer_group", spec["signal_group"]),
        "summary": summary,
        "quality_summary": quality_summary,
        "post_trigger_summary": post_trigger_summary,
        "quality_post_trigger_summary": quality_post_trigger_summary,
        "anchor_summary": anchor_summary,
        "quality_anchor_summary": quality_anchor_summary,
        "ov_summary": ov_summary,
        "detected_cycle_summary": detected_cycle_summary,
        "quality_detected_cycle_summary": quality_detected_cycle_summary,
    }


def _rule_candidate(name, group_name, sigs, inverts):
    return {
        "name": name,
        "family": "rule-fused-tt",
        "signal_group": group_name,
        "use_stability_gate": True,
        "fn": lambda cs, lh, sigs=sigs, inverts=inverts: detect_multi_signal_fused_ttest_prefix_daily(
            cs,
            sigs,
            sigma=PREFIX_BENCHMARK_RULE_SIGMA,
            inverts=inverts,
        ),
    }


def _phase_candidate(
    group_name,
    model_type="rf",
    localizer_sigs_override=None,
    localizer_label_override=None,
    trigger_alpha=PHASECLS_TRIGGER_ALPHA,
    trigger_prob=PHASECLS_TRIGGER_PROB,
    confirm_days=PHASECLS_CONFIRM_DAYS,
    lookback_localize=PHASECLS_LOOKBACK_LOCALIZE,
    stabilization_policy=PHASECLS_STABILIZATION_POLICY,
    clamp_radius=PHASECLS_CLAMP_RADIUS,
    trigger_mode="baseline",
    enter_threshold=None,
    stay_threshold=None,
    hybrid_k=None,
    hybrid_tol=None,
    hybrid_lower_prob=None,
    localizer_score_min=PHASECLS_LOCALIZER_SCORE_MIN,
    localizer_shift_min=PHASECLS_LOCALIZER_SHIFT_MIN,
    localizer_agreement_days=PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    localizer_agreement_tol=PHASECLS_LOCALIZER_AGREEMENT_TOL,
    sticky_radius=PHASECLS_STICKY_RADIUS,
    sticky_improve_margin=PHASECLS_STICKY_IMPROVE_MARGIN,
    name_suffix="",
    phase_ensemble_models=None,
    localizer_lookback_fusion=None,
    localizer_smooth_window_m=0,
    monotone_back_margin=None,
    use_bayesian_localizer=False,
    prior_weight=2.0,
    bayesian_prior_overrides=None,
):
    sigs = _group_lookup()[group_name]
    localizer_sigs = (
        list(localizer_sigs_override)
        if localizer_sigs_override is not None
        else PHASECLS_LOCALIZER_OVERRIDES.get(group_name)
    )
    if phase_ensemble_models:
        model_label = "PhaseCls-ENS"
        fam = "phasecls-ens"
    else:
        model_label = "PhaseCls" if model_type == "rf" else f"PhaseCls-{model_type.upper()}"
        fam = f"phasecls-{model_type}"
    localizer_label = (
        localizer_label_override
        if localizer_label_override is not None
        else ("+".join(localizer_sigs) if localizer_sigs is not None else group_name)
    )
    method_name = f"{model_label}-{group_name}{name_suffix}"

    def _fn(cs, lh):
        return prefix_phase_classify_loso(
            cs,
            lh,
            sigs=sigs,
            localizer_sigs=localizer_sigs,
            model_type=model_type,
            sigma=PREFIX_BENCHMARK_ML_SIGMA,
            trigger_prob=trigger_prob,
            trigger_alpha=trigger_alpha,
            confirm_days=confirm_days,
            lookback_localize=lookback_localize,
            stabilization_policy=stabilization_policy,
            clamp_radius=clamp_radius,
            trigger_mode=trigger_mode,
            enter_threshold=enter_threshold,
            stay_threshold=stay_threshold,
            hybrid_k=hybrid_k,
            hybrid_tol=hybrid_tol,
            hybrid_lower_prob=hybrid_lower_prob,
            localizer_score_min=localizer_score_min,
            localizer_shift_min=localizer_shift_min,
            localizer_agreement_days=localizer_agreement_days,
            localizer_agreement_tol=localizer_agreement_tol,
            sticky_radius=sticky_radius,
            sticky_improve_margin=sticky_improve_margin,
            monotone_back_margin=monotone_back_margin,
            localizer_smooth_window_m=localizer_smooth_window_m,
            phase_ensemble_models=phase_ensemble_models,
            localizer_lookback_fusion=localizer_lookback_fusion,
            use_bayesian_localizer=use_bayesian_localizer,
            prior_weight=prior_weight,
            bayesian_prior_overrides=bayesian_prior_overrides,
        )

    return {
        "name": method_name,
        "family": fam,
        "signal_group": group_name,
        "localizer_group": localizer_label,
        "trigger_alpha": trigger_alpha,
        "trigger_prob": trigger_prob,
        "confirm_days": confirm_days,
        "lookback_localize": lookback_localize,
        "stabilization_policy": stabilization_policy,
        "clamp_radius": clamp_radius,
        "trigger_mode": trigger_mode,
        "enter_threshold": enter_threshold,
        "stay_threshold": stay_threshold,
        "hybrid_k": hybrid_k,
        "hybrid_tol": hybrid_tol,
        "hybrid_lower_prob": hybrid_lower_prob,
        "localizer_score_min": localizer_score_min,
        "localizer_shift_min": localizer_shift_min,
        "localizer_agreement_days": localizer_agreement_days,
        "localizer_agreement_tol": localizer_agreement_tol,
        "sticky_radius": sticky_radius,
        "sticky_improve_margin": sticky_improve_margin,
        "use_stability_gate": False,
        "fn": _fn,
    }


def _rule_state_candidate(
    group_name,
    localizer_sigs_override=None,
    localizer_label_override=None,
    lookback_localize=PHASECLS_LOOKBACK_LOCALIZE,
    localizer_score_min=PHASECLS_LOCALIZER_SCORE_MIN,
    localizer_shift_min=PHASECLS_LOCALIZER_SHIFT_MIN,
    localizer_agreement_days=PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    localizer_agreement_tol=PHASECLS_LOCALIZER_AGREEMENT_TOL,
    sticky_radius=PHASECLS_STICKY_RADIUS,
    sticky_improve_margin=PHASECLS_STICKY_IMPROVE_MARGIN,
    name_suffix="",
):
    sigs = (
        list(localizer_sigs_override)
        if localizer_sigs_override is not None
        else PHASECLS_LOCALIZER_OVERRIDES.get(group_name, _group_lookup()[group_name])
    )
    localizer_label = (
        localizer_label_override
        if localizer_label_override is not None
        else "+".join(sigs)
    )
    return {
        "name": f"RuleState-{group_name}{name_suffix}",
        "family": "rule-state",
        "signal_group": group_name,
        "localizer_group": localizer_label,
        "use_stability_gate": False,
        "fn": lambda cs, lh, sigs=sigs, lookback_localize=lookback_localize, localizer_score_min=localizer_score_min, localizer_shift_min=localizer_shift_min, localizer_agreement_days=localizer_agreement_days, localizer_agreement_tol=localizer_agreement_tol, sticky_radius=sticky_radius, sticky_improve_margin=sticky_improve_margin: prefix_rule_state_detect(
            cs,
            sigs=sigs,
            lookback_localize=lookback_localize,
            localizer_score_min=localizer_score_min,
            localizer_shift_min=localizer_shift_min,
            localizer_agreement_days=localizer_agreement_days,
            localizer_agreement_tol=localizer_agreement_tol,
            sticky_radius=sticky_radius,
            sticky_improve_margin=sticky_improve_margin,
        ),
    }


def _fast_candidate_specs(bayesian_overrides=None):
    """Fast benchmark pool.

    This list intentionally mixes:
    - mainline shipped comparators
    - stronger non-personalized Bayesian variants
    - personalized Bayesian comparators retained for explicit comparison

    The personalized candidates are named explicitly with `BayesianPersonalized`
    so their status is visible in logs and downstream reports.
    """
    rule_groups = _rule_group_lookup()
    return [
        _rule_candidate(
            "Rule-TempOnly-ftt_prefix",
            "TempOnly",
            rule_groups["TempOnly"][0],
            rule_groups["TempOnly"][1],
        ),
        _rule_candidate(
            "Rule-HROnly-ftt_prefix",
            "HROnly",
            rule_groups["HROnly"][0],
            rule_groups["HROnly"][1],
        ),
        _phase_candidate("HROnly", model_type="rf"),
        _phase_candidate("TempOnly", model_type="rf"),
        _phase_candidate("Temp+HR+HRV", model_type="rf"),
        _phase_candidate("AllSignals", model_type="rf"),
        _phase_candidate(
            "Temp+HR",
            model_type="rf",
            use_bayesian_localizer=True,
            prior_weight=2.0,
            name_suffix="[Bayesian]",
        ),
        _phase_candidate(
            "Temp+HR",
            model_type="rf",
            use_bayesian_localizer=True,
            prior_weight=2.0,
            bayesian_prior_overrides=bayesian_overrides,
            name_suffix="[BayesianPersonalized]",
        ),
        _phase_candidate(
            "Temp+HR",
            model_type="rf",
            phase_ensemble_models=PHASECLS_PHASE_ENSEMBLE_MODELS,
            use_bayesian_localizer=True,
            prior_weight=2.0,
            bayesian_prior_overrides=bayesian_overrides,
            name_suffix="[Champion-BayesianPersonalized]",
        ),
        _phase_candidate("Temp+HR", model_type="rf", name_suffix="[RF-baseline]"),
        _phase_candidate(
            "Temp+HR",
            model_type="rf",
            phase_ensemble_models=PHASECLS_PHASE_ENSEMBLE_MODELS,
            stabilization_policy="score_smooth",
            localizer_smooth_window_m=int(PHASECLS_LOCALIZER_SCORE_SMOOTH_M),
            name_suffix="[Champion]",
        ),
        _phase_candidate(
            "Temp+HR",
            model_type="rf",
            trigger_mode="evidence",
            stabilization_policy="sticky",
            localizer_score_min=PHASECLS_LOCALIZER_SCORE_MIN,
            localizer_shift_min=PHASECLS_LOCALIZER_SHIFT_MIN,
            localizer_agreement_days=PHASECLS_LOCALIZER_AGREEMENT_DAYS,
            localizer_agreement_tol=PHASECLS_LOCALIZER_AGREEMENT_TOL,
            sticky_radius=PHASECLS_STICKY_RADIUS,
            sticky_improve_margin=PHASECLS_STICKY_IMPROVE_MARGIN,
            name_suffix="[EvidenceSticky]",
        ),
        _rule_state_candidate(
            "Temp+HR",
            localizer_sigs_override=PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"],
            localizer_label_override="+".join(PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"]),
        ),
    ]


def _full_candidate_specs(bayesian_overrides=None):
    specs = []
    for short_name, sig_key, invert in PREFIX_SINGLE_SIGNAL_SPECS:
        specs.append(
            {
                "name": f"{short_name}-tt_prefix",
                "family": "rule-single-tt",
                "signal_group": short_name,
                "use_stability_gate": True,
                "fn": lambda cs, lh, sig_key=sig_key, invert=invert: detect_ttest_prefix_daily(
                    cs,
                    sig_key,
                    sigma=PREFIX_BENCHMARK_RULE_SIGMA,
                    invert=invert,
                ),
            }
        )
        specs.append(
            {
                "name": f"{short_name}-cusum_prefix",
                "family": "rule-single-cusum",
                "signal_group": short_name,
                "use_stability_gate": True,
                "fn": lambda cs, lh, sig_key=sig_key, invert=invert: detect_cusum_prefix_daily(
                    cs,
                    sig_key,
                    sigma=PREFIX_BENCHMARK_RULE_SIGMA,
                    invert=invert,
                ),
            }
        )

    for group_name, sigs, inverts in PREFIX_RULE_SIGNAL_GROUPS:
        specs.append(
            {
                "name": f"{group_name}-ftt_prefix",
                "family": "rule-fused-tt",
                "signal_group": group_name,
                "use_stability_gate": True,
                "fn": lambda cs, lh, sigs=sigs, inverts=inverts: detect_multi_signal_fused_ttest_prefix_daily(
                    cs,
                    sigs,
                    sigma=PREFIX_BENCHMARK_RULE_SIGMA,
                    inverts=inverts,
                ),
            }
        )
        specs.append(
            {
                "name": f"{group_name}-cusum_prefix",
                "family": "rule-fused-cusum",
                "signal_group": group_name,
                "use_stability_gate": True,
                "fn": lambda cs, lh, sigs=sigs, inverts=inverts: detect_multi_cusum_fused_prefix_daily(
                    cs,
                    sigs,
                    sigma=PREFIX_BENCHMARK_RULE_SIGMA,
                    inverts=inverts,
                ),
            }
        )

    for model_type in PHASECLS_MODEL_TYPES:
        for group_name, _sigs in PREFIX_ML_SIGNAL_GROUPS:
            specs.append(_phase_candidate(group_name, model_type=model_type))

    for model_type, model_label in PREFIX_ML_MODELS:
        for group_name, sigs in PREFIX_ML_SIGNAL_GROUPS:
            specs.append(
                {
                    "name": f"{model_label}-{group_name}",
                    "family": f"ml-{model_type}",
                    "signal_group": group_name,
                    "use_stability_gate": True,
                    "fn": lambda cs, lh, model_type=model_type, sigs=sigs: prefix_ml_detect_loso(
                        cs,
                        lh,
                        model_type=model_type,
                        sigs=sigs,
                        sigma=PREFIX_BENCHMARK_ML_SIGMA,
                    ),
                }
            )
    return specs


def _candidate_specs(mode="fast", bayesian_overrides=None):
    if mode == "fast":
        return _fast_candidate_specs(bayesian_overrides=bayesian_overrides)
    return _full_candidate_specs(bayesian_overrides=bayesian_overrides)



def _evaluate_candidate(cs, lh, subj_order, labeled, quality, spec):
    print(f"\n  Candidate: {spec['name']}")
    if spec["family"].startswith("phasecls-"):
        print(
            "    "
            f"[{spec['name']} Config] feature_group={spec['signal_group']}"
            f" localizer={spec.get('localizer_group', spec['signal_group'])}"
        )
    t0 = time.time()
    det_by_day, conf_by_day = spec["fn"](cs, lh)
    elapsed = time.time() - t0
    det, confs = _collapse_daily_to_cycle_estimate(det_by_day, conf_by_day)

    print("    Main summary:")
    summary = evaluate_prefix_current_day(
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
        use_stability_gate=spec["use_stability_gate"],
    )
    print("    Ovulation estimate summary:")
    ov_summary = _ovulation_accuracy_summary(det_by_day, lh, spec["name"])
    print(f"    [{spec['name']} Time] detector_seconds={elapsed:.2f}")
    quality_summary = evaluate_prefix_current_day(
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label=f"{spec['name']} Quality",
        use_stability_gate=spec["use_stability_gate"],
    )
    print("    Post-trigger summary:")
    post_trigger_summary = evaluate_prefix_post_trigger(
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
        use_stability_gate=spec["use_stability_gate"],
    )
    quality_post_trigger_summary = evaluate_prefix_post_trigger(
        cs,
        det_by_day,
        conf_by_day,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label=f"{spec['name']} Quality",
        use_stability_gate=spec["use_stability_gate"],
    )
    print("    Anchor-day summary:")
    anchor_summary = predict_menses_by_anchors(
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label=spec["name"],
    )
    quality_anchor_summary = predict_menses_by_anchors(
        cs,
        det,
        confs,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label=f"{spec['name']} Quality",
    )
    print("    Detected-cycle summary:")
    detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        det_by_day,
        conf_by_day,
        labeled,
        spec["name"],
        spec["use_stability_gate"],
    )
    quality_detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        det_by_day,
        conf_by_day,
        quality,
        f"{spec['name']} Quality",
        spec["use_stability_gate"],
    )
    return {
        "name": spec["name"],
        "family": spec["family"],
        "signal_group": spec["signal_group"],
        "summary": summary,
        "quality_summary": quality_summary,
        "post_trigger_summary": post_trigger_summary,
        "quality_post_trigger_summary": quality_post_trigger_summary,
        "anchor_summary": anchor_summary,
        "quality_anchor_summary": quality_anchor_summary,
        "ov_summary": ov_summary,
        "detected_cycle_summary": detected_cycle_summary,
        "quality_detected_cycle_summary": quality_detected_cycle_summary,
        "elapsed_sec": elapsed,
    }


def _evaluate_baselines(cs, lh, subj_order, labeled, quality):
    print(f"\n{SEP}\n  C. BASELINES\n{SEP}")
    oracle_det, oracle_conf = _build_oracle_prefix_daily(cs, lh)
    oracle_cycle_det, oracle_cycle_conf = _collapse_daily_to_cycle_estimate(oracle_det, oracle_conf)
    print("  Oracle-prefix")
    print("    Main summary:")
    oracle_summary = evaluate_prefix_current_day(
        cs,
        oracle_det,
        oracle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Oracle-prefix",
    )
    print("    Ovulation estimate summary:")
    oracle_ov_summary = _ovulation_accuracy_summary(oracle_det, lh, "Oracle-prefix")
    oracle_quality_summary = evaluate_prefix_current_day(
        cs,
        oracle_det,
        oracle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Oracle-prefix Quality",
    )
    print("    Post-trigger summary:")
    oracle_post_trigger = evaluate_prefix_post_trigger(
        cs,
        oracle_det,
        oracle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Oracle-prefix",
    )
    oracle_quality_post_trigger = evaluate_prefix_post_trigger(
        cs,
        oracle_det,
        oracle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Oracle-prefix Quality",
    )
    print("    Anchor-day summary:")
    oracle_anchor = predict_menses_by_anchors(
        cs,
        oracle_cycle_det,
        oracle_cycle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Oracle-prefix",
    )
    oracle_quality_anchor = predict_menses_by_anchors(
        cs,
        oracle_cycle_det,
        oracle_cycle_conf,
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Oracle-prefix Quality",
    )
    print("    Detected-cycle summary:")
    oracle_detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        oracle_det,
        oracle_conf,
        labeled,
        "Oracle-prefix",
        False,
    )
    oracle_quality_detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        oracle_det,
        oracle_conf,
        quality,
        "Oracle-prefix Quality",
        False,
    )
    print("\n  Calendar")
    print("    Main summary:")
    calendar_summary = evaluate_prefix_current_day(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Calendar",
    )
    print("    Ovulation estimate summary:")
    calendar_ov_summary = _ovulation_accuracy_summary({}, lh, "Calendar")
    calendar_quality_summary = evaluate_prefix_current_day(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Calendar Quality",
    )
    print("    Post-trigger summary:")
    calendar_post_trigger = evaluate_prefix_post_trigger(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Calendar",
    )
    calendar_quality_post_trigger = evaluate_prefix_post_trigger(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Calendar Quality",
    )
    print("    Anchor-day summary:")
    calendar_anchor = predict_menses_by_anchors(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=labeled,
        label="Calendar",
    )
    calendar_quality_anchor = predict_menses_by_anchors(
        cs,
        {},
        {},
        subj_order,
        lh,
        fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
        eval_subset=quality,
        label="Calendar Quality",
    )
    print("    Detected-cycle summary:")
    calendar_detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        {},
        {},
        labeled,
        "Calendar",
        False,
    )
    calendar_quality_detected_cycle_summary = _detected_cycle_bundle(
        cs,
        lh,
        subj_order,
        {},
        {},
        quality,
        "Calendar Quality",
        False,
    )
    return (
        {
            "name": "Oracle-prefix",
            "signal_group": "Oracle",
            "summary": oracle_summary,
            "quality_summary": oracle_quality_summary,
            "post_trigger_summary": oracle_post_trigger,
            "quality_post_trigger_summary": oracle_quality_post_trigger,
            "anchor_summary": oracle_anchor,
            "quality_anchor_summary": oracle_quality_anchor,
            "ov_summary": oracle_ov_summary,
            "detected_cycle_summary": oracle_detected_cycle_summary,
            "quality_detected_cycle_summary": oracle_quality_detected_cycle_summary,
            "elapsed_sec": 0.0,
        },
        {
            "name": "Calendar",
            "signal_group": "Calendar",
            "summary": calendar_summary,
            "quality_summary": calendar_quality_summary,
            "post_trigger_summary": calendar_post_trigger,
            "quality_post_trigger_summary": calendar_quality_post_trigger,
            "anchor_summary": calendar_anchor,
            "quality_anchor_summary": calendar_quality_anchor,
            "ov_summary": calendar_ov_summary,
            "detected_cycle_summary": calendar_detected_cycle_summary,
            "quality_detected_cycle_summary": calendar_quality_detected_cycle_summary,
            "elapsed_sec": 0.0,
        },
    )

def _get_bayesian_prior_overrides(cs, lh, subj_order):
    """Build per-cycle prior overrides for explicit personalized comparators only."""
    import sys
    from pathlib import Path
    research_code_path = Path(__file__).resolve().parents[2] / "research" / "code"
    if str(research_code_path) not in sys.path:
        sys.path.append(str(research_code_path))
    
    try:
        from personalization import build_zero_shot_calibration_table, L1Config
    except ImportError:
        # Fallback to absolute FYP root based pathing
        fyp_root = Path(__file__).resolve().parents[4]
        alt_path = fyp_root / "new_workspace" / "record" / "research" / "code"
        if str(alt_path) not in sys.path:
            sys.path.append(str(alt_path))
        from personalization import build_zero_shot_calibration_table, L1Config

    cfg = L1Config()
    cal_df = build_zero_shot_calibration_table(cs, lh, subj_order, cfg)

    overrides = {}
    for row in cal_df.itertuples():
        # Map sgk to (mean_frac, std_frac)
        # build_zero_shot_calibration_table puts these in the df
        if np.isfinite(row.ov_frac_prior_mean):
            overrides[row.small_group_key] = (row.ov_frac_prior_mean, row.ov_frac_prior_std)
    return overrides


def run_prefix_benchmark(cs, lh, subj_order, labeled, quality_subset, mode="fast"):
    """Evaluate main candidate grid + Oracle/Calendar baselines (prints to stdout)."""
    overrides = _get_bayesian_prior_overrides(cs, lh, subj_order)
    candidate_rows = []
    for spec in _candidate_specs(mode, bayesian_overrides=overrides):

        candidate_rows.append(
            _evaluate_candidate(cs, lh, subj_order, labeled, quality_subset, spec)
        )
    oracle_baseline, calendar_baseline = _evaluate_baselines(
        cs, lh, subj_order, labeled, quality_subset
    )
    return {
        "candidate_rows": candidate_rows,
        "oracle_baseline": oracle_baseline,
        "calendar_baseline": calendar_baseline,
    }

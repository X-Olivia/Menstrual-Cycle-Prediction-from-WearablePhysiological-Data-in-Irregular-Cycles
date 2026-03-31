from __future__ import annotations

import contextlib
import io
import itertools
import math
import time
import numpy as np

from data import CYCLE_OV_CSV, SIGNALS_DIR, load_all_signals
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
    PHASE_POLICY_SWEEP_CONFIRM_DAYS,
    PHASE_POLICY_SWEEP_LOOKBACKS,
    PHASE_POLICY_SWEEP_TRIGGER_ALPHAS,
    PHASE_POLICY_SWEEP_TRIGGER_PROBS,
    PREFIX_BENCHMARK_ML_SIGMA,
    PREFIX_BENCHMARK_RULE_SIGMA,
    PREFIX_ML_MODELS,
    PREFIX_ML_SIGNAL_GROUPS,
    PREFIX_RULE_SIGNAL_GROUPS,
    PREFIX_SINGLE_SIGNAL_SPECS,
)


SEP = "=" * 76


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


def _fast_candidate_specs():
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


def _full_candidate_specs():
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


def _candidate_specs(mode):
    if mode == "fast":
        return _fast_candidate_specs()
    return _full_candidate_specs()


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


def _print_header(mode):
    print(f"\n{SEP}")
    print("  Multi-Signal Prefix Benchmark Selector")
    print(f"{SEP}")
    print(f"  Mode: {mode}")
    print(f"  Cycle: {CYCLE_OV_CSV}")
    print(f"  Signals: {SIGNALS_DIR}")
    print(f"  Rule σ={PREFIX_BENCHMARK_RULE_SIGMA} | ML σ={PREFIX_BENCHMARK_ML_SIGMA}")
    print(
        "  Fast mode candidates: Calendar, Oracle-prefix, "
        "Rule-TempOnly-ftt_prefix, Rule-HROnly-ftt_prefix, "
        "PhaseCls-HROnly, PhaseCls-TempOnly, PhaseCls-Temp+HR, "
        "PhaseCls-Temp+HR[EvidenceSticky], RuleState-Temp+HR"
    )


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
    candidate_rows = []
    for spec in _candidate_specs(mode):
        candidate_rows.append(
            _evaluate_candidate(cs, lh, subj_order, labeled, quality_subset, spec)
        )

    oracle_baseline, calendar_baseline = _evaluate_baselines(
        cs,
        lh,
        subj_order,
        labeled,
        quality_subset,
    )
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
    import sys

    main(compare_localizer_refinement="--localizer-refinement" in sys.argv)

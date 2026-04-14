"""Unified benchmark candidate registry.

Full mode is the total benchmark pool.
Fast mode is a curated subset of the full pool.
Slow candidates remain part of the full pool, but can be run separately.
"""
from __future__ import annotations

from functools import lru_cache

from protocol import (
    PHASECLS_LOCALIZER_OVERRIDES,
    PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    PHASECLS_LOCALIZER_AGREEMENT_TOL,
    PHASECLS_LOCALIZER_SCORE_MIN,
    PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
    PHASECLS_LOCALIZER_SHIFT_MIN,
    PHASECLS_PHASE_ENSEMBLE_MODELS,
    PHASECLS_STICKY_IMPROVE_MARGIN,
    PHASECLS_STICKY_RADIUS,
    PREFIX_ML_MODELS,
    PREFIX_ML_SIGNAL_GROUPS,
    PREFIX_RULE_SIGNAL_GROUPS,
    PREFIX_SINGLE_SIGNAL_SPECS,
)


def _entry(
    name,
    *,
    kind,
    family,
    builder,
    in_fast=False,
    in_full=True,
    runtime_class="fast",
    **kwargs,
):
    return {
        "name": name,
        "kind": kind,
        "family": family,
        "builder": builder,
        "in_fast": in_fast,
        "in_full": in_full,
        "runtime_class": runtime_class,
        **kwargs,
    }


@lru_cache(maxsize=1)
def benchmark_candidate_registry():
    entries = []

    for short_name, sig_key, invert in PREFIX_SINGLE_SIGNAL_SPECS:
        entries.append(
            _entry(
                f"{short_name}-tt_prefix",
                kind="rule-single",
                family="rule-single-tt",
                builder="rule_single_ttest",
                short_name=short_name,
                sig_key=sig_key,
                invert=invert,
            )
        )
        entries.append(
            _entry(
                f"{short_name}-cusum_prefix",
                kind="rule-single",
                family="rule-single-cusum",
                builder="rule_single_cusum",
                short_name=short_name,
                sig_key=sig_key,
                invert=invert,
            )
        )

    for group_name, _sigs, _inverts in PREFIX_RULE_SIGNAL_GROUPS:
        rule_name = (
            f"Rule-{group_name}-ftt_prefix"
            if group_name in {"TempOnly", "HROnly"}
            else f"{group_name}-ftt_prefix"
        )
        entries.append(
            _entry(
                rule_name,
                kind="rule-fused",
                family="rule-fused-tt",
                builder="rule_fused_ttest",
                group_name=group_name,
                in_fast=group_name in {"TempOnly", "HROnly"},
            )
        )
        entries.append(
            _entry(
                f"{group_name}-cusum_prefix",
                kind="rule-fused",
                family="rule-fused-cusum",
                builder="rule_fused_cusum",
                group_name=group_name,
            )
        )

    for group_name, _sigs in PREFIX_ML_SIGNAL_GROUPS:
        phase_name = (
            "PhaseCls-Temp+HR[RF-baseline]"
            if group_name == "Temp+HR"
            else f"PhaseCls-{group_name}"
        )
        phase_kwargs = {"model_type": "rf"}
        if group_name == "Temp+HR":
            phase_kwargs["name_suffix"] = "[RF-baseline]"
        entries.append(
            _entry(
                phase_name,
                kind="phase",
                family="phasecls-rf",
                builder="phase_candidate",
                group_name=group_name,
                phase_kwargs=phase_kwargs,
                in_fast=group_name in {"HROnly", "TempOnly", "Temp+HR", "Temp+HR+HRV", "AllSignals"},
            )
        )

    entries.extend(
        [
            _entry(
                "PhaseCls-Temp+HR[Bayesian]",
                kind="phase-bayesian",
                family="phasecls-rf",
                builder="phase_candidate",
                group_name="Temp+HR",
                in_fast=True,
                phase_kwargs={
                    "model_type": "rf",
                    "use_bayesian_localizer": True,
                    "prior_weight": 2.0,
                    "name_suffix": "[Bayesian]",
                },
            ),
            _entry(
                "PhaseCls-Temp+HR[BayesianPersonalized]",
                kind="phase-bayesian-personalized",
                family="phasecls-rf",
                builder="phase_candidate",
                group_name="Temp+HR",
                in_fast=True,
                inject_bayesian_overrides=True,
                phase_kwargs={
                    "model_type": "rf",
                    "use_bayesian_localizer": True,
                    "prior_weight": 2.0,
                    "name_suffix": "[BayesianPersonalized]",
                },
            ),
            _entry(
                "PhaseCls-ENS-Temp+HR[Champion-BayesianPersonalized]",
                kind="phase-ensemble-bayesian-personalized",
                family="phasecls-ens",
                builder="phase_candidate",
                group_name="Temp+HR",
                in_fast=True,
                inject_bayesian_overrides=True,
                phase_kwargs={
                    "model_type": "rf",
                    "phase_ensemble_models": PHASECLS_PHASE_ENSEMBLE_MODELS,
                    "use_bayesian_localizer": True,
                    "prior_weight": 2.0,
                    "name_suffix": "[Champion-BayesianPersonalized]",
                },
            ),
            _entry(
                "PhaseCls-ENS-Temp+HR[Champion]",
                kind="phase-ensemble",
                family="phasecls-ens",
                builder="phase_candidate",
                group_name="Temp+HR",
                in_fast=True,
                phase_kwargs={
                    "model_type": "rf",
                    "phase_ensemble_models": PHASECLS_PHASE_ENSEMBLE_MODELS,
                    "stabilization_policy": "score_smooth",
                    "localizer_smooth_window_m": int(PHASECLS_LOCALIZER_SCORE_SMOOTH_M),
                    "name_suffix": "[Champion]",
                },
            ),
            _entry(
                "PhaseCls-Temp+HR[EvidenceSticky]",
                kind="phase-evidence",
                family="phasecls-rf",
                builder="phase_candidate",
                group_name="Temp+HR",
                in_fast=True,
                phase_kwargs={
                    "model_type": "rf",
                    "trigger_mode": "evidence",
                    "stabilization_policy": "sticky",
                    "localizer_score_min": PHASECLS_LOCALIZER_SCORE_MIN,
                    "localizer_shift_min": PHASECLS_LOCALIZER_SHIFT_MIN,
                    "localizer_agreement_days": PHASECLS_LOCALIZER_AGREEMENT_DAYS,
                    "localizer_agreement_tol": PHASECLS_LOCALIZER_AGREEMENT_TOL,
                    "sticky_radius": PHASECLS_STICKY_RADIUS,
                    "sticky_improve_margin": PHASECLS_STICKY_IMPROVE_MARGIN,
                    "name_suffix": "[EvidenceSticky]",
                },
            ),
            _entry(
                "RuleState-Temp+HR",
                kind="rule-state",
                family="rule-state",
                builder="rule_state_candidate",
                group_name="Temp+HR",
                in_fast=True,
                localizer_sigs_override=PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"],
                localizer_label_override="+".join(PHASECLS_LOCALIZER_OVERRIDES["Temp+HR"]),
            ),
        ]
    )

    for model_type, model_label in PREFIX_ML_MODELS:
        for group_name, _sigs in PREFIX_ML_SIGNAL_GROUPS:
            entries.append(
                _entry(
                    f"{model_label}-{group_name}",
                    kind="legacy-ml-prefix",
                    family=f"ml-{model_type}",
                    builder="ml_prefix",
                    group_name=group_name,
                    model_type=model_type,
                    runtime_class="slow",
                )
            )

    return tuple(entries)


def candidate_defs_for_pool(mode="fast", include_slow=True, family=None):
    defs = list(benchmark_candidate_registry())
    if family is not None:
        defs = [entry for entry in defs if entry["family"] == family]
    if mode == "fast":
        return [entry for entry in defs if entry["in_fast"]]
    if mode == "slow":
        return [entry for entry in defs if entry["in_full"] and entry["runtime_class"] == "slow"]
    if mode != "full":
        raise ValueError(f"Unknown benchmark mode: {mode}")
    if include_slow:
        return [entry for entry in defs if entry["in_full"]]
    return [
        entry
        for entry in defs
        if entry["in_full"] and entry["runtime_class"] != "slow"
    ]


def candidate_names_for_pool(mode="fast", include_slow=True, family=None):
    return [entry["name"] for entry in candidate_defs_for_pool(mode, include_slow=include_slow, family=family)]


def slow_candidate_families():
    return sorted(
        {
            entry["family"]
            for entry in benchmark_candidate_registry()
            if entry["runtime_class"] == "slow"
        }
    )

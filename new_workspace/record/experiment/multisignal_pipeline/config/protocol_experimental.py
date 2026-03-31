"""
EXPERIMENTAL / research-only protocol constants.

Sweeps, optional ensembles, extra stabilization hyperparameters, and orchestration
knobs not used by the default mainline benchmark.
"""
from __future__ import annotations

# Sweeps & orchestration (historical experiment harness)
RULE_SIGMAS = [1.5, 2.0, 2.5]
STACKING_TOP_NS = [5, 10, 15, 20]
ENSEMBLE_TOP_NS = [3, 5, 7, 10, 15]
FINAL_RANKING_KEY_INDEX = 4
FINAL_MENSES_TOP_K = 5

PHASE_POLICY_SWEEP_TRIGGER_ALPHAS = [0.20, 0.35, 0.50, 0.65]
PHASE_POLICY_SWEEP_TRIGGER_PROBS = [0.50, 0.55, 0.60]
PHASE_POLICY_SWEEP_CONFIRM_DAYS = [1, 2]
PHASE_POLICY_SWEEP_LOOKBACKS = [6, 8, 10, 12]

# Optional phase-probability ensemble (research)
PHASECLS_PHASE_ENSEMBLE_MODELS = ("rf", "hgb")
PHASECLS_LOCALIZER_LOOKBACK_FUSION = (8, 10, 12)

# Extra stabilization hyperparameters (used only by experimental policies)
PHASECLS_SOFT_STICKY_RADIUS = 2
PHASECLS_SOFT_STICKY_MARGIN = 0.10
PHASECLS_MONOTONE_BACK_MARGIN = 0.18

EXPERIMENTAL_CONFIG = {
    "phase_ensemble_models": PHASECLS_PHASE_ENSEMBLE_MODELS,
    "localizer_lookback_fusion": PHASECLS_LOCALIZER_LOOKBACK_FUSION,
    "phase_policy_sweep": {
        "trigger_alphas": PHASE_POLICY_SWEEP_TRIGGER_ALPHAS,
        "trigger_probs": PHASE_POLICY_SWEEP_TRIGGER_PROBS,
        "confirm_days": PHASE_POLICY_SWEEP_CONFIRM_DAYS,
        "lookbacks": PHASE_POLICY_SWEEP_LOOKBACKS,
    },
}

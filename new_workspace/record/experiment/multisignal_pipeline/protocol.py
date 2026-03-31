from __future__ import annotations

# Shared protocol and structural constants for the multisignal experiment.
# Phase 2 cleanup only: values are preserved exactly from the pre-refactor code.

# Rule-detector priors / defaults
EXPECTED_OVULATION_FRACTION = 0.575
POSITION_PRIOR_WIDTH = 4.0
SINGLE_SIGNAL_CUSUM_THRESHOLD = 1.0
MULTI_SIGNAL_CUSUM_THRESHOLD = 0.5
MIN_CYCLE_LEN_FOR_DETECTION = 12
MIN_DETECTION_DAY = 5
MAX_RIGHT_MARGIN_DAYS = 3
MIN_EXPECTED_OVULATION_DAY = 8
SAVGOL_FALLBACK_SIGMA = 2.0
HMM_TRANSITION_STAY_PROB = 0.95

# Shared defaults derived from prior cycles
DEFAULT_HISTORY_CYCLE_LEN = 28.0
DEFAULT_HISTORY_CYCLE_STD = 4.0

# Label / subset definitions
OVULATION_PROBABILITY_THRESHOLD = 0.5
LABEL_LUTEAL_MIN = 8
LABEL_LUTEAL_MAX = 20
QUALITY_PRE_WINDOW = 5
QUALITY_POST_START_OFFSET = 2
QUALITY_POST_WINDOW = 5
QUALITY_MIN_OV_DAY = 3
QUALITY_MIN_TEMP_SHIFT = 0.2

# Menses-prediction protocol
# Population prior aligned with cohort mean luteal (~12d); improves per-cycle countdown vs 14d folklore.
DEFAULT_POPULATION_LUTEAL_LENGTH = 12.0
ALTERNATE_LUTEAL_LENGTHS = [12, 13]
MENSES_LUTEAL_UPDATE_MIN = 8
MENSES_LUTEAL_UPDATE_MAX = 22
ANCHORS_PRE = [-7, -3, -1]
ANCHORS_POST = [2, 5, 10]
ANCHORS_ALL = ANCHORS_PRE + ANCHORS_POST
COUNTDOWN_MIN_OVULATION_DAY = 3
COUNTDOWN_POST_OVULATION_OFFSET = 2

# Orchestration protocol
RULE_SIGMAS = [1.5, 2.0, 2.5]
STACKING_TOP_NS = [5, 10, 15, 20]
ENSEMBLE_TOP_NS = [3, 5, 7, 10, 15]
FINAL_RANKING_KEY_INDEX = 4  # ±2d accuracy in the final tuple
FINAL_MENSES_TOP_K = 5
REPORT_DAY_THRESHOLDS = [1, 2, 3, 5]

# ML / CNN feature-engineering constants kept unchanged
FEATURE_SIGMA = 1.5
FEATURE_TTEST_DAYS = [8, 10, 12, 14, 16, 18, 20]
FEATURE_AUTOCORR_LAGS = [1, 3, 5]
PHASE_CLASSIFIER_BOUNDARY_THRESHOLD = 0.5
CNN_MAX_LEN = 40

PREFIX_BENCHMARK_RULE_SIGMA = 2.0
PREFIX_BENCHMARK_ML_SIGMA = 1.5
FAST_PREFIX_BENCHMARK = True
PHASECLS_DEFAULT_GROUPS = ["HROnly", "TempOnly", "Temp+HR"]
PHASECLS_TRIGGER_PROB = 0.60
PHASECLS_TRIGGER_ALPHA = 0.20
PHASECLS_CONFIRM_DAYS = 2
PHASECLS_LOOKBACK_LOCALIZE = 10
PHASECLS_STABILIZATION_POLICY = "none"
PHASECLS_CLAMP_RADIUS = 2
PHASECLS_STICKY_RADIUS = 2
PHASECLS_STICKY_IMPROVE_MARGIN = 0.25
# Post-trigger localizer refinement (prefix-valid; no future reads)
PHASECLS_SOFT_STICKY_RADIUS = 2
PHASECLS_SOFT_STICKY_MARGIN = 0.10
PHASECLS_MONOTONE_BACK_MARGIN = 0.18
PHASECLS_LOCALIZER_SCORE_SMOOTH_M = 3
# Equal-weight LOSO phase-probability ensemble (no leakage if members are LOSO).
PHASECLS_PHASE_ENSEMBLE_MODELS = ("rf", "hgb")
# Fuse deterministic localizers at multiple prefix lookbacks (each view is prefix-valid).
PHASECLS_LOCALIZER_LOOKBACK_FUSION = (8, 10, 12)
PHASECLS_LOCALIZER_SCORE_MIN = 1.50
PHASECLS_LOCALIZER_SHIFT_MIN = 0.10
PHASECLS_LOCALIZER_AGREEMENT_DAYS = 3
PHASECLS_LOCALIZER_AGREEMENT_TOL = 2
PHASECLS_MODEL_CACHE_VERSION = "phaseprob_v1"
PHASECLS_LOCALIZER_CACHE_VERSION = "phaseloc_v2"
PHASECLS_LOCALIZER_OVERRIDES = {
    # The phase classifier still uses the full Temp+HR causal feature group, but
    # a narrower deterministic localizer improves benchmark accuracy on this
    # dataset while staying strictly prefix-valid.
    "Temp+HR": ["nightly_temperature", "rhr", "noct_hr_min"],
}
# Keep the active benchmark pool to models that are stable on this machine.
# Boosting families remain implemented in detectors_ml.py but are not enabled
# here because OpenMP-backed builds crash in the local runtime.
PHASECLS_MODEL_TYPES = ["rf"]
PREFIX_CACHE_VERSION = "phasecls_v3"
PHASE_POLICY_SWEEP_TRIGGER_ALPHAS = [0.20, 0.35, 0.50, 0.65]
PHASE_POLICY_SWEEP_TRIGGER_PROBS = [0.50, 0.55, 0.60]
PHASE_POLICY_SWEEP_CONFIRM_DAYS = [1, 2]
PHASE_POLICY_SWEEP_LOOKBACKS = [6, 8, 10, 12]

# Single-signal rule candidates only use signals with established directionality
# from the old codebase, to avoid inventing new invert assumptions.
PREFIX_SINGLE_SIGNAL_SPECS = [
    ("NT", "nightly_temperature", False),
    ("NocT", "noct_temp", False),
    ("NocHR", "noct_hr_mean", False),
    ("RHR", "rhr", False),
    ("RMSSD", "rmssd_mean", True),
    ("HF", "hf_mean", True),
    ("LFHF", "lf_hf_ratio", False),
]

# Controlled fused rule groups, again only using signals with established directionality.
PREFIX_RULE_SIGNAL_GROUPS = [
    ("TempOnly", ["nightly_temperature", "noct_temp"], [False, False]),
    ("HROnly", ["rhr", "noct_hr_mean"], [False, False]),
    ("HRVOnly", ["rmssd_mean", "hf_mean", "lf_hf_ratio"], [True, True, False]),
    ("Temp+HR", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean"], [False, False, False, False]),
    ("Temp+HRV", ["nightly_temperature", "noct_temp", "rmssd_mean", "hf_mean", "lf_hf_ratio"], [False, False, True, True, False]),
    ("Temp+HR+HRV", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean", "rmssd_mean", "hf_mean", "lf_hf_ratio"], [False, False, False, False, True, True, False]),
]

# Prefix ML signal groups. These are direction-agnostic feature groups and may
# include the wider loaded signal space.
PREFIX_ML_SIGNAL_GROUPS = [
    ("TempOnly", ["nightly_temperature", "noct_temp"]),
    ("HROnly", ["rhr", "noct_hr_mean", "noct_hr_std", "noct_hr_min"]),
    ("HRVOnly", ["rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio", "hrv_coverage"]),
    ("Temp+HR", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean", "noct_hr_std", "noct_hr_min"]),
    ("Temp+HRV", ["nightly_temperature", "noct_temp", "rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio", "hrv_coverage"]),
    ("Temp+HR+HRV", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean", "rmssd_mean", "lf_hf_ratio"]),
    ("AllSignals", ["nightly_temperature", "noct_temp", "rhr", "rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio", "noct_hr_mean", "noct_hr_std", "noct_hr_min", "hrv_coverage"]),
]

PREFIX_ML_MODELS = [
    ("ridge", "ML-ridge-prefix"),
    ("rf", "ML-rf-prefix"),
]

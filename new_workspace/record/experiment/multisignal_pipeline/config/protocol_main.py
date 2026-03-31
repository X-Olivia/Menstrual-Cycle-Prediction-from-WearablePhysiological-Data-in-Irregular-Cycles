"""
MAINLINE / production-side protocol constants.

Used by the default benchmark pipeline (RF phase classifier, baseline trigger,
deterministic localizer, prefix evaluation). Do not import experimental sweeps from here.
"""
from __future__ import annotations

# --- Classifier (single main family for shipped benchmark) ---
MAIN_CLASSIFIER = "rf"

# --- Evaluation: what we optimize for in reports (math unchanged; keys into summary dicts) ---
PRIMARY_METRIC_KEY = "post_ov_days"  # use .get("mae") on summary[PRIMARY_METRIC_KEY]
SECONDARY_METRIC_KEY = "post_trigger"  # use post_trigger_summary["mae"]

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

DEFAULT_HISTORY_CYCLE_LEN = 28.0
DEFAULT_HISTORY_CYCLE_STD = 4.0

OVULATION_PROBABILITY_THRESHOLD = 0.5
LABEL_LUTEAL_MIN = 8
LABEL_LUTEAL_MAX = 20
QUALITY_PRE_WINDOW = 5
QUALITY_POST_START_OFFSET = 2
QUALITY_POST_WINDOW = 5
QUALITY_MIN_OV_DAY = 3
QUALITY_MIN_TEMP_SHIFT = 0.2

DEFAULT_POPULATION_LUTEAL_LENGTH = 12.0
ALTERNATE_LUTEAL_LENGTHS = [12, 13]
MENSES_LUTEAL_UPDATE_MIN = 8
MENSES_LUTEAL_UPDATE_MAX = 22
ANCHORS_PRE = [-7, -3, -1]
ANCHORS_POST = [2, 5, 10]
ANCHORS_ALL = ANCHORS_PRE + ANCHORS_POST
COUNTDOWN_MIN_OVULATION_DAY = 3
COUNTDOWN_POST_OVULATION_OFFSET = 2

REPORT_DAY_THRESHOLDS = [1, 2, 3, 5]

FEATURE_SIGMA = 1.5
FEATURE_TTEST_DAYS = [8, 10, 12, 14, 16, 18, 20]
FEATURE_AUTOCORR_LAGS = [1, 3, 5]
PHASE_CLASSIFIER_BOUNDARY_THRESHOLD = 0.5
CNN_MAX_LEN = 40

PREFIX_BENCHMARK_RULE_SIGMA = 2.0
PREFIX_BENCHMARK_ML_SIGMA = 1.5
FAST_PREFIX_BENCHMARK = True
PHASECLS_DEFAULT_GROUPS = ["HROnly", "TempOnly", "Temp+HR"]

# Phase / localizer defaults (mainline)
PHASECLS_TRIGGER_PROB = 0.60
PHASECLS_TRIGGER_ALPHA = 0.20
PHASECLS_CONFIRM_DAYS = 2
PHASECLS_LOOKBACK_LOCALIZE = 10
PHASECLS_STABILIZATION_POLICY = "none"
PHASECLS_CLAMP_RADIUS = 2
PHASECLS_STICKY_RADIUS = 2
PHASECLS_STICKY_IMPROVE_MARGIN = 0.25

PHASECLS_LOCALIZER_SCORE_MIN = 1.50
PHASECLS_LOCALIZER_SHIFT_MIN = 0.10
PHASECLS_LOCALIZER_AGREEMENT_DAYS = 3
PHASECLS_LOCALIZER_AGREEMENT_TOL = 2
PHASECLS_MODEL_CACHE_VERSION = "phaseprob_v1"
PHASECLS_LOCALIZER_CACHE_VERSION = "phaseloc_v2"
PHASECLS_LOCALIZER_OVERRIDES = {
    "Temp+HR": ["nightly_temperature", "rhr", "noct_hr_min"],
}

PHASECLS_MODEL_TYPES = ["rf"]
PREFIX_CACHE_VERSION = "phasecls_v3"

PREFIX_SINGLE_SIGNAL_SPECS = [
    ("NT", "nightly_temperature", False),
    ("NocT", "noct_temp", False),
    ("NocHR", "noct_hr_mean", False),
    ("RHR", "rhr", False),
    ("RMSSD", "rmssd_mean", True),
    ("HF", "hf_mean", True),
    ("LFHF", "lf_hf_ratio", False),
]

PREFIX_RULE_SIGNAL_GROUPS = [
    ("TempOnly", ["nightly_temperature", "noct_temp"], [False, False]),
    ("HROnly", ["rhr", "noct_hr_mean"], [False, False]),
    ("HRVOnly", ["rmssd_mean", "hf_mean", "lf_hf_ratio"], [True, True, False]),
    ("Temp+HR", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean"], [False, False, False, False]),
    ("Temp+HRV", ["nightly_temperature", "noct_temp", "rmssd_mean", "hf_mean", "lf_hf_ratio"], [False, False, True, True, False]),
    ("Temp+HR+HRV", ["nightly_temperature", "noct_temp", "rhr", "noct_hr_mean", "rmssd_mean", "hf_mean", "lf_hf_ratio"], [False, False, False, False, True, True, False]),
]

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

# Authoritative mainline phase-classifier + trigger + localizer update (no ensemble).
MAIN_PHASE_CONFIG = {
    "classifier": MAIN_CLASSIFIER,
    "trigger_mode": "baseline",
    "trigger_prob": PHASECLS_TRIGGER_PROB,
    "trigger_alpha": PHASECLS_TRIGGER_ALPHA,
    "confirm_days": PHASECLS_CONFIRM_DAYS,
    "lookback_localize": PHASECLS_LOOKBACK_LOCALIZE,
    "stabilization_policy": PHASECLS_STABILIZATION_POLICY,
    "phase_ensemble_models": None,
    "localizer_lookback_fusion": None,
}

# Allowed post-trigger localizer update policies in **core** (mainline + one supported comparator).
MAINLINE_STABILIZATION = "none"
SUPPORTED_COMPARATOR_STABILIZATION = "score_smooth"
PHASECLS_LOCALIZER_SCORE_SMOOTH_M = 3

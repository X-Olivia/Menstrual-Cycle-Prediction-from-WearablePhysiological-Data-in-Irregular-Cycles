"""model_v3 config: paths, feature groups, hyperparameters."""
import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data (v4 is default: per-cycle-early z-norm + frac fix + RHR median + boundary removal + temp_std)
FEATURES_CSV = os.path.join(WORKSPACE, "processed_data", "v4", "daily_features_v4.csv")
FEATURES_V3_CSV = os.path.join(WORKSPACE, "processed_data", "v3", "daily_features_v3.csv")
FEATURES_V4_CSV = FEATURES_CSV
FEATURES_V5_CSV = os.path.join(WORKSPACE, "processed_data", "v5", "daily_features_v5.csv")
CYCLE_CSV = os.path.join(WORKSPACE, "subdataset", "cycle_clean_2.csv")

# ── Feature groups (for ablation) ────────────────────────────────────────────

FEAT_WEARABLE_Z = [
    "rmssd_mean_z", "lf_mean_z", "hf_mean_z", "lf_hf_ratio_z",
    "hr_mean_z", "hr_std_z", "hr_min_z", "hr_max_z",
    "wt_mean_z", "wt_std_z", "wt_min_z", "wt_max_z",
    "nightly_temperature_z", "resting_hr_z",
]

FEAT_RESPIRATORY_Z = ["full_sleep_br_z", "deep_sleep_br_z"]

FEAT_SLEEP_Z = ["sleep_score_z", "deep_sleep_min_z", "restlessness_z"]

FEAT_SYMPTOMS = ["cramps", "bloating", "sorebreasts", "moodswing"]

FEAT_DELTAS = [
    "delta_wt_mean_1d", "delta_nightly_temperature_1d",
    "delta_rmssd_mean_1d", "delta_hf_mean_1d",
    "delta_hr_mean_1d", "delta_full_sleep_br_1d",
]

FEAT_SHIFTS = ["wt_shift_7v3", "temp_shift_7v3"]

FEAT_CYCLE_PRIOR = [
    "day_in_cycle", "day_in_cycle_frac",
    "hist_cycle_len_mean", "hist_cycle_len_std",
    "days_remaining_prior", "days_remaining_prior_log",
]

# Rolling window features (16 z-sources * 4 stats)
_ROLLING_BASES = [
    "rmssd_mean", "lf_mean", "hf_mean", "lf_hf_ratio",
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "wt_mean", "wt_std", "wt_min", "wt_max",
    "nightly_temperature", "resting_hr",
    "full_sleep_br", "deep_sleep_br",
]
FEAT_ROLLING = [
    f"{base}_{stat}"
    for base in _ROLLING_BASES
    for stat in ("rmean5", "rstd5", "rslope5", "dev5")
]

FEAT_PREV_CYCLE = ["prev_cycle_len", "prev_cycle_deviation"]

# Default: 23 features (v4: 22 ablation-validated + nightly_temperature_std_z)
ALL_FEATURES = (
    FEAT_CYCLE_PRIOR
    + FEAT_WEARABLE_Z
    + FEAT_RESPIRATORY_Z
    + ["nightly_temperature_std_z"]
)

# v5: 25 features (v4 default + prev_cycle features)
ALL_FEATURES_V5 = ALL_FEATURES + FEAT_PREV_CYCLE

# Legacy v3: 22 features without nightly_temperature_std_z
ALL_FEATURES_V3 = (
    FEAT_CYCLE_PRIOR
    + FEAT_WEARABLE_Z
    + FEAT_RESPIRATORY_Z
)

# Rolling window variant (70 = 6 prior + 64 rolling). Multi-seed eval
# showed no significant improvement over the 22-feature baseline.
ALL_FEATURES_ROLLING = (
    FEAT_CYCLE_PRIOR
    + FEAT_ROLLING
)

# Full 37 features (for ablation experiments only)
ALL_FEATURES_FULL = (
    FEAT_WEARABLE_Z
    + FEAT_RESPIRATORY_Z
    + FEAT_SLEEP_Z
    + FEAT_SYMPTOMS
    + FEAT_DELTAS
    + FEAT_SHIFTS
    + FEAT_CYCLE_PRIOR
)

# ── LightGBM hyperparameters ─────────────────────────────────────────────────

LGB_PARAMS = {
    "objective": "huber",
    "metric": "mae",
    "huber_delta": 3.0,
    "learning_rate": 0.03,
    "num_leaves": 20,
    "max_depth": 5,
    "min_child_samples": 30,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "min_split_gain": 0.05,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 80,
}

# ── Split ─────────────────────────────────────────────────────────────────────

TEST_SUBJECT_RATIO = 0.15
RANDOM_SEED = 42
MAX_CYCLE_LEN = 45

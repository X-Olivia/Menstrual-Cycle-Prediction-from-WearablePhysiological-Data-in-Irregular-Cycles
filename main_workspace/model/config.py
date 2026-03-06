"""model_v3 config: paths, feature groups, hyperparameters."""
import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data (v4 is default: per-cycle-early z-norm + frac fix + RHR median + boundary removal + temp_std)
FEATURES_CSV = os.path.join(WORKSPACE, "processed_data", "v4", "daily_features_v4.csv")
FEATURES_V3_CSV = os.path.join(WORKSPACE, "processed_data", "v3", "daily_features_v3.csv")
FEATURES_V4_CSV = FEATURES_CSV
FEATURES_V5_CSV = os.path.join(WORKSPACE, "processed_data", "v5", "daily_features_v5.csv")
FEATURES_V6_CSV = os.path.join(WORKSPACE, "processed_data", "v6", "daily_features_v6.csv")
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

FEAT_PHASE = [
    "estimated_phase",
    "days_since_estimated_ovulation",
    "is_luteal_estimate",
]

FEAT_TRANSITION = [
    "hr_mean_max_pos_5d", "hr_mean_max_neg_5d",
    "nightly_temperature_max_pos_5d", "nightly_temperature_max_neg_5d",
    "rmssd_mean_max_pos_5d", "rmssd_mean_max_neg_5d",
    "wt_mean_max_pos_5d", "wt_mean_max_neg_5d",
    "hr_mean_trend_reversal", "nightly_temperature_trend_reversal",
    "hr_temp_concordance",
]

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

# v6: 23 base + 3 phase + 11 transition = 37 features
ALL_FEATURES_V6 = ALL_FEATURES + FEAT_PHASE + FEAT_TRANSITION

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

# Optuna-tuned params (80 trials × 3 seeds, v4 features)
# 10-seed eval: MAE 3.291±0.569, ±3d 66.4%±7.5% (baseline: 3.337±0.562, 64.6%±6.4%)
LGB_PARAMS_TUNED = {
    "objective": "huber",
    "metric": "mae",
    "huber_delta": 3.9956,
    "learning_rate": 0.0682,
    "num_leaves": 44,
    "max_depth": 3,
    "min_child_samples": 44,
    "subsample": 0.8748,
    "colsample_bytree": 0.5012,
    "reg_alpha": 0.3204,
    "reg_lambda": 7.6198,
    "min_split_gain": 0.1753,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 80,
}

# Optuna-tuned params (80 trials × 3 seeds, v6 features)
# 10-seed eval: MAE 3.295±0.583, ±3d 65.4%±7.7%
LGB_PARAMS_TUNED_V6 = {
    "objective": "huber",
    "metric": "mae",
    "huber_delta": 3.5353,
    "learning_rate": 0.0545,
    "num_leaves": 30,
    "max_depth": 3,
    "min_child_samples": 49,
    "subsample": 0.9854,
    "colsample_bytree": 0.5298,
    "reg_alpha": 1.8001,
    "reg_lambda": 1.6175,
    "min_split_gain": 0.1160,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 80,
}

# v7: subdaily + luteal features
FEATURES_V7_CSV = os.path.join(WORKSPACE, "processed_data", "v7", "daily_features_v7.csv")

FEAT_SUBDAILY_Z = [
    "hr_nocturnal_nadir_z", "hr_nadir_timing_frac_z",
    "hr_onset_mean_z", "hr_wake_mean_z",
    "hr_onset_to_nadir_z", "hr_wake_surge_z",
    "hr_nocturnal_iqr_z", "hr_nocturnal_range_z",
    "hr_circadian_amplitude_z",
    "wt_nocturnal_plateau_z", "wt_rise_time_frac_z",
    "wt_nocturnal_auc_z", "wt_pre_wake_drop_z",
    "wt_nocturnal_range_sub_z", "wt_nocturnal_std_sub_z",
    "hrv_early_night_z", "hrv_late_night_z",
    "hrv_night_slope_z", "lf_hf_early_vs_late_z",
    "hrv_nocturnal_range_z",
]

FEAT_LUTEAL_PERSONAL = [
    "personal_avg_luteal_len",
    "personal_luteal_std",
    "temp_shift_detected",
    "days_since_temp_shift",
    "est_days_remaining_luteal",
]

ALL_FEATURES_V7_FULL = ALL_FEATURES + FEAT_SUBDAILY_Z + FEAT_LUTEAL_PERSONAL

# v7 best: v4 base + single luteal countdown (validated via 10-seed ablation)
ALL_FEATURES_V7 = ALL_FEATURES + ["est_days_remaining_luteal"]

# Optuna-tuned params (80 trials × 3 seeds, v7 = v4 + est_days_remaining_luteal)
# 10-seed eval: MAE 3.245±0.456, ±3d 65.6%±6.1% (baseline: 3.337±0.562, 64.6%±6.4%)
LGB_PARAMS_TUNED_V7 = {
    "objective": "huber",
    "metric": "mae",
    "huber_delta": 7.0804,
    "learning_rate": 0.1127,
    "num_leaves": 16,
    "max_depth": 4,
    "min_child_samples": 44,
    "subsample": 0.9257,
    "colsample_bytree": 0.3049,
    "reg_alpha": 0.4903,
    "reg_lambda": 3.3859,
    "min_split_gain": 0.1957,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 2000,
    "early_stopping_rounds": 80,
}

# ── Split ─────────────────────────────────────────────────────────────────────

TEST_SUBJECT_RATIO = 0.15
RANDOM_SEED = 42
MAX_CYCLE_LEN = 45

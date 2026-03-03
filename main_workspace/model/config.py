"""Experiment config: paths, feature columns, hyperparameters."""
import os

# Data paths (relative to main_workspace)
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CYCLE_CSV  = os.path.join(WORKSPACE, "subdataset", "cycle_clean_2.csv")
FULL_CSV   = os.path.join(WORKSPACE, "processed_data", "2", "full.csv")
# P2: 主要输入切换为 sleep 窗口（整夜睡眠期 HRV/HR/WT，与 Wang 2025 / Hamidovic 2023 对齐）
SLEEP_CSV  = os.path.join(WORKSPACE, "processed_data", "2", "sleep.csv")

# Input features v3: 14-dim z + P1 周期位置 (2) + P5 双相转折 (2) + P6 历史周期长度 (2) = 20 维
# day_in_cycle / day_in_cycle_frac: 不做 z（原值，周期位置先验）
# wt_shift_7v3 / temp_shift_7v3:   双相转折指标（°C 差分，不做 z）
# hist_cycle_len_mean / hist_cycle_len_std: 历史均值/标准差（天），不做 z，Wang 2025 核心特征
FEATURE_COLS = [
    # 原 14 维 z（来自 sleep.csv）
    "rmssd_mean_z", "lf_mean_z", "hf_mean_z", "lf_hf_ratio_z",
    "hr_mean_z", "hr_std_z", "hr_min_z", "hr_max_z",
    "wt_mean_z", "wt_std_z", "wt_min_z", "wt_max_z",
    "nightly_temperature_z", "resting_hr_z",
    # P1: 周期位置先验（不做 z）
    # day_in_cycle: 本周期内天序号（0-indexed），只编码「今天是第几天」，不泄露周期总长
    # day_in_cycle_frac: day_in_cycle / 28.0（固定28天先验归一化，与实际周期长度无关）
    # ⚠️ day_in_cycle_norm = day_in_cycle / actual_cycle_len 是数据泄露——禁止使用
    "day_in_cycle",
    "day_in_cycle_frac",
    # P5: 腕温/夜温双相转折衍生特征（°C 差分）
    "wt_shift_7v3",
    "temp_shift_7v3",
    # P6: 历史周期长度先验（Wang 2025 核心特征，无数据泄露：仅用当前周期之前的历史）
    # 第一个周期使用群体先验（~25天）；后续周期使用该被试的实际历史均值/标准差
    "hist_cycle_len_mean",
    "hist_cycle_len_std",
    # P6b: 先验剩余天数（直接可用的预测锚点，解决 21+ horizon 预测天花板问题）
    # days_remaining_prior     = hist_cycle_len_mean - day_in_cycle  (有符号，天)
    # days_remaining_prior_log = sign * log1p(|prior|)  与目标 log1p(days) 同量纲
    #   模型初始只需学习小修正 δ，而无需从原始特征从零计算倒计时
    "days_remaining_prior",
    "days_remaining_prior_log",
]
INPUT_DIM = len(FEATURE_COLS)

# Missing values: downstream fill for any remaining NaN after interpolation
FILL_MISSING = 0.0

# Model
HIDDEN_SIZE = 64
RNN_TYPE = "gru"        # "gru" | "lstm"
BIDIRECTIONAL = True    # Bidirectional backbone: lets model see HRV decline (before) AND temp rise (after)
DROPOUT = 0.3           # Dropout for regularization (critical: only 29 subjects, ~130 training cycles)

# Data quality
MAX_CYCLE_LEN = 45      # Filter cycles > 45 days (oligomenorrhea; only ~2 cycles in dataset, outlier MAE)

# Training
LAMBDA_OV        = 0.5   # Stage1: ovulation auxiliary weight
LAMBDA_OV_STAGE2 = 0.05  # Stage2: keep a small ovulation signal so backbone representation doesn't degrade
STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 50
LR_STAGE1 = 1e-3
LR_STAGE2_HEAD = 1e-3
LR_STAGE2_BACKBONE = 1e-5
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 20
GRAD_CLIP = 1.0  # Gradient clipping for RNN stability

# Ovulation class imbalance: ~90 positive days / 1391 total ≈ 6.5% positive rate
# pos_weight = (1 - pos_rate) / pos_rate ≈ 14.4; cap at 20 to avoid over-weighting noise
POS_WEIGHT_OV = 14.0

# Split (plan 2.4.1)
TEST_SUBJECT_RATIO = 0.15
N_FOLDS = 5
RANDOM_SEED = 42

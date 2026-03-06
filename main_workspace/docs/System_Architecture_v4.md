# 经期预测系统架构文档（v4 更新版）

## 目录

1. [数据处理](#1-数据处理)
2. [特征提取](#2-特征提取)
3. [模型架构](#3-模型架构)
4. [代码架构](#4-代码架构)
5. [实验记录](#5-实验记录)

---

## 1. 数据处理

数据处理分为四个阶段，输入为 mcPHASES 原始数据集（42 名受试者），输出为模型可用的日级特征表。

### 1.1 阶段一：周期清洗 (`cycle_clean.ipynb`)

**输入**：`hormones_and_selfreport.csv`（5659 行，42 人）

**处理流程**：

1. **列提取**：保留 `id`, `study_interval`, `day_in_study`, `phase`, `lh`, `estrogen`
2. **周期定义**：
   - 大组 = `id` + `study_interval`（62 组）
   - 小组（周期）= 以 `Menstrual` 阶段首次出现为起点，每次从非 Menstrual 回到 Menstrual 时递增周期号
   - 首次 Menstrual 之前的天标记为 `cycle0`
3. **LH / Estrogen 插补**：周期内前后相邻均值填补；若周期内连续缺失 > 5 天则整个周期丢弃（6 个周期被移除）
4. **清洗规则**（仅对 LH/Estrogen 完整的周期执行）：
   - 丢弃 `cycle0`
   - 丢弃周期天数 < 6 天
   - 丢弃周期内 `day_in_study` 缺失 > 1 天
5. **排卵概率标注**：
   - 根据 LH baseline（经期后 1-4 天均值）和 LH surge（LH/baseline ≥ 2.5）检测排卵窗口
   - 利用截断正态分布 + τ~U(7,24) 建模排卵日不确定性
   - 输出 `ovulation_prob_fused`（软标签）

**输出**：`subdataset/cycle_clean_2.csv`（4825 行，173 个周期，42 人）

### 1.2 阶段二：穿戴数据过滤 (`body_data_clean_2.ipynb`)

**输入**：`cycle_clean_2.csv`（日期锚点）+ mcPHASES 原始 5 张表

**处理**：以 `(id, study_interval, day_in_study)` 为 key，inner join 仅保留落在已清洗周期内的行。

| 源表                           | 原始行数   | 过滤后行数          |
| ------------------------------ | ---------- | ------------------- |
| resting_heart_rate             | 13,737     | 4,825（多值取均值） |
| heart_rate_variability_details | 436,262    | 376,056             |
| computed_temperature           | 5,575      | 4,623               |
| wrist_temperature              | 6,856,019  | 5,979,250           |
| heart_rate                     | 63,100,276 | 53,949,571          |

**输出**：`subdataset/2/*.csv`（5 张过滤后的穿戴数据文件）

### 1.3 阶段三：日级特征聚合 (`daily_data_2.ipynb`)

**输入**：`subdataset/2/*.csv` + `cycle_clean_2.csv`

**处理流程**：

1. **HRV 质量过滤**：保留 `coverage ≥ 0.6`, `rmssd > 0`, `lf > 0`, `hf > 0`
2. **时间窗口划分**：根据睡眠起止时间划分 sleep / morning / evening / full 四个窗口
3. **日级聚合**：
   - HRV → `rmssd_mean`, `lf_mean`, `hf_mean`, `lf_hf_ratio`
   - HR → `hr_mean`, `hr_std`, `hr_min`, `hr_max`（bpm 30-220, confidence ≥ 0.5）
   - WT → `wt_mean`, `wt_std`, `wt_min`, `wt_max`（±5°C 内）
   - 夜间温度 → `nightly_temperature`
   - 静息心率 → `resting_hr`
4. **双相转折特征**：`wt_shift_7v3` = 近3日均值 − 前3-9日均值
5. **周期位置**：`day_in_cycle`（0-indexed），`day_in_cycle_frac` = day_in_cycle / 28
6. **缺失处理**：线性插值（limit=3 天），剩余 NaN → 0，生成 `*_missing` 标志
7. **Z-normalization**：以周期前 5 天为基线（v2 实现）
8. **历史周期长度**：`hist_cycle_len_mean/std`（首周期使用群体先验 ~25 天），`days_remaining_prior` = hist_mean − day_in_cycle

**输出**：`processed_data/2/sleep.csv`（4825 行 × 54 列，sleep 窗口）

### 1.4 阶段四：特征管线 v4（当前默认，`build_features_v4.py`）

**输入**：

- `processed_data/2/sleep.csv`（14 维基础特征）
- mcPHASES 原始表：`respiratory_rate_summary.csv`, `sleep_score.csv`, `sleep.csv`, `hormones_and_selfreport.csv`, `computed_temperature.csv`, `resting_heart_rate.csv`
- `subdataset/cycle_clean_2.csv`（周期元数据）

**处理流程（10 步）**：

| 步骤 | 操作                                      | 说明                                                                                      |
| ---- | ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| ①   | 加载基础数据                              | 从 `sleep.csv` 恢复 NaN，清除旧 z-score                                                 |
| ②   | **RHR 重聚合**                      | 从原始 `resting_heart_rate.csv` 重新加载，**用 median 替代 mean**，过滤 value > 0 |
| ③   | 合并新数据源                              | 呼吸频率、睡眠架构、PMS 症状、**夜间温度标准差**                                    |
| ④   | **排除边界周期**                    | 移除每人每 study_interval 最后一个周期（62 个周期，1549 行）                              |
| ⑤   | **修复 day_in_cycle_frac**          | 从 `day/28` 改为 `day/hist_cycle_len_mean`（clip≥15）                                |
| ⑥   | 修复双相转折                              | within-cycle groupby 防止跨周期泄漏                                                       |
| ⑦   | 修复插值                                  | within-cycle groupby 防止跨周期边界插值                                                   |
| ⑧   | **Per-cycle-early z-normalization** | A/B/C 三类特征分别处理（详见 §2.1）                                                      |
| ⑨   | 变化率特征                                | 6 日差分 + 2 双相转折                                                                     |
| ⑩   | 组装输出                                  | 106 维特征（模型实际使用 23 维）                                                          |

**新增数据源**：

| 数据源             | 来源文件                            | 聚合方式                                                                 | 覆盖率  |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------ | ------- |
| 呼吸频率           | `respiratory_rate_summary.csv`    | 日均值 →`full_sleep_br`, `deep_sleep_br`                            | 100%    |
| 睡眠架构           | `sleep_score.csv` + `sleep.csv` | 日均值 →`sleep_score`, `deep_sleep_min`, `restlessness`           | 95-100% |
| PMS 症状           | `hormones_and_selfreport.csv`     | 序数编码(0-5) →`cramps`, `bloating`, `sorebreasts`, `moodswing` | 59%     |
| **温度波动** | `computed_temperature.csv`        | 日中位数 →`nightly_temperature_std`                                   | 98%     |
| **静息心率** | `resting_heart_rate.csv`（重载）  | **日中位数** → `resting_hr`（替换原 mean 聚合）                 | 85%     |

**输出**：`processed_data/v4/daily_features_v4.csv`（3276 行 × 106 维，模型使用 23 维）

---

## 2. 特征提取

### 2.1 Z-normalization 策略（v4）

v4 将特征分为三类，采用不同的归一化策略：

| 类别                          | 特征                                                                                                         | 处理方式                                                                                     | 理由                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **A类**：已基线化温度   | `wt_mean/std/min/max`                                                                                      | per-cycle 中心化（减前 5 天均值），**不除以 std**                                      | 来自 `temperature_diff_from_baseline`，避免双重归一化 |
| **B类**：绝对值生理信号 | `nightly_temperature`, HRV 4 项, HR 4 项, RRS 2 项, 睡眠 3 项, `resting_hr`, `nightly_temperature_std` | per-cycle-early-days z-norm（前 5 天 mean/std 为基线），覆盖率不足时 fallback 到 per-subject | 与 Wang 2025 / Hamidovic 2023 论文方法一致              |
| **C类**：非生理特征     | 症状 (4)、周期先验 (6)                                                                                       | 不归一化                                                                                     | 无需标准化                                              |

实际运行统计：96% 的周期使用了 per-cycle-early 基线（2130/2220），仅 4% 回退到 per-subject。

### 2.2 最终 23 维特征清单（当前默认）

| 组别                             | 维度 | 特征                                                                                                                                                                                                                             | 说明                                                    |
| -------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **A. 周期位置先验**        | 6    | `day_in_cycle`, `day_in_cycle_frac`, `hist_cycle_len_mean`, `hist_cycle_len_std`, `days_remaining_prior`, `days_remaining_prior_log`                                                                                 | `day_in_cycle_frac` = day / hist_mean（非固定 28 天） |
| **B. 穿戴设备 z-score**    | 14   | `rmssd_mean_z`, `lf_mean_z`, `hf_mean_z`, `lf_hf_ratio_z`, `hr_mean_z`, `hr_std_z`, `hr_min_z`, `hr_max_z`, `wt_mean_z`, `wt_std_z`, `wt_min_z`, `wt_max_z`, `nightly_temperature_z`, `resting_hr_z` | per-cycle-early z-norm                                  |
| **C. 呼吸频率 + 温度波动** | 3    | `full_sleep_br_z`, `deep_sleep_br_z`, `nightly_temperature_std_z`                                                                                                                                                          | 黄体期生理信号变化                                      |

### 2.3 消融移除的特征组

以下特征保留在管线输出中供后续实验，但默认不参与训练：

| 组别             | 维度 | 移除原因（基于 v3 消融实验） |
| ---------------- | ---- | ---------------------------- |
| 双相转折         | 2    | Test MAE +0.031              |
| 变化率 (日差分)  | 6    | Test MAE −0.006（不显著）   |
| 睡眠架构 z-score | 3    | Test MAE +0.008              |
| PMS 症状         | 4    | Test MAE +0.021              |

---

## 3. 模型架构

### 3.1 单阶段模型：LightGBM 日级回归

**任务定义**：给定当天的 23 维特征，预测 `days_until_next_menses`（下次经期开始的天数）。

**标签构造**：`days_until_next_menses = cycle_end + 1 − day_in_study`，其中 `cycle_end` 是该周期最后一天的 `day_in_study`。

**模型**：LightGBM (Gradient Boosted Decision Trees)

**损失函数**：Huber loss

- 误差 < δ 天：L2（梯度平滑，精确拟合）
- 误差 > δ 天：L1（抗极端值，不过度惩罚长 horizon 误差）

**超参数**（两套配置）：

| 参数                   | 原始 `LGB_PARAMS` | Optuna 调参 `LGB_PARAMS_TUNED` |
| ---------------------- | ------------------- | ------------------------------- |
| huber_delta            | 3.0                 | 3.9956                          |
| learning_rate          | 0.03                | 0.0682                          |
| num_leaves             | 20                  | 44                              |
| max_depth              | 5                   | 3                               |
| min_child_samples      | 30                  | 44                              |
| subsample              | 0.75                | 0.8748                          |
| colsample_bytree       | 0.75                | 0.5012                          |
| reg_alpha / reg_lambda | 0.5 / 3.0           | 0.3204 / 7.6198                 |
| min_split_gain         | 0.05                | 0.1753                          |
| early_stopping_rounds  | 80                  | 80                              |

**评估指标**：

- MAE（天数）
- ±1d / ±2d / ±3d 准确率（±3d 定义：绝对误差 < 3.5 天）
- 按 horizon 分层评估：1-5, 6-10, 11-15, 16-20, 21+ 天

**数据分割**：Subject-level split（15% test, 剩余 80/20 train/val），防止同一受试者的数据泄漏到训练和测试集。

**当前性能**（v4 数据，10-seed 评估）：

| 配置          | Test MAE        | Test ±3d Acc    | Val MAE         |
| ------------- | --------------- | --------------- | --------------- |
| 原始超参      | 3.337 ± 0.562   | 64.6% ± 6.4%   | 4.027 ± 0.840   |
| **Optuna 调参** | **3.289 ± 0.572** | **66.5% ± 7.6%** | **3.940 ± 0.836** |

分层 MAE（Optuna 调参）：1-5d: 3.04, 6-10d: 2.62, 11-15d: 3.00, 16-20d: 3.12, 21+d: 3.97

---

## 4. 代码架构

### 4.1 目录结构

```
main_workspace/
├── data_process/                      # 数据处理管线
│   ├── cycle_clean.ipynb              # 阶段一：周期清洗
│   ├── body_data_clean_2.ipynb        # 阶段二：穿戴数据过滤
│   ├── daily_data_2.ipynb             # 阶段三：日级聚合
│   ├── build_features_v4.py           # 阶段四：特征管线 v4（当前默认）
│   ├── build_features_v7.py           # v7 特征管线（v4 + 子日特征 + 黄体特征，实验性）
│   └── build_subdaily_features.py     # 从高频 HR/WT/HRV 提取夜间特征（v7 依赖）
│
├── model/                             # 模型代码（原 model_v3，已重命名）
│   ├── __init__.py
│   ├── __main__.py                    # python -m model 入口
│   ├── config.py                      # 路径、特征组、超参数配置（含 Optuna 调参结果）
│   ├── dataset.py                     # 数据加载 + 标签构造 + Subject/LLCO 分割
│   ├── train_lgb.py                   # LightGBM 训练 + 预测 + 特征重要性
│   ├── evaluate.py                    # MAE / ±kd / 分层评估
│   ├── run_experiment.py              # 单次实验 + 消融实验
│   ├── robust_eval.py                 # 多种子稳健评估（Subject Split / LLCO）
│   └── tune_optuna.py                 # Optuna 超参数自动调优
│
├── archive/                           # 归档（不再使用的实验代码）
│   ├── data_process/                  # build_features_v3/v5/v6.py 等
│   └── model/                         # A/B 对比、残差实验、GRU、两阶段等
│
├── subdataset/
│   ├── cycle_clean_2.csv              # 阶段一输出（4825 行，173 周期）
│   └── 2/                             # 阶段二输出
│
├── processed_data/
│   ├── 2/sleep.csv                    # 阶段三输出（4825 行 × 54 列）
│   └── v4/daily_features_v4.csv       # v4 管线输出（当前默认，3276 行 × 106 维）
│
└── mcphases-.../                      # mcPHASES 原始数据集（.gitignore）
```

### 4.2 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: 数据预处理（Notebook，手动执行一次）                    │
│                                                             │
│  hormones_and_selfreport.csv                                │
│           │                                                 │
│           ▼                                                 │
│  [cycle_clean.ipynb]                                        │
│  周期定义 → LH/Estrogen 插补 → 清洗规则 → 排卵概率           │
│           │                                                 │
│           ▼                                                 │
│  cycle_clean_2.csv (4825 行, 173 周期, 42 人)                │
│           │                                                 │
│      ┌────┴────┐                                            │
│      ▼         ▼                                            │
│  [body_data_clean_2]        [daily_data_2]                  │
│  5表 inner join 过滤         HRV/HR/WT 日级聚合              │
│      │                           │                          │
│      ▼                           ▼                          │
│  subdataset/2/*.csv         processed_data/2/sleep.csv      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: 特征管线 v4（脚本，可重复执行）                        │
│                                                             │
│  python data_process/build_features_v4.py                   │
│                                                             │
│  sleep.csv + 6 张 mcPHASES 原始表 + cycle_clean_2.csv       │
│           │                                                 │
│           ▼                                                 │
│  ① 加载基础 14 维 + 恢复 NaN                                 │
│  ② RHR 重聚合（median）                                      │
│  ③ 合并新数据源（RRS/睡眠/症状/温度波动）                      │
│  ④ 排除边界周期（-62 周期, -1549 行）                         │
│  ⑤ 修复 day_in_cycle_frac（基于 hist_mean）                  │
│  ⑥ 修复双相转折 + 插值（within-cycle）                        │
│  ⑦ Per-cycle-early z-normalization (A/B/C 三类)             │
│  ⑧ 滚动窗口 + 变化率特征                                     │
│  ⑨ 组装输出                                                  │
│           │                                                 │
│           ▼                                                 │
│  daily_features_v4.csv (3276 行 × 106 维)                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: 模型训练与评估                                       │
│                                                             │
│  daily_features_v4.csv + cycle_clean_2.csv                  │
│           │                                                 │
│           ▼                                                 │
│  [dataset.py]                                               │
│  加载特征 → 构造标签 days_until_next_menses                  │
│  → 过滤 >45 天周期 → Subject-level split                    │
│           │                                                 │
│           ▼                                                 │
│  [train_lgb.py]                                             │
│  23 维 → LightGBM (Huber loss + early stopping)             │
│  支持传入自定义超参 (params=LGB_PARAMS_TUNED)                │
│           │                                                 │
│           ▼                                                 │
│  [evaluate.py]                                              │
│  MAE / ±1d / ±2d / ±3d / 按 horizon 分层                    │
│                                                             │
│  [robust_eval.py] ← 推荐入口                                │
│  10-seed 多种子评估，消除小样本方差                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 运行命令

所有命令均在 `main_workspace/` 目录下执行。

#### 数据处理

```bash
cd main_workspace

# 阶段一~三：在 Jupyter 中依次运行 notebook（仅首次需要）
# 阶段四：特征管线 v4（当前默认，可重复执行）
conda run -n menstrual python data_process/build_features_v4.py
```

#### 模型训练与评估

提供三种运行模式：

| 命令                                       | 运行次数                  | 适用场景                                 |
| ------------------------------------------ | ------------------------- | ---------------------------------------- |
| `python -m model`                        | 1 次（固定 seed=42 分组） | 快速验证，看单次结果                     |
| `run_multi_seed(10, params=...)`         | 10 次（每次不同随机分组） | **推荐**，消除小样本方差，结果可靠 |
| `run_ablation()`                         | ~6 次（逐步添加特征组）   | 分析各特征组贡献                         |

**为什么推荐 `run_multi_seed`**：数据集仅 40 人，单次实验中 6 人进入测试集，随机分到哪些人对结果影响很大（MAE 波动 ±0.5-1 天）。多种子跑 10 次取均值后结果更稳定。

```bash
cd main_workspace

# ① 快速查看单次结果（使用默认超参）
conda run -n menstrual python -m model

# ② 多种子稳健评估 —— 默认超参（MAE 3.337, ±3d 64.6%）
conda run -n menstrual python -c "
from model.robust_eval import run_multi_seed
run_multi_seed(10)
"

# ③ 多种子稳健评估 —— Optuna 调参超参（MAE 3.289, ±3d 66.5%，推荐）
conda run -n menstrual python -c "
from model.robust_eval import run_multi_seed
from model.config import LGB_PARAMS_TUNED
run_multi_seed(10, params=LGB_PARAMS_TUNED)
"

# ④ 消融实验（逐步添加特征组，分析各组贡献）
conda run -n menstrual python -c "
from model.run_experiment import run_ablation
run_ablation()
"

# ⑤ Optuna 超参数自动调优（80 trials × 3 seeds，约 15-30 分钟）
conda run -n menstrual python -c "
from model.tune_optuna import tune_and_evaluate
tune_and_evaluate(n_trials=80)
"
```

### 4.4 各模块职责

**`model/` 目录（10 个文件）**：

| 文件                     | 职责                                              | 核心函数 / 接口                                                                |
| ------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------ |
| `config.py`            | 路径、特征列表、超参数（含 Optuna 调参结果）      | `ALL_FEATURES`(23维), `LGB_PARAMS`, `LGB_PARAMS_TUNED`                  |
| `dataset.py`           | 数据加载、标签构造、Subject/LLCO 分割             | `load_data(features_csv=)`, `subject_split()`, `cycle_split()`         |
| `train_lgb.py`         | LightGBM 训练逻辑                                 | `train_lightgbm(params=)`, `predict()`, `feature_importance()`         |
| `evaluate.py`          | 评估指标计算与格式化输出                          | `compute_metrics()`, `stratified_metrics()`, `print_metrics()`         |
| `run_experiment.py`    | 单次实验与消融实验                                | `run_experiment(feature_list=, features_csv=)`, `run_ablation()`         |
| `robust_eval.py`       | 多种子评估减少小样本方差                          | `run_multi_seed(n_seeds=, params=)`, `run_multi_seed_llco(params=)`      |
| `tune_optuna.py`       | Optuna 超参数自动调优 + 评估                      | `tune(n_trials=)`, `tune_and_evaluate()`                                   |

**`data_process/` 目录（6 个文件）**：

| 文件                         | 职责                                              | 核心函数 / 接口                |
| ---------------------------- | ------------------------------------------------- | ------------------------------ |
| `build_features_v4.py`    | 特征管线 v4：5 项修复 + per-cycle-early z-norm    | `main()` → 10 步顺序执行    |
| `build_features_v7.py`    | v7 管线：v4 + 子日特征 + 黄体特征（实验性）       | `main()`                     |
| `build_subdaily_features.py` | 从高频 HR/WT/HRV 提取夜间特征                   | `process_hr/wt/hrv_features()` |

**`archive/` 归档目录（13 个文件）**：

| 子目录           | 文件                                           | 归档原因                              |
| ---------------- | ---------------------------------------------- | ------------------------------------- |
| `data_process/` | `build_features_v3/v5/v6.py`                 | 被 v4 取代或实验未产生改善            |
| `data_process/` | `baseline_ovulation_probe.py`                | 依赖已删除的 `split.py`，无法运行   |
| `model/`        | `run_ab_v3v4.py`, `run_ab_v4_fixes/rolling.py` | 导入已不存在的 `model_v3` 包        |
| `model/`        | `compare_all.py`, `residual_experiment.py`   | 一次性对比 / 实验已验证无效           |
| `model/`        | `two_stage.py`                               | 两阶段实验                            |
| `model/`        | `seq_model.py`, `seq_dataset.py`, `run_seq_experiment.py` | GRU 序列模型实验         |

### 4.5 数据规模概要

| 阶段            | 行数   | 受试者 | 周期数 | 说明                         |
| --------------- | ------ | ------ | ------ | ---------------------------- |
| mcPHASES 原始   | 5,659  | 42     | ~241   | 含 cycle0 和不完整周期       |
| cycle_clean_2   | 4,825  | 42     | 173    | 清洗后                       |
| v4 边界周期移除 | 3,276  | 40     | 111    | 排除每人每 interval 最后周期 |
| 模型最终使用    | ~3,100 | 40     | ~108   | 过滤 >45 天周期后            |

---

## 5. 实验记录

### 5.1 v3 → v4 → Optuna 改进历程

| 版本                        | 改动                                           | Test MAE              | Test ±3d Acc         | 说明                        |
| --------------------------- | ---------------------------------------------- | --------------------- | -------------------- | --------------------------- |
| v3 baseline                 | per-subject z-norm, 22 feat, 固定 28 天 frac   | 4.295 ± 0.609         | 46.1%                | 原始基线                    |
| v4 归一化实验               | per-cycle-early z-norm                         | 4.377 ± 0.506         | 46.1%                | 仅改 z-norm，无显著改善     |
| v4 + 4 fixes                | + frac 修复 + RHR median + 边界移除 + temp_std | 3.337 ± 0.562         | 64.6% ± 6.4%        | MAE -22%                    |
| **v4 + 4 fixes + Optuna** | + 超参数自动调优 (80 trials × 3 seeds)         | **3.289 ± 0.572** | **66.5% ± 7.6%** | **当前最佳，MAE -23.4%** |

### 5.2 v5/v6/v7 实验记录（均未超越 v4 + Optuna，已归档）

| 版本 | 改动                                                       | Test MAE         | Test ±3d      | 结论                                    |
| ---- | ---------------------------------------------------------- | ---------------- | ------------- | --------------------------------------- |
| v5   | v4 + prev_cycle_len / prev_cycle_deviation                 | ~3.33            | ~64%          | 无显著改善                              |
| v6   | v4 + 3 相位估计 + 11 转变特征 (共 37 维)                   | 3.295 ± 0.583   | 65.4%         | 特征过多导致过拟合                      |
| v7 full | v4 + 20 子日特征 + 5 黄体特征 (共 48 维)                | 3.481            | ~60%          | 严重过拟合，反而变差                    |
| v7 slim | v4 + est_days_remaining_luteal (共 24 维)               | 3.245 ± 0.456   | 65.6%         | MAE 略降但 ±3d 降低，且**存在数据泄漏** |

**v7 数据泄漏问题**：
- `personal_avg_luteal_len`：使用了全部周期（含测试周期）的 cycle_end 计算黄体期长度
- `detect_temp_shift`：回溯标注温度转变日，使周期早期天数看到了"未来"温度信息
- 因此 v7 的 MAE 改善不可信。**当前推荐使用 v4 + Optuna 调参作为基线**

### 5.3 v4 数据质量修复详情

基于 [mcPHASES 官方 GitHub](https://github.com/chai-toronto/mcphases) 示例代码分析，发现并修复了以下问题：

1. **`day_in_cycle_frac`**（feature importance 第 2）：原实现 `day/28` 对长周期（如 35 天）在 day 28 时 frac=1.0，但实际周期才完成 80%。修复为 `day/hist_cycle_len_mean`。
2. **RHR 聚合**：官方代码使用 median，我们原用 mean。Median 对异常值更鲁棒。
3. **边界周期**：每人每 study_interval 最后一个周期可能被研究截断而不完整，官方代码也排除首尾周期。
4. **`nightly_temperature_std`**：来自 `computed_temperature.csv` 的 `baseline_relative_nightly_standard_deviation`，反映夜间温度波动程度。

### 5.4 分层 MAE 对比

| Horizon  | v3 MAE | v4 MAE | v4+Optuna MAE | 改善 (v3→v4+Optuna) |
| -------- | ------ | ------ | ------------- | -------------------- |
| 1-5 天   | 4.396  | 3.151  | 3.038         | -31%                 |
| 6-10 天  | 3.907  | 2.819  | 2.616         | -33%                 |
| 11-15 天 | 3.560  | 3.001  | 3.002         | -16%                 |
| 16-20 天 | 3.573  | 3.120  | 3.123         | -13%                 |
| 21+ 天   | 5.228  | 3.942  | 3.965         | -24%                 |

### 5.5 消融实验结果（基于 v3 数据）

详见 `docs/Ablation_Study_Results.md`。核心结论：

- `day_in_cycle` + `hist_cycle_len_mean/std` 占据 feature importance 前 4 名
- 穿戴设备最强信号：`resting_hr_z`（第 5）、`deep_sleep_br_z`（第 8）、`lf_mean_z`（第 9）
- 睡眠架构、PMS 症状、变化率、双相转折在当前数据规模下无正向贡献

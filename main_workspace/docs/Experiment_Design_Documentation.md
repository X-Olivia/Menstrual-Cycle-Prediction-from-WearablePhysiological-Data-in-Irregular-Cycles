# 实验设计文档 — model/experiment/ 全部实验详解

## Experiment Design Documentation — Detailed Design of All Experiments

---

## 目录 / Table of Contents

1. [run_experiment.py — LightGBM 基础实验](#1-run_experimentpy) *(已归档至 archive/model/experiment/)*
2. [robust_eval.py — 多种子鲁棒性评估](#2-robust_evalpy)
3. [tune_optuna.py — Optuna 超参数调优](#3-tune_optunapy) *(已归档至 archive/model/experiment/)*
4. [run_oracle_experiment.py — Oracle 上界实验](#4-run_oracle_experimentpy)
5. [run_detected_ov_experiment.py — 排卵检测混合模型实验](#5-run_detected_ov_experimentpy)
6. [run_highfreq_temp_experiment.py — 高频温度排卵检测实验](#6-run_highfreq_temp_experimentpy) *(已归档)*
7. [run_ovulation_experiments.py — 排卵检测综合对比实验](#7-run_ovulation_experimentspy) *(已归档)*
8. [run_leakage_check.py — 数据泄漏审计实验](#8-run_leakage_checkpy) *(已归档，脚本在 archive/model/experiment/)*
9. [ovulation_cnn.py — 1D-CNN 排卵回归模型实验](#9-ovulation_cnnpy-位于-model)

---

## 1. run_experiment.py

### LightGBM 端到端基础实验 / LightGBM End-to-End Baseline Experiment

**目的**: 建立月经周期预测的基线模型，作为后续所有实验的参照。

**实验设计**:

- **任务**: 回归预测 `days_until_next_menses`（距离下次月经的天数）
- **模型**: LightGBM (Huber loss)
- **特征**: `config.py` 中定义的 `ALL_FEATURES`，包括周期先验 (cycle prior)、可穿戴信号 z-score、变化量 (shifts/deltas)、呼吸/睡眠特征、症状特征等
- **数据划分**: 按被试 (subject) 三分 — 训练集 / 验证集 / 测试集，比例由 `TEST_SUBJECT_RATIO` 控制，固定随机种子
- **评估指标**: MAE、±1d/±2d/±3d 准确率、分层评估 (按 horizon 分组)

**关键函数**:

| 函数 | 功能 |
|------|------|
| `run_experiment()` | 完整流程：加载→划分→训练→评估→特征重要性 |
| `run_ablation()` | 消融实验：逐步添加特征组，观察边际贡献 |

**消融实验特征组递增顺序**:
1. Prior only → 2. + Wearable → 3. + Shifts → 4. + Deltas → 5. + Respiratory → 6. + Sleep → 7. + Symptoms (ALL)

**运行**: `python -m model.experiment.run_experiment`

---

## 2. robust_eval.py

### 多种子鲁棒性评估 / Multi-Seed Robust Evaluation

**目的**: 消除单次随机划分带来的方差，获得更可靠的性能估计。

**实验设计**:

该文件包含三种评估方案，复杂度递增：

### 方案 A: run_multi_seed (Subject Split)

- **划分方式**: 按被试随机划分 (Subject-level split)
- **重复次数**: n_seeds=10（seed 42~51）
- **评估**: 每个 seed 独立训练+测试，汇报 MAE 和 ±3d 的 均值±标准差
- **分层评估**: 按 horizon 分桶（1-5、6-10、11-15、16-20、21+ 天）

### 方案 B: run_multi_seed_llco (Leave-Last-Cycle-Out)

- **划分方式**: Leave-Last-Cycle-Out — 每个被试的最后一个周期作为测试集
- **动机**: 模拟真实使用场景（用历史周期预测最新周期）
- **重复次数**: n_seeds=10

### 方案 C: run_multi_seed_llco_bias (LLCO + 个体偏差校正)

- **划分方式**: 同方案B (LLCO)
- **额外步骤**: 训练后，计算每个被试在训练集上的残差均值（`subject_bias`），在测试时加回该偏差
  ```
  pred_test[uid] += mean(y_train[uid] - pred_train[uid])
  ```
- **动机**: 探索个性化校准能否降低系统性偏差

**运行**: `python -m model.experiment.robust_eval`

---

## 3. tune_optuna.py

### Optuna 超参数调优 / Optuna Hyperparameter Optimization

**目的**: 自动搜索 LightGBM 最优超参数组合。

**实验设计**:

- **搜索框架**: Optuna (Tree-structured Parzen Estimator)
- **搜索空间**:
  | 超参数 | 范围 |
  |--------|------|
  | huber_delta | [1.0, 5.0] |
  | learning_rate | [0.01, 0.1] (log) |
  | num_leaves | [8, 48] |
  | max_depth | [3, 8] |
  | min_child_samples | [10, 60] |
  | subsample | [0.6, 1.0] |
  | colsample_bytree | [0.5, 1.0] |
  | reg_alpha | [0.0, 2.0] |
  | reg_lambda | [0.5, 10.0] |
  | min_split_gain | [0.0, 0.2] |

- **目标函数**: `mean(val_MAE)` over n_seeds=3 subject splits
- **early stopping**: 80 rounds，最多 2000 boost rounds
- **试验次数**: 默认 80 trials
- **输出**: 自动输出最优参数的 `LGB_PARAMS_TUNED` dict，可直接粘贴到 `config.py`

**关键函数**:

| 函数 | 功能 |
|------|------|
| `tune()` | 纯调参，输出最优参数 |
| `tune_and_evaluate()` | 调参后，用最优参数做 10-seed robust eval |

**运行**: `python -m model.experiment.tune_optuna`

---

## 4. run_oracle_experiment.py

### Oracle 上界实验 / Oracle Upper-Bound Experiment

**目的**: 测量"如果我们能完美检测排卵日"，月经预测性能能达到什么水平。建立理论天花板。

**实验设计**:

- **假设**: 使用 LH 检测试纸标注的排卵日作为完美 ground truth（Oracle）
- **两阶段混合模型**:
  1. **排卵前** (day_in_cycle < ov_day + 2): 使用 LightGBM 预测
  2. **排卵后** (day_in_cycle ≥ ov_day + 2): 使用黄体期倒计时
     ```
     pred = personal_luteal_mean - (day_in_cycle - ov_day)
     ```
- **黄体期长度**: 从 LH 标签计算每个被试的个人平均黄体期（8~20天范围内的有效周期）
- **+2天延迟**: 模拟真实场景中排卵检测存在的最低确认延迟
- **评估**: 10-seed GroupShuffleSplit (15% test)

**关键结果**:

| 模型 | MAE | ±3d |
|------|-----|------|
| LightGBM only | ~3.55 | ~60.6% |
| LightGBM + Oracle 倒计时 | ~3.02 | ~68.7% |
| 排卵后倒计时阶段单独 | ~1.14 | ~93.9% |

**意义**: 证明排卵检测对月经预测具有巨大潜力（+8pp ±3d），且排卵后倒计时本身非常精确。后续所有排卵检测实验的目标就是逼近这个 Oracle 上界。

**运行**: `python -m model.experiment.run_oracle_experiment`

---

## 5. run_detected_ov_experiment.py

### 排卵检测混合模型实验 / Detected-Ovulation Hybrid Experiment

**目的**: 用真实的可穿戴信号（而非 LH ground truth）自动检测排卵，插入混合预测模型，测量实际收益。

**实验设计（四阶段流水线）**:

### Stage 1: 可穿戴信号加载与日级聚合

从 5 个数据源加载并合并日级特征:
- `computed_temperature_cycle.csv` → `nightly_temperature`
- `resting_heart_rate_cycle.csv` → `resting_hr`
- `heart_rate_cycle.csv` → `hr_mean`, `hr_std`, `hr_min`
- `heart_rate_variability_details_cycle.csv` → `rmssd_mean`, `lf_mean`, `hf_mean`
- `wrist_temperature_cycle.csv` → `wt_mean`, `wt_max`

### Stage 2: 因果特征工程

对 9 个原始信号各生成 5 个因果滚动特征（共 45 + 1 = 46 个特征）:
- `_rm3`: 3日滚动均值
- `_rm7`: 7日滚动均值
- `_svl`: 短期 vs 长期差 (rm3 - rm7)
- `_d1`: 1日差分
- `_d3`: 3日差分
- `cycle_frac`: 周期位置先验 (day_in_cycle / hist_cycle_len)

### Stage 3: LOSO 排卵分类器

- **模型**: GradientBoostingClassifier (n_estimators=150, max_depth=3)
- **目标**: 二分类 — P(当前日属于排卵后)
- **交叉验证**: Leave-One-Subject-Out (被试级别)
- **检测策略**: 三种将概率转换为排卵日的规则:
  | 策略 | 逻辑 |
  |------|------|
  | threshold | 连续2天 prob > 0.5 → 排卵日 = 当天 - 1 |
  | cumulative | 累积得分 (decay=0.85, trigger=1.5) 超过阈值 → 排卵日 |
  | bayesian | 结合周期位置先验 × 后验概率 > 0.4 → 排卵日 |

### Stage 4: 混合模型评估

- **基线**: 纯 LightGBM
- **检测混合**: LightGBM + 检测到的排卵日倒计时
- **Oracle 混合**: LightGBM + LH ground truth 倒计时
- **软融合**: `pred = (1-w) × lgb + w × countdown`，权重 w 由排卵概率动态决定
- **评估**: 10-seed GroupShuffleSplit

**运行**: `python -m model.experiment.run_detected_ov_experiment`

---

## 6. run_highfreq_temp_experiment.py

### 高频温度排卵检测实验 / High-Frequency Temperature Ovulation Detection

**目的**: 利用分钟级原始腕温数据提取更精细的特征，对比日级聚合数据的排卵检测效果。

**核心假设**: 分钟级数据 → 更低噪声 → 更高 SNR → 更好的排卵检测。

**实验设计**:

### 高频特征提取 (`extract_daily_highfreq_features`)

从分钟级腕温 (`wrist_temperature_cycle.csv`) 中为每天提取 18 个特征:

| 类别 | 特征 | 说明 |
|------|------|------|
| 基础统计 | temp_hf_mean, std, min, max, range, median | 全天分钟级温度统计 |
| 夜间窗口 | night_mean, min, max, std, range | 0:00-6:00 窗口 |
| 斜率 | rise_slope, drop_slope | 前/后 120 分钟线性拟合斜率 |
| 稳定窗口 | stable_mean, stable_std, stable_iqr | 2:00-5:00 最稳定时段 |
| 分布 | half_diff, p10, p90, p90_p10 | 前后半差异、分位数差 |

### 三组对比实验

| 实验 | 信号源 | 特征数 |
|------|--------|--------|
| [A] Baseline | 日级聚合信号 (10个原始信号 × 5个滚动特征 + cycle_frac) | 51 |
| [B] HF only | 高频温度特征 (12个 × 5 + cycle_frac) | 61 |
| [C] Combined | HF温度 + HR/HRV (12+7 = 19个 × 5 + cycle_frac) | 96 |

### 检测流水线

每组均经过相同流程:
1. 因果滚动特征构建 (rm3, rm7, svl, d1, d3)
2. LOSO GradientBoostingClassifier (n_estimators=200, max_depth=4)
3. 三策略排卵检测 (threshold / cumulative / bayesian)
4. 月经预测 (10-seed 日历法 vs 检测倒计时 vs Oracle 倒计时)

**运行**: `python -m model.experiment.run_highfreq_temp_experiment`

---

## 7. run_ovulation_experiments.py

### 排卵检测综合对比实验 / Comprehensive Ovulation Detection Suite

**目的**: 系统性测试两大路径（规则法 + ML法）共 10+ 种排卵检测算法，在多种参数配置和数据源下的表现。

**这是最大的实验文件（1217行），包含以下所有子实验:**

### PATH 1: 规则法 / Rule-based Methods

#### Exp 1a: ruptures 变点检测

- **算法**: PELT / BinSeg / DynP (from `ruptures` library)
- **原理**: 寻找温度时间序列中最显著的水平变化点，该点即为排卵引发的升温开始
- **参数扫描**:
  - method: pelt(pen=2,3,5), binseg, dynp
  - 数据源: agg (日级聚合), noct (夜间均值)
- **后处理**: 在所有变点中选择前后温差最大且 post > pre 的变点
- **实验数**: 5 method × 2 data = 10 组

#### Exp 1b: CUSUM (累积和控制图)

- **算法**: 单侧 CUSUM 检测
- **原理**: 对温度序列 z-标准化后，计算累积正向偏移量 `S(t) = max(0, S(t-1) + z(t) - drift)`，当 S(t) 超过阈值即触发
- **参数扫描**:
  - threshold: 0.3, 0.5, 1.0, 1.5
  - drift: 0.02, 0.05, 0.1
  - 数据源: agg, noct
- **回溯**: 触发后回溯找 S < 0.3×threshold 的位置作为实际排卵点
- **实验数**: 4 threshold × 3 drift × 2 data = 24 组

#### Exp 1c: BOCPD (贝叶斯在线变点检测)

- **算法**: Bayesian Online Changepoint Detection
- **原理**: 维护运行长度 (run length) 的后验分布，当后验突然 reset 时检测到变点
- **参数扫描**:
  - hazard_rate: 1/15, 1/20, 1/30
  - 数据源: agg, noct
- **先验**: Normal-Inverse-Gamma
- **实验数**: 3 hazard × 2 data = 6 组

#### Exp 1d: 增强型 3-over-6 Coverline

- **算法**: 经典排卵检测规则的增强版
- **原理**: 对每天计算前 `baseline_days` 天均值 + shift_threshold 作为 coverline，若连续 `confirm_days` 天都在 coverline 之上，则检测到排卵
- **参数扫描**:
  - shift_threshold: 0.05, 0.1, 0.15, 0.2 (°C)
  - confirm_days: 2, 3, 4
  - 数据源: agg, noct
- **约束**: day_in_cycle ≥ 8 (排除早期误检)
- **实验数**: 4 threshold × 3 confirm × 2 data = 24 组

### PATH 2: ML / 数据驱动方法

#### Exp 2a: 2-state HMM

- **算法**: GaussianHMM (hmmlearn)
- **原理**: 假设月经周期由两个隐状态组成 — 低温状态 (卵泡期) 和高温状态 (黄体期)，HMM 自动推断状态转移点
- **初始化**: 强先验 — startprob=[0.9, 0.1], transmat=[[0.95, 0.05], [0.02, 0.98]] (倾向长期停留在同一状态)
- **变体**:
  - 单信号温度 (agg / noct)
  - 多信号 (temp + HR + RMSSD)

#### Exp 2b: 增强型 GBDT

- **算法**: GradientBoostingClassifier (n_estimators=300, max_depth=4)
- **特征**: 从分钟级数据提取的夜间增强特征:
  - noct_stable_mean/std: 2:00-5:00 稳定窗口
  - noct_nadir/nadir_time: 体温最低点及时间
  - noct_rise_slope: 上升斜率
  - noct_range, p90, p10, iqr: 分布特征
  - noct_mean_2h, noct_mean_4h: 2/4小时窗口均值
  - (可选) + nightly_temperature, HR, HRV 夜间信号
- **交叉验证**: LOSO
- **检测策略**: threshold / cumulative / bayesian

#### Exp 2c: 1D-CNN（日级分类版）

- **架构**: Conv1d(1→32, k=15) → MaxPool(4) → Conv1d(32→64, k=7) → MaxPool(4) → Conv1d(64→64, k=5) → AdaptiveAvgPool → FC(512→64→1)
- **输入**: 单日 480 分钟夜间温度序列 (22:00-06:00)，edge-padding 至固定长度
- **目标**: 二分类 — 该日是否在排卵后 (BCEWithLogitsLoss)
- **训练**: LOSO，每个被试独立测试，30 epochs
- **检测**: 将逐日概率送入 threshold/cumulative/bayesian 策略得到排卵日

#### Exp 2d: 多信号 HMM

- **信号**: nightly_temperature + hr_mean + rmssd_mean
- **其余同 Exp 2a**

### 集成方法 / Ensemble

- **加权中位数**: 取 top-3 方法的检测结果，以各自 ±3d 准确率为权重做加权平均
- **Top-5 均值**: 取 top-5 方法的简单平均
- **多数投票**: 取 top-5 中 ≥3 方法有结果的，取中位数

### 质量过滤评估

- **过滤条件**: 只保留温度 shift ≥ 0.2°C 的周期（排卵后5天均值 - 排卵前5天均值 ≥ 0.2°C）
- **动机**: 并非所有周期都有可检测的温度升高，过滤后评估方法在"有信号"周期上的真实表现

### 总结输出

按 ±3d 准确率排序输出所有方法的完整评估表（Method / Recall / MAE / ±1d~±5d）

**运行**: `python -m model.experiment.run_ovulation_experiments`

---

## 8. run_leakage_check.py

### 数据泄漏审计实验 / Data Leakage & Baseline Audit

**目的**: 系统性验证 1D-CNN 排卵检测模型的高准确率是否源于真正的温度模式学习，还是统计捷径利用。

**背景**: 1D-CNN 模型报告 ~92% ±3d 准确率，但需排除以下泄漏源。

### Check 1: 标签分布分析

- **检查内容**: 目标变量 `ov_frac = ov_day / cycle_len` 的分布是否过于集中
- **方法**: 计算 ov_frac 统计量 + 常数预测基线
- **预期发现**: 若 std 极小，常数输出即可达高准确率（天花板效应）

### Check 2: LOSO 划分正确性

- **检查内容**: 测试被试的周期是否出现在其训练集中
- **方法**: 遍历每个 fold，验证 `test_id not in train_ids`
- **预期**: PASS（不存在被试级泄漏）

### Check 3: Z-归一化范围

- **检查内容**: per-cycle z-norm 是否泄漏了排卵后的信息
- **分析**: 回测场景下使用全周期 z-norm 是合理的（整个周期已完成），但因果场景需改用 expanding z-norm
- **结论**: 分析性判断，不涉及模型训练

### Check 4: 目标变量与周期长度代理

- **检查内容**: 
  1. `ov_day` 与 `cycle_len` 的 Pearson 相关性
  2. 线性代理 `0.575 × cycle_len` 的准确率
  3. CNN 能否通过 zero-padding 推断 cycle_len
- **方法**: 统计相关性 + 计算 baseline 指标

### Check 5: 打乱温度消融

- **实验**: 随机打乱每个周期内的温度值（破坏时序模式），保留 zero-padding 结构
- **操作**: `np.random.shuffle(seq[:cycle_len])`
- **训练**: 与原始 CNN 完全相同的 LOSO 训练流程（60 epochs, lr=3e-4）
- **预期**: 若模型主要利用温度时序模式，准确率应大幅下降

### Check 6: 噪声消融（三组对照）

| 组别 | 温度信号 | 周期长度信息 | 实现方式 |
|------|----------|-------------|----------|
| 6a: NOISE + zero-pad | ✗ (i.i.d. Gaussian) | ✓ (可见) | 零填充噪声序列 |
| 6b: REAL + edge-pad | ✓ (真实温度) | ✗ (隐藏) | 用最后一个有效值填充尾部 |
| 6c: NOISE + full-len | ✗ (i.i.d. Gaussian) | ✗ (隐藏) | 全长度噪声 |

每组均执行完整 LOSO 训练 (1 seed)。

### 结论框架

通过比较 6 组结果，分解准确率来源:
```
总准确率 ≈ ov_frac分布贡献 + cycle_len代理贡献 + 温度信号贡献
 ~92%    ≈     ~83%           +     ~5-6%         +    ~2-5%
```

**运行**: `python -m model.experiment.run_leakage_check`

---

## 9. ovulation_cnn.py (位于 model/)

### 1D-CNN 排卵回归模型实验 / 1D-CNN Ovulation Regression

**注意**: 此文件同时包含模型定义和实验代码，位于 `model/` 而非 `model/experiment/`，因为它的模型类 (`OvulationCNN`) 被其他实验文件引用。

**目的**: 使用 1D-CNN 直接从周期级温度序列回归排卵日位置。

### 模型架构

```
OvulationCNN:
  Conv1d(1→32, k=7, pad=3) → BN → ReLU
  Conv1d(32→64, k=5, pad=2) → BN → ReLU
  Conv1d(64→128, k=3, pad=1) → BN → ReLU
  AdaptiveAvgPool1d(8)
  FC(128×8 → 128) → ReLU → Dropout(0.3) → FC(128 → 1) → Sigmoid
```

- **输入**: 长度=45 的 z-normalized 夜间温度序列（per-cycle z-norm，zero-padding 至 MAX_CYCLE_LEN=45）
- **输出**: `ov_frac ∈ [0, 1]`，代表排卵日在周期中的相对位置
- **损失**: MSE on ov_frac

### 数据准备 (`prepare_temp_samples`)

1. 遍历所有有 LH 标签的周期
2. 提取该周期的夜间温度序列，插值缺失值
3. z-normalize：`(T - mean) / std`
4. 零填充至固定长度 45
5. 计算 `ov_frac = ov_day / cycle_len`, `ov_day = lh_ov_dic[sgk]`

### 训练流程 (`train_and_evaluate_loso`)

- **验证方式**: LOSO（按 participant_id 划分）
- **训练细节**: Adam(lr=3e-4, wd=1e-4), StepLR(step=30, gamma=0.5), 60 epochs
- **多种子**: 默认 n_seeds=5，每个 seed 独立 LOSO 训练
- **Batch size**: 16

### 实验 1: CNN 温度回归

5 个种子独立 LOSO 训练，输出每个种子的排卵检测指标，以及 5-seed ensemble（取均值）。

### 实验 2: t-test Split（规则法互补）

`detect_ov_ttest_split` — 对每个可能的分割点做前后两段 t-test，选择 t 统计量最大的点:
```python
for split in range(5, len-3):
    t_stat, p = ttest_ind(temps[:split], temps[split:])
    if t_stat > best and post_mean > pre_mean:
        best_split = split
```
输出全集结果 + 高置信度（t-score > 1.5）子集。

### 实验 3: 混合策略

```
hybrid = CNN_ensemble优先 → t-test回退 → 日历法兜底 (0.50 × cycle_len)
```

### 月经预测集成

使用 CNN 检测的排卵日 + 个人黄体期长度进行月经预测，与日历法和 Oracle 对比，计算 "Oracle gap closed" 百分比。

**运行**: `python -m model.ovulation_cnn`

---

## 实验间关系图 / Experiment Dependency Map

```
┌─────────────────────────────────────┐
│       run_experiment.py (基线)       │
│  LightGBM → MAE, ±3d baseline       │
└──────────────┬──────────────────────┘
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
  robust_eval  tune_   run_oracle
  (多种子)     optuna   (上界测试)
               (调参)    │
                         │ 证明: 排卵检测有巨大价值
                         ▼
┌─────────────────────────────────────┐
│    排卵检测实验群                      │
│                                       │
│  run_detected_ov  ←── GBDT + 多信号  │
│  run_highfreq     ←── 分钟级温度特征  │
│  run_ovulation_experiments            │
│     ├── 规则法: ruptures, CUSUM,      │
│     │   BOCPD, coverline              │
│     ├── ML法: HMM, GBDT, 1D-CNN      │
│     └── ensemble + quality filter     │
│                                       │
│  ovulation_cnn.py ←── 最佳 CNN 模型  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    run_leakage_check.py (审计)       │
│  验证 CNN 高准确率的真实性             │
│  结论: 真实信号贡献仅 ~2-5pp         │
└─────────────────────────────────────┘
```

---

## 核心文件依赖 / Core File Dependencies

所有实验文件依赖 `model/` 下的核心模块:

| 核心模块 | 提供的功能 |
|----------|-----------|
| `config.py` | 路径、特征列表、超参数 |
| `dataset.py` | 数据加载、标签构建、划分 |
| `evaluate.py` | MAE、±k天准确率、分层评估 |
| `ovulation_detect.py` | 排卵检测工具函数、LH标签、黄体期计算 |
| `train_lgb.py` | LightGBM 训练与预测 |
| `ovulation_cnn.py` | CNN 模型类、数据准备、评估 |

---

## 运行命令汇总 / Run Commands

```bash
cd /Users/xujing/FYP/main_workspace

# 基线实验
python -m model.experiment.run_experiment

# 鲁棒性评估
python -m model.experiment.robust_eval

# 超参数调优
python -m model.experiment.tune_optuna

# Oracle 上界
python -m model.experiment.run_oracle_experiment

# GBDT 排卵检测混合模型
python -m model.experiment.run_detected_ov_experiment

# 高频温度实验
python -m model.experiment.run_highfreq_temp_experiment

# 排卵检测综合对比 (最大实验)
python -m model.experiment.run_ovulation_experiments

# 数据泄漏审计
python -m model.experiment.run_leakage_check

# 1D-CNN 排卵回归 (模型定义+实验)
python -m model.ovulation_cnn
```

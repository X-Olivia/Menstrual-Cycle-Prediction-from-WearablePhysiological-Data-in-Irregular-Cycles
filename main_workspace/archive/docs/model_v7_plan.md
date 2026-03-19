# 模型优化方案 v7 — 基于数据优势与文献调研

## 当前状态

### 模型表现 (v4 + Optuna, 10-seed)
- MAE: 3.291 ± 0.569
- ±3d 准确率 (PAE±3d): 66.4% ± 7.5%
- 可穿戴信号贡献极小（Prior only MAE 3.365 vs Full 3.333）

### 数据资产（未充分利用）

| 数据源 | 原始采样率 | 当前使用方式 | 信息损失 |
|--------|-----------|-------------|---------|
| 心率 (HR) | 每 5 秒 (~17,280 点/天) | 压缩为 4 个日级统计量 (mean/std/min/max) | **极大** |
| 腕温 (WT) | 每 1 分钟 (~1,440 点/天) | 压缩为 4 个日级统计量 | **极大** |
| HRV (RMSSD/LF/HF) | 每 5 分钟 (睡眠期) | 压缩为 1 个日级均值 | **大** |
| LH (黄体生成素) | 每日尿液检测，94.3% 覆盖率 | 仅用于标注排卵日 | **完全未作为训练信号** |
| E3G (雌激素代谢物) | 每日尿液检测，94.3% 覆盖率 | 仅用于辅助排卵标注 | **完全未作为训练信号** |
| PdG (孕激素代谢物) | 33% 覆盖率 | 未使用 | **完全未使用** |

### 关键约束
- **LH/E3G/PdG 不可作为模型输入特征**（部署时用户无法获取）
- 但可以作为**内部标签**用于辅助训练任务
- 排卵标记已根据 LH 标注（`ovulation_prob_fused` 字段）
- 42 名被试，中位数 4 个周期/人（范围 1-8），共 111 个有效周期
- 128/173 个周期检测到排卵

---

## 改进方向

### 方向 A: 子日级特征工程 — 从高频原始数据提取 [高优先级]

**问题**: 当前将 ~17,280 个 HR 采样点压缩为 4 个数字（mean/std/min/max），损失了昼夜节律模式、夜间最低值时机、晨起反应等关键信息。文献（Ava Bracelet, Oura Ring, Apple Watch）一致表明这些子日级模式是月经周期最敏感的生物标记。

**实现**: 创建 `data_process/build_subdaily_features.py`

#### A.1 夜间心率特征（从 5 秒 HR 数据）

需要的原始文件: `heart_rate.csv` (63M 行) + `computed_temperature.csv`（含睡眠起止时间）

| 特征名 | 计算方式 | 文献依据 |
|--------|---------|---------|
| `hr_nocturnal_nadir` | 睡眠窗口内 HR 30 分钟滑动平均最低值 | Goodale 2019 (Ava): nadir 是排卵前后最显著变化 |
| `hr_nadir_timing_frac` | nadir 出现时间 / 睡眠总时长 | 黄体期 nadir 提前 |
| `hr_sleep_onset_delta` | 入睡后 30min HR 均值 - 入睡前 30min HR 均值 | 入睡 HR 下降幅度随周期变化 |
| `hr_wake_surge` | 醒后 30min HR 均值 - 睡眠最后 1h HR 均值 | 晨起交感激活强度 |
| `hr_circadian_amplitude` | 白天 HR 90th percentile - 夜间 HR 10th percentile | 黄体期振幅增大 |
| `hr_nocturnal_iqr` | 夜间 HR 的 IQR（四分位距） | 夜间 HR 稳定性指标 |

#### A.2 夜间腕温特征（从 1 分钟 WT 数据）

需要的原始文件: `wrist_temperature.csv` (6.9M 行)

| 特征名 | 计算方式 | 文献依据 |
|--------|---------|---------|
| `wt_nocturnal_plateau` | 夜间中间 4 小时温度均值 | 类 BBT 效果，更精细 |
| `wt_rise_time_min` | 从入睡到温度达到夜间峰值 90% 的时间（分钟） | 温度上升速率随周期相位变化 |
| `wt_nocturnal_auc` | 夜间温度曲线下面积（梯形法） | 捕获整体夜间温度水平 |
| `wt_pre_wake_drop` | 醒前 30min 温度变化幅度 | 昼夜节律恢复指标 |
| `wt_nocturnal_range` | 夜间温度 max - min | 温度波动幅度 |

#### A.3 夜间 HRV 时序特征（从 5 分钟 HRV 数据）

需要的原始文件: `heart_rate_variability_details.csv` (436K 行)

| 特征名 | 计算方式 | 文献依据 |
|--------|---------|---------|
| `hrv_early_night` | 入睡后前 2h 的 RMSSD 均值 | 副交感优势期，对周期变化最敏感 |
| `hrv_late_night` | 醒前 2h 的 RMSSD 均值 | 与前者对比反映 HRV 夜间趋势 |
| `hrv_night_slope` | 全夜 RMSSD 线性斜率 | 黄体期 HRV 夜间下降斜率更陡 |
| `lf_hf_early_vs_late` | 前半夜 LF/HF - 后半夜 LF/HF | 自主神经夜间调节模式 |

**涉及文件修改**:
- 新建 `data_process/build_subdaily_features.py`
- 修改 `data_process/build_features_v4.py` 中的 `merge_all()`，增加子日级特征合并
- 修改 `model_v3/config.py`，新增 `FEAT_SUBDAILY` 特征组

**预期提升**: MAE 降低 0.3-0.5，±3d 提升 5-8%（基于 Ava Bracelet 和 Oura 研究中夜间特征的效果）

---

### 方向 B: 激素监督的辅助训练 — 用 LH/E3G 增强训练信号 [高优先级]

**问题**: LH 和 E3G 覆盖率高达 94%，是周期阶段最强的生物标记，但目前仅用于排卵标注。在不将其作为模型输入的前提下，可以利用激素数据增强训练过程。

#### B.1 基于 LH 的排卵锚定特征（训练时可用）

**核心思路**: 在训练数据中，我们知道每个周期的 LH surge 位置（即排卵日）。可以计算**个人平均黄体期长度**，然后在测试时用**可穿戴信号**检测排卵，结合个人黄体期长度预测月经。

**实现**:
1. 从 `cycle_clean_2.csv` 的 `ovulation_day_method1/method2` 计算每个周期的实际黄体期长度
2. 计算 `personal_avg_luteal_len`（每人历史平均黄体期长度）
3. 在 v6 的 `estimate_phase()` 基础上改进：用 `personal_avg_luteal_len` 替代固定 14 天
4. 新增特征: `personal_avg_luteal_len`, `personal_luteal_std`

| 特征名 | 计算方式 | 说明 |
|--------|---------|------|
| `personal_avg_luteal_len` | 该被试历史周期中排卵日到周期结束的平均天数 | 个体黄体期长度高度一致（std ~1-2 天） |
| `personal_luteal_std` | 个人黄体期长度标准差 | 反映个体黄体期稳定性 |
| `days_since_temp_shift` | 当前周期中温度双相升高的位置（可穿戴检测排卵） | 替代 LH 的部署时排卵定位 |
| `est_days_remaining_luteal` | `personal_avg_luteal_len - days_since_temp_shift` | 基于个人黄体期的倒计时估计 |

**注意**: `personal_avg_luteal_len` 在 LLCO 设置中可以从训练周期中安全计算，不会泄露测试信息。

#### B.2 多任务学习 — 激素标签作为辅助目标

**核心思路**: 训练时用 LH 衍生的**周期阶段标签**作为辅助预测目标。模型被迫学习区分周期阶段的可穿戴特征模式，间接提升主任务性能。

**实现方案（LightGBM 兼容）**:

方案一：**两阶段训练**
1. 第一阶段：训练一个阶段分类器（LightGBM 分类），用 LH 衍生的阶段标签（经期/卵泡期/黄体期）作为 target
2. 将第一阶段模型的**预测概率**（3 个浮点值）作为新特征加入主回归模型
3. 主回归模型仍预测 `days_until_next_menses`

方案二：**Soft label 注入**
1. 从 LH/E3G 曲线计算每天的**周期进度置信度**（0-1 连续值）
2. 例如：`lh_phase_progress = days_since_lh_surge / personal_avg_luteal_len`
3. 将此 soft label 的**模型预测值**（而非真实值）作为特征
4. 避免在部署时需要 LH 数据

**涉及文件修改**:
- 修改 `data_process/build_features_v4.py`（或新建 v7）：增加 `personal_avg_luteal_len` 等特征
- 新建 `model_v3/multitask.py`：实现两阶段训练
- 修改 `model_v3/config.py`：新增 `FEAT_LUTEAL_PERSONAL` 和 `FEAT_PHASE_PROB` 特征组

**预期提升**: MAE 降低 0.5-1.0，±3d 提升 8-15%（黄体期长度个体内变异仅 1-2 天，理论上限接近 MAE ~1.5）

---

### 方向 C: 简单个性化 — Per-Subject 偏差校正 [中优先级，极低工作量]

**问题**: 全局模型对所有被试预测同样的偏差模式，但个体差异显著（如某人总是晚 2 天来月经）。

**实现**: 修改 `model_v3/train_lgb.py` 和 `robust_eval.py`

```python
# 训练阶段：记录每个被试在训练集中的平均残差
subject_bias = {}
for subj_id in train_subjects:
    mask = (df["id"] == subj_id) & train_mask
    if mask.sum() > 0:
        residuals = y_train[mask] - pred_train[mask]
        subject_bias[subj_id] = np.mean(residuals)

# 测试阶段：校正预测
for subj_id in test_subjects:
    if subj_id in subject_bias:
        pred_test[test_df["id"] == subj_id] += subject_bias[subj_id]
```

在 LLCO 设置中尤其有效，因为测试被试的训练周期数据可用。

**涉及文件修改**:
- 修改 `model_v3/robust_eval.py`：在预测后增加偏差校正步骤

**预期提升**: MAE 降低 0.1-0.3，±3d 提升 1-3%

---

### 方向 D: PAE±3d 指标增强与优化 [中优先级]

**问题**: 当前用 Huber loss 优化 MAE，但 ±3d 准确率（PAE±3d）才是最终关注的指标。两者并非完全一致 — 减小 MAE 不一定最大化 PAE±3d。

#### D.1 增加 Per-Cycle PAE 评估

当前评估是 per-day（每个样本是一天），论文通常 per-cycle 评估（每个周期取一个预测值）。

**实现**: 在 `evaluate.py` 中新增 `compute_cycle_metrics()`
- 对每个周期，取特定时间点的预测（如 day 7、day 14 的 `days_until_next_menses` 预测）
- 计算 per-cycle MAE 和 PAE±3d
- 这更接近论文报告方式

#### D.2 自定义 Focal Regression Loss

**核心思路**: 类 Focal Loss，让模型更关注 error 在 2-4 天边界的样本（这些样本决定 PAE±3d）。

```python
def focal_huber_loss(y_pred, dataset):
    y_true = dataset.get_label()
    err = y_true - y_pred
    abs_err = np.abs(err)
    
    # Focal weight: error 在 3 天附近的样本权重更高
    focal_weight = 1.0 + 2.0 * np.exp(-((abs_err - 3.0) ** 2) / 2.0)
    
    # Huber loss gradient
    delta = 3.0
    huber_grad = np.where(abs_err <= delta, -err, -delta * np.sign(err))
    huber_hess = np.where(abs_err <= delta, 1.0, 0.0)
    
    return focal_weight * huber_grad, focal_weight * huber_hess
```

#### D.3 回归 + 分类集成

- 模型 1: LightGBM 回归 → 预测连续天数
- 模型 2: LightGBM 分类 → 预测 "是否在 3 天内来月经"
- 集成: 当分类器高置信度预测"3 天内"时，将回归预测 clip 到 [1, 3]

**涉及文件修改**:
- 修改 `model_v3/evaluate.py`：新增 `compute_cycle_metrics()`
- 修改 `model_v3/train_lgb.py`：支持自定义 loss
- 新建 `model_v3/ensemble.py`：回归+分类集成

**预期提升**: PAE±3d 提升 2-5%

---

### 方向 E: 温度双相变化点检测 — 改进排卵定位 [中优先级]

**问题**: 当前的 `temp_shift_7v3`（近 3 天均值 vs 前 3-9 天均值）过于简单，无法精确定位排卵后的温度双相转变点。

**实现**: 在 `build_subdaily_features.py` 或单独脚本中

基于 BBT 研究（三日温升规则 + CUSUM 算法）：

| 特征名 | 计算方式 | 说明 |
|--------|---------|------|
| `temp_changepoint_day` | CUSUM 或 Bayesian changepoint 检测 | 温度从低温相转高温相的天数 |
| `temp_changepoint_magnitude` | 变化点前后 3 天均温差 | 变化幅度越大越确信排卵 |
| `temp_days_since_shift` | 当前日距温度变化点的天数 | 比 `days_since_estimated_ovulation` 更精确 |
| `temp_high_phase_duration` | 已在高温相持续的天数 | 黄体期进度指标 |
| `temp_high_phase_stability` | 高温相温度的变异系数 | 黄体期温度稳定性 |

**涉及文件修改**:
- 新建或修改 `data_process/build_subdaily_features.py` 中的温度部分
- 修改 `model_v3/config.py`：新增 `FEAT_TEMP_CHANGEPOINT` 特征组

**预期提升**: MAE 降低 0.2-0.4（对黄体期预测特别有效）

---

## 实施优先级与路线图

```
Phase 1 (核心改进，预计效果最大):
  ├── 方向 A: 子日级特征工程
  │   └── A.1 夜间 HR 特征 + A.2 夜间温度特征 + A.3 HRV 时序特征
  ├── 方向 B.1: 排卵锚定个性化特征 (personal_avg_luteal_len)
  └── 方向 C: Per-subject 偏差校正

Phase 2 (辅助优化):
  ├── 方向 B.2: 多任务学习（阶段分类器 soft labels）
  ├── 方向 D: PAE±3d 指标增强与优化
  └── 方向 E: 温度双相变化点检测

Phase 3 (集成与评估):
  └── 10-seed robust evaluation 对比所有配置
```

| 阶段 | 方向 | 预期 MAE | 预期 PAE±3d | 工作量 |
|------|-----|---------|------------|-------|
| Baseline | v4 + Optuna | 3.29 | 66.4% | — |
| Phase 1 | + 子日级 + 排卵锚定 + 偏差校正 | **2.3-2.8** | **72-78%** | 中 |
| Phase 2 | + 多任务 + PAE优化 + 变化点 | **1.8-2.3** | **78-85%** | 中-高 |
| Phase 3 | 最优集成 | **1.5-2.0** | **82-88%** | 低 |

---

## 涉及的文件修改总结

### 新建文件
- `data_process/build_subdaily_features.py` — 子日级特征提取（方向 A）
- `model_v3/multitask.py` — 多任务学习/两阶段训练（方向 B.2）
- `model_v3/ensemble.py` — 回归+分类集成（方向 D.3）

### 修改文件
- `data_process/build_features_v4.py` 或新建 `build_features_v7.py` — 合并子日级特征 + 排卵锚定特征
- `model_v3/config.py` — 新增特征组: `FEAT_SUBDAILY`, `FEAT_LUTEAL_PERSONAL`, `FEAT_TEMP_CHANGEPOINT`, `FEAT_PHASE_PROB`
- `model_v3/evaluate.py` — 新增 `compute_cycle_metrics()` per-cycle 评估
- `model_v3/robust_eval.py` — 新增偏差校正逻辑
- `model_v3/train_lgb.py` — 支持自定义 loss function

### 依赖的原始数据文件
- `heart_rate.csv` (63M rows, ~5s intervals)
- `wrist_temperature.csv` (6.9M rows, ~1min intervals)
- `heart_rate_variability_details.csv` (436K rows, ~5min intervals)
- `computed_temperature.csv` (睡眠起止时间)
- `cycle_clean_2.csv` (排卵标注: `ovulation_day_method1/2`, `ovulation_prob_fused`)
- `hormones_and_selfreport.csv` (LH/E3G 数据，仅用于方向 B 的辅助标签)

---

## 关于个性化的可行性分析

**结论: 可行，且在 LLCO 设置下效果应显著。**

| 方法 | 需要的最少周期数 | 我们的数据 (中位 4 周期) | 可行性 |
|------|----------------|----------------------|-------|
| Per-subject 偏差校正 | 1-2 个训练周期 | 在 LLCO 中有 3 个训练周期 | **完全可行** |
| `personal_avg_luteal_len` | 1 个有排卵检测的周期 | 128/173 周期有排卵 | **高度可行** |
| Subject clustering | 全局特征即可 | 42 人 → 3-4 个聚类 | **可行** |
| Neural net + subject embedding | 3-4 个周期 (~100 天数据点) | 刚好够 8 维 embedding | **边界可行** |
| 完全个体化模型 | 10+ 个周期 | 最多 8 个 | **不可行** |

最有效的个性化路径是 **方向 B.1 + 方向 C** 的组合: 利用 LH 衍生的排卵标注计算个人黄体期长度（B.1），再通过偏差校正（C）微调预测值。这种方法在每人仅有 2-3 个训练周期时即可工作。

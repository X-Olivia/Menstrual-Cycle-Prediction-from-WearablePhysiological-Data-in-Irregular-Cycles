# new_workspace 可穿戴排卵与经期预测实验情况说明

本文档汇总当前仓库中两条主实验线的**目的、数据、评估协议与一次完整跑通的主要结果**（与 `predict_menses_by_anchors` 及 `oracle_luteal_countdown_experiment.py` 的终端输出一致）。

---

## 1. 概述

| 实验线 | 入口脚本 | 核心内容 |
|--------|----------|----------|
| **多信号排卵与经期** | `record/multisignal_ovulation_detection_and_menses_experiment.py` | 规则 / 多信号规则 / ML LOSO / CNN / stacking / 加权集成；输出排卵日排名（E）与锚点剩余天数（F） |
| **Oracle 与黄体倒计时 + LightGBM** | `record/oracle_luteal_countdown_experiment.py` | 日级特征上 LightGBM 预测剩余天数；排卵分类器 LOSO；三种策略（仅 LGB、检测混合、Oracle 混合）及锚点拆分 |

两条线共用 **LH 标注的周期与可穿戴信号**，但特征形态不同：多信号以**周期内序列**为主；Oracle 脚本以 **`daily_features_v4.csv` 日级表**为主。

---

## 2. 数据规模（多信号脚本启动日志）

- 周期 CSV + `signals/` 加载完成后：**Cycles: 111，Labeled: 79，Quality: 42**  
- 多信号全量约 **189 个配置**，一次完整运行约 **800 s** 量级（以本机为准）

---

## 3. 多信号实验

### 3.1 流程结构（终端 A–F）

- **A**：单信号规则（T-test、CUSUM、Bayesian、HMM、SavGol 等 × 多 σ、多通道）  
- **B**：多信号规则融合（fused t-test、multi-HMM、multi-CUSUM）  
- **C**：ML LOSO（ridge、elastic、SVR、RF、GBDT、bayridge、KNN、Huber、XGB、LGBM）、相位分类、1D-CNN  
- **D**：stack-topN、wens-topN  
- **E**：在 **带 LH 的周期**上汇总各方法的排卵日误差（每周期一条，**n=79**），按 ±2d 等排序  
- **F**：对排名靠前的检测器，用 **固定黄体 12/13 天** 与 `predict_menses_by_anchors` 评 **剩余天数**（锚点见下）

### 3.2 评估定义（多信号）

**排卵日（E）**：预测排卵日与 LH 日的绝对误差；报告 MAE、±1d/±2d/±3d/±5d 覆盖率（每周期一个样本）。

**剩余天数（F）**：以 **LH 排卵日为基准**，在 `ov-7、ov-3、ov-1、ov+2、ov+5、ov+10`（若在 `[0, cycle_len)` 内）各算一次 `pred_remaining` 与 `true_remaining`；`Pre_all` / `Post_all` 分别为前三、后三锚点上的误差合并，样本数约为 **237 / 228**（随周期长度略变）。排卵前锚点多用日历长度，排卵后锚点在满足 `anchor_day ≥ ov_est+2` 等条件时用 **countdown（ov_est + lut）**。±kd 为绝对剩余天数误差 ≤ k 的比例。

### 3.3 主要结果摘录（与当前一次跑通一致）

**E 节 — ALL LABELED（n=79）**

| 方法 | n | MAE | ±3d | ±5d |
|------|---|-----|-----|-----|
| wens-top7 | 79 | 1.92 | 84.8% | 97.5% |
| wens-top3 / wens-top5 | 79 | 1.94–1.95 | 86.1% | 97.5% |
| stack-top5 | 79 | 2.08 | 84.8% | 97.5% |
| CNN-multi | 72 | 1.92 | 83.3% | 95.8% |

**E 节 — QUALITY（n=42）**：`CNN-multi`（n=37）MAE=1.76，±3d=91.9%，±5d=97.3%。

**F 节 — 基线（lut13）**

| 标签 | n | MAE | ±3d | ±5d |
|------|---|-----|-----|-----|
| Oracle+lut13 Pre_all | 237 | 4.28 | 46.8% | 68.4% |
| Oracle+lut13 Post_all | 228 | 1.27 | 96.1% | 100% |
| Calendar Post_all | 228 | 4.34 | 46.5% | 68.0% |

**F 节 — 可穿戴估排卵 + lut13（Post_all，节选）**

| 配置 | MAE | ±3d | ±5d |
|------|-----|-----|-----|
| wens-top5+lut13 | 2.35 | 77.2% | 90.8% |
| wens-top3+lut13 | 2.31 | 76.8% | 90.4% |
| wens-top7+lut13 | 2.40 | 74.1% | 90.8% |
| stack-top5+lut13 | 2.50 | 73.7% | 91.2% |

E 节 ±3d 与 F 节 Post_all ±3d **指标对象不同**（排卵日 vs 锚点剩余天数），数值不作直接等同。

---

## 4. Oracle + Luteal Countdown 实验

### 4.1 目的与设定

在 **日级特征**上训练 **LightGBM** 预测 `days_until_next_menses`，并比较：

1. **LightGBM only**：全程使用回归预测。  
2. **Detected-ov hybrid**：由排卵分类器得到估测排卵日；若当前日在 **估测排卵日 +2** 之后，用 **估测排卵日 + 黄体长度** 修正剩余天数，否则仍用 LightGBM。  
3. **Oracle hybrid**：与 2 相同，但排卵日替换为 **LH 真值**，用于隔离「排卵检测误差」对黄体倒计时段的影响。

脚本另输出：排卵分类器 **LOSO AUC**、多种检测策略下的排卵日偏移与 ±3d/±5d；**人口黄体均值**（日志示例约 13.1 天）。

### 4.2 主要结果摘录（与当前一次跑通一致）

- 日表规模示例：3276 行，40 受试者，79 个带 LH 周期；排卵分类器 **LOSO AUC ≈ 0.931**  
- 检测策略示例（threshold）：77/79 周期给出检测，平均偏移约 0.5 d，排卵日 ±3d 约 53.2%  
- **10-seed 汇总，threshold 策略 — 剩余天数整体**：LightGBM only MAE≈3.51、±3d≈59.6%；Detected-ov hybrid ±3d≈57.1%；**Oracle hybrid MAE≈3.13、±3d≈66.7%**  
- **锚点拆分（同一策略下）**：Pre 锚点（ov-7/−3/−1）上 Oracle hybrid 与 LGB 接近；Post 锚点（ov+2/+5/+10）上 **Oracle hybrid ±3d 可达约 96.9%–100%**（与多信号 F 节排卵后 Oracle 高位一致）

---

## 5. 代码位置（`new_workspace/`）

| 模块 | 路径 |
|------|------|
| 多信号入口 | `record/multisignal_ovulation_detection_and_menses_experiment.py` |
| 编排 | `record/experiment/multisignal_ovulation_detection_and_menses_experiment_entry.py` |
| 规则检测 | `record/experiment/multisignal_detectors_rule.py` |
| ML/CNN/stacking | `record/experiment/multisignal_ovulation_detection_and_menses_experiment.py` |
| 数据加载 | `record/experiment/multisignal_data.py` |
| F 节锚点 | `record/experiment/multisignal_menses.py`（`predict_menses_by_anchors`） |
| Oracle + LGB | `record/oracle_luteal_countdown_experiment.py` |

---

## 6. 数据依赖与日志

- 多信号：`processed_dataset/cycle_cleaned_ov.csv`，`processed_dataset/signals/*.csv`  
- Oracle 脚本：上述 cycle + **`processed_dataset/daily_features/daily_features_v4.csv`**  
- 完整分项数字以终端全文或自行重定向的 `.log` 为准；本目录下可有 `multisignal_run_*.log` 等备查。

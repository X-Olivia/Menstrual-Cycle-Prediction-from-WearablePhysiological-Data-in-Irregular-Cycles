# 数据泄漏审计报告 — 1D-CNN 排卵检测模型

## Data Leakage Audit Report — 1D-CNN Ovulation Detection Model

---

## 1. 背景 / Background

在构建 1D-CNN 排卵日回归模型时，我们使用 LOSO（Leave-One-Subject-Out）交叉验证获得了 **~92% ±3d 准确率 (MAE≈1.7d)** 的看似优异结果。为验证这些结果是否源于模型对温度时序模式的真正学习，我们进行了系统性的数据泄漏审计。

When building the 1D-CNN ovulation day regression model, LOSO cross-validation yielded an impressive-looking **~92% ±3d accuracy (MAE≈1.7d)**. To verify whether these results stem from genuine learning of temperature patterns, we conducted a systematic data leakage audit.

---

## 2. 审计项目 / Audit Checks

### Check 1: 标签分布分析 / Label Distribution Analysis

**发现 / Finding: ov_frac 分布极度集中**

| 统计量 | 值 |
|---------|-----|
| ov_frac 均值 (mean) | 0.575 |
| ov_frac 标准差 (std) | 0.076 |
| ov_frac 范围 (range) | [0.35, 0.80] |

由于目标变量 `ov_frac = ov_day / cycle_len` 分布非常窄（标准差仅 0.076），一个不需要任何模型的**常数预测**（固定输出 0.575）即可达到：

- **MAE = 1.91d**
- **±3d 准确率 = 89.5%**

**结论**: 高准确率的「天花板效应」——标签分布本身就使得随机猜测都能获得很高的 ±3d 准确率。

Since the target `ov_frac = ov_day / cycle_len` has a very narrow distribution (std=0.076), a **constant prediction** of 0.575 (no model at all) achieves 89.5% ±3d accuracy.

---

### Check 2: LOSO 划分正确性 / LOSO Split Integrity

**发现 / Finding: PASS ✓**

验证了所有折叠中，测试被试的周期从未出现在训练集中。LOSO 按被试 (participant_id) 划分实现正确，不存在被试级别的数据泄漏。

Confirmed that test subject cycles never appear in their training fold. LOSO is correctly implemented at the subject level.

---

### Check 3: Z-归一化范围 / Z-normalisation Scope

**发现 / Finding: 可接受 (ACCEPTABLE)**

当前使用**周期级 z-归一化**（per-cycle z-norm），即使用整个周期的温度均值和标准差来标准化。在回测（retrospective）场景下，这是可接受的，因为整个周期数据在检测时点已经可用。

若需因果（prospective）检测，应改用**扩展窗口归一化**（expanding z-norm）。

Per-cycle z-normalisation is acceptable for retrospective detection since the full cycle is observed. For causal (prospective) use, expanding-window z-norm would be needed.

---

### Check 4: 目标变量与周期长度 / Target Variable vs Cycle Length

**发现 / Finding: 周期长度与排卵日高度相关，且可从 padding 推断**

| 检查项 | 结果 |
|--------|------|
| Pearson r(cycle_len, ov_day) | **0.921** (p < 1e-50) |
| 线性代理 0.575 × L 的 ±3d | **84.2%** |
| 常数代理 round(0.575 × L) 的 ±3d | **89.5%** |

**关键问题**: CNN 的输入序列使用 **零值填充（zero-padding）** 至固定长度 `MAX_CYCLE_LEN=45`。模型可以通过检测非零值结束位置轻松推断 `cycle_len`，从而将预测简化为：

```
pred_ov_day = pred_frac × cycle_len ≈ 0.575 × cycle_len
```

这本质上是一个「智能日历法」，而非基于温度模式的检测。

The CNN input uses **zero-padding** to `MAX_CYCLE_LEN=45`. The model can trivially infer `cycle_len` from where zeros begin, reducing the task to a "smart calendar method".

---

### Check 5: 打乱温度序列 / Shuffled Temperature Ablation

**实验**: 随机打乱每个周期内的温度值顺序（破坏时序模式），但保留 zero-padding 结构（保留周期长度信息）。

**Experiment**: Randomly shuffle temperature values within each cycle (destroying temporal pattern) while keeping zero-padding (preserving cycle length info).

| 条件 / Condition | ±3d | MAE |
|-------------------|------|------|
| 原始 CNN / Original CNN | ~92% | ~1.7d |
| 打乱温度 / Shuffled temp (len kept) | **86.3%** | 1.89d |

**结论**: 破坏温度时序模式后，准确率仅下降 ~6 个百分点。这说明模型的大部分准确率来源于周期长度信息，而非温度模式。

Destroying temporal patterns causes only ~6pp drop, confirming most accuracy comes from cycle-length cues.

---

### Check 6: 噪声消融实验 / Noise Ablation Experiments

| 条件 / Condition | 温度信号 | 周期长度 | ±3d | MAE |
|-------------------|----------|----------|------|------|
| CNN — 原始 / Original | ✓ 真实 | ✓ 可见 (zero-pad) | **~92%** | ~1.7d |
| CNN — 打乱 / Shuffled | ✗ 破坏 | ✓ 可见 (zero-pad) | **86.3%** | 1.89d |
| CNN — 纯噪声+零填充 / Noise+zero-pad | ✗ 噪声 | ✓ 可见 (zero-pad) | **88.4%** | 1.98d |
| CNN — 真实温度+边缘填充 / Real+edge-pad | ✓ 真实 | ✗ 隐藏 (edge-pad) | **91.6%** | 1.69d |
| CNN — 纯噪声+全填充 / Noise+full-len | ✗ 噪声 | ✗ 隐藏 (random fill) | **83.2%** | 2.12d |
| 常数基线 / Constant baseline (0.575) | — | — | **89.5%** | 1.91d |

**关键解读 / Key Interpretations:**

1. **纯噪声 + 保留长度 = 88.4%**: 完全没有温度信号，仅靠 padding 推断周期长度，准确率就接近 90%。证实模型主要利用周期长度。

2. **真实温度 + 隐藏长度 = 91.6%**: 当用 edge-padding 隐藏周期长度时，使用真实温度仍能达到高准确率。这说明温度信号**确实有一定贡献**（+2-5pp over noise baseline）。

3. **纯噪声 + 隐藏长度 = 83.2%**: 信号和长度都去除后，准确率降至最低。但仍高达 83%，这是因为 `ov_frac` 分布本身极窄。

---

## 3. 泄漏机制图解 / Leakage Mechanism

```
                    ┌─────────────────────────────────────┐
                    │      CNN Input (padded to L=45)      │
                    │                                       │
                    │  [T₁, T₂, ..., Tₙ, 0, 0, ..., 0]   │
                    │   ←── n values ──→ ←── zeros ──→     │
                    │                                       │
                    │  CNN detects: n = cycle_len            │
                    │  CNN learns:  ov_frac ≈ 0.575         │
                    │  Output:      ov_day ≈ 0.575 × n      │
                    └─────────────────────────────────────┘
                              ↑ This is essentially a
                                "smart calendar method"
```

---

## 4. 真实信号贡献估计 / True Signal Contribution Estimate

通过消融实验，可以分解模型准确率的来源：

| 来源 / Source | 对 ±3d 的贡献 | 说明 |
|---------------|---------------|------|
| ov_frac 分布窄（天花板效应） | ~83% | 标签分布本身保证的下限 |
| 周期长度代理 | ~5-6% | 从 zero-padding 推断 cycle_len |
| **温度时序信号** | **~2-5%** | **模型的真实贡献** |
| **合计** | **~90-92%** | |

**温度信号的真实贡献仅约 2-5 个百分点的 ±3d 准确率。**

The true marginal contribution of temperature pattern recognition is approximately **2-5 percentage points** in ±3d accuracy above the calendar-based baseline.

---

## 5. 结论与建议 / Conclusions & Recommendations

### 结论 / Conclusions

1. **不存在被试级数据泄漏**: LOSO 划分正确实现。
2. **存在严重的「天花板效应」泄漏**: 由于 `ov_frac` 分布集中 + zero-padding 泄露周期长度，一个不使用任何温度信号的 baseline 就能达到 ~89% ±3d。
3. **±3d 准确率不适合作为唯一评估指标**: 该指标在 ov_frac 分布窄时会严重膨胀，无法区分真正有效的模型和简单的日历法。
4. **温度信号确实有贡献，但贡献有限**: edge-pad 实验表明真实温度相比噪声确实能提升 2-5pp，说明模型学到了一些双相模式。

### 建议 / Recommendations

1. **改进评估方式**: 
   - 报告相对于常数基线（0.575 × L）的改进量，而非绝对 ±3d 数值。
   - 使用 MAE 作为主要指标，补充 ±1d、±2d 精度。

2. **改进建模方式**:
   - 使用 edge-padding 或 mask-based padding 消除长度泄漏。
   - 考虑分类模型（逐日概率）替代回归 ov_frac。
   - 评估排卵检测的增量贡献（在已知 cycle_len 基础上）。

3. **改进数据处理**:
   - 对 ov_frac 进行去均值处理 → 预测残差 `ov_frac - 0.575`。
   - 这使模型必须从温度信号中学习 *偏离均值的部分*。

---

## 6. 实验代码 / Experiment Code

完整实验代码: `main_workspace/model/experiment/run_leakage_check.py`

运行方式:
```bash
cd /Users/xujing/FYP/main_workspace
python -m model.experiment.run_leakage_check
```

---

## 附录: 实验环境 / Appendix: Environment

- 数据集 / Dataset: Apple Women's Health Study (subdataset)
- 排卵标签 / Ovulation labels: LH-based annotation
- 验证方式 / Validation: Leave-One-Subject-Out (LOSO)
- 模型 / Model: 3-layer 1D-CNN (Conv1d-32 → Conv1d-64 → Conv1d-128 → FC)
- 框架 / Framework: PyTorch
- 日期 / Date: 2026-03-03

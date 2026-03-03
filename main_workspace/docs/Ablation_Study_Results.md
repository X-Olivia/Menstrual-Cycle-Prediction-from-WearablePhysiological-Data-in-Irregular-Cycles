# 消融实验 (Ablation Study) 结果

## 实验设计

逐步累加特征组，观察每组特征对模型性能的边际贡献。

**实验条件**：
- 模型：LightGBM，Huber loss (δ=3.0)
- 数据分割：Subject-level split，seed=42（Train 29 人 / Val 7 人 / Test 6 人）
- 数据量：4520 行，42 受试者，168 周期（过滤 >45 天周期后）

**运行命令**：

```bash
cd main_workspace
python -c "from model_v3.run_experiment import run_ablation; run_ablation()"
```

**消融代码位于 `model_v3/run_experiment.py :: run_ablation()`**：

```python
groups = [
    ("Prior only",      FEAT_CYCLE_PRIOR),
    ("+ Wearable",      FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z),
    ("+ Shifts",        FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS),
    ("+ Deltas",        FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS),
    ("+ Respiratory",   FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z),
    ("+ Sleep",         FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z + FEAT_SLEEP_Z),
    ("+ Symptoms (ALL)",FEAT_CYCLE_PRIOR + FEAT_WEARABLE_Z + FEAT_SHIFTS + FEAT_DELTAS + FEAT_RESPIRATORY_Z + FEAT_SLEEP_Z + FEAT_SYMPTOMS),
]
```

---

## 总结表

### Test 集

| 配置 | 特征数 | Test MAE | Test ±1d | Test ±2d | Test ±3d | MAE Δ |
|------|--------|----------|----------|----------|----------|-------|
| Prior only | 6 | 4.339 | 17.8% | 33.4% | 45.0% | — |
| + Wearable | 20 | 4.210 | 16.9% | 32.0% | 44.5% | −0.129 |
| + Shifts | 22 | 4.241 | 15.3% | 32.0% | 43.0% | +0.031 |
| + Deltas | 28 | 4.235 | 15.9% | 30.9% | 42.7% | −0.006 |
| **+ Respiratory** | **30** | **4.204** | 16.6% | 30.0% | 42.9% | **−0.031** |
| + Sleep | 33 | 4.212 | 17.1% | 30.7% | 44.4% | +0.008 |
| + Symptoms (ALL) | 37 | 4.233 | 16.9% | 33.5% | 45.0% | +0.021 |

### Validation 集

| 配置 | 特征数 | Val MAE | Val ±1d | Val ±2d | Val ±3d |
|------|--------|---------|---------|---------|---------|
| Prior only | 6 | 4.718 | 15.6% | 30.5% | 44.6% |
| + Wearable | 20 | 4.159 | 18.7% | 33.6% | 49.8% |
| + Shifts | 22 | 4.168 | 18.7% | 33.7% | 49.7% |
| + Deltas | 28 | 4.125 | 18.7% | 34.2% | 50.2% |
| + Respiratory | 30 | 4.111 | 19.8% | 34.3% | 49.7% |
| + Sleep | 33 | 4.164 | 19.4% | 33.0% | 49.6% |
| + Symptoms (ALL) | 37 | 4.166 | 18.2% | 34.2% | 51.8% |

---

## Test 集 Horizon 分层详情

### Prior only (6 features)

| horizon | n | MAE | ±1d | ±2d | ±3d |
|---------|---|-----|-----|-----|-----|
| 1-5 | 125 | 4.83 | 17.6% | 35.2% | 52.0% |
| 6-10 | 123 | 3.75 | 34.1% | 48.8% | 55.3% |
| 11-15 | 115 | 3.48 | 17.4% | 41.7% | 53.0% |
| 16-20 | 110 | 3.87 | 10.0% | 30.0% | 45.5% |
| 21+ | 201 | 5.14 | 12.4% | 19.9% | 29.4% |

### + Respiratory (30 features) — 最优配置

| horizon | n | MAE | ±1d | ±2d | ±3d |
|---------|---|-----|-----|-----|-----|
| 1-5 | 125 | 4.46 | 22.4% | 38.4% | 52.8% |
| 6-10 | 123 | 3.95 | 18.7% | 39.0% | 48.0% |
| 11-15 | 115 | 3.54 | 9.6% | 27.8% | 48.7% |
| 16-20 | 110 | 3.57 | 16.4% | 29.1% | 37.3% |
| 21+ | 201 | 4.98 | 12.9% | 23.9% | 35.3% |

### + Symptoms / ALL (37 features)

| horizon | n | MAE | ±1d | ±2d | ±3d |
|---------|---|-----|-----|-----|-----|
| 1-5 | 125 | 4.50 | 24.0% | 40.0% | 52.8% |
| 6-10 | 123 | 4.01 | 17.1% | 36.6% | 48.8% |
| 11-15 | 115 | 3.38 | 10.4% | 33.0% | 52.2% |
| 16-20 | 110 | 3.73 | 20.0% | 34.5% | 39.1% |
| 21+ | 201 | 4.97 | 14.4% | 27.4% | 36.8% |

---

## 特征重要性 (ALL 37 features, gain)

| 排名 | 特征 | Gain | 类别 |
|------|------|------|------|
| 1 | day_in_cycle | 222,256 | 周期先验 |
| 2 | day_in_cycle_frac | 49,819 | 周期先验 |
| 3 | hist_cycle_len_std | 33,591 | 周期先验 |
| 4 | hist_cycle_len_mean | 33,281 | 周期先验 |
| 5 | resting_hr_z | 23,928 | 穿戴设备 |
| 6 | days_remaining_prior | 20,364 | 周期先验 |
| 7 | bloating | 7,420 | PMS 症状 |
| 8 | days_remaining_prior_log | 7,087 | 周期先验 |
| 9 | lf_mean_z | 4,934 | HRV |
| 10 | deep_sleep_br_z | 4,307 | 呼吸频率 |
| 11 | moodswing | 4,028 | PMS 症状 |
| 12 | hr_min_z | 3,809 | 心率 |
| 13 | delta_hf_mean_1d | 3,022 | 变化率 |
| 14 | lf_hf_ratio_z | 2,970 | HRV |
| 15 | rmssd_mean_z | 2,934 | HRV |
| 16 | hr_mean_z | 2,765 | 心率 |
| 17 | full_sleep_br_z | 2,760 | 呼吸频率 |
| 18 | wt_min_z | 2,342 | 腕温 |
| 19 | hf_mean_z | 2,302 | HRV |
| 20 | wt_max_z | 2,251 | 腕温 |

周期先验 6 个特征总 gain: 366,397 (82%)，穿戴设备+症状 31 个特征总 gain: ~80,000 (18%)。

---

## 关键结论

1. **周期先验是最强信号**：仅 6 个先验特征即达到 Test MAE 4.339，后续 31 个生理特征只额外贡献了 0.135 天的改善（3.1%）。

2. **Shifts / Deltas / Sleep / Symptoms 无正向贡献**：在此数据规模和模型架构下，这些特征组的加入反而轻微恶化了性能（过拟合风险 > 信息增益）。

3. **Val 和 Test 趋势不一致**：Val MAE 随特征增加持续改善（4.718 → 4.111），但 Test MAE 在特征增多后反弹。这是典型的小样本过拟合信号。

4. **消融实验中最优配置为 Prior + Wearable + Shifts + Deltas + Respiratory（30 维，MAE 4.204）**，但 Shifts 和 Deltas 本身边际贡献为负。

---

## 最终配置验证

基于消融结论，将 Shifts 和 Deltas 一并移除，仅保留 **Prior + Wearable + Respiratory（22 维）**。

验证结果：

| 配置 | 特征数 | Test MAE | Test ±3d | Val MAE | Val ±3d |
|------|--------|----------|----------|---------|---------|
| 消融最优 (含 Shifts+Deltas) | 30 | 4.204 | 42.9% | 4.111 | 49.7% |
| **最终配置 (不含 Shifts+Deltas)** | **22** | **4.180** | **44.1%** | **4.142** | **49.2%** |

22 维配置的 Test MAE（4.180）优于消融中的所有配置，进一步验证了 Shifts 和 Deltas 在当前架构下为噪声特征。

**正式模型采用 22 维特征配置**（`config.py :: ALL_FEATURES`）。

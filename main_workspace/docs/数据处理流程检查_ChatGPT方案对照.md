# ChatGPT 数据处理流程检查与对照

本文档对照 **mcPHASES 实际数据表** 与 **《数据处理计划_基于中期报告》**，对 ChatGPT 给出的 7 步流程做合理性检查与列名/表名对应整理。**不涉及代码实现。**

---

## 一、STEP 1 — 数据质量过滤：是否合理、是否对应表/列

### 1.1 HRV 表过滤

| ChatGPT 规则 | 实际表/列 | 是否对应 | 说明 |
|--------------|-----------|----------|------|
| coverage >= 0.6 | **heart_rate_variability_details**（周期版：heart_rate_variability_details_cycle.csv）的 **coverage** 列 | ✅ 对应 | 列名一致，阈值 0.6 与计划文档「剔除覆盖不足」一致，合理。 |
| rmssd > 0 | 同表的 **rmssd** 列 | ✅ 对应 | 列名一致；>0 可剔除无效/异常块，合理。 |
| LF > 0 | 同表的 **low_frequency** 列 | ✅ 对应 | 表中列名为 **low_frequency**，不是 "LF"，实现时用 `low_frequency`。 |
| HF > 0 | 同表的 **high_frequency** 列 | ✅ 对应 | 表中列名为 **high_frequency**，不是 "HF"，实现时用 `high_frequency`。 |

- **验收标准**：过滤后行数 / 原行数 > 0.7。合理；若 <0.7 需放宽阈值。

---

### 1.2 HR 表过滤

| ChatGPT 规则 | 实际表/列 | 是否对应 | 说明 |
|--------------|-----------|----------|------|
| HR < 30 or HR > 220 → drop | **heart_rate**（周期版：heart_rate_cycle.csv）的 **bpm** 列 | ✅ 对应 | 表中心率列名为 **bpm**，不是 "HR"；30–220 bpm 为常见生理范围，合理。 |
| confidence < 0.5 → drop | 同表的 **confidence** 列 | ✅ 对应 | 列名一致；低置信度剔除合理。 |

- **验收标准**：删除比例 < 15%。合理。

---

### 1.3 WT 表过滤

| ChatGPT 规则 | 实际表/列 | 是否对应 | 说明 |
|--------------|-----------|----------|------|
| \|temp_diff\| > 5°C → drop | **wrist_temperature**（周期版：wrist_temperature_cycle.csv）的 **temperature_diff_from_baseline** 列 | ✅ 对应 | 表中列名为 **temperature_diff_from_baseline**（相对基线的偏差，单位 °C），不是 "temp_diff"；\|偏差\| > 5°C 属极端异常，合理。 |

- **验收标准**：删除比例 < 5%。合理。

---

### STEP 1 小结

- 规则与现有周期表（*_cycle.csv）均可对应，**先做质量过滤再后续步骤** 的顺序正确。
- **列名映射**（实现时需用实际列名）：
  - HR → **bpm**（heart_rate）
  - LF → **low_frequency**，HF → **high_frequency**（heart_rate_variability_details）
  - temp_diff → **temperature_diff_from_baseline**（wrist_temperature）

---

## 二、STEP 2 — 时间窗口切分：是否可行、数据从哪来

### 2.1 窗口定义与数据来源

| 窗口 | ChatGPT 定义 | 需要的数据 | 实际数据来源 |
|------|--------------|------------|--------------|
| **morning** | wake → +30min | 每晚「醒来时间」 | **computed_temperature_cycle** 或 **sleep** 表：**sleep_end_timestamp**（即 wake）。 |
| **evening** | sleep_start −30min → sleep_start | 每晚「入睡时间」 | 同上：**sleep_start_timestamp**。 |
| **sleep** | sleep_start → sleep_end | 整段睡眠起止时间 | 同上：**sleep_start_timestamp**、**sleep_end_timestamp**。 |
| **full** | 全天 | 当日所有时间点 | HR/HRV/WT 表内该 **day_in_study** 下所有 **timestamp**，无需睡眠表。 |

- **结论**：morning / evening / sleep 依赖「每夜睡眠起止时间」。当前可用的来源为：
  - **computed_temperature_cycle.csv**：已有 **sleep_start_timestamp**、**sleep_end_timestamp**、**sleep_end_day_in_study**（归属日），可对应用 **day_in_study = sleep_end_day_in_study** 做窗口切分；
  - 或 **sleep.csv**（需同样按 sleep_end_day_in_study 归属到 day_in_study）。
- **注意**：computed_temperature 仅覆盖「有夜间温度的那一夜」，不是所有 (id, study_interval, day_in_study) 都有睡眠起止；因此「至少 80% day 有数据」应理解为：在**有睡眠边界的 day** 上，各窗口内样本满足一定覆盖率，或整体上 80% 的周期内日至少有一个窗口有数据。需在实现时明确「80%」的分子分母（例如：周期内日数 vs 有该窗口数据的日数）。

### 2.2 时间与 day_in_study 的对应关系

- HR / HRV / WT 表中 **timestamp** 为**当日时间**（如 23:25:00），与 **day_in_study** 一起表示「该研究日内某时刻」。
- 睡眠可能**跨日**（如 23:00 day7 → 07:00 day8）。数据处理计划约定：一夜归属到 **sleep_end_day_in_study**（醒来那天）。因此：
  - 对 **day_in_study = 8**：sleep 窗口取「结束在 day 8 的那一夜」的 sleep_start / sleep_end；
  - morning = 该夜 sleep_end → sleep_end+30min（均在 day 8 的时间轴上可表示）；
  - evening = 该夜 sleep_start−30min → sleep_start（可能落在**前一个** day_in_study 的 22:30–23:00）。若原始数据按「日期」存，则 22:30–23:00 可能在 day_in_study=7；若按「醒来日」统一归属，则需在实现时统一约定 evening 归属到 day 7 还是 day 8，并在文档中写明。

### 2.3 验收标准

- 「每个 window 至少 80% day 有数据」：需明确是「占 cycle 内总日数」还是「占该 id×study_interval 内总日数」，以及「有数据」指「该窗口内至少有一条 HR/HRV/WT 记录」。

---

## 三、STEP 3 — 日级聚合：与表/列对应

### 3.1 HR 聚合

- 数据来源：**heart_rate_cycle**（过滤后），列 **bpm**。
- 生成：hr_mean, hr_std, hr_min, hr_max → 与 ChatGPT 一致，**需按 (id, study_interval, day_in_study) × 窗口** 各做一次（即每个 window 一套日级 HR 统计）。

### 3.2 HRV 聚合

- 数据来源：**heart_rate_variability_details_cycle**（过滤后），列 **rmssd**, **low_frequency**, **high_frequency**。
- 生成：rmssd_mean, lf_mean, hf_mean, **lf_hf_ratio**。  
  - lf_hf_ratio = low_frequency / high_frequency（同日内先聚合 LF、HF 再比，或先算每条 ratio 再聚合，需统一）。列名实现时用 **low_frequency** / **high_frequency**。

### 3.3 WT 聚合

- 数据来源：**wrist_temperature_cycle**（过滤后），列 **temperature_diff_from_baseline**。
- 生成：wt_mean, wt_std, wt_max, wt_min → 与 ChatGPT 一致；实现时该列即「WT」含义。

### 3.4 已日级、直接保留

| ChatGPT 名称 | 实际来源与列 | 说明 |
|--------------|--------------|------|
| nightly_temperature | **computed_temperature_cycle** 的 **nightly_temperature** 列；按 **day_in_study**（= sleep_end_day_in_study）已归属到日 | 每夜一行，直接 merge 到日级表。 |
| resting_hr | **resting_heart_rate_cycle** 的 **value** 列 | 表中列名为 **value**，不是 "resting_hr"；按 (id, study_interval, day_in_study) 已日级。 |

- **验收**：每张日级表行数 = 唯一 (id, study_interval, **day_in_study**)，无重复。注意：**统一用 day_in_study**，不要用 "day" 以免与其它表混淆。

---

## 四、STEP 4 — 合并特征表

- Merge key：**on = (id, study_interval, day_in_study)**（与数据处理计划一致）。ChatGPT 写的 "day" 应统一为 **day_in_study**。
- 产出：daily_features_fullwindow.csv、daily_features_sleep.csv、daily_features_morning.csv、daily_features_evening.csv（或按你最终命名的 4 个窗口表）。
- **验收**：所有 merge 后行数一致（以 cycle 内日集合为基准做外连接或右连接时，行数应等于周期内日数）。

---

## 五、STEP 5 — 缺失值处理

- 「少量缺失 mean impute、整天缺失 mask」+ **feature_missing_flag**：合理。
- 「没有 NaN 但有 mask 列」：即用均值（或其它填充）把 NaN 填掉，同时用 mask 列标记「该处原为缺失」，便于模型或评估时区分。实现时需约定：mask=1 表示该特征在该日为缺失/已填充。

---

## 六、STEP 6 — 个体归一化（**已采用做法 B：滚动、仅用历史**）

- **ChatGPT 原方案（不采用）**：z = (x − mean_subject) / std_subject，即用该 subject **全时段**的均值和标准差，会造成时序泄漏。
- **本流程采用（做法 B）**：与《数据处理计划_基于中期报告》阶段 3 一致，使用**仅用 t 之前 K 天的滚动窗口**做 within-individual 标准化：
  - 按 (id, study_interval) 分组，按 day_in_study 排序；
  - 对每个时刻 t：μ_t、σ_t **只由 t 之前**的观测计算（如前 K 天；不足 K 天时用扩展窗口）；
  - 公式：**z_t = (x_t − μ_t) / (σ_t + ε)**，σ_t 设下界避免除零。
- **结论**：采用做法 B，避免用未来信息，与经期预测等时序任务一致。

---

## 七、STEP 7 — 输出与文件结构

- **X shape = (N_days, N_features)**、**index.csv** 含 id, study_interval, day，且 **len(X) == len(index)**：合理。
- **index 列名**：建议与全流程统一为 **id, study_interval, day_in_study**（不要仅写 "day"）。
- **最终文件结构**：ChatGPT 给出的是 processed_data/ 下 morning.csv, evening.csv, sleep.csv, full.csv + index.csv。若 4 个文件对应 4 个窗口的**日级特征表**，则与 STEP 4 的 4 个 daily_features_* 一致；仅命名风格不同（可二选一或加说明）。

---

## 八、总表：列名/表名速查（实现时对照）

| 概念/称呼 | 实际表（周期版） | 实际列名 |
|-----------|------------------|----------|
| HR | heart_rate_cycle | bpm, confidence |
| HRV (LF/HF/rmssd, coverage) | heart_rate_variability_details_cycle | rmssd, coverage, low_frequency, high_frequency |
| WT / temp_diff | wrist_temperature_cycle | temperature_diff_from_baseline |
| 夜间温度（日级） | computed_temperature_cycle（按 day_in_study 归属） | nightly_temperature |
| 静息心率（日级） | resting_heart_rate_cycle | value（即 resting_hr）, error |
| 睡眠起止时间 | computed_temperature_cycle 或 sleep 表 | sleep_start_timestamp, sleep_end_timestamp；归属日用 sleep_end_day_in_study |
| 日主键 | 所有日级表 | id, study_interval, **day_in_study** |

---

## 九、建议优先落实的修正与约定

1. **全流程统一使用 day_in_study**，index 与 merge key 均用 (id, study_interval, day_in_study)。
2. **STEP 1**：用实际列名（bpm, low_frequency, high_frequency, temperature_diff_from_baseline）写过滤条件。
3. **STEP 2**：明确 morning/evening/sleep 的睡眠边界来源（computed_temperature_cycle 或 sleep.csv），以及「80% day 有数据」的统计口径；明确 evening 跨日时归属到哪一个 day_in_study。
4. **STEP 6**：**已采用做法 B**——滚动、仅用历史的 within-individual 标准化（见第六节）；实现时按 (id, study_interval) 分组、按 day_in_study 排序后，对每个 t 仅用 t 之前 K 天计算 μ_t、σ_t。
5. **3.4**：nightly_temperature 来自 computed_temperature；resting_hr 来自 resting_heart_rate 的 **value** 列。

以上整理可直接用于评审和实现时的对照，无需先写代码即可按此检查与调整流程。

# Data Processing Flow Check and Mapping (ChatGPT vs Actual)

This document checks the **7-step flow** suggested by ChatGPT against **mcPHASES tables** and the **《数据处理计划_基于中期报告》**, and maps column/table names. **No code implementation.**

---

## 1. STEP 1 — Data quality filters: tables/columns

### 1.1 HRV

| ChatGPT rule | Actual table/column | Match | Note |
|--------------|---------------------|-------|------|
| coverage >= 0.6 | **heart_rate_variability_details_cycle** → **coverage** | ✅ | Same name; 0.6 matches plan. |
| rmssd > 0 | same table **rmssd** | ✅ | Valid. |
| LF > 0 | **low_frequency** (not "LF") | ✅ | Use `low_frequency`. |
| HF > 0 | **high_frequency** (not "HF") | ✅ | Use `high_frequency`. |

### 1.2 HR

| ChatGPT rule | Actual | Match | Note |
|--------------|--------|-------|------|
| HR < 30 or > 220 → drop | **heart_rate_cycle** → **bpm** | ✅ | Use **bpm**; range 30–220 bpm. |
| confidence < 0.5 → drop | **confidence** | ✅ | Valid. |

### 1.3 WT

| ChatGPT rule | Actual | Match | Note |
|--------------|--------|-------|------|
| \|temp_diff\| > 5°C → drop | **wrist_temperature_cycle** → **temperature_diff_from_baseline** | ✅ | Use **temperature_diff_from_baseline**; \|deviation\| > 5°C. |

**STEP 1 summary**: All rules map to cycle tables; **column names**: HR → **bpm**, LF → **low_frequency**, HF → **high_frequency**, temp_diff → **temperature_diff_from_baseline**.

---

## 2. STEP 2 — Time windows and data source

- **Morning**: wake → +30 min → need **sleep_end_timestamp** (e.g. from **computed_temperature_cycle** or **sleep**).
- **Evening**: sleep_start −30 min → sleep_start → **sleep_start_timestamp**.
- **Sleep**: sleep_start → sleep_end.
- **Full**: full calendar day; all timestamps for that day_in_study.

**Note**: computed_temperature covers only nights with temperature; “80% day have data” should be defined (e.g. 80% of cycle days with at least one window). Evening may span two calendar days; decide attribution to day_in_study (e.g. sleep_end_day_in_study).

---

## 3. STEP 3 — Daily aggregation

- **HR**: heart_rate_cycle (filtered), **bpm** → hr_mean, hr_std, hr_min, hr_max per (id, study_interval, day_in_study) × window.
- **HRV**: heart_rate_variability_details_cycle → rmssd_mean, lf_mean, hf_mean, **lf_hf_ratio** (use **low_frequency**, **high_frequency**).
- **WT**: wrist_temperature_cycle → wt_mean, wt_std, wt_min, wt_max.
- **nightly_temperature**: computed_temperature_cycle **nightly_temperature**, by day_in_study = sleep_end_day_in_study.
- **resting_hr**: resting_heart_rate_cycle **value** (not "resting_hr" column name).

---

## 4. STEP 4 — Merge

- Merge key: **(id, study_interval, day_in_study)**. Use **day_in_study** everywhere, not "day".
- Output: four window tables (e.g. full, sleep, morning, evening) + index; row count = cycle-internal days.

---

## 5. STEP 5 — Missing values

- “Few missing → mean impute; whole day missing → mask” + **feature_missing_flag**. No NaN in final table; mask=1 means that position was missing/filled.

---

## 6. STEP 6 — Within-individual normalisation (Approach B: rolling, history only)

- **Not used**: z = (x − mean_subject) / std_subject (full history) → temporal leakage.
- **Used (Approach B)**: z_t = (x_t − μ_t) / (σ_t + ε), μ_t and σ_t from **past K days only** for that (id, study_interval), ordered by day_in_study.

---

## 7. STEP 7 — Output

- X shape = (N_days, N_features); index.csv with id, study_interval, **day_in_study**; len(X) == len(index). Final layout: e.g. processed_data/ with full.csv, sleep.csv, morning.csv, evening.csv, index.csv.

---

## 8. Quick reference: concept → actual table/column

| Concept | Actual table (cycle) | Actual column |
|---------|----------------------|---------------|
| HR | heart_rate_cycle | bpm, confidence |
| HRV (LF/HF, rmssd, coverage) | heart_rate_variability_details_cycle | rmssd, coverage, low_frequency, high_frequency |
| WT / temp_diff | wrist_temperature_cycle | temperature_diff_from_baseline |
| Nightly temperature (daily) | computed_temperature_cycle (by day_in_study) | nightly_temperature |
| Resting HR (daily) | resting_heart_rate_cycle | value, error |
| Sleep start/end | computed_temperature_cycle or sleep | sleep_start_timestamp, sleep_end_timestamp; day = sleep_end_day_in_study |
| Day key | All daily tables | id, study_interval, **day_in_study** |

---

## 9. Recommended conventions

1. Use **day_in_study** everywhere for index and merge keys.
2. STEP 1: Use actual column names (bpm, low_frequency, high_frequency, temperature_diff_from_baseline).
3. STEP 2: Define sleep-boundary source and “80% day” denominator; define evening attribution when it spans days.
4. STEP 6: Use **Approach B** (rolling, past-only) for within-individual standardisation.
5. Section 3.4: nightly_temperature from computed_temperature; resting_hr from resting_heart_rate **value** column.

# Data Processing Plan Based on Interim Report

This document uses only the **interim report** and **mcPHASES dataset documentation** to define data processing needed for the reported method, target data structures, and an executable plan. It does not depend on open_source content.

---

## 1. Interim report requirements (summary)

### 1.1 Data and use

- **Source**: mcPHASES (PhysioNet), 42 participants, two study periods (2022 / 2024). Cycles and ovulation: **ovulation** as time anchor; natural cycle = between two consecutive menses starts; keep **clear, complete** cycles only.
- **Ovulation labels**: From daily urine LH (and estrogen) surge rules; **not** model input (annotation only).
- **Model input**: **Wearable-derived daily features only**; **no** LH, PdG, E3G, etc., to mimic real deployment and avoid leakage.

### 1.2 Physiological features (report §3.2)

- **HRV**: During sleep, ~5 min resolution → **daily** aggregate; exclude low-coverage or poor-quality nights.
- **WST (wrist skin temperature)**: Deviation from personal baseline → **daily** aggregate (e.g. main sleep or defined night window).
- **Standardisation**: **Within-individual, rolling baseline using only past**: z_t = (x_t − μ_t) / (σ_t + ε); μ_t, σ_t from **past K days**; extend window if fewer than K days; σ_t lower-bounded.

### 1.3 Other modalities and alignment

- Sleep and self-report: aggregate to daily and align with physiological features and cycle anchors. Output: one **daily feature vector** per day; **no hormone measurements**.

### 1.4 Supervision

- **Main task**: Menses prediction → each day needs “**days until next menses start**” (time-to-event).
- **Auxiliary**: Ovulation → 0/1 only on **days with ovulation annotation**; other days masked in auxiliary loss.

---

## 2. Current data status

| Item | Status | Output |
|------|--------|--------|
| Hormones and cycle structure | ✅ | cycle_anchor_clean.csv: full cycles, id, study_interval, day_in_study, phase, lh, estrogen, ovulation_day_method1/2 |
| Cycle boundaries and ovulation labels | ✅ | Cycle definition (Menstrual→Menstrual), LH surge detection, two ovulation-day definitions |

**Not yet done** (report §4.1.1): systematic cleaning and feature extraction for continuous wearable signals (HRV, WST); daily alignment with cycle anchors; within-individual standardisation; merging with sleep/self-report; and supervision labels (days_until_next_menses, ovulation 0/1).

---

## 3. mcPHASES tables and join keys

All use **(id, study_interval, day_in_study)** as daily key, aligned with cycle_anchor_clean.

- **hormones_and_selfreport**, **cycle_anchor_clean**: direct day_in_study.
- **heart_rate_variability_details**: (id, study_interval, day_in_study, timestamp); aggregate by day + quality filter (e.g. coverage).
- **computed_temperature**: one row per night; use **sleep_end_day_in_study** (or sleep_start) to attribute to a day_in_study.
- **sleep**: attribute to one day_in_study (e.g. **sleep_end_day_in_study**).
- **resting_heart_rate**: already daily by (id, study_interval, day_in_study).

**Temperature/sleep attribution**: One night → **sleep_end_day_in_study** (wake day). Same for sleep segments (e.g. main sleep only).

---

## 4. Target outputs

### 4.1 Daily multimodal feature table (no hormones)

- **Key**: id, study_interval, day_in_study.
- **Columns**: identifiers; optional cycle_id, day_in_cycle, phase; HRV_z, WST_z, sleep (optional), self-report (optional); **no** lh, estrogen, pdg.
- **Standardisation**: Within-individual rolling (past only) for HRV, WST.

### 4.2 Cycle-level labels and supervision

- **days_until_next_menses**: next menses start day − day_in_study (or store next_menses_day_in_study).
- **ovulation_label**: 0/1 only on annotated days; NA elsewhere (mask in training). Optional: keep method1/method2 for ablation.

### 4.3 Single training view

- Join 4.1 and 4.2 on (id, study_interval, day_in_study). Rows = days in complete cycles only. Sequence model: group by (id, study_interval, cycle_id), sort by day_in_cycle or day_in_study; mask auxiliary loss on steps without ovulation label.

---

## 5. Processing stages (concise)

- **Stage 0**: Confirm cycle anchors; add cycle_id, next_menses_start_day if missing; optional method3 ovulation.
- **Stage 1**: Daily aggregation for HRV, WST (computed_temperature + optional wrist_temperature), sleep; quality filters; output per (id, study_interval, day_in_study).
- **Stage 2**: Right-join to cycle_anchor_clean; keep only in-cycle days; leave NA for missing wearable/sleep.
- **Stage 3**: Within-individual rolling standardisation (past K days only) for HRV, WST.
- **Stage 4**: Compute days_until_next_menses; set ovulation_label from method1/method2 (1 only on ovulation day, else 0; NA if cycle has no annotation).
- **Stage 5**: Merge self-report (no hormones); export daily_features_with_labels or daily_features + daily_labels. Ensure **no hormone columns**.
- **Stage 6** (optional): Missing and quality report per id/cycle.

---

## 6. Deliverables

- Daily feature table (no hormones), cycle/supervision columns, single training table/view; optional cycle meta table. All aligned with report §3.2–3.4 and subject-level split for evaluation.

Original Chinese document: `docs/数据处理计划_基于中期报告.md`.

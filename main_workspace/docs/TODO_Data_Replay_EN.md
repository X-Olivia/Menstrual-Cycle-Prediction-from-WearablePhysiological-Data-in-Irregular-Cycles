# TODO: Data Replay and Potential Issues

**Purpose**: Read the interim report first, then check experiment code and existing data item by item, record “unresolved issues”, then continue experiments.

---

## Part A: Execution Checklist (Step by Step)

Work through in order; aim to complete Part A steps 1–4 in half a day, step 5 in the afternoon or next day.

---

### Step 1: Read the interim report (~30–45 min)

- [ ] **1.1** Read **Abstract + §1 Introduction**: Note what the report says is “done” vs “not yet”; highlight or tag:
  - Claims of “completed” work;
  - “Not yet” or “next phase” items.
- [ ] **1.2** Read **§2 Literature Review**: Skim; focus on 2.2 ovulation physiology, 2.3 methods (calendar vs physiological).
- [ ] **1.3** Read **§3 Methodology** (main focus):
  - 3.2 Data and features: cycle definition, ovulation label definition, **what is model input vs annotation only**, standardisation;
  - 3.3 Model: main task (period prediction), auxiliary (ovulation), masking;
  - 3.4 Evaluation: subject-level split, metrics.
- [ ] **1.4** Read **§4 Implementation and Preliminary Results**:
  - Compare 4.1.1–4.1.4 with “what was actually done / known issues”;
  - Note **report vs reality** (e.g. step only half-done, data anomaly, parameter not in report).
- [ ] **1.5** Read **§5 Progress + Reflection**: Timeline and reflection; recall where you got stuck and what was compromised.

**Output**: Short “report vs actual” notes + list of doubts (Step 4 will complete these).

---

### Step 2: Walk through cycle_anchor code (~45–60 min)

Path: `main_workspace/data_process/cycle_anchor.ipynb`. Run top to bottom, no skipping.

- [X] **2.1** **Load and extract** (Sections 1–3): Confirm input is `hormones_and_selfreport.csv`, columns id, study_interval, day_in_study, phase, lh, estrogen; no wrong file or missing columns.
- [X] **2.2** **Cycle identification and grouping** (Sections 4–5): Cycle = from Menstrual to next Menstrual; how cycle 0 is handled; group = id + study_interval, subgroup = cycle; note whether **cycle_id is written to final CSV** (if not, downstream must recompute or add).
- [X] **2.3** **Missing and complete cycles** (Sections 5–6): Complete = no missing lh/estrogen/phase in that cycle; whole-cycle drop if any missing; note whether missing is per-row or “any day missing → drop whole cycle”.
- [X] **2.4** **Day continuity and removal** (Sections 6–7.1): Non-contiguous = gap in day_in_study within cycle; report says “removed 18_2024 Cycle 1”; confirm **only** Cycle 1 was removed and **18_2024 Cycle 2** (also non-contiguous) is still in the 109 cycles (e.g. missing day 974)—sequence models assuming contiguous days would be wrong.
- [X] **2.5** **LH surge and ovulation day labels** (Sections 8–9): Baseline = LH mean in days 1–6 after menses (or first 6 days), denoised; surge = from day 7, LH/baseline ≥ threshold, consecutive days; **threshold**: code 2.0 vs 2.5 in comments/docs → log as issue; Method 1 = surge start +1, Method 2 = surge peak +1; note **how many cycles have baseline/surge** (e.g. 103 baseline, 81 surge → 28 cycles with no ovulation label).
- [X] **2.6** **Save and export**: Which columns are written to `cycle_anchor_clean.csv`; whether cycle_id, day_in_cycle, next_menses_start_day are included (if not, add later for modelling).

**Output**: Fill “Experimental data potential issues” (Part B) with Step 2 findings.

---

### Step 3: Check output data files (~20–30 min)

- [X] **3.1** **cycle_anchor.csv**: Row count, column names; missing rates for lh/estrogen vs report.
- [X] **3.2** **cycle_anchor_clean.csv**: Rows (expect 2796 + header), columns; only id, study_interval, day_in_study, phase, lh, estrogen, ovulation_day_method1/2? Any cycle_id / day_in_cycle / next_menses_start_day? Spot-check 2–3 (id, study_interval) for contiguous day_in_study (especially 18_2024 cycle 2); how many rows with ovulation_day_method1/2 = 1 (~78/77?) and how many cycles all 0.
- [X] **3.3** Other tables: Column names and purpose; consistency with report and data plan.

**Output**: Add “missing columns”, “non-contiguous cycle still present”, “sparse ovulation labels” to the issues list.

---

### Step 4: Maintain “Experimental data potential issues” (Part B)

- [X] **4.1** Create or open `main_workspace/docs/实验数据潜在问题清单.md` (template in Part B).
- [X] **4.2** Enter all points from Steps 1–3: brief description, source (report/code/data), must-fix vs can-defer, recommended action.
- [X] **4.3** Mark **1–3 must-fix items** as “before continuing experiments”.

**Output**: A maintained issues list + priority fixes.

---

### Step 5: Fix 1–3 priority items, then continue

- [ ] **5.1** Choose 1–3 “must fix” items (e.g. non-contiguous cycle, missing cycle_id, threshold mismatch).
- [ ] **5.2** Fix: change code or re-run notebook, add columns at export, or remove 18_2024 Cycle 2; update `cycle_anchor_clean.csv` or docs.
- [ ] **5.3** Mark those items “done” or “documented”; then follow `数据处理计划_基于中期报告.md` for wearable features and modelling.

---

## Part B: Experimental data potential issues (template)

Template to fill during replay. Some “possible issues” are pre-filled; verify and edit.

(See `docs/实验数据潜在问题清单.md` or `docs/Experimental_Data_Issues_EN.md` for the table.)

---

## Part C: After replay – where to continue

- Data processing plan: `main_workspace/docs/数据处理计划_基于中期报告.md` (or `docs/Data_Processing_Plan_Interim_Report_EN.md`)
- Next: After Step 5, start **Stage 1 (wearable daily aggregation)** for HRV/WST/sleep features and align with cycle_anchor_clean (or fixed version).

---

**Suggested order**: Step 1 → Step 2 → Step 3 → Step 4 (fill issues as you go) → complete 1–3 items in Step 5 → then continue the data processing plan.

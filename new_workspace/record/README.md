# record

Experiment and feature scripts that use **new_workspace** data (processed_dataset, signals, processed_data).

## build_sleep_daily.py

**Produces `processed_data/2/sleep.csv`** (and full.csv, morning.csv, evening.csv, index.csv).  
Equivalent to main_workspace `data_process/daily_data_2.ipynb`. Reads cycle anchor and wearable signals from `processed_dataset/cycle_cleaned_ov.csv` and `processed_dataset/signals/*.csv`; computes sleep/morning/evening/full window aggregates, cycle position (day_in_cycle, hist_cycle_len, days_remaining_prior), biphasic shift, missing fill, and per-cycle-early z-normalization.

**Prerequisite:** `data_clean.py` → `ovulation_labels.py` → `wearable_signals.py` (so cycle_cleaned_ov.csv and signals/*.csv exist).

**Run:**  
`python record/build_sleep_daily.py` (from new_workspace).

---

## build_features_v4.py

**Produces `daily_features_v4.csv`** (used by oracle_luteal_countdown and other experiments).  
Reads base daily aggregates from `processed_data/sleep.csv`, mcPHASES raw tables, and `processed_dataset/cycle_cleaned_ov.csv`; applies v4 fixes (per-cycle-early z-norm, RHR median, boundary removal, etc.); writes `processed_data/v4/daily_features_v4.csv`.

**Prerequisite:** `processed_data/2/sleep.csv` must exist. Run **build_sleep_daily.py** first (from new_workspace), or copy from main_workspace:  
`main_workspace/processed_data/2/sleep.csv` → `new_workspace/processed_data/sleep.csv`.

**Run:**  
`python record/build_features_v4.py` 

---

## oracle_luteal_countdown_experiment.py

**Oracle + Luteal Countdown**: menstrual prediction comparing LightGBM-only, detected-ovulation hybrid, and **Oracle hybrid** (post-ovulation days use **LH ground truth** for luteal countdown; pre-ovulation use LightGBM).

### Data (new_workspace)

- **Cycle + LH labels**: `processed_dataset/cycle_cleaned_ov.csv` (from data_clean → ovulation_labels).
- **Wearable signals**: `processed_dataset/signals/*.csv` (from wearable_signals.py).
- **Daily features**: `processed_data/v4/daily_features_v4.csv` if present; otherwise falls back to main_workspace path (requires main_workspace feature pipeline to have been run).

### Run order

1. `data_clean.py` → `processed_dataset/cycle_cleaned.csv`
2. `ovulation_labels.py` → `cycle_cleaned_ov.csv`
3. `wearable_signals.py` → `processed_dataset/signals/*.csv`
4. (Optional) Build v4 daily features: run `python record/build_sleep_daily.py`, then `python record/build_features.py`.
5. `python record/oracle_luteal_countdown_experiment.py`

### Evaluation (anchor-day split)

The script reports, in addition to overall metrics, a breakdown of **remaining-days error** at LH anchor days:

- Pre: `ov-7`, `ov-3`, `ov-1`
- Post: `ov+2`, `ov+5`, `ov+10`
Metrics (MAE and ±3d) are printed separately for `LightGBM only`, `Detected-ov hybrid`, and `Oracle hybrid`.

---

## multisignal_ovulation_detection_and_menses_experiment.py

**Multi-signal ovulation detection and menstrual prediction**: temperature, HR, HRV from wearables → rule-based (t-test, CUSUM, HMM, SavGol), ML (Ridge, GBDT, etc.), 1D-CNN, stacking/weighted ensemble for ovulation day; then ovulation + luteal countdown for next-menses prediction. Compares to Oracle (LH truth) and calendar baseline.

### Data (new_workspace)

- **Cycle + LH labels**: `processed_dataset/cycle_cleaned_ov.csv`
- **Wearable signals**: `processed_dataset/signals/*.csv` (same as above)

### Run order

1. `data_clean.py` → `ovulation_labels.py` → `wearable_signals.py`
2. `python record/multisignal_ovulation_detection_and_menses_experiment.py` (from new_workspace)

### Evaluation (anchor-day split)

`multisignal_ovulation_detection_and_menses_experiment.py` now evaluates menstrual prediction **at specific LH anchor days** and prints **separate** metrics for:

- Pre anchors: `ov-7`, `ov-3`, `ov-1`
- Post anchors: `ov+2`, `ov+5`, `ov+10`
The switch between calendar vs luteal countdown is decided at the anchor day (using the same countdown-enabled rule as the script).


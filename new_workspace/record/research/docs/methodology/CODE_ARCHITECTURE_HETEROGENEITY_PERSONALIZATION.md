# Code Architecture for Heterogeneity and Detector Personalization Research

## 1. Purpose

This document specifies how the research plan should be implemented in code without being constrained by the current irregular or personalization branch.

The architecture should satisfy five requirements:

1. keep the current multisignal strict-prefix pipeline as the reusable baseline
2. add subgroup analysis cleanly
3. add detector personalization in explicit `L0-L3` layers
4. keep evaluation and subgroup definitions auditable
5. avoid entangling subgroup construction, detector redesign, and result reporting

## 2. Architectural Principle

The implementation should be layered.

### Layer A. Baseline detector engine

This layer contains the current strict-prefix detector and evaluation logic.

### Layer B. Research metadata layer

This layer constructs:

- subgroup labels
- personalization metadata
- experiment manifests

### Layer C. Personalization adapters

This layer modifies detector calibration according to `L0-L3` without rewriting the detector itself.

### Layer D. Reporting layer

This layer produces:

- overall tables
- subgroup tables
- detector decomposition tables
- exportable result artifacts

This separation is important because the research question is primarily about comparison across subgroups and personalization regimes, not about replacing the core detector each time.

## 3. Recommended Directory Layout

The new research code should live in a separate package area under the current multisignal workspace.

Recommended structure:

```text
new_workspace/record/experiment/multisignal_pipeline/
  research/
    subgrouping.py
    personalization.py
    experiment_specs.py
    experiment_runner.py
    analysis_tables.py
    exports.py
    paths.py
```

This should remain distinct from the current detector implementation modules so that:

- the baseline detector can still run unchanged
- subgroup and personalization logic remain explicit and reviewable

## 4. Module Responsibilities

## 4.1 `research/subgrouping.py`

### Responsibility

Construct history-only subgroup labels.

### Inputs

- cycle table
- user id
- historical cycle history only

### Core outputs

- per-user subgroup assignment
- subgroup-defining statistics
- subgroup eligibility flags

### Recommended functions

- `build_user_history_table(...)`
- `compute_cycle_length_level_group(...)`
- `compute_cycle_variability_group(...)`
- `compute_ovulatory_status_group(...)`
- `build_user_subgroup_table(...)`

### Required output columns

- `user_id`
- `n_history_cycles`
- `hist_cycle_len_mean`
- `hist_cycle_len_std`
- `hist_cycle_len_cv`
- `cycle_length_group`
- `cycle_variability_group`
- `ovulatory_status_group`
- `subgroup_version`

This module must not inspect target-cycle future information.

## 4.2 `research/personalization.py`

### Responsibility

Implement the `L0-L3` detector-personalization regimes.

### Design principle

This module should not redefine the detector. It should provide calibration parameters and adapters that the detector can consume.

### Recommended abstractions

- `DetectorCalibrationProfile`
- `UserHistoryCalibration`
- `PersonalizationLevel`

### Recommended functions

- `build_zero_shot_calibration(...)`
- `build_one_shot_calibration(...)`
- `build_few_shot_calibration(...)`
- `apply_detector_calibration(...)`

### Example calibration fields

- `temp_shift_scale`
- `hr_baseline`
- `ov_frac_prior_mean`
- `ov_frac_prior_std`
- `trigger_bias`
- `localizer_refine_radius`

This module should explicitly map:

- `L0` -> no user calibration
- `L1` -> history priors only
- `L2` -> one prior cycle calibration
- `L3` -> few-shot calibration

## 4.3 `research/experiment_specs.py`

### Responsibility

Declare experiments as structured configs rather than ad hoc code branches.

### Recommended contents

- baseline experiments
- subgroup experiments
- personalization experiments
- sensitivity experiments

### Recommended functions

- `overall_baseline_specs()`
- `subgroup_baseline_specs()`
- `personalization_specs()`
- `secondary_sensitivity_specs()`

### Design rule

Experiment specification should be declarative. The runner should consume specs, not embed experimental logic inline.

## 4.4 `research/experiment_runner.py`

### Responsibility

Run experiments under a common interface and collect standardized outputs.

### Recommended workflow

1. load data
2. build subgroup table
3. build personalization metadata
4. run selected experiment specs
5. save per-cycle outputs
6. aggregate tables

### Recommended functions

- `run_overall_baseline_experiments(...)`
- `run_subgroup_baseline_experiments(...)`
- `run_personalization_experiments(...)`
- `run_secondary_ovulatory_status_analysis(...)`

### Required output granularity

The runner should save:

- per-cycle outputs
- per-user outputs
- subgroup aggregates
- overall aggregates

This is necessary because subgroup questions cannot be answered from top-line averages alone.

## 4.5 `research/analysis_tables.py`

### Responsibility

Produce the paper-facing tables in a stable format.

### Recommended functions

- `make_overall_table(...)`
- `make_detector_table(...)`
- `make_cycle_length_group_table(...)`
- `make_variability_group_table(...)`
- `make_ovulatory_status_table(...)`
- `make_gain_table(...)`

### Design rule

This module should accept normalized result tables and should not rerun detectors.

## 4.6 `research/exports.py`

### Responsibility

Write versioned artifacts to disk for reproducibility.

### Recommended artifact types

- subgroup table CSV
- per-cycle prediction CSV
- overall summary CSV
- subgroup summary CSV
- Markdown report tables
- machine-readable JSON metadata

### Recommended functions

- `export_subgroup_table(...)`
- `export_cycle_predictions(...)`
- `export_overall_summary(...)`
- `export_subgroup_summary(...)`
- `export_experiment_manifest(...)`

## 4.7 `research/paths.py`

### Responsibility

Centralize file and output paths.

### Recommended output root

```text
new_workspace/record/research/results/
```

Recommended structure:

```text
results/
  subgroup_tables/
  overall/
  personalization/
  sensitivity/
  manifests/
```

This avoids scattering outputs across the detector codebase.

## 5. How the Existing Baseline Should Be Reused

The current multisignal pipeline should remain the detector backend for `L0`.

Recommended reuse points:

- detector inference from current `run.py` / `detectors_ml.py`
- current strict-prefix evaluation functions from `menses.py`
- current baseline methods such as `Calendar`

The new research modules should wrap these rather than rewrite them.

## 6. Personalization Integration Strategy

The key architectural rule is:

**personalization must be injected as calibration, not as a separate detector rewrite for each level.**

That means the detector should accept an optional personalization object such as:

```python
calibration = {
    "temp_shift_scale": ...,
    "hr_baseline": ...,
    "ov_frac_prior_mean": ...,
    "trigger_bias": ...,
    "localizer_refine_radius": ...,
}
```

Then:

- `L0` passes `None`
- `L1` passes history-derived priors
- `L2` passes one-shot calibrated values
- `L3` passes few-shot calibrated values

This architecture keeps the scientific comparison clean:

- same detector backbone
- different calibration regime

## 7. Data Objects That Should Exist

The research code should explicitly define the following data objects.

### 7.1 `UserHistorySummary`

Contains:

- count of prior cycles
- historical mean cycle length
- historical variability
- historical ovulatory frequency
- historical wearable baseline summaries

### 7.2 `UserSubgroupRecord`

Contains:

- user id
- subgroup assignments
- subgroup-defining values
- eligibility flags

### 7.3 `DetectorCalibrationProfile`

Contains:

- personalization level
- calibrated detector parameters
- provenance of calibration

### 7.4 `CyclePredictionRecord`

Contains:

- user id
- cycle id
- personalization level
- subgroup labels
- detector outputs
- downstream outputs
- evaluation metrics

These objects are useful because the project’s central claims depend on linking user subgroup, detector behavior, and prediction outcome.

## 8. Required Result Artifacts

Every experiment run should persist the following outputs.

### 8.1 Manifest

A machine-readable experiment manifest with:

- code version
- detector version
- subgroup version
- personalization version
- data version
- timestamp

### 8.2 Per-cycle predictions

A CSV or parquet table with one row per evaluated cycle.

### 8.3 Per-user subgroup table

A versioned table with subgroup assignments and summary stats.

### 8.4 Aggregated summary tables

- overall
- cycle-length level
- cycle variability
- ovulatory status

This ensures reproducibility and paper-traceability.

## 9. Suggested Command-Line Entry Points

The research code should expose a small number of explicit entry points.

Recommended commands:

- `python -m research.experiment_runner --mode baseline`
- `python -m research.experiment_runner --mode subgroup-baseline`
- `python -m research.experiment_runner --mode personalize-l1`
- `python -m research.experiment_runner --mode personalize-l2`
- `python -m research.experiment_runner --mode personalize-l3`
- `python -m research.experiment_runner --mode ovulatory-sensitivity`

Each command should save outputs into the research results directory.

## 10. Recommended Development Order

The build order should mirror the experimental order.

### Step 1

Implement `research/subgrouping.py`

Goal:

- freeze subgroup definitions

### Step 2

Implement `research/experiment_runner.py` in baseline mode

Goal:

- reproduce current detector results with subgroup exports

### Step 3

Implement `research/analysis_tables.py`

Goal:

- produce stable subgroup tables

### Step 4

Implement `research/personalization.py` for `L1`

Goal:

- add history-calibrated personalization without changing detector backbone

### Step 5

Extend to `L2` and `L3`

Goal:

- compare one-shot and few-shot gains

### Step 6

Add secondary ovulatory-status analysis

Goal:

- produce sensitivity tables

## 11. Boundaries: What This Architecture Should Avoid

The research architecture should avoid the following:

- mixing subgroup construction with detector code
- creating a new detector family for each personalization level
- using one-off notebook logic as the source of truth
- embedding subgroup-specific rules directly inside the main detector
- using quality labels as online routing features

These shortcuts would undermine interpretability and make the paper harder to defend.

## 12. Final Architectural Summary

The final research codebase should have the following shape:

1. the current multisignal detector remains the reusable strict-prefix baseline engine
2. subgrouping becomes an explicit and versioned metadata layer
3. personalization is implemented as a calibrated detector adapter with `L0-L3` levels
4. experiment specs and runners make comparisons reproducible
5. subgroup and detector-decomposition tables become first-class outputs

This architecture best supports the paper’s real scientific aim:

not simply building a stronger predictor, but determining **who benefits from wearable physiology, who benefits from detector personalization, and why**.

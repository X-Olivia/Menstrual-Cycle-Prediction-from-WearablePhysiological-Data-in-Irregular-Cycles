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
    experiment_runner.py
    report_builder.py
    exports.py
    paths.py
```

This should remain distinct from the current detector implementation modules so that:

- the baseline detector can still run unchanged
- subgroup and personalization logic remain explicit and reviewable

## 4. Module Responsibilities

## 4.1 `research/subgrouping.py`

### Responsibility

Construct retrospective user-level subgroup labels for post hoc analysis while
preserving per-cycle historical summary fields for cold-start-aware baselines.

### Inputs

- cycle table
- user id
- observed cycle summaries for retrospective subgroup assignment
- per-cycle historical summaries for baseline priors

### Core outputs

- per-user retrospective subgroup assignment
- subgroup-defining statistics
- subgroup eligibility flags
- per-cycle historical summary fields used by history-based baselines
- descriptive user-level variability statistics including both `SD` and `CV`

### Recommended functions

- `build_user_history_table(...)`
- `compute_cycle_length_level_group(...)`
- `compute_cycle_variability_group(...)`
- `build_subgroup_summary(...)`

### Required output columns

- `user_id`
- `n_history_cycles`
- `hist_cycle_len_mean`
- `hist_cycle_len_std`
- `hist_cycle_len_cv`
- `user_cycle_len_std`
- `user_cycle_len_cv`
- `cycle_length_level_group`
- `cycle_variability_group`
- `subgroup_version`

This module has two timing rules:

- retrospective subgroup labels may use observed user-level cycle summaries across the user's available cycles
- per-cycle historical summary fields used by baselines and personalization must remain forward-only and must not inspect target-cycle future information

Implementation note:

For the current main subgroup labels, cycle-variability groups are still
assigned from fixed `SD` thresholds. `CV` is exported alongside `SD` in the
subgroup artifacts for descriptive reporting and possible sensitivity analyses,
but it does not currently change the subgroup label itself.

The earlier `ovulatory_status_group` sketch has been removed from the active
research code because it relied on LH-label presence as a proxy phenotype. That
proxy is too tightly coupled to the same supervision and evaluation labels used
elsewhere in the study.

## 4.2 `research/personalization.py`

### Responsibility

Implement the `L0-L3` detector-personalization regimes.

### Design principle

This module should not redefine the detector. It should provide personalization profile parameters and adapters that the detector can consume.

### Recommended abstractions

- `L1Config`, `L2Config`, `L3Config`
- `PersonalizationProfileTable`

### Recommended functions

- `build_zero_shot_personalization_profile_table(...)`
- `build_one_shot_personalization_profile_table(...)`
- `build_few_shot_personalization_profile_table(...)`
- `apply_l1_zero_shot_personalization(...)`
- `apply_l2_one_shot_personalization(...)`
- `apply_l3_few_shot_personalization(...)`

### Example profile fields

- `temp_shift_scale`
- `hr_baseline`
- `ov_frac_prior_mean`
- `ov_frac_prior_std`
- `trigger_bias`
- `localizer_refine_radius`

This module explicitly maps:

- `L0` -> no user-specific profile
- `L1` -> history priors only
- `L2` -> one prior cycle profile
- `L3` -> few-shot profile

## 4.3 `research/experiment_runner.py`

### Responsibility

Declare and execute experiments, and collect standardized outputs. 

Note: This module replaces the earlier planned `experiment_specs.py` by embedding
experiment methods and configurations directly as functions and internal
definitions.

### Recommended workflow

1. load data
2. build subgroup table
3. build personalization metadata
4. execute specific analysis modes (baseline, L1, L2, L3)
5. save per-cycle outputs
6. aggregate tables

### Main analysis modes

- `run_subgroup_baseline_analysis(...)`
- `run_l1_zero_shot_analysis(...)`
- `run_l2_one_shot_analysis(...)`
- `run_l3_few_shot_analysis(...)`

### Required output granularity

The runner should save:

- per-cycle results (MAE, accuracy)
- subgroup-wise aggregates
- personalization profile tables

This is necessary because subgroup questions cannot be answered from top-line averages alone.

## 4.4 `research/report_builder.py`

### Responsibility

Produce the paper-facing tables and Markdown reports.

Note: This module replaces the earlier planned `analysis_tables.py`. It
aggregates results from multiple experiment runs to generate cross-regime
comparisons.

### Recommended functions

- `build_report()`
- `_load_results(...)`
- `_method_short_name(...)`

### Reporting tasks

- Produce `Wearable Gain Table` (Calendar vs History vs L0)
- Produce `Personalization Table` (L0 vs L1 vs L2 vs L3)
- Produce `Diagnostic Ablation Table` (L2 vs L2a vs L2b)

### Design rule

This module should accept normalized CSV result files and should not rerun detectors.

## 4.5 `research/exports.py`

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

**personalization must be injected as a profile object, not as a separate detector rewrite for each level.**

That means the detector should accept an optional personalization object such as:

```python
profile = {
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
- `L2` passes one-shot profile values
- `L3` passes few-shot profile values

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

Do not add an ovulatory-status subgroup analysis unless the project obtains a
stronger phenotype label than LH-only proxy membership.

Goal:

- avoid coupling subgroup definition to the same label system used for evaluation

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

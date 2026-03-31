# Personalization Level Specification

## 1. Purpose

This document freezes the operational definition of personalization levels `L0-L3`.

Its purpose is to prevent scope drift. The levels below must be treated as a fixed adaptation contract for the main paper. Later experiments may add exploratory variants, but they must not overwrite these primary definitions.

## 2. General Rule

All personalization levels must satisfy the following:

- the detector backbone remains the same
- the feature family remains the same
- the strict subject-wise protocol remains the same
- the strict prefix inference protocol remains the same
- only the explicitly allowed calibration parameters may differ across levels

The intention is to compare personalization levels, not to compare a sequence of increasingly different detectors.

## 3. Personalization Levels

## L0. Population

### Definition

No user-specific detector calibration.

### Allowed user-specific inputs

- none beyond the standard population-model inputs already used by the baseline

### Allowed adaptations

- none

### Role in the paper

This is the main non-personalized wearable baseline.

## L1. Zero-shot history-calibrated

### Definition

No parameter fitting on prior user cycles. Only history-derived user priors may be used to calibrate the detector.

### Allowed adaptations

- `ov_frac_prior_mean`
- `ov_frac_prior_std`
- `trigger_bias`

### Disallowed adaptations

- `temp_shift_scale`
- `hr_baseline`
- `localizer_refine_radius`
- `localizer_refine_weight`

### Rationale

`L1` should represent the weakest and safest form of personalization: user-specific timing priors and decision calibration, but no user-specific physiological response calibration.

## L2. One-shot personalized

### Definition

Detector calibration using one prior completed cycle from the same user.

### Allowed adaptations

All `L1` adaptations, plus:

- `temp_shift_scale`
- `hr_baseline`

### Disallowed adaptations

- `localizer_refine_radius`
- `localizer_refine_weight`

### Rationale

`L2` introduces a single prior-cycle physiological calibration step without changing localizer dynamics yet.

## L3. Few-shot personalized

### Definition

Detector calibration using two to three prior completed cycles from the same user.

### Allowed adaptations

All `L2` adaptations, plus:

- `localizer_refine_radius`
- `localizer_refine_weight`

### Rationale

`L3` is the strongest planned personalization level. It is the first level allowed to personalize localizer refinement behavior, because that requires more stable user-specific evidence than one-shot calibration.

## 4. Parameter Semantics

To keep the levels interpretable, the main allowed parameters should be understood as follows.

### `ov_frac_prior_mean`

User-specific prior mean for typical ovulation timing expressed as a cycle fraction.

### `ov_frac_prior_std`

User-specific prior uncertainty for ovulation timing.

### `trigger_bias`

User-specific calibration term that shifts the detector’s tendency to trigger.

### `temp_shift_scale`

User-specific normalization or scaling factor for temperature-shift evidence.

### `hr_baseline`

User-specific baseline for resting or nocturnal heart-rate-related features.

### `localizer_refine_radius`

User-specific bound controlling how far localizer refinement may move after an initial estimate.

### `localizer_refine_weight`

User-specific weighting parameter controlling how strongly new localizer evidence can update the existing estimate.

## 5. What Must Stay Fixed Across Levels

The following must remain fixed in the main paper comparison:

- detector architecture family
- feature definition
- subgroup definition
- train/test protocol
- evaluator definitions
- baseline set

If any of these change, the experiment must be labelled as a detector-variant study rather than a personalization-level comparison.

## 6. Reporting Requirement

Every personalization result table in the paper or appendix must explicitly indicate:

- the level label (`L0-L3`)
- the allowed adaptations at that level

This is necessary so that the reader can distinguish:

- added personalization strength
- added detector flexibility

## 7. What Counts as a Violation

The following would violate this specification in the main study:

- introducing a new personalized parameter at `L1` that is not listed above
- changing the detector family between `L0` and `L3`
- changing feature sets across levels
- changing trigger logic only in personalized levels unless the trigger bias is part of the allowed calibration

Any such experiment may still be run, but it must be reported as exploratory and not as part of the main `L0-L3` progression.

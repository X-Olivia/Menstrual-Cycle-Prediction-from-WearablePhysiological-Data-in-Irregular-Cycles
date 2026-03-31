# Experiment Plan for Heterogeneity-Sensitive Menstrual Cycle Prediction

## 1. Purpose

This plan translates the methodology into an executable research program.

The objective is not to maximize one aggregate score as quickly as possible. The objective is to answer, with a controlled experimental design:

1. which irregularity profiles are hardest
2. where wearable physiology improves prediction over calendar history
3. whether detector personalization adds any reliable value beyond the population wearable baseline
4. if such gains exist, whether they occur mainly in the detector rather than downstream countdown

## 2. Fixed Principles

All experiments in this plan must preserve the following constraints.

### 2.1 Subject-wise evaluation

All user-level generalization claims must use strict subject-wise splits.

### 2.2 Strict prefix inference

At cycle day `d`, only observations available up to day `d` may be used.

### 2.3 No subgroup leakage

All subgroup definitions must be constructed from user history only and must not use target-cycle future information.

### 2.4 Quality subgroup use

Signal-quality labels may be used only for offline analysis and sensitivity reporting. They must not be used for online routing or for subgroup definitions that claim deployment realism.

## 3. Main Experimental Axes

The research design has three main axes.

### 3.1 Information regime

- `B0 Calendar-only`
- `B1 Population wearable detector`
- `B2 Personalized wearable detector`

### 3.2 Personalization regime

- `L0 Population`
- `L1 Zero-shot history-calibrated`
- `L2 One-shot personalized`
- `L3 Few-shot personalized`

Note:

- personalization is a secondary analysis axis
- the primary positive claim of the paper should not depend on `L1-L3` outperforming `L0`

### 3.3 Irregularity subgroup regime

Primary subgroup axes:

- `A1 cycle-length level`
- `A2 cycle variability`

Secondary subgroup axis:

- `A3 ovulatory status`

## 4. Subgroup Definition Strategy

Subgroups must be constructed in a staged way.

### 4.1 Phase 1 subgroup set: mandatory core

These are the subgroups that must be implemented first.

#### A1. Cycle-length level

User-level grouping based on historical mean cycle length:

- `short-stable`
- `typical-stable`
- `long-stable`

Recommended implementation:

- use the user’s pre-target historical mean cycle length
- use fixed interpretable thresholds
- require minimum history count before assigning a subgroup

#### A2. Cycle variability

User-level grouping based on historical cycle-length variation:

- `low-variability`
- `medium-variability`
- `high-variability`

Recommended implementation:

- use user-level historical cycle-length standard deviation and coefficient of variation
- require minimum history count before assigning a subgroup

### 4.2 Phase 2 subgroup set: secondary

#### A3. Ovulatory status

User-level grouping:

- `consistently ovulatory`
- `mixed ovulatory`
- `frequently anovulatory`

This should be implemented only after the core subgroup analyses are stable, and it must be reported as a secondary or sensitivity analysis.

### 4.3 Phase 3 subgroup set: compact joint profiles

Only if sample sizes permit:

- `short-stable ovulatory`
- `long-stable ovulatory`
- `high-variability ovulatory`
- `high-variability mixed/anovulatory`
- `anovulation-dominant`

These joint profiles should be exploratory unless each cell is adequately populated.

## 5. Experiment Sequence

The work should proceed in phases. Each phase must produce a saved result artifact before the next phase starts.

## Phase A. Reproducible baseline lock

### Goal

Establish the current multisignal strict-prefix detector as the baseline reference.

### Models

- `Calendar`
- `Population wearable detector`

Recommended baseline detector:

- current `Temp+HR` strict-prefix detector
- current best deployable mainline, not hindsight Oracle

### Required outputs

- overall all-labeled table
- overall quality table
- detector metrics table
- detected-cycle table
- per-user raw prediction export

### Deliverable

A frozen baseline result file that all later subgroup and personalization results can be compared against.

## Phase B. History-only subgroup construction

### Goal

Create subgroup labels without touching the detector yet.

### Input

- user history only
- no target-cycle future information

### Output

A user-level subgroup table containing:

- user id
- history count
- historical mean cycle length
- historical cycle-length SD
- historical cycle-length CV
- cycle-length level group
- cycle-variability group
- optional ovulatory-status group

### Required checks

- subgroup counts
- subgroup balance
- minimum cycle history per subgroup
- users excluded due to insufficient history

### Deliverable

A versioned subgroup definition file to be reused across all later experiments.

## Phase C. Baseline subgroup analysis

### Goal

Answer whether wearable physiology helps different irregularity profiles differently, before personalization is introduced. This is the main positive analysis track of the paper.

### Comparisons

- `Calendar` vs `Population wearable detector`

### Required reporting

For each subgroup under `A1` and `A2`:

- `PostOvDays MAE`
- `PostOvDays ±2d / ±3d`
- `PostTrigger MAE`
- `PostTrigger ±2d / ±3d`
- `OvFirst MAE`
- `OvFinal MAE`
- detected-cycle rate
- latency

### Required analysis

- gain over Calendar by subgroup
- ranking of hardest subgroups
- whether gains occur through better detector metrics, downstream metrics, or both

### Deliverable

A subgroup baseline report answering:

- which subgroup is hardest
- whether wearable signals help all subgroups or only some

## Phase D. L1 Zero-shot history-calibrated personalization

### Goal

Test the weakest and safest form of detector personalization.

### Meaning of L1

No parameter fitting on target-user prior cycles. Use only history-derived priors to calibrate the detector.

### Allowed calibration targets

- user-specific cycle-fraction prior
- user-specific temperature-shift scale normalization
- user-specific resting HR baseline normalization
- user-specific trigger confidence calibration
- user-specific localizer refinement bounds

### Comparisons

- `L0 Population`
- `L1 Zero-shot history-calibrated`

### Required analysis

- overall gain
- subgroup-specific gain
- whether gain is larger in high-variability and shifted users

### Deliverable

A report showing whether history-only personalization already improves the detector.

## Phase E. L2 One-shot personalization

### Goal

Test whether one prior completed cycle materially improves the detector.

### Meaning of L2

The detector is calibrated using one prior completed cycle from the same user.

### Allowed calibration targets

- user-specific temperature response scale
- user-specific ovulation timing prior
- user-specific trigger calibration
- user-specific localizer refinement parameters

### Comparisons

- `L0`
- `L1`
- `L2`

### Required analysis

- overall gain from `L1 -> L2`
- subgroup-specific gain from `L1 -> L2`
- whether one-shot gain is concentrated in harder subgroups

### Deliverable

A one-shot personalization report focused on detector-side gain.

## Phase F. L3 Few-shot personalization

### Goal

Test whether a small number of prior cycles provides additional benefit beyond one-shot personalization.

### Meaning of L3

Detector calibration is adapted using two to three prior completed cycles.

### Comparisons

- `L0`
- `L1`
- `L2`
- `L3`

### Required analysis

- incremental gain from `L2 -> L3`
- subgroup-specific marginal gains
- whether personalization saturates early

### Deliverable

A few-shot personalization report showing whether detector personalization has diminishing returns.

## Phase G. Secondary ovulatory-status analysis

### Goal

Test whether the above findings change across ovulatory-status groups.

### Caution

This phase must be reported as secondary because the subgroup axis shares a noisy label source with supervision and evaluation.

### Required reporting

Repeat subgroup comparison tables for:

- consistently ovulatory
- mixed ovulatory
- frequently anovulatory

### Deliverable

A sensitivity-analysis appendix or secondary-results section.

## 6. Metrics by Phase

To avoid inconsistent reporting, all phases should use the same metric families.

### 6.1 Main downstream metrics

- `PostOvDays MAE`
- `PostOvDays ±2d / ±3d`
- `PostTrigger MAE`
- `PostTrigger ±2d / ±3d`
- `anchor post_all MAE`

### 6.2 Detector metrics

- `OvFirst MAE`
- `OvFirst ±2d / ±3d`
- `OvFinal MAE`
- `OvFinal ±2d / ±3d`
- detected-cycle rate
- first-detection day
- latency

### 6.3 Secondary metrics

- `AllDays MAE`
- `AllDays ±2d / ±3d`

These must still be reported, but not treated as the main success criterion.

## 7. Main Tables for the Paper

The paper should commit in advance to a stable reporting structure.

### Table 1. Overall comparison

Rows:

- Calendar
- L0 Population wearable
- L1 Zero-shot
- L2 One-shot
- L3 Few-shot

Columns:

- PostOvDays MAE
- PostTrigger MAE
- OvFirst MAE
- OvFinal MAE
- detected-cycle rate
- latency

### Table 2. Cycle-length level subgroup gains

Rows:

- short-stable
- typical-stable
- long-stable

Columns:

- Calendar
- L0
- L1
- L2
- L3
- gain over Calendar
- gain from personalization

### Table 3. Cycle-variability subgroup gains

Rows:

- low variability
- medium variability
- high variability

Columns parallel Table 2.

### Table 4. Detector decomposition

Rows:

- L0
- L1
- L2
- L3

Columns:

- OvFirst
- OvFinal
- detected-cycle rate
- latency
- PostTrigger

This table is the key to testing whether personalization works mainly through the detector.

### Table 5. Secondary ovulatory-status analysis

Rows:

- consistently ovulatory
- mixed ovulatory
- frequently anovulatory

This table should be clearly marked as secondary.

## 8. Statistical Comparison Strategy

The paper should not rely only on mean differences.

Recommended comparisons:

- paired subgroup deltas where applicable
- bootstrap confidence intervals for MAE differences
- subgroup-wise uncertainty bars

The purpose is not to overstate significance, but to avoid purely descriptive claims when subgroup sizes differ.

## 9. What Counts as Success

The research program is successful if it can answer the following clearly:

1. Which irregularity profiles are hardest under Calendar?
2. Which irregularity profiles benefit most from wearable physiology?
3. Which irregularity profiles benefit most from personalization?
4. Does personalization mainly improve detector quality rather than only downstream countdown?

This is the primary scientific success criterion, even if absolute aggregate MAE does not yet match the best proprietary commercial systems.

## 10. Risks and Mitigations

### Risk 1. Subgroups become too small

Mitigation:

- prioritise single-axis subgroup analysis
- treat joint profiles as exploratory

### Risk 2. Personalization definition drifts over time

Mitigation:

- freeze `L0-L3` definitions before implementation
- document exactly which detector parameters each level may adapt

### Risk 3. Ovulatory-status analysis is overinterpreted

Mitigation:

- keep it secondary
- state label-source dependence explicitly

### Risk 4. Results become detector-specific and hard to interpret

Mitigation:

- keep Calendar and the same main wearable baseline throughout
- isolate personalization changes from detector-family changes

## 11. Immediate Build Order

The engineering work should proceed in this order:

1. implement subgroup construction from history-only statistics
2. generate subgroup baseline reports for current multisignal detector
3. implement `L1` detector calibration
4. implement `L2`
5. implement `L3`
6. generate subgroup gain tables
7. run secondary ovulatory-status analysis

This order ensures that the study answers the subgroup question before adding more adaptive machinery.

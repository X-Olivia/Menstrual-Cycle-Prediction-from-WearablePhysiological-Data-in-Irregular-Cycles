# Methodology for Heterogeneity-Sensitive Menstrual Cycle Prediction

## 1. Study Focus

This study does not aim to build a single universally optimal menstrual predictor. Its purpose is narrower and more clinically meaningful:

**to test whether wearable physiology and detector personalization are uniformly useful, or whether their benefits depend on structured forms of menstrual irregularity under strict prefix prediction.**

This framing treats menstrual prediction as a heterogeneity-sensitive problem rather than an average-case forecasting problem.

## 2. Main Research Question

The paper should be organised around one central question:

**How do wearable physiology and detector personalization interact with structured menstrual irregularity under strict subject-wise, strict-prefix prediction?**

This question can be answered through two concrete sub-questions:

1. Do wearable physiological signals improve prediction equally across irregularity profiles, or mainly in some subgroups?
2. Does detector personalization provide broad benefit, or does it mainly help harder irregular profiles?

## 3. Core Hypotheses

Only two main hypotheses should be carried by the paper.

### H1. Wearable physiology helps selectively

Wearable physiological signals improve prediction beyond calendar history, but the gain is not uniform across users. It depends on the type of irregularity.

### H2. Personalization mainly helps harder subgroups, and mainly upstream

Detector personalization is expected to provide the largest benefit in harder irregular profiles, and this benefit is expected to arise primarily through improvements in ovulation detection rather than only through downstream menstrual countdown.

## 4. Task Formulation

The study should use a two-stage physiological pipeline:

1. ovulation timing estimation from wearable signals under strict prefix constraints
2. menses onset prediction conditioned on the ovulation estimate

This formulation is preferred over direct end-to-end menstruation prediction because it is more physiologically interpretable and supports error decomposition.

It makes it possible to separate:

- detector-side failure
- downstream countdown-side failure

This separation is central to the paper, because personalization may help one stage more than the other.

## 5. Evaluation Protocol

### 5.1 Strict subject-wise evaluation

All main experiments must remain subject-wise. Test users must not contribute training data to population models.

### 5.2 Strict prefix evaluation

For a target cycle day `d`, the model may use only data visible up to day `d`. No future information may enter:

- features
- smoothing
- trigger logic
- localizer updates
- state transitions

This is a methodological requirement, not an optional robustness check.

### 5.3 Main metric roles

To avoid ambiguity, the paper should assign different roles to different metrics.

#### Primary scientific metric

- `PostOvDays`

This metric is best aligned with the physiological question of how well the system performs after the true ovulation point.

#### Primary deployment metric

- `PostTrigger`

This reflects performance after the model itself has started giving usable countdowns.

#### Supplementary stress-test metric

- `AllDays`

This should still be reported, but it should not be framed as the primary success criterion.

### 5.4 Additional detector metrics

The paper should also report:

- `OvFirst`
- `OvFinal`
- detected-cycle coverage
- first-detection day
- detection latency

These metrics are necessary to determine whether personalization helps by improving the detector itself.

## 6. Irregularity Taxonomy

Irregularity should not be reduced to a single `regular vs irregular` label.

The main analysis should use two primary axes defined from user history only.

### 6.1 Primary axis A: cycle-length level

This axis captures users whose cycles are internally stable but systematically shifted away from the conventional calendar assumption.

Recommended categories:

- short-stable
- typical-stable
- long-stable

These categories should be defined using each user’s historical mean cycle length only, without using the target cycle.

### 6.2 Primary axis B: cycle-to-cycle variability

This axis captures users whose cycles fluctuate substantially across cycles.

Recommended categories:

- low variability
- medium variability
- high variability

These categories should be defined using history-based dispersion measures such as standard deviation or coefficient of variation.

### 6.3 Secondary axis: ovulatory status

Ovulatory status should be analysed, but it should **not** be one of the main grouping axes in the core results.

Recommended categories:

- consistently ovulatory
- mixed ovulatory
- frequently anovulatory

However, this axis should be treated as a **secondary or sensitivity analysis**, because it is partly entangled with the same LH-based label system used for supervision and evaluation.

## 7. Personalization: Operational Definition

The paper must define personalization operationally rather than conceptually.

Personalization in this study refers to **detector personalization**, not primarily to personalized luteal countdown.

### 7.1 Four personalization levels

The paper should compare exactly four levels:

- `L0 Population`
- `L1 Zero-shot history-calibrated`
- `L2 One-shot personalized`
- `L3 Few-shot personalized`

### 7.2 Meaning of each level

#### L0 Population

Cross-user detector with no user-specific adaptation beyond globally shared modeling.

#### L1 Zero-shot history-calibrated

No parameter fitting on user-specific prior cycles, but user-specific historical priors are used for calibration.

#### L2 One-shot personalized

Detector calibration is adapted using one prior completed cycle from the same user.

#### L3 Few-shot personalized

Detector calibration is adapted using two to three prior completed cycles from the same user.

### 7.3 What is allowed to be personalized

Personalization should act on detector calibration parameters such as:

- user-specific temperature shift scale
- user-specific resting HR or nocturnal HR baseline
- user-specific ovulation timing prior in cycle-fraction form
- user-specific trigger confidence calibration
- user-specific localizer refinement bounds

### 7.4 What should not define personalization

The paper should not define personalization primarily as:

- historical average cycle length alone
- historical cycle-length variability alone
- personalized luteal length alone

These may still be used as supporting variables, but they should not stand in for detector personalization.

## 8. Model Comparison Structure

Models should be compared by information structure, not only by algorithm name.

### 8.1 Calendar-only baseline

Uses historical cycle information only, with no wearable physiology.

Purpose:

- quantify the calendar ceiling
- show where irregularity breaks average-based assumptions

### 8.2 Population wearable detector

Uses wearable signals under strict prefix constraints without user-specific adaptation.

Purpose:

- quantify the value of wearable physiology alone

### 8.3 Personalized wearable detector

Uses wearable signals under strict prefix constraints plus detector personalization at levels `L1-L3`.

Purpose:

- quantify whether personalization adds value
- identify which irregularity profiles benefit most

## 9. Detector Architecture Recommendation

The detector should be described as a hybrid physiological detector, not as a generic tabular classifier.

Recommended structure:

- supervised phase model for `has ovulated by today`
- stateful ovulation localizer for concrete day estimation
- detector calibration layer for personalization

This framing is consistent with both the current multisignal baseline and the broader wearable ovulation literature, where strong performance usually comes from coordinated signal interpretation rather than from a single model family alone.

## 10. Experimental Design

The paper should not use an overly fragmented full factorial design as the main presentation. Instead, it should use a staged design.

### Stage 1: overall baseline comparison

Compare:

- Calendar-only
- Population wearable detector
- Personalized detector levels `L1-L3`

This establishes whether wearable signals and personalization help at all.

### Stage 2: single-axis subgroup analysis

Evaluate gains separately across:

- cycle-length level groups
- cycle variability groups

This should be the core subgroup analysis.

### Stage 3: secondary analyses

Only after Stage 2, and only if sample sizes are sufficient:

- compact joint profile view
- ovulatory-status analysis

These analyses should be labelled exploratory or sensitivity analyses if subgroup sizes are small.

## 11. Primary Outcome Tables

The main paper should include:

### Table A. Overall performance

For each model level:

- `PostOvDays MAE`
- `PostOvDays ±2d / ±3d`
- `PostTrigger MAE`
- `PostTrigger ±2d / ±3d`
- `AllDays MAE`

### Table B. Detector performance

For each model level:

- `OvFirst MAE`
- `OvFinal MAE`
- detected-cycle rate
- first-detection day
- latency

### Table C. Subgroup gains

For each irregularity subgroup:

- gain over Calendar
- gain from wearable physiology
- gain from personalization

This table should be the main answer to the research question.

## 12. What the Paper Should Claim

The paper should claim only the following:

1. Wearable physiology is useful, but not equally useful for all irregular users.
2. Detector personalization is not uniformly beneficial.
3. Personalization gains are concentrated in some harder irregularity profiles.
4. These gains arise mainly through the detector rather than only through downstream countdown refinement.

## 13. What the Paper Should Not Claim

The paper should not claim:

- that personalization is universally beneficial
- that all irregular users form one meaningful category
- that aggregate MAE alone is sufficient to characterise success
- that ovulatory-status subgroup findings are free from label-source dependence

The paper should explicitly acknowledge that ovulation truth is inferred rather than clinically definitive, and that ovulatory-status subgrouping partially shares this noisy label source.

## 14. Immediate Experimental Roadmap

The next implementation steps should follow this order:

1. construct history-only subgroup definitions for:
   - cycle-length level
   - cycle variability
2. evaluate the current multisignal strict-prefix detector across those subgroup definitions
3. implement detector personalization levels `L1-L3`
4. compare subgroup-specific gains
5. add ovulatory-status only as a secondary analysis

This order keeps the study focused and avoids mixing subgroup construction, detector redesign, and personalization claims all at once.

## 15. Recommended Contribution Statement

The paper’s contribution should be framed as follows:

1. It reframes menstrual cycle prediction as a heterogeneity-sensitive problem.
2. It evaluates structured irregularity instead of collapsing users into a single irregular category.
3. It quantifies where wearable physiology improves prediction and where it does not.
4. It tests whether detector personalization is broadly useful or mainly valuable for harder irregularity profiles.
5. It does so under strict subject-wise and strict-prefix evaluation, reducing hindsight bias and improving methodological credibility.

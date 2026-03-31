# Subgroup Reporting Rules

## 1. Purpose

This document defines the reporting eligibility rules for subgroup analysis in the main paper.

Its purpose is to prevent over-fragmentation, unstable subgroup claims, and inconsistent movement between main-text and appendix reporting.

## 2. Core Principle

Not every subgroup that can be defined should appear in the main paper tables.

Subgroup reporting must follow a fixed hierarchy:

1. primary subgroup tables
2. secondary subgroup tables
3. exploratory subgroup tables

The category depends on sample size and methodological reliability, not on whether the subgroup produces an interesting result.

## 3. Reporting Tiers

## Tier 1. Primary subgroup reporting

These are the subgroups intended for main-text reporting.

### Allowed subgroup families

- cycle-length level groups
- cycle-variability groups

### Eligibility conditions

A subgroup may enter Tier 1 only if it satisfies both:

- `n_users >= U_min`
- `n_cycles >= C_min`

These thresholds must be fixed before running the full study and must be reported in the paper.

### Recommended use

Tier 1 groups should be used for:

- main subgroup performance tables
- main gain-over-calendar tables
- main personalization gain tables

## Tier 2. Secondary subgroup reporting

These are subgroups that are scientifically relevant but methodologically more fragile.

### Allowed subgroup families

- ovulatory-status groups

### Eligibility conditions

Tier 2 groups should still satisfy basic minimum sample requirements, but even when they do, they should be labelled as secondary or sensitivity analyses.

### Reason

These groups are partly entangled with the same noisy label system used for supervision and evaluation.

## Tier 3. Exploratory subgroup reporting

These are subgroups that are potentially informative but too small or too fragmented for strong claims.

### Typical examples

- compact joint profiles
- rare mixed subgroups
- small anovulation-heavy subgroups

### Use

These groups may be shown in:

- appendix
- supplementary materials
- exploratory result sections

They should not be used for the paper’s main claims.

## 4. Minimum Size Rule

The study must freeze minimum subgroup size thresholds before final analysis.

The thresholds should include:

- `U_min`: minimum number of users
- `C_min`: minimum number of cycles

No subgroup should be promoted to main-text reporting unless both thresholds are met.

If a subgroup fails either threshold:

- it must be removed from Tier 1
- it may be retained as exploratory if scientifically relevant

## 5. Joint Profile Rule

Joint profiles must not be part of the main paper by default.

They may enter the main paper only if:

- every reported joint cell satisfies the minimum size rule
- the number of joint cells remains small enough to interpret clearly

Otherwise, joint profiles must be placed in exploratory reporting only.

## 6. Missing-History Rule

Users with insufficient historical data to support subgroup assignment must not be forced into a subgroup.

They should be assigned to:

- `unassigned-history-insufficient`

This category should be tracked during preprocessing, but it should not be interpreted as a scientific subgroup.

The paper should report:

- how many users were unassigned
- why they were unassigned

## 7. Main-Text Table Policy

The main paper should include only:

- overall comparison table
- cycle-length level subgroup table
- cycle-variability subgroup table
- detector decomposition table

Secondary tables may include:

- ovulatory-status subgroup table

Exploratory tables may include:

- compact joint profile tables

## 8. Claim Strength Rule

The strength of subgroup claims must depend on the reporting tier.

### Tier 1

Allowed claim style:

- main comparative conclusions
- subgroup-specific gain claims
- personalization selectivity claims

### Tier 2

Allowed claim style:

- sensitivity analysis
- consistency or inconsistency with primary findings

### Tier 3

Allowed claim style:

- exploratory patterns
- hypothesis-generating observations

Tier 3 results must not be used as the main evidence for the paper’s central conclusions.

## 9. No Outcome-Driven Promotion

A subgroup must never be promoted into the main paper because it shows the strongest improvement.

Tier assignment must be determined by:

- subgroup family
- sample size
- methodological reliability

not by effect size.

## 10. Required Metadata Export

Each experiment run should export a subgroup metadata table that includes:

- subgroup family
- subgroup name
- reporting tier
- user count
- cycle count
- eligibility flag

This table should serve as the source of truth for deciding what goes into the main text and what remains supplementary.

## 11. Paper Wording Rule

The paper should clearly distinguish:

- primary subgroup analyses
- secondary analyses
- exploratory analyses

This distinction should be made explicit both in the Methods section and in table captions.

## 12. Immediate Next Step

Before implementing subgroup experiments, the project should choose and freeze:

- `U_min`
- `C_min`

and then apply these rules consistently across all subsequent subgroup reporting.

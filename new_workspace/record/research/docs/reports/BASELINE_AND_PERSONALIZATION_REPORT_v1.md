# Baseline and Personalization Research Report

## 1. Scope

This report summarizes the first completed research-stage experiments under the heterogeneity-sensitive methodology.

Completed stages:

- history-only subgroup construction
- subgroup baseline analysis
- `L1` zero-shot detector calibration
- `L2` one-shot detector calibration
- `L3` few-shot detector calibration

Primary subgroup axes used here:

- `cycle_length_level_group`
- `cycle_variability_group`

## 2. Main Findings

### 2.1 Wearable physiology is not uniformly useful

The current strict-prefix wearable baseline (`L0 Population`) helps some subgroups substantially more than others.

Most visible gains appear in:

- shifted cycle-length groups (`short`, `long`)
- `medium-variability` users
- the small `high-variability` subgroup

The weakest gain appears in:

- `low-variability` users, where wearable performance is nearly identical to Calendar on `PostOvDays`

### 2.2 Current detector personalization does not improve results

At this stage:

- `L1` makes no effective change
- `L2` generally worsens performance
- `L3` also generally worsens performance

This means detector personalization should not be assumed to be beneficial. In the current implementation, personalization is either neutral (`L1`) or harmful (`L2/L3`).

### 2.3 Current evidence supports the wearable subgroup claim, but not a positive personalization claim

The current results support the paper's primary wearable-benefit claim much more clearly than any positive personalization claim.

- primary claim supported: wearable physiology helps selectively across irregularity profiles
- secondary personalization claim not supported: current personalization does not improve harder subgroups and is not yet a positive finding

## 3. Wearable Gain over Calendar by Subgroup

| subgroup_family          | subgroup_name      |   calendar_post_ov |   l0_post_ov |   wearable_gain_post_ov |   l0_post_trigger |   l0_ov_first |   l0_detect_rate |
|:-------------------------|:-------------------|-------------------:|-------------:|------------------------:|------------------:|--------------:|-----------------:|
| cycle_length_level_group | long               |              6.465 |        4.802 |                   1.663 |             3.189 |         3.5   |            1     |
| cycle_length_level_group | short              |              9.56  |        3.901 |                   5.658 |             2.48  |         3.333 |            0.75  |
| cycle_length_level_group | typical            |              3.065 |        2.764 |                   0.301 |             2.677 |         3.233 |            0.977 |
| cycle_variability_group  | high-variability   |              6.418 |        3.643 |                   2.775 |             1.765 |         3.5   |            1     |
| cycle_variability_group  | low-variability    |              2.589 |        2.588 |                   0.001 |             2.748 |         2.739 |            1     |
| cycle_variability_group  | medium-variability |              5.721 |        2.974 |                   2.747 |             3.244 |         4.25  |            1     |

Interpretation:

- `wearable_gain_post_ov > 0` means the wearable baseline reduces `PostOvDays MAE` relative to Calendar
- the largest gains are currently observed in `short`, `long`, and `medium-variability` groups
- `low-variability` shows almost no gain, suggesting Calendar remains competitive there

## 4. Personalization Comparison by Subgroup

### cycle_length_level_group

| subgroup_name   | method_short   |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:----------------|:---------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| long            | L0 Population  |          4 |         5 |         4.802 |              3.189 |          3.5   |          3.5   |                 1     |               4.25  |
| short           | L0 Population  |          4 |         5 |         3.901 |              2.48  |          3.333 |          5     |                 0.75  |               5.667 |
| typical         | L0 Population  |         44 |        28 |         2.764 |              2.677 |          3.233 |          4.837 |                 0.977 |               6.07  |
| long            | L1 Zero-shot   |          4 |         5 |         4.802 |              3.189 |          3.5   |          3.5   |                 1     |               4.25  |
| short           | L1 Zero-shot   |          4 |         5 |         3.901 |              2.48  |          3.333 |          5     |                 0.75  |               5.667 |
| typical         | L1 Zero-shot   |         44 |        28 |         2.764 |              2.677 |          3.233 |          4.837 |                 0.977 |               6.07  |
| long            | L2 One-shot    |          4 |         5 |         4.802 |              3.189 |          3.5   |          3.5   |                 1     |               4.25  |
| short           | L2 One-shot    |          4 |         5 |         3.901 |              2.48  |          3.333 |          5     |                 0.75  |               5.667 |
| typical         | L2 One-shot    |         44 |        28 |         3.119 |              3.568 |          3.814 |          4.837 |                 0.977 |               3.488 |
| long            | L3 Few-shot    |          4 |         5 |         4.802 |              3.189 |          3.5   |          3.5   |                 1     |               4.25  |
| short           | L3 Few-shot    |          4 |         5 |         3.901 |              2.48  |          3.333 |          5     |                 0.75  |               5.667 |
| typical         | L3 Few-shot    |         44 |        28 |         2.992 |              3.422 |          3.791 |          4.837 |                 0.977 |               4.465 |

### cycle_variability_group

| subgroup_name      | method_short   |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:-------------------|:---------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| high-variability   | L0 Population  |          2 |         1 |         3.643 |              1.765 |          3.5   |          5     |                     1 |               6.5   |
| low-variability    | L0 Population  |         23 |        17 |         2.588 |              2.748 |          2.739 |          4.652 |                     1 |               6.304 |
| medium-variability | L0 Population  |          4 |         6 |         2.974 |              3.244 |          4.25  |          3.5   |                     1 |               3.25  |
| high-variability   | L1 Zero-shot   |          2 |         1 |         3.643 |              1.765 |          3.5   |          5     |                     1 |               6.5   |
| low-variability    | L1 Zero-shot   |         23 |        17 |         2.588 |              2.748 |          2.739 |          4.652 |                     1 |               6.304 |
| medium-variability | L1 Zero-shot   |          4 |         6 |         2.974 |              3.244 |          4.25  |          3.5   |                     1 |               3.25  |
| high-variability   | L2 One-shot    |          2 |         1 |         3.85  |              5.378 |          6     |          5     |                     1 |               1.5   |
| low-variability    | L2 One-shot    |         23 |        17 |         2.927 |              3.388 |          3.174 |          4.652 |                     1 |               4.043 |
| medium-variability | L2 One-shot    |          4 |         6 |         3.592 |              4.167 |          5.5   |          3.5   |                     1 |               1.5   |
| high-variability   | L3 Few-shot    |          2 |         1 |         3.85  |              4.192 |          6     |          5     |                     1 |               2.5   |
| low-variability    | L3 Few-shot    |         23 |        17 |         2.899 |              3.521 |          3.348 |          4.652 |                     1 |               4.043 |
| medium-variability | L3 Few-shot    |          4 |         6 |         3.592 |              4.44  |          5.5   |          3.5   |                     1 |               1     |

## 5. Interpretation by Research Question

### Q1. Which irregularity profiles are hardest?

Under Calendar, the hardest currently observed groups are:

- `short` cycle-length level
- `long` cycle-length level
- `medium-variability` and `high-variability`

### Q2. Where do wearable signals help?

They help most where Calendar assumptions are structurally weakest:

- users with shifted but stable cycle length
- users with non-trivial between-cycle variability

### Q3. Where does personalization help?

At the current implementation stage, personalization does not help. This is a meaningful negative result rather than a missing result.

- `L1` is effectively neutral
- `L2` and `L3` degrade the current baseline

### Q4. What does this imply for the paper?

The paper can already support a strong claim that wearable physiology provides selective benefit across irregularity profiles.

However, the detector-personalization claim must remain secondary and provisional. The current code does not yet support a positive personalization result.

## 6. Methodological Cautions

- subgroup sizes are still small in several cells, especially `high-variability`
- current subgroup tables are useful for directional evidence, not strong final subgroup claims
- `ovulatory-status` has not yet been incorporated into the main report and should remain secondary

## 7. Immediate Next Step

The next research step should not be to claim personalization works.

Instead, it should be to:

1. freeze subgroup reporting thresholds (`U_min`, `C_min`)
2. keep the current wearable subgroup-baseline result as the main positive finding
3. redesign detector personalization before making any stronger personalization claims

In practical terms, the current evidence says:

**wearable physiology already shows selective value; current detector personalization is currently a negative or null finding rather than a positive contribution.**

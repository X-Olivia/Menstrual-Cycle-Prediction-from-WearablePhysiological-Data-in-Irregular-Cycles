# Baseline Subgroup Analysis

## cycle_length_level_group

### Calendar

| subgroup_name   |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:----------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| long            |          4 |         5 |       6.46521 |                nan |            nan |            nan |                     0 |                 nan |
| short           |          4 |         5 |       9.55954 |                nan |            nan |            nan |                     0 |                 nan |
| typical         |         44 |        28 |       3.06529 |                nan |            nan |            nan |                     0 |                 nan |

### PhaseCls-Temp+HR[L1-zero-shot]

| subgroup_name   |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:----------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| long            |          4 |         5 |       4.80191 |            3.18919 |        3.5     |        3.5     |              1        |             4.25    |
| short           |          4 |         5 |       3.90141 |            2.48    |        3.33333 |        5       |              0.75     |             5.66667 |
| typical         |         44 |        28 |       2.76445 |            2.67673 |        3.23256 |        4.83721 |              0.977273 |             6.06977 |

### PhaseCls-Temp+HR[RF-baseline]

| subgroup_name   |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:----------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| long            |          4 |         5 |       4.80191 |            3.18919 |        3.5     |        3.5     |              1        |             4.25    |
| short           |          4 |         5 |       3.90141 |            2.48    |        3.33333 |        5       |              0.75     |             5.66667 |
| typical         |         44 |        28 |       2.76445 |            2.67673 |        3.23256 |        4.83721 |              0.977273 |             6.06977 |

## cycle_variability_group

### Calendar

| subgroup_name      |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:-------------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| high-variability   |          2 |         1 |       6.41839 |                nan |            nan |            nan |                     0 |                 nan |
| low-variability    |         23 |        17 |       2.58866 |                nan |            nan |            nan |                     0 |                 nan |
| medium-variability |          4 |         6 |       5.72098 |                nan |            nan |            nan |                     0 |                 nan |

### PhaseCls-Temp+HR[L1-zero-shot]

| subgroup_name      |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:-------------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| high-variability   |          2 |         1 |       3.64319 |            1.76471 |        3.5     |        5       |                     1 |             6.5     |
| low-variability    |         23 |        17 |       2.58776 |            2.74769 |        2.73913 |        4.65217 |                     1 |             6.30435 |
| medium-variability |          4 |         6 |       2.97401 |            3.2439  |        4.25    |        3.5     |                     1 |             3.25    |

### PhaseCls-Temp+HR[RF-baseline]

| subgroup_name      |   n_cycles |   n_users |   post_ov_mae |   post_trigger_mae |   ov_first_mae |   ov_final_mae |   detected_cycle_rate |   latency_days_mean |
|:-------------------|-----------:|----------:|--------------:|-------------------:|---------------:|---------------:|----------------------:|--------------------:|
| high-variability   |          2 |         1 |       3.64319 |            1.76471 |        3.5     |        5       |                     1 |             6.5     |
| low-variability    |         23 |        17 |       2.58776 |            2.74769 |        2.73913 |        4.65217 |                     1 |             6.30435 |
| medium-variability |          4 |         6 |       2.97401 |            3.2439  |        4.25    |        3.5     |                     1 |             3.25    |


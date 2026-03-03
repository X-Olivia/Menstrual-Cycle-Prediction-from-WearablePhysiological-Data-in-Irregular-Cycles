# Model Code Implementation Plan (Based on cycle_clean_2, processed_data/2 and Design)

This document aligns current data (`cycle_clean_2.csv`, `processed_data/2`) with the **sequential latent-state multi-task model** from the paper and outlines the **model implementation plan** (analysis and structure only, no code). The plan is aligned with the paper’s formulas and training protocol.

---

## Zero: Paper Model and Formulas (Implementation Checklist)

| Paper | Formula / convention | Implementation |
|-------|----------------------|----------------|
| Daily input | {x_1,…,x_t}, physiological aggregation + **within-individual standardisation** | Daily features from processed_data/2; only *_z columns, **no lh/estrogen** |
| Recurrent state | h_t = f(x_t, h_{t-1}), f = recurrent update | **GRU** main; **LSTM** for comparison; same protocol and data |
| Main: menses | ŷ_t = W_m h_t + b_m, “days until next menses” | Regression head, scalar; time-to-event; only complete cycles supervised; mask at truncation |
| Aux: ovulation | p(t_ov) = σ(W_ov h_t + b_ov), classification | BCE only on steps with **ovulation annotation**; other steps **masked**, no gradient |
| Aux loss | L_ov = Σ_{t∈T_ov} BCE(p(t_ov), o_t) | **Scheme B**: o_t ∈ [0,1] soft label (ovulation_prob_fused); T_ov = all annotated steps in cycle |
| Total loss | L = L_menses + λ L_ovulation | λ hyperparameter; auxiliary as **inductive bias** |
| Two-stage | Stage 1: backbone + both heads; Stage 2: main-task fine-tune, backbone frozen or low LR | Avoids sparse/noisy auxiliary dominating main task |

---

## One: Data and Model Mapping

### 1.1 Data assets

- **cycle_clean_2.csv**: (id, study_interval, day_in_study) by small_group_key → cycle; daily phase, lh/estrogen, ovulation_day_method1/2, ovulation_prob_fused. **Annotations and sequence boundaries**; hormones **only for labels/eval**, not model input.
- **processed_data/2/full.csv** (and morning/evening/sleep): same keys; wearable daily features (HRV, HR, wrist temp, nightly_temperature, resting_hr); *_missing and **_z** (within-individual). **Model input x_t**.

**Scale**: cycle_clean_2 ~4825 rows (in-cycle days); full.csv larger—subset full by cycle keys so only **in-cycle** days are used. Sequences by small_group_key, day_in_study ascending.

### 1.2 Feature columns and input dim

- **full.csv** wearable z-columns (14): rmssd_mean_z, lf_mean_z, hf_mean_z, lf_hf_ratio_z, hr_mean_z, hr_std_z, hr_min_z, hr_max_z, wt_mean_z, wt_std_z, wt_min_z, wt_max_z, nightly_temperature_z, resting_hr_z.
- **Missing**: agree on strategy (e.g. fill 0, mask, or extra missing feature).
- **Outliers**: clip or fix upstream if *_z has extremes.

### 1.3 Labels in cycle_clean_2

- **Cycle boundary**: each small_group_key = one cycle; next cycle start = next menses start.
- **Main label days_until_next_menses**: y_t = next_menses_day − day_in_study(t). Mask at truncated end (no future menses info).
- **Ovulation (aux)**: **Scheme B** — ovulation_prob_fused as o_t ∈ [0,1]; BCE. T_ov = steps with ovulation annotation; ovulation_mask = True only there.

---

## Two: Implementation Plan (Modular)

### 2.1 Data pipeline (D1–D5)

- **D1**: In-cycle day set and cycle meta (next_menses_day, etc.).
- **D2**: Load daily features; keep only in-cycle days; align columns; missing strategy.
- **D3**: Add days_until_next_menses and **mask_menses** (False where truncated).
- **D4**: Add ovulation_label (= ovulation_prob_fused) and **ovulation_mask** (True only in T_ov).
- **D5**: Build sequences by (id, small_group_key), day_in_study ascending; each sample = one cycle: X, y_menses, y_ov, mask_menses, mask_ov; h_0 = 0 per sequence; train/val/test by **subject (id)**.

### 2.2 Features and input

- x_t from processed_data/2 only; no lh/estrogen. Standardisation already in *_z; optional global scale. Missing: fill 0 or mask. Sequence order: strict day_in_study ascending.

### 2.3 Model (aligned with paper)

- **Backbone**: h_t = f(x_t, h_{t-1}); GRU (main) or LSTM; h_0 = 0 per cycle.
- **Menses head**: ŷ_t = W_m h_t + b_m; L1/MSE only on non-truncated steps.
- **Ovulation head**: p(t_ov) = σ(W_ov h_t + b_ov); BCE only on ovulation_mask=True.
- **Total**: L = L_menses + λ L_ovulation.

### 2.4 Two-stage training

- **Stage 1**: Joint optimisation of backbone + both heads.
- **Stage 2**: Fine-tune main task only; backbone frozen or low LR; ovulation head not updated (or small LR).

### 2.4.1 Split and leakage

- Split by **subject (id)**; same id only in one of train/val/test.
- **Chosen**: Fixed test (hold out some subjects) + **K-fold only for train/val**; final evaluation **once** on fixed test.

### 2.5 Ovulation label: Scheme B

- o_t = ovulation_prob_fused ∈ [0,1]; BCE; T_ov = annotated steps. Report as “ovulation soft classification, BCE, target [0,1]”.

### 2.6 Evaluation

- **Menses**: MAE, ±1/2/3 day accuracy; correctness w.r.t. **that cycle’s** true length (next_menses day). Optional stratification by horizon.
- **Ovulation**: F1, accuracy, ±1 day; correlation if soft; only on annotated cycles/days.

### 2.7 Checklist (anti-error)

- Input: only *_z from processed_data/2; no hormones.
- Main mask: truncated steps do not contribute to L_menses.
- Aux mask: only T_ov contributes to BCE.
- Sequences: (id, small_group_key), day_in_study ascending; h_0 = 0 per sequence.
- Two-stage: Stage 2 only main task; backbone frozen or low LR.
- Split: fixed test + K-fold train/val; key = id.
- Ovulation: Scheme B, o_t = ovulation_prob_fused, BCE.
- Menses evaluation: correctness w.r.t. that cycle’s next_menses; compare stepwise within cycle.
- Ovulation evaluation: only on annotated cycles/days; hormones for annotation/eval only, not input.

---

## Three: Implementation Order

1. **Data**: Implement D1–D5 → DataLoader/Dataset with cycle sequences, features, dual labels, dual masks.
2. **Model**: GRU backbone + menses head + ovulation head + total loss (with masks).
3. **Training**: Stage 1 joint → Stage 2 main-task fine-tune; save best val.
4. **Evaluation**: Report main (MAE, ±k day) and auxiliary (F1, ±1 day).
5. **Ablation** (optional): LSTM, λ, ovulation source, missing strategy.

This plan allows systematic implementation of the sequential multi-task latent model on the current data while staying consistent with the paper.

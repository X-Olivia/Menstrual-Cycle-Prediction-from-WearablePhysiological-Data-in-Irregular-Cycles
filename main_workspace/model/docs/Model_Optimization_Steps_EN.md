# Model Optimization Steps 

**Principle**: First confirm the signal is learnable, then tune the model, then the loss. Otherwise you may be optimizing a task that is not learnable.

---

## Stage Zero: Ovulation Signal Learnability (Must Do First)

Before changing any loss, λ, or architecture, answer:

> **Can ovulation be distinguished by the current features?**

### Approach: Logistic Regression Probe

- **Input**: Same-day features x_t (same 14 dims as the model, or per-day features from the same sequences).
- **Output**: Ovulation label (binary: e.g. max-prob day per cycle = 1, else 0; or soft label thresholded).
- **Task**: Per-day samples (no sequence), predict “is this day ovulation day” from x_t.
- **Metrics**: **AUC**, **PR-AUC**.

### Interpretation

**AUC** (probe uses `class_weight=None` to reflect true signal strength):

| AUC        | Conclusion    |
| ---------- | ------------- |
| < 0.55     | No signal     |
| 0.55–0.65 | Weak signal   |
| 0.65–0.75 | Learnable     |
| > 0.75     | Strong signal |

**PR-AUC**: Interpret relative to **baseline PR (test positive ratio)**. Use **PR-AUC / baseline_PR**:

| Ratio  | Meaning |
| ------ | ------- |
| < 1.5  | Weak    |
| 1.5–3 | Present |
| > 3    | Strong  |

### Implementation Notes

- **Low-confidence cycle filter**: Keep only cycles with `max(ovulation_prob_fused) >= 0.6` to avoid noisy labels.
- **class_weight=None**: Probe aims to detect if signal exists; do not change class distribution.
- **Sanity check**: Script prints `Feature mean |x|`; if ≈ 0, check for over-filling with 0 or scaling issues.

Run:

```bash
cd main_workspace && python -m model.baseline_ovulation_probe
```

For full-feature probe only: `python -m model.baseline_ovulation_probe full`. If AUC < 0.55, do Stage One (ablation, missing handling) or accept ceiling, then re-run probe.

---

## Stage One: Features and Signal Source

**Goal**: After confirming some single-day separability, identify **which features** carry ovulation signal.

| Step | Action                                       | Files                                                                                                         |
| ---- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1.1  | **Feature ablation (ovulation probe)** | Run probe with temp-only, temp+HR, HRV-only, all; report AUC/PR-AUC.                                          |
| 1.2  | **Missing imputation**                 | Try within-subject interpolation or “missing indicator + neighbour mean” instead of global 0; re-run probe. |

Run: `python -m model.baseline_ovulation_probe` (default = Stage One ablation); `python -m model.baseline_ovulation_probe full` for Stage Zero.

**Acceptance**: At least one feature set (e.g. temp or temp+HR) has probe AUC > 0.55; if all < 0.55, improve features/annotations or missing handling first.

---

## Stage Two: Loss and Training Protocol (When Signal Is Learnable)

| Step | Action                                       | Files                                                                                                                                          |
| ---- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1  | **Ovulation BCE positive weighting**   | Weight steps with ovulation_prob_fused > 0 (e.g. pos_weight = #neg / #pos).**Required.**                                                 |
| 2.2  | **Keep ovulation in Stage 2**          | In Stage 2 still optimise ovulation head at small LR (e.g. 1e-4) or keep small λ (e.g. 0.2) so ovulation signal is not dropped.**Key.** |
| 2.3  | **Early stopping by ov BCE, not corr** | Use ovulation BCE (or total loss) for checkpoint selection; do not use ovulation_corr for early stopping (noisy).                              |
| 2.4  | **Increase Stage 1 λ (optional)**     | Only after Stage Zero/One show learnability. Raise LAMBDA_OV from 0.5 to 1.0–2.0.                                                             |

**Do not**: Compute L_ov only on positive steps—that removes negative boundary; keep all samples and weight positives.

**Acceptance**: Val/Test ovulation_corr improves; main-task MAE/acc_1d does not degrade.

---

## Stage Three: Data and Labels (After Stage Two)

| Step | Action                             |
| ---- | ---------------------------------- |
| 3.1  | **Focal Loss**               |
| 3.2  | **Binary labels (optional)** |

Again: do not switch to “loss only on positive steps”; keep negatives and weight.

---

## Stage Four: Evaluation and Monitoring (Throughout)

| Step | Action                                                 |
| ---- | ------------------------------------------------------ |
| 4.1  | **Always report ovulation_corr**                 |
| 4.2  | **Stratify by horizon; focus 6–20**             |
| 4.3  | **Optional: ovulation day ±1 recall/precision** |

---

## Stage Five: Structure (Only After Learnable Signal and Gains in Two–Three)

| Step | Action                        |
| ---- | ----------------------------- |
| 5.1  | **Two-phase modelling** |
| 5.2  | **Luteal length prior** |

---

## Execution Order (by Day)

| Day              | Action                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Day 1**  | Run**logistic regression baseline** (ovulation probe); report AUC, PR-AUC.                                                                                   |
| **Day 2**  | If**AUC > 0.6** → signal present, do Stage One (ablation) and Stage Two (loss/Stage2). If **AUC < 0.55** → improve features/data, then re-run probe. |
| **Day 3+** | After probe and ablation conclusions, consider λ, Focal Loss, two-phase model.                                                                                    |

---

## Common Mistake to Avoid

- **Wrong**: “Model didn’t learn → model must be bad.”
- **Often**: “Model didn’t learn → **signal was not learnable** (or very weak).”
  Tuning loss, λ, or architecture without first checking signal can waste days with no improvement and no way to tell if the issue is tuning or task ceiling.

---

## Baseline Record (for comparison)

- **Val**: MAE=6.97, acc_1d=13.8%, ovulation_corr=-0.079
- **Test**: MAE=7.79, acc_1d=15.0%, ovulation_corr=-0.047
- **Test by horizon**: 1–5d acc_1d≈22.5%, 21+ acc_1d≈7.5%

**Goals**: Confirm learnability via **probe AUC**; then improve **ovulation_corr** from negative to positive; improve **acc_1d for horizon 6–20**; move overall **acc_1d** toward ±1 day.

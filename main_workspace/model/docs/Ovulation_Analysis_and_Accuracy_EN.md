# Why the Model Does Not Learn Ovulation Well — CS and Medical Perspective

## 0. Ovulation Head: Discrimination vs Prediction

The ovulation head is a **discriminative/probability head**, not a predictive head:

- **Given**: x₁, x₂, …, x_t (physiological signals up to today).
- **Output**: P(today is ovulation day | signals up to today).

So it is **online discrimination**: from current and past sequence, judge whether “today” is ovulation day; it does **not** predict a future ovulation day. Ovulation is a single event; wearables are lagged (temperature rises after ovulation; no LH in input). So in **wearable-only** settings the mapping **x_t → ovulation_t** is very weak; the ovulation head can only provide weak inductive bias; the main task remains menses prediction. “Not learning ovulation” here means this **discrimination** task (ovulation_corr ≈ 0) is not learned well.

---

## 1. Results Summary and Problem

### 1.1 Current metrics

| Set | MAE | acc_±1d | acc_±2d | acc_±3d | **ovulation_corr** |
|-----|-----|---------|---------|---------|--------------------|
| Val | 6.97 | 13.8% | 24.3% | 36.1% | **-0.079** |
| Test | 7.79 | 15.0% | 24.9% | 35.6% | **-0.047** |

By **horizon** (true days until next menses):

- **1–5 d**: MAE ≈ 4.3–5.4, acc_1d ≈ 22–23% (best).
- **6–10, 11–15, 16–20**: MAE ≈ 4.3–5.6, acc_1d ≈ 12–23%.
- **21+**: MAE ≈ 12.1–13.4, acc_1d ≈ 6.5–7.5% (**worst**).

So **ovulation probability is almost uncorrelated (slightly negative)** with true ovulation probability; the auxiliary task does not learn “is today ovulation day”. **Larger horizon (more follicular)** predicts worse, consistent with “if we could identify ovulation first, prediction would improve”.

### 1.2 Design vs reality

- Design: use ovulation as **inductive bias** so latent state h_t encodes ovulation-related transition and improves main task (menses).
- Goal: **±1 day accuracy**; ideal flow “sequence → detect ovulation → then predict accurately”.
- Reality: ovulation **discrimination** head barely learns (ovulation_corr ≈ 0, slightly negative); overall acc_1d ≈ 15%, far from ±1 day.

---

## 2. CS View: Why Ovulation Discrimination Fails

### 2.1 Extreme label sparsity and class imbalance (main cause)

- Total in-cycle steps: **4825**.
- Steps with **ovulation_prob_fused non-zero**: **360** (~**7.5%**).
- So ~**92.5%** of steps have target 0; only **7.5%** have soft label in (0, 1].
- Per cycle: days with non-zero ovulation label: **mean 2.1, max 4**; cycle length ~28 days → only 2–4 positive days per cycle.
- BCE is computed on every (masked) step, but most targets are 0 → optimizer pushes predictions toward 0; the 360 positive steps cannot dominate gradients → ovulation head outputs low probability everywhere → ovulation_corr ≈ 0 or negative.

### 2.2 Main task dominates and representation conflict

- Main task (menses): **every** step has supervision (except truncation); target days_until_next_menses is smooth (monotone in t).
- Auxiliary (ovulation): **sparse** and **peak-like** (few days high).
- L = L_menses + λ L_ov; L_menses has many more steps and is easier to minimise → **main task dominates**; h_t encodes “days to menses” rather than “today ovulation?”. Same representation must fit both smooth regression and a single-day peak → conflict; main task wins. Wearable-only x_t → ovulation_t signal is weak (no LH, temperature lag, HRV phase-level), so the discrimination head has little to use.

### 2.3 Two-stage training further weakens ovulation

- Stage 1: joint optimisation; best model by **total val loss**.
- Stage 2: only main task; backbone low LR or frozen; **ovulation head not updated**.
- So the final model’s ovulation head and backbone are tuned only for main task in Stage 2; no mechanism preserves or strengthens ovulation signal → ovulation_corr fails to improve or worsens (consistent with slight negative correlation).

### 2.4 Features: dimension and missing

- Input 14 dims (HRV, HR, wrist temp, nightly temp, resting_hr); 14 is not high for GRU. Issues: (1) **Missing**: many days HRV etc. filled with 0 (FILL_MISSING=0); if missing is high, effective signal is mostly temp/HR. (2) **Most ovulation-relevant** (temp/wrist temp) is mixed in 14 dims; if missing or low variance, model may rely more on HR/HRV, while literature says **temp/wrist temp** are most reliable for ovulation in wearables. So the problem is not “too many features” but **ovulation-relevant features (especially temp) having insufficient weight or too much noise** in the sequence, plus sparsity and main-task dominance.

---

## 3. Medical View: Discriminability of Ovulation in Wearables

### 3.1 Temperature and ovulation: lag and individual variation

- **BBT / wrist skin temperature** rises **after** ovulation (biphasic, ~0.3–0.5°C) and stays until next menses.
- Only ~**70%** of ovulatory cycles show clear biphasic BBT; ~20% have ovulation but no typical BBT. BBT nadir relates to LH peak but **daily BBT** often **confirms** ovulation after the fact (rise = already ovulated); LH is not in input.
- So for “is **today** ovulation day?”, the best事前 signal (LH surge) is unavailable; temperature is **post-ovulation** → weak and lagged for same-day discrimination. If nightly temp is often missing or variable, wearable temp is a **weak, retrospective** target for day-level ovulation discrimination.

### 3.2 Luteal phase stability: why “detect ovulation then predict” is a good goal

- **Luteal phase** (ovulation to next menses) is relatively stable (often 12–16 days), less variable than follicular.
- If the model could **identify “past ovulation”** (or ovulation day), “days until menses” ≈ remaining luteal days → prediction would be much better; **follicular** (horizon 21+) has high length variance and no ovulation anchor → naturally worse—consistent with stratified results (21+ worst).

### 3.3 Literature on wearables and ovulation/menses

- Wearables + ML for **fertility window / ovulation**: accuracy ~85–87% (often window, not single day); **phase classification** (menstrual / ovulatory / luteal) ~87% accuracy, 0.96 AUC. **Single-day ovulation** is rarely the main reported metric; most work uses “window” or “phase”, not “±1 day ovulation”.
- **Menses prediction**: e.g. ~83–90% accuracy 3 days ahead; strict **±1 day** is less common in public work → going from wearables to “±1 day” is a high bar.
- Conclusion: **Learning “is today ovulation day?” from wearables alone, without hormones, is physiologically and literatively difficult** (x_t → ovulation_t very weak); current design (sparse soft labels + main-task dominance) amplifies this.

---

## 4. Summary of Causes

| Level | Cause |
|-------|--------|
| **Data/labels** | Ovulation labels extremely sparse (~7.5% non-zero steps, 2–4 days per cycle); BCE dominated by 0s; positive gradient weak. |
| **Task design** | Main task dense and smooth; ovulation sparse and peak-like; shared h_t → main task wins; representation fits “days to menses” not “ovulation day”. |
| **Training** | Stage 2 only trains main task; ovulation head not updated; early stop by main/total loss → ovulation signal marginalised. |
| **Physiology/signal** | Ovulation is discrimination not prediction; no LH; temperature post-ovulation; HRV phase-level → wearable-only x_t→ovulation_t very weak; temp retrospective; ~20% cycles no typical biphasic. |
| **Features** | Dimension not the main issue; temp and other ovulation-related features may be missing or not emphasised, reducing learnability. |

So the issue is **not** “too many features” but **sparse ovulation labels, main-task dominance, two-stage not protecting ovulation, and physiologically weak wearable-only discrimination for “today is ovulation day”**; hence no ovulation anchor for luteal prediction and limited ±1 day accuracy.

---

## 5. Improvement Directions (CS and Medical)

### 5.1 Reduce sparsity and imbalance

- **Reweight BCE**: higher weight for steps with ovulation_prob_fused > 0 (e.g. pos_weight = #neg / #pos).
- Keep **all** samples; do **not** compute L_ov only on positive steps (loses negative boundary). Consider higher weight for “±1 day around ovulation”.
- **Binarise + Focal Loss**: binarise to ovulation day / non-ovulation (e.g. max-prob day = 1), use Focal Loss to reduce negative dominance; or focal-style weight on positives with soft labels.

### 5.2 Explicit “ovulation then menses” modelling

- **Two-phase inference/training**: (1) Train “is today ovulation day?” (or “past ovulation”) from sequence (or temp/HR subset); (2) On steps “past ovulation”, use “days since ovulation” or luteal phase as extra input and predict days_until_next_menses. This makes “detect ovulation” an explicit intermediate.
- **Structural prior**: Add “luteal length” prior (e.g. 12–16 days) in loss or output; or heavier penalty for error near ovulation day to encourage learning ovulation first.

### 5.3 Multi-task and training protocol

- **Stage 1**: Increase λ (e.g. 1.0–2.0) or higher LR for ovulation head; consider ovulation_corr in early stopping so “main good but ovulation lost” is avoided.
- **Stage 2 keep ovulation**: Do not freeze ovulation head; or keep L_ov with small weight so ovulation signal is not dropped in fine-tuning.

### 5.4 Features and input

- **Emphasise temp**: Separate branch or normalisation for nightly_temperature / wrist temp so temp is not drowned by HRV/HR; or **feature ablation** (temp-only, temp+HR) to see effect on ovulation metrics.
- **Missing**: For days with missing temp, consider interpolation or “missing indicator + neighbour mean” instead of global 0. (Note: Stage One 1.2 tried HRV/temp/HR imputation; “HRV-only” and “all” did not improve—see Model_Optimization_Steps; current suggestion is keep fill 0 or light handling.)

### 5.5 Evaluation

- Report **stratified by horizon**; focus on **horizon 6–20** (luteal) acc_1d; monitor **ovulation_corr** and “±1 day around ovulation” recall/precision so gains are from ovulation use, not only main-task overfitting.
- If gold-standard ovulation is limited, “phase classification” (menstrual/follicular/ovulatory/luteal) can be an intermediate metric (e.g. vs 87% in literature), then connect “ovulation discrimination → menses prediction”.

---

## 6. Short Summary

- **Task**: Ovulation head is **discriminative** — P(today is ovulation day | x₁..x_t); online, does not predict future ovulation. In wearable-only setting x_t→ovulation_t is very weak.
- **Why it does not learn**: Sparse labels (~7.5% non-zero, 2–4 days/cycle), main-task gradient dominates shared representation, Stage 2 only optimises main task, and wearable discrimination for “today ovulation” is weak and retrospective → ovulation_corr ≈ 0 or slightly negative.
- **“Too many features”**: Not the main cause; more important are weight/quality of ovulation-related signal (especially temp) and task/training/label design.
- **Why ±1 day matters**: Luteal phase is relatively fixed; **detecting** ovulation first would improve accuracy; current bad 21+ horizon shows “missing ovulation anchor” is the bottleneck.
- **Suggestions**: Reweight ovulation BCE / positive weighting; explicit two-phase “detect ovulation then predict menses” or structural prior; Stage 2 keep ovulation; strengthen temp and missing handling; evaluate with ovulation_corr and stratified acc_1d.

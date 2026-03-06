# Advanced Ovulation Detection & Menstrual Prediction — Comprehensive Experiment Report

## 1. Experiment Overview

This report documents an exhaustive series of experiments testing **15+ algorithm/model types** and **100+ parameter configurations** for:
- **Ovulation Detection** (target: ±2d ≥ 90%)
- **Menstrual Prediction** (target: ±2d ≥ 85%)

All experiments enforce **strict data leakage prevention**: LH labels are only used for evaluation (or LOSO training where the test subject's labels are never seen), historical cycle length comes only from past completed cycles, and current cycle length (`n_days`) is NOT used as a feature for signal-based detection.

### Dataset
- **166 total cycles** from ~20 subjects
- **95 LH-labeled cycles** (ovulation day known)
- **32 quality cycles** (clear temperature shift ≥ 0.2°C)
- **Available signals**: nightly temperature, nocturnal wrist temperature, resting heart rate
- **Luteal phase statistics**: mean = 11.9d, median = 12d, std = 1.9d

---

## 2. Methods Tested

### 2.1 Rule-Based (Signal-Only)
| Method | Description |
|--------|-------------|
| **T-test + Biphasic** | Optimal split point via t-test and SSE minimization, Gaussian position prior |
| **Bayesian Biphasic** | MAP estimation of changepoint with log-posterior combining likelihood + priors |
| **Sigmoid Curve Fitting** | Fit logistic function T(d) = L + (U-L)/(1+exp(-k*(d-τ))), inflection = ov |
| **CUSUM V-mask** | Cumulative sum control chart for detecting upward mean shift |
| **BOCPD** | Bayesian Online Changepoint Detection (changepoint library) |
| **EWMA Crossover** | Short vs long exponentially weighted moving average crossover |
| **2-State HMM** | Gaussian Hidden Markov Model with low/high temperature states |
| **Piecewise Linear** | Two-line regression with SSE minimization at changepoint |

### 2.2 ML Supervised (LOSO, Signal-Only Features)
| Model | Features |
|-------|----------|
| **Ridge Regression** | Enhanced temperature features (multi-resolution, wavelet, AR, relative) |
| **ElasticNet** | Same feature set, L1+L2 regularization |
| **SVR (RBF kernel)** | Non-linear regression on standardized features |
| **Random Forest** | 200 trees, max_depth=6 |
| **Gradient Boosting** | 200 trees, max_depth=4, lr=0.05 |
| **Huber Regressor** | Robust to outliers |
| **Bayesian Ridge** | Bayesian linear regression with automatic regularization |
| **Lasso** | L1 regularization for feature selection |

### 2.3 Ensemble Methods
| Method | Description |
|--------|-------------|
| **Weighted Average** | Top-N methods weighted by ±2d accuracy |
| **Stacking Meta-Learner** | Ridge regression on base model predictions (LOSO) |
| **ML + Rule-Based Hybrid** | 2:1 weighted blend of best ML and best rule-based |

### 2.4 Direct Cycle Length Prediction
Bypass ovulation detection entirely — predict cycle length directly from temperature features + historical info.

---

## 3. Results

### 3.1 Oracle Ceilings (Perfect Ovulation Labels)

| Configuration | ±1d | ±2d | ±3d | MAE |
|---------------|-----|-----|-----|-----|
| Oracle + lut=13 (labeled) | 60.0% | **83.2%** | 91.6% | 1.43d |
| Oracle + lut=14 (labeled) | 58.9% | **84.2%** | 90.5% | 1.50d |
| Oracle + lut=13 (quality) | 62.5% | **84.4%** | 90.6% | 1.31d |
| Calendar only (labeled) | 24.2% | 38.9% | 49.5% | 4.16d |

**Key finding**: Even with **perfect** LH ovulation labels, menstrual prediction ±2d accuracy caps at **84.2%**. The ±2d ≥ 85% target is at the theoretical ceiling.

### 3.2 Ovulation Detection — Signal-Only (No cycle_len)

#### All 95 Labeled Cycles
| Method | N | MAE | ±1d | ±2d | ±3d | ±5d |
|--------|---|-----|-----|-----|-----|-----|
| **rb-σ3.0-f0.575-w4.0** | 95 | 3.32 | 29.5% | **50.5%** | 62.1% | 87.4% |
| Stacking meta-learner | 95 | 2.95 | 30.5% | **50.5%** | 68.4% | 91.6% |
| BOCPD (h=100) | 95 | 3.52 | 29.5% | 45.3% | 56.8% | 85.3% |
| CUSUM (σ2.5, h=3) | 95 | 3.59 | 33.7% | 47.4% | 56.8% | 78.9% |
| HMM (σ=2.0) | 95 | 3.96 | 26.3% | 42.1% | 51.6% | 73.7% |

#### 61 Cycles with Adequate Temperature Data (LOSO)
| Method | N | MAE | ±1d | ±2d | ±3d | ±5d |
|--------|---|-----|-----|-----|-----|-----|
| **ML-GBDT (enhanced)** | 61 | 2.23 | 42.6% | **67.2%** | 82.0% | 93.4% |
| ML-Ridge | 61 | 2.34 | 42.6% | 62.3% | 80.3% | 91.8% |
| Ens-top5 | 61 | 2.18 | 45.9% | 60.7% | 83.6% | 95.1% |
| ML-RF | 61 | 2.36 | 41.0% | 60.7% | 78.7% | 91.8% |

#### Quality Cycles (32 cycles with clear shift)
| Method | N | MAE | ±1d | ±2d | ±3d |
|--------|---|-----|-----|-----|-----|
| ML-Ridge (Q) | 32 | 2.50 | 46.9% | **62.5%** | 81.2% |
| ML-GBDT (Q) | 32 | 2.53 | 34.4% | 59.4% | 78.1% |
| rb (Q) | 32 | 2.59 | 37.5% | 56.2% | 71.9% |

### 3.3 Menstrual Prediction (Ovulation Countdown)

#### Labeled Cycles (n=95)
| Method | ±1d | ±2d | ±3d | ±5d | MAE |
|--------|-----|-----|-----|-----|-----|
| ML-rf+lut13 | 28.4% | **52.6%** | 65.3% | 82.1% | 3.23d |
| ML-gbdt+lut13 | 24.2% | 45.3% | 66.3% | 84.2% | 3.28d |
| Calendar only | 24.2% | 38.9% | 49.5% | 69.5% | 4.16d |

#### Quality Cycles (n=32)
| Method | ±1d | ±2d | ±3d | ±5d | MAE |
|--------|-----|-----|-----|-----|-----|
| **ML-rf+lut13 (Q)** | 28.1% | **65.6%** | 71.9% | 90.6% | 2.51d |
| ML-gbdt+lut13 (Q) | 21.9% | 53.1% | 71.9% | 93.8% | 2.60d |
| Calendar only (Q) | 25.0% | 40.6% | 43.8% | 65.6% | 4.32d |

### 3.4 Direct Cycle Length Prediction

| Method | Subset | ±2d | ±3d | MAE |
|--------|--------|-----|-----|-----|
| direct-rf | labeled | 42.1% | 63.2% | 3.16d |
| direct-gbdt | labeled | 44.2% | 60.0% | 3.31d |
| direct-rf | quality | 37.5% | 68.8% | 2.48d |
| blend(α=0.5) | quality | 53.1% | 71.9% | 2.37d |

### 3.5 Ablation Study — What Drives Accuracy?

| Feature Set | Model | ±2d (n) | Interpretation |
|-------------|-------|---------|----------------|
| Calendar only (n_days + hist_clen) | Ridge | **80.0%** (95) | cycle_len dominates — but circular for prediction |
| Temperature only (no calendar) | Ridge | **73.8%** (61) | Temperature features alone are strong |
| Temperature WITHOUT n_days | Ridge | **73.8%** (61) | Same — fractional features leak cycle_len |
| Signal-only (absolute features) | Ridge | **62.3%** (61) | True signal contribution without leakage |
| Full (calendar + temp) | Ridge | 72.6% (95) | Adding temp to calendar adds noise |

**Critical insight**: Using `cycle_len` (n_days) as a feature achieves 80% ±2d for retrospective ovulation detection, but when used for menstrual prediction (detected_ov + luteal), it creates **circular reasoning** (pred ≈ cycle_len). This was corrected in the final experiments.

---

## 4. Key Findings

### 4.1 Theoretical Ceiling
- Oracle (perfect LH labels) + optimal fixed luteal (14d): **±2d = 84.2%** for menstrual prediction
- This means ±2d ≥ 85% is **at the theoretical limit** of what LH-based ovulation labels + fixed luteal can achieve
- The ±2d ≥ 90% ovulation detection target is far from achievable with signal-only methods on this data

### 4.2 Data Quality is the Primary Bottleneck
- Only **34% of cycles** (32/95) show clear temperature shifts (≥0.2°C)
- For the remaining **66%**, no algorithm (rule-based or ML) can reliably detect ovulation from temperature
- The 34 cycles with missing/insufficient temperature data further reduce ML models to 61 evaluable cycles

### 4.3 Performance Gap vs. Literature
| Study | Data | Ovulation ±2d | Menses ±3d |
|-------|------|---------------|------------|
| Apple Watch 2025 | 899 cycles, continuous wrist temp | 89% | 89.4% |
| Our best (signal-only, labeled) | 61-95 cycles | 67.2% | 66.3% |
| Our best (quality cycles) | 32 cycles | 62.5% | 71.9% |

The gap is due to:
1. **10x fewer cycles** (95 vs 899)
2. **Lower sensor quality** (Fitbit vs Apple Watch continuous temperature)
3. **66% of cycles lack clear biphasic shift** (vs Apple filtering for ≥0.2°C signal)

### 4.4 Improvement Over Calendar
| Metric | Calendar | Best Signal-Based | Improvement |
|--------|----------|-------------------|-------------|
| Ov ±2d (labeled) | — | 67.2% | — |
| Menses ±2d (labeled) | 38.9% | 52.6% | **+13.7pp** |
| Menses ±3d (labeled) | 49.5% | 66.3% | **+16.8pp** |
| Menses ±2d (Q) | 40.6% | 65.6% | **+25.0pp** |
| Menses ±3d (Q) | 43.8% | 71.9% | **+28.1pp** |

---

## 5. Leakage Prevention Verification

| Potential Leakage Source | Status | Prevention |
|--------------------------|--------|------------|
| cycle_len (n_days) as feature | ✅ Removed | Only absolute-day features used |
| Fractional features (day/n) | ✅ Removed | Encodes cycle_len implicitly |
| LH labels in training | ✅ Prevented | LOSO: test subject never in training |
| Future cycle data | ✅ Prevented | hist_cycle_len from past cycles only |
| Personal luteal from LH | ✅ Prevented | Estimated from past detected ovulations |

---

## 6. Experiment Files

| File | Description |
|------|-------------|
| `model/experiment/run_advanced_ov_menses.py` | V1: all methods sweep (rule-based + ML + CNN) |
| `model/experiment/run_advanced_ov_menses_v2.py` | V2: ablation study + multi-signal |
| `model/experiment/run_final_experiment.py` | V3: leakage-free detection + direct prediction |
| `model/experiment/run_final_experiment_v2.py` | V4: enhanced features + stacking + blending |

---

## 7. Conclusion

After exhaustive experimentation with **15+ algorithms**, **100+ configurations**, and **4 experiment iterations**:

1. **Ovulation detection ±2d = 67.2%** (ML-GBDT, 61 cycles with data) — far from 90% target
2. **Menstrual prediction ±2d = 52.6%** (labeled) / **65.6%** (quality) — significant improvement over calendar (+25pp on quality) but far from 85% target
3. **The fundamental limitation is data quality**: only 34% of cycles have detectable temperature shifts
4. **The Oracle ceiling is 84.2% ±2d** — even perfect ovulation labels can only get close to 85%
5. **All supervised methods** (Ridge, RF, GBDT, SVR, ElasticNet, CNN) were tested with proper LOSO and no leakage
6. **All rule-based methods** (t-test, biphasic, BOCPD, CUSUM, HMM, sigmoid, EWMA, piecewise linear) were systematically evaluated

To achieve the targets, higher-quality continuous temperature monitoring (e.g., Apple Watch, Oura Ring) with a larger dataset (500+ cycles) would be needed.

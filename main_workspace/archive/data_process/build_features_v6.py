"""
build_features_v6.py — 经期预测特征管线 v6
===========================================
基于 v4，包含以下改进（受论文 Kilungeja et al. 2025 & BBT+HR 2022 启发）：

1. 阶段估计特征 (方向 B):
   - estimated_phase: 基于 day_in_cycle / hist_cycle_len_mean 估计月经阶段 (0-3)
   - days_since_estimated_ovulation: 距估计排卵日的天数
   - is_luteal_estimate: 是否处于黄体期

2. 突变检测与拐点特征 (方向 C):
   - max_positive_change_5d / max_negative_change_5d: 5天窗口内最大正/负变化
   - trend_reversal: HR/温度趋势反转信号
   - hr_temp_concordance: HR与温度是否同向变化

输出:
  - processed_data/v6/daily_features_v6.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_features_v4 import (
    load_base, load_rrs, load_sleep_architecture, load_symptoms,
    reload_rhr_median, load_nightly_temp_std, remove_boundary_cycles,
    fix_day_in_cycle_frac, merge_all, compute_biphasic_shift,
    interpolate_within_cycle, per_cycle_early_z_normalize,
    compute_rolling_features, compute_rate_of_change, assemble_features,
    WORKSPACE, KEY, CYCLE_GROUP, RAW_WEARABLE, ROLLING_Z_SOURCES,
    DELTA_SOURCES, SYMPTOM_COLS,
)

OUTPUT_DIR = os.path.join(WORKSPACE, "processed_data", "v6")

# Signals used for transition feature extraction
TRANSITION_SIGNALS = ["hr_mean", "nightly_temperature", "rmssd_mean", "wt_mean"]


# ── Direction B: Phase estimation features ───────────────────────────────────

def estimate_phase(df):
    """Estimate menstrual cycle phase from day_in_cycle and hist_cycle_len_mean.

    Phase encoding:
      0 = Menstrual   (day 1-5)
      1 = Follicular   (day 6 to estimated_ovulation - 2)
      2 = Ovulation    (estimated_ovulation -1 to +1)
      3 = Luteal       (estimated_ovulation + 2 to cycle end)

    Estimated ovulation = hist_cycle_len_mean - 14 (luteal phase ~14 days).
    """
    day = df["day_in_cycle"].values
    hist_len = df["hist_cycle_len_mean"].values

    est_ov = np.clip(hist_len - 14.0, 10.0, 35.0)

    phase = np.full(len(df), 1, dtype=np.int8)  # default follicular
    phase[day <= 5] = 0                           # menstrual
    phase[day >= (est_ov - 1)] = 2                # ovulation window
    phase[day >= (est_ov + 2)] = 3                # luteal

    df["estimated_phase"] = phase
    df["days_since_estimated_ovulation"] = (day - est_ov).astype(np.float32)
    df["is_luteal_estimate"] = (phase == 3).astype(np.int8)

    phase_counts = pd.Series(phase).value_counts().sort_index()
    print(f"[phase] Estimated phase distribution: "
          f"menstrual={phase_counts.get(0, 0)}, follicular={phase_counts.get(1, 0)}, "
          f"ovulation={phase_counts.get(2, 0)}, luteal={phase_counts.get(3, 0)}")
    print(f"[phase] days_since_estimated_ovulation range: "
          f"{df['days_since_estimated_ovulation'].min():.1f} ~ "
          f"{df['days_since_estimated_ovulation'].max():.1f}")

    return df


# ── Direction C: Transition / inflection-point features ──────────────────────

def compute_transition_features(df, window=5):
    """Compute transition-detection features inspired by Kilungeja et al. (2025).

    For each signal in TRANSITION_SIGNALS, within each cycle:
    1. max_positive_change: largest positive day-over-day difference in window
    2. max_negative_change: largest negative day-over-day difference in window
    """
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    n_feats = 0

    for col in TRANSITION_SIGNALS:
        z_col = f"{col}_z"
        src = z_col if z_col in df.columns else col
        if src not in df.columns:
            continue

        g = df.groupby(CYCLE_GROUP, sort=False)[src]

        # Day-over-day difference
        diff = g.transform(lambda s: s.diff())

        # Max positive/negative change in rolling window
        max_pos = diff.rolling(window, min_periods=1).max()
        max_neg = diff.rolling(window, min_periods=1).min()

        base = col.replace("_z", "")
        df[f"{base}_max_pos_{window}d"] = max_pos.astype(np.float32)
        df[f"{base}_max_neg_{window}d"] = max_neg.astype(np.float32)
        n_feats += 2

    print(f"[transition] Computed {n_feats} max-change features (window={window})")
    return df


def compute_trend_reversal(df, short=3, long=7):
    """Detect trend reversals: when short-term slope switches sign vs long-term slope.

    Inspired by the "inflection point" concept from the BBT+HR study.
    A reversal from falling to rising in HR/temp suggests ovulation.
    """
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    n_feats = 0

    for col in ["hr_mean", "nightly_temperature"]:
        z_col = f"{col}_z"
        src = z_col if z_col in df.columns else col
        if src not in df.columns:
            continue

        g = df.groupby(CYCLE_GROUP, sort=False)[src]

        slope_short = g.transform(lambda s: s.rolling(short, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0,
            raw=True
        ))
        slope_long = g.transform(lambda s: s.rolling(long, min_periods=3).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0,
            raw=True
        ))

        # Reversal: long-term falling, short-term rising (or vice versa)
        reversal = ((slope_short > 0) & (slope_long < 0)).astype(np.float32)

        base = col
        df[f"{base}_trend_reversal"] = reversal
        n_feats += 1

    print(f"[reversal] Computed {n_feats} trend-reversal features")
    return df


def compute_hr_temp_concordance(df, window=3):
    """Compute concordance between HR and temperature trends.

    When both signals move in the same direction, it strengthens phase
    transition signals (e.g., both rising post-ovulation).
    """
    hr_col = "hr_mean_z" if "hr_mean_z" in df.columns else "hr_mean"
    temp_col = "nightly_temperature_z" if "nightly_temperature_z" in df.columns else "nightly_temperature"

    if hr_col not in df.columns or temp_col not in df.columns:
        print("[concordance] Skipped — missing HR or temperature columns")
        return df

    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])

    hr_diff = df.groupby(CYCLE_GROUP, sort=False)[hr_col].transform(
        lambda s: s.rolling(window, min_periods=1).mean().diff()
    )
    temp_diff = df.groupby(CYCLE_GROUP, sort=False)[temp_col].transform(
        lambda s: s.rolling(window, min_periods=1).mean().diff()
    )

    # +1 if both rising, -1 if both falling, 0 if divergent
    concordance = np.sign(hr_diff) * np.sign(temp_diff)
    df["hr_temp_concordance"] = concordance.astype(np.float32).fillna(0.0)

    n_pos = (df["hr_temp_concordance"] > 0).sum()
    n_neg = (df["hr_temp_concordance"] < 0).sum()
    n_zero = (df["hr_temp_concordance"] == 0).sum()
    print(f"[concordance] hr_temp_concordance: +1={n_pos}, -1={n_neg}, 0={n_zero}")

    return df


# ── Assembly ─────────────────────────────────────────────────────────────────

def assemble_features_v6(df):
    """v6 assembly: v4 features + phase estimation + transition features."""
    # v4 feature groups
    feat_a = [f"{c}_z" for c in RAW_WEARABLE]
    feat_b = ["full_sleep_br_z", "deep_sleep_br_z", "nightly_temperature_std_z"]
    feat_c = ["sleep_score_z", "deep_sleep_min_z", "restlessness_z"]
    feat_d = SYMPTOM_COLS
    feat_e = [f"delta_{c}_1d" for c in DELTA_SOURCES] + ["wt_shift_7v3", "temp_shift_7v3"]
    feat_f = [
        "day_in_cycle", "day_in_cycle_frac",
        "hist_cycle_len_mean", "hist_cycle_len_std",
        "days_remaining_prior", "days_remaining_prior_log",
    ]
    feat_g = []
    for col_z in ROLLING_Z_SOURCES:
        base = col_z.replace("_z", "")
        for suffix in ["rmean5", "rstd5", "rslope5", "dev5"]:
            feat_g.append(f"{base}_{suffix}")

    # v6 new features: phase estimation
    feat_phase = ["estimated_phase", "days_since_estimated_ovulation", "is_luteal_estimate"]

    # v6 new features: transition detection
    feat_transition = []
    for col in TRANSITION_SIGNALS:
        base = col
        feat_transition.append(f"{base}_max_pos_5d")
        feat_transition.append(f"{base}_max_neg_5d")
    feat_transition += ["hr_mean_trend_reversal", "nightly_temperature_trend_reversal"]
    feat_transition += ["hr_temp_concordance"]

    all_features = feat_a + feat_b + feat_c + feat_d + feat_e + feat_f + feat_g + feat_phase + feat_transition
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    present = [f for f in all_features if f in df.columns]
    out = df[KEY + present].copy()
    print(f"[assemble-v6] {len(present)} model features + {len(KEY)} keys = {len(out.columns)} cols")
    return out, present


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  build_features_v6.py")
    print("  (v6: phase estimation + transition features)")
    print("=" * 60)

    # Same v4 pipeline up to z-normalization
    df = load_base()
    df = reload_rhr_median(df)

    rrs = load_rrs()
    sleep_arch = load_sleep_architecture()
    symptoms = load_symptoms()
    temp_std = load_nightly_temp_std()

    df = merge_all(df, rrs, sleep_arch, symptoms, temp_std)
    df = remove_boundary_cycles(df)
    df = fix_day_in_cycle_frac(df)

    df = compute_biphasic_shift(df)
    df = interpolate_within_cycle(df)
    df = per_cycle_early_z_normalize(df)

    df = compute_rolling_features(df, window=5)
    df = compute_rate_of_change(df)

    # v6: Phase estimation features (Direction B)
    df = estimate_phase(df)

    # v6: Transition features (Direction C)
    df = compute_transition_features(df, window=5)
    df = compute_trend_reversal(df)
    df = compute_hr_temp_concordance(df)

    # Assemble
    out, feature_names = assemble_features_v6(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "daily_features_v6.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Features: {len(feature_names)}")

    print("\n[QC] NaN rates per feature:")
    for f in feature_names:
        nan_rate = out[f].isna().mean()
        if nan_rate > 0:
            print(f"  {f:40s}: {nan_rate:.1%} NaN")

    # Phase feature distribution check
    print("\n[QC] Phase features:")
    for f in ["estimated_phase", "days_since_estimated_ovulation", "is_luteal_estimate"]:
        if f in out.columns:
            print(f"  {f}: mean={out[f].mean():.2f}, std={out[f].std():.2f}, "
                  f"range=[{out[f].min():.1f}, {out[f].max():.1f}]")

    return out, feature_names


if __name__ == "__main__":
    main()

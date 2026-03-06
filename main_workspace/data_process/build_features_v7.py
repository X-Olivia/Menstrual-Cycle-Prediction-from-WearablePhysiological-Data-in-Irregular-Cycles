"""
build_features_v7.py — 经期预测特征管线 v7
===========================================
基于 v4，新增以下改进：

1. 子日级特征 (方向 A): 从高频 HR/WT/HRV 提取的 20 个夜间特征
2. 排卵锚定个性化特征 (方向 B.1):
   - personal_avg_luteal_len: 每人历史平均黄体期长度
   - personal_luteal_std: 黄体期长度个人标准差
   - temp_shift_detected: 温度双相变化点检测
   - est_days_remaining_luteal: 基于个人黄体期的倒计时估计

输出:
  - processed_data/v7/daily_features_v7.csv
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

OUTPUT_DIR = os.path.join(WORKSPACE, "processed_data", "v7")
CYCLE_CSV = os.path.join(WORKSPACE, "subdataset", "cycle_clean_2.csv")
SUBDAILY_CSV = os.path.join(OUTPUT_DIR, "subdaily_features.csv")

SUBDAILY_FEATURES = [
    "hr_nocturnal_nadir", "hr_nadir_timing_frac",
    "hr_onset_mean", "hr_wake_mean",
    "hr_onset_to_nadir", "hr_wake_surge",
    "hr_nocturnal_iqr", "hr_nocturnal_range",
    "hr_circadian_amplitude",
    "wt_nocturnal_plateau", "wt_rise_time_frac",
    "wt_nocturnal_auc", "wt_pre_wake_drop",
    "wt_nocturnal_range_sub", "wt_nocturnal_std_sub",
    "hrv_early_night", "hrv_late_night",
    "hrv_night_slope", "lf_hf_early_vs_late",
    "hrv_nocturnal_range",
]

LUTEAL_FEATURES = [
    "personal_avg_luteal_len",
    "personal_luteal_std",
    "temp_shift_detected",
    "days_since_temp_shift",
    "est_days_remaining_luteal",
]


# ── Direction B.1: Ovulation-Anchored Personal Features ──────────────────────

def compute_personal_luteal_lengths(df):
    """Compute per-subject average luteal phase length from LH-derived ovulation labels.

    Uses ovulation_prob_fused from cycle_clean_2.csv (LH-based),
    filtered to high-confidence (prob > 0.5) and physiologically
    plausible luteal lengths (8-18 days).
    """
    cycle_data = pd.read_csv(CYCLE_CSV)

    # Find ovulation day per cycle (highest probability > 0.5)
    ov_candidates = cycle_data[cycle_data["ovulation_prob_fused"] > 0.5].copy()
    if len(ov_candidates) == 0:
        print("[luteal] No ovulation detected, skipping")
        return {}

    ov_day = (
        ov_candidates.groupby("small_group_key")
        .apply(lambda g: g.loc[g["ovulation_prob_fused"].idxmax(), "day_in_study"],
               include_groups=False)
        .reset_index()
        .rename(columns={0: "ov_day"})
    )

    cycle_end = (
        cycle_data.groupby("small_group_key")["day_in_study"]
        .max().reset_index()
        .rename(columns={"day_in_study": "cycle_end"})
    )

    merged = ov_day.merge(cycle_end, on="small_group_key")
    merged["luteal_len"] = merged["cycle_end"] - merged["ov_day"] + 1

    # Filter to physiologically plausible range
    reasonable = merged[(merged["luteal_len"] >= 8) & (merged["luteal_len"] <= 18)]

    # Get subject id
    subj_map = cycle_data[["small_group_key", "id"]].drop_duplicates()
    reasonable = reasonable.merge(subj_map, on="small_group_key")

    # Per-subject stats
    subj_luteal = (
        reasonable.groupby("id")["luteal_len"]
        .agg(personal_avg_luteal_len="mean", personal_luteal_std="std", _n_cycles="count")
        .reset_index()
    )
    subj_luteal["personal_luteal_std"] = subj_luteal["personal_luteal_std"].fillna(2.0)

    print(f"[luteal] {len(reasonable)} cycles with valid ovulation, "
          f"{len(subj_luteal)} subjects with personal luteal data, "
          f"mean luteal={subj_luteal['personal_avg_luteal_len'].mean():.1f}d")

    return subj_luteal[["id", "personal_avg_luteal_len", "personal_luteal_std"]]


def detect_temp_shift(df, min_shift=0.15, lookback=5, confirm=3):
    """Detect temperature biphasic shift within each cycle using wrist temp data.

    Uses a simplified 3-over-6 rule: a shift is detected when the last
    `confirm` days' mean temp exceeds the previous `lookback` days' mean
    by at least `min_shift` degrees.
    """
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])

    temp_col = "nightly_temperature"
    if temp_col not in df.columns:
        temp_col = "wt_mean"

    shift_day = np.full(len(df), 0, dtype=np.int32)  # 0 = not detected

    for _, grp in df.groupby(CYCLE_GROUP):
        idx = grp.index.values
        temps = grp[temp_col].values.astype(np.float64)
        n = len(temps)

        detected = False
        detect_pos = 0
        for i in range(lookback + confirm, n):
            if np.isnan(temps[i]):
                continue
            baseline = temps[max(0, i - lookback - confirm):i - confirm]
            recent = temps[i - confirm:i]

            valid_bl = baseline[~np.isnan(baseline)]
            valid_rc = recent[~np.isnan(recent)]

            if len(valid_bl) >= 3 and len(valid_rc) >= 2:
                if np.mean(valid_rc) - np.mean(valid_bl) >= min_shift:
                    if not detected:
                        detected = True
                        detect_pos = i - confirm  # shift started `confirm` days ago
                    break

        if detected:
            for j, ix in enumerate(idx):
                day_offset = j - detect_pos
                shift_day[ix] = max(0, day_offset)

    df["temp_shift_detected"] = (shift_day > 0).astype(np.int8)
    df["days_since_temp_shift"] = shift_day.astype(np.float32)
    df.loc[shift_day == 0, "days_since_temp_shift"] = np.nan

    n_detected = df.groupby(CYCLE_GROUP)["temp_shift_detected"].max().sum()
    n_total = df.groupby(CYCLE_GROUP).ngroups
    print(f"[temp_shift] Detected in {n_detected}/{n_total} cycles")
    return df


def add_luteal_countdown(df, subj_luteal):
    """Add est_days_remaining_luteal = personal_avg_luteal_len - days_since_temp_shift."""
    if subj_luteal is None or len(subj_luteal) == 0:
        df["est_days_remaining_luteal"] = np.nan
        return df

    df = df.merge(subj_luteal, on="id", how="left")

    # Fill missing personal luteal with global median
    global_median = subj_luteal["personal_avg_luteal_len"].median()
    df["personal_avg_luteal_len"] = df["personal_avg_luteal_len"].fillna(global_median)
    df["personal_luteal_std"] = df["personal_luteal_std"].fillna(2.0)

    # Countdown: how many days until expected menses based on personal luteal phase
    df["est_days_remaining_luteal"] = np.where(
        df["days_since_temp_shift"].notna(),
        df["personal_avg_luteal_len"] - df["days_since_temp_shift"],
        np.nan,
    )

    valid_n = df["est_days_remaining_luteal"].notna().sum()
    print(f"[luteal_countdown] {valid_n}/{len(df)} rows with luteal countdown estimate")
    return df


# ── Subdaily feature z-normalization ─────────────────────────────────────────

def z_normalize_subdaily(df, subdaily_cols, clip=5.0, eps=1e-8):
    """Per-subject z-normalize subdaily features."""
    for col in subdaily_cols:
        if col not in df.columns:
            continue
        subj_stats = df.groupby("id")[col].agg(["mean", "std"])
        for subj_id, row in subj_stats.iterrows():
            mask = df["id"] == subj_id
            mu, sigma = row["mean"], row["std"]
            vals = df.loc[mask, col].values.astype(np.float64)
            if np.isnan(sigma) or sigma < eps:
                z = np.where(np.isnan(vals), 0.0, 0.0)
            else:
                z = np.where(np.isnan(vals), 0.0, np.clip((vals - mu) / sigma, -clip, clip))
            df.loc[mask, f"{col}_z"] = z.astype(np.float32)
    z_cols = [f"{c}_z" for c in subdaily_cols if f"{c}_z" in df.columns]
    print(f"[z-norm-subdaily] Normalized {len(z_cols)} subdaily features")
    return df, z_cols


# ── Assembly ─────────────────────────────────────────────────────────────────

def assemble_features_v7(df, subdaily_z_cols):
    """v7 assembly: v4 features + subdaily z-scores + luteal features."""
    # v4 features
    feat_a = [f"{c}_z" for c in RAW_WEARABLE]
    feat_b = ["full_sleep_br_z", "deep_sleep_br_z", "nightly_temperature_std_z"]
    feat_d = SYMPTOM_COLS
    feat_e = [f"delta_{c}_1d" for c in DELTA_SOURCES] + ["wt_shift_7v3", "temp_shift_7v3"]
    feat_f = [
        "day_in_cycle", "day_in_cycle_frac",
        "hist_cycle_len_mean", "hist_cycle_len_std",
        "days_remaining_prior", "days_remaining_prior_log",
    ]

    # v7 new features
    feat_subdaily = subdaily_z_cols
    feat_luteal = [f for f in LUTEAL_FEATURES if f in df.columns]

    all_features = feat_a + feat_b + feat_d + feat_e + feat_f + feat_subdaily + feat_luteal
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    present = [f for f in all_features if f in df.columns]
    out = df[KEY + present].copy()
    print(f"[assemble-v7] {len(present)} model features + {len(KEY)} keys = {len(out.columns)} cols")
    print(f"  v4 base: {len(feat_a) + len(feat_b) + len(feat_d) + len(feat_e) + len(feat_f)}")
    print(f"  subdaily: {len([f for f in feat_subdaily if f in df.columns])}")
    print(f"  luteal: {len([f for f in feat_luteal if f in df.columns])}")
    return out, present


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  build_features_v7.py")
    print("  (v7: subdaily + ovulation-anchored personal features)")
    print("=" * 60)

    # v4 pipeline
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

    # v7: Merge subdaily features
    print("\n--- v7: Merging subdaily features ---")
    if os.path.exists(SUBDAILY_CSV):
        subdaily = pd.read_csv(SUBDAILY_CSV)
        join_key = ["id", "study_interval", "day_in_study"]
        before = len(df)
        df = df.merge(subdaily, on=join_key, how="left")
        print(f"[subdaily] Merged {len(subdaily)} rows, {len(df)} total (was {before})")
    else:
        print(f"[WARN] {SUBDAILY_CSV} not found. Run build_subdaily_features.py first.")

    # v7: Z-normalize subdaily features
    subdaily_present = [c for c in SUBDAILY_FEATURES if c in df.columns]
    df, subdaily_z_cols = z_normalize_subdaily(df, subdaily_present)

    # v7: Personal luteal length
    print("\n--- v7: Computing personal luteal features ---")
    subj_luteal = compute_personal_luteal_lengths(df)

    # v7: Temperature shift detection
    df = detect_temp_shift(df)

    # v7: Luteal countdown
    df = add_luteal_countdown(df, subj_luteal)

    # Assemble
    out, feature_names = assemble_features_v7(df, subdaily_z_cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "daily_features_v7.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Features: {len(feature_names)}")

    print("\n[QC] NaN rates per feature:")
    for f in feature_names:
        nan_rate = out[f].isna().mean()
        if nan_rate > 0.01:
            print(f"  {f:40s}: {nan_rate:.1%} NaN")

    return out, feature_names


if __name__ == "__main__":
    main()

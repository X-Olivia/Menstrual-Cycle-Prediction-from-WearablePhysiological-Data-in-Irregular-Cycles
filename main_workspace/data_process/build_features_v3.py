"""
build_features_v3.py — 经期预测特征管线 v3
===========================================
从已有的日级穿戴设备聚合数据出发，修复 bug、接入新数据源、重构特征工程。

输入:
  - processed_data/2/sleep.csv          (已有 14 维日级穿戴设备特征)
  - cycle_clean_2.csv                   (周期元数据)
  - mcPHASES raw: respiratory_rate_summary.csv
  - mcPHASES raw: sleep_score.csv
  - mcPHASES raw: sleep.csv
  - mcPHASES raw: hormones_and_selfreport.csv

输出:
  - processed_data/v3/daily_features_v3.csv  (37 维模型输入特征)

变更 vs v2:
  - 修复 biphasic shift 跨周期 bug
  - 修复插值跨周期 bug
  - 新增: 呼吸频率 (2), 睡眠架构 (3), PMS 症状 (4)
  - 新增: 变化率特征 (6 Δ + 2 shift = 8)
  - z-normalization 改为 per-subject
"""

import os
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCPHASES_DIR = os.path.join(
    WORKSPACE,
    "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-"
    "and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0",
)
EXISTING_SLEEP = os.path.join(WORKSPACE, "processed_data", "2", "sleep.csv")
CYCLE_CSV = os.path.join(WORKSPACE, "subdataset", "cycle_clean_2.csv")
OUTPUT_DIR = os.path.join(WORKSPACE, "processed_data", "v3")

KEY = ["id", "study_interval", "day_in_study", "small_group_key"]
CYCLE_GROUP = ["id", "study_interval", "small_group_key"]

# 14 raw wearable features from existing pipeline
RAW_WEARABLE = [
    "rmssd_mean", "lf_mean", "hf_mean", "lf_hf_ratio",
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "wt_mean", "wt_std", "wt_min", "wt_max",
    "nightly_temperature", "resting_hr",
]

# New raw features to z-normalize
RAW_NEW = [
    "full_sleep_br", "deep_sleep_br",
    "sleep_score", "deep_sleep_min", "restlessness",
]

ALL_RAW = RAW_WEARABLE + RAW_NEW

# PMS symptom columns (ordinal 0-5, keep NaN for LightGBM)
SYMPTOM_COLS = ["cramps", "bloating", "sorebreasts", "moodswing"]
SYMPTOM_MAP = {
    "Not at all": 0, "Very Low/Little": 1, "Low": 2,
    "Moderate": 3, "High": 4, "Very High": 5,
}

# Features for which to compute daily deltas
DELTA_SOURCES = [
    "wt_mean", "nightly_temperature", "rmssd_mean",
    "hf_mean", "hr_mean", "full_sleep_br",
]

# ── Step 0: Load base data ───────────────────────────────────────────────────

def load_base():
    """Load existing daily features and restore pre-interpolation NaN."""
    df = pd.read_csv(EXISTING_SLEEP)

    # Restore NaN where _missing flag is True (undo interpolation + zero-fill)
    for feat in RAW_WEARABLE:
        flag = f"{feat}_missing"
        if flag in df.columns:
            df.loc[df[flag] == 1, feat] = np.nan

    # Drop old z-scores and old shift features (will recompute)
    drop_cols = [c for c in df.columns if c.endswith("_z")]
    drop_cols += ["wt_shift_7v3", "temp_shift_7v3"]
    drop_cols += [c for c in df.columns if c.endswith("_missing")]
    df = df.drop(columns=drop_cols, errors="ignore")

    print(f"[base] Loaded {len(df)} rows, {len(df.columns)} cols")
    return df


# ── Step 1: Load new data sources ────────────────────────────────────────────

def load_rrs():
    """Load respiratory rate summary → daily aggregate by (id, study_interval, day_in_study)."""
    path = os.path.join(MCPHASES_DIR, "respiratory_rate_summary.csv")
    rrs = pd.read_csv(path)

    # Aggregate: mean per day (there may be multiple sleep sessions)
    rrs_daily = (
        rrs.groupby(["id", "study_interval", "day_in_study"])
        .agg(
            full_sleep_br=("full_sleep_breathing_rate", "mean"),
            deep_sleep_br=("deep_sleep_breathing_rate", "mean"),
        )
        .reset_index()
    )
    print(f"[RRS] {len(rrs_daily)} daily rows")
    return rrs_daily


def load_sleep_architecture():
    """Load sleep score + sleep raw → daily features."""
    # Sleep score
    ss_path = os.path.join(MCPHASES_DIR, "sleep_score.csv")
    ss = pd.read_csv(ss_path)
    ss_daily = (
        ss.groupby(["id", "study_interval", "day_in_study"])
        .agg(
            sleep_score=("overall_score", "mean"),
            deep_sleep_min=("deep_sleep_in_minutes", "mean"),
            restlessness=("restlessness", "mean"),
        )
        .reset_index()
    )

    # Sleep raw: main sleep only, aggregate by wake day
    sl_path = os.path.join(MCPHASES_DIR, "sleep.csv")
    sl = pd.read_csv(sl_path)
    sl = sl[sl["mainsleep"] == True].copy()
    sl = sl.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    sl_daily = (
        sl.groupby(["id", "study_interval", "day_in_study"])
        .agg(
            minutesasleep=("minutesasleep", "mean"),
            sleep_efficiency=("efficiency", "mean"),
        )
        .reset_index()
    )

    merged = ss_daily.merge(
        sl_daily, on=["id", "study_interval", "day_in_study"], how="outer"
    )
    print(f"[Sleep] {len(merged)} daily rows")
    return merged


def load_symptoms():
    """Load PMS symptoms → ordinal encoding."""
    path = os.path.join(MCPHASES_DIR, "hormones_and_selfreport.csv")
    hr = pd.read_csv(path)
    out = hr[["id", "study_interval", "day_in_study"]].copy()
    for col in SYMPTOM_COLS:
        out[col] = hr[col].map(SYMPTOM_MAP)  # unmapped → NaN
    print(f"[Symptoms] {len(out)} rows, valid rates: "
          + ", ".join(f"{c}={out[c].notna().mean():.0%}" for c in SYMPTOM_COLS))
    return out


# ── Step 2: Merge ────────────────────────────────────────────────────────────

def merge_all(df, rrs, sleep_arch, symptoms):
    """Left-join new sources onto base daily data."""
    join_key = ["id", "study_interval", "day_in_study"]
    df = df.merge(rrs, on=join_key, how="left")
    df = df.merge(sleep_arch, on=join_key, how="left")
    df = df.merge(symptoms, on=join_key, how="left")
    print(f"[merge] {len(df)} rows, {len(df.columns)} cols after merge")
    return df


# ── Step 3: Fix biphasic shift (within cycle) ───────────────────────────────

def compute_biphasic_shift(df):
    """wt_shift_7v3 and temp_shift_7v3: recent 3-day mean minus prior 3-9 day mean.
    FIXED: groupby includes small_group_key to prevent cross-cycle leakage."""
    for raw_col, shift_col in [
        ("wt_mean", "wt_shift_7v3"),
        ("nightly_temperature", "temp_shift_7v3"),
    ]:
        df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
        grouped = df.groupby(CYCLE_GROUP, sort=False)[raw_col]
        recent_3 = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
        prior_3_9 = grouped.transform(
            lambda s: s.shift(3).rolling(4, min_periods=1).mean()
        )
        df[shift_col] = recent_3 - prior_3_9
    print("[shift] Biphasic shift features recomputed (within-cycle)")
    return df


# ── Step 4: Interpolation within cycle ───────────────────────────────────────

def interpolate_within_cycle(df, limit=3):
    """Linear interpolation within (id, study_interval, small_group_key), max gap=3.
    FIXED: interpolation no longer crosses cycle boundaries."""
    interp_cols = [c for c in ALL_RAW if c in df.columns]
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    for col in interp_cols:
        df[col] = df.groupby(CYCLE_GROUP)[col].transform(
            lambda s: s.interpolate(method="linear", limit=limit, limit_direction="both")
        )
    n_still_nan = df[interp_cols].isna().sum().sum()
    print(f"[interp] Within-cycle interpolation done, {n_still_nan} NaN remaining")
    return df


# ── Step 5: Per-subject z-normalization ──────────────────────────────────────

def per_subject_z_normalize(df, clip=5.0, eps=1e-8):
    """Z-normalize using per-subject mean/std across all their data.
    Advantages over per-cycle-early-days:
      - stable baseline (more data per subject)
      - consistent with Wang 2025 'personal baseline' concept
    """
    z_cols = [c for c in ALL_RAW if c in df.columns]
    for col in z_cols:
        subj_stats = df.groupby("id")[col].agg(["mean", "std"])
        subj_stats.columns = ["_mean", "_std"]
        df = df.merge(subj_stats, on="id", how="left")
        valid = df["_std"] > eps
        df[f"{col}_z"] = 0.0
        df.loc[valid, f"{col}_z"] = (
            (df.loc[valid, col] - df.loc[valid, "_mean"]) / df.loc[valid, "_std"]
        ).clip(-clip, clip)
        df = df.drop(columns=["_mean", "_std"])
    print(f"[z-norm] Per-subject z-normalization for {len(z_cols)} features")
    return df


# ── Step 5.5: Rolling window features ────────────────────────────────────────

ROLLING_Z_SOURCES = [f"{c}_z" for c in RAW_WEARABLE] + ["full_sleep_br_z", "deep_sleep_br_z"]

def _ols_slope(arr):
    """OLS slope for a rolling window of variable length."""
    n = len(arr)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(n, dtype=np.float64)
    xm, ym = x[mask], arr[mask]
    xm = xm - xm.mean()
    denom = np.dot(xm, xm)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xm, ym - ym.mean()) / denom)


def compute_rolling_features(df, window=5):
    """Compute rolling mean, std, slope, and deviation for physiological z-features.
    All computations are within-cycle to prevent leakage."""
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    n_feats = 0
    for col in ROLLING_Z_SOURCES:
        if col not in df.columns:
            continue
        g = df.groupby(CYCLE_GROUP, sort=False)[col]

        rmean = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
        rstd = g.transform(lambda s: s.rolling(window, min_periods=2).std())
        rslope = g.transform(
            lambda s: s.rolling(window, min_periods=2).apply(_ols_slope, raw=True)
        )

        base = col.replace("_z", "")
        df[f"{base}_rmean{window}"] = rmean
        df[f"{base}_rstd{window}"] = rstd
        df[f"{base}_rslope{window}"] = rslope
        df[f"{base}_dev{window}"] = df[col] - rmean
        n_feats += 4

    print(f"[rolling] Computed {n_feats} rolling features (window={window})")
    return df


# ── Step 6: Rate-of-change features ─────────────────────────────────────────

def compute_rate_of_change(df):
    """Day-over-day deltas within cycle. NaN on first day of cycle (LightGBM handles it)."""
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    for col in DELTA_SOURCES:
        if col not in df.columns:
            continue
        delta_col = f"delta_{col}_1d"
        df[delta_col] = df.groupby(CYCLE_GROUP)[col].diff(1)
    print(f"[delta] Computed {len(DELTA_SOURCES)} rate-of-change features")
    return df


# ── Step 7: Assemble final feature set ──────────────────────────────────────

def assemble_features(df):
    """Select and order all model input features + primary keys."""

    # A. Wearable z-scores (14)
    feat_a = [f"{c}_z" for c in RAW_WEARABLE]

    # B. Respiratory z-scores (2)
    feat_b = ["full_sleep_br_z", "deep_sleep_br_z"]

    # C. Sleep architecture z-scores (3)
    feat_c = ["sleep_score_z", "deep_sleep_min_z", "restlessness_z"]

    # D. PMS symptoms ordinal (4)
    feat_d = SYMPTOM_COLS

    # E. Rate-of-change (6 deltas + 2 shifts = 8)
    feat_e_delta = [f"delta_{c}_1d" for c in DELTA_SOURCES]
    feat_e_shift = ["wt_shift_7v3", "temp_shift_7v3"]
    feat_e = feat_e_delta + feat_e_shift

    # F. Cycle position + history (6)
    feat_f = [
        "day_in_cycle", "day_in_cycle_frac",
        "hist_cycle_len_mean", "hist_cycle_len_std",
        "days_remaining_prior", "days_remaining_prior_log",
    ]

    # G. Rolling window features (16 sources * 4 stats = 64)
    feat_g = []
    for col_z in ROLLING_Z_SOURCES:
        base = col_z.replace("_z", "")
        for suffix in ["rmean5", "rstd5", "rslope5", "dev5"]:
            feat_g.append(f"{base}_{suffix}")

    all_features = feat_a + feat_b + feat_c + feat_d + feat_e + feat_f + feat_g
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    present = [f for f in all_features if f in df.columns]

    out = df[KEY + present].copy()
    print(f"[assemble] {len(present)} model features + {len(KEY)} keys = {len(out.columns)} cols")
    return out, present


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  build_features_v3.py")
    print("=" * 60)

    # Step 0: Load base
    df = load_base()

    # Step 1: Load new sources
    rrs = load_rrs()
    sleep_arch = load_sleep_architecture()
    symptoms = load_symptoms()

    # Step 2: Merge
    df = merge_all(df, rrs, sleep_arch, symptoms)

    # Step 3: Fix biphasic shift
    df = compute_biphasic_shift(df)

    # Step 4: Fix interpolation
    df = interpolate_within_cycle(df)

    # Step 5: Per-subject z-normalization
    df = per_subject_z_normalize(df)

    # Step 5.5: Rolling window features
    df = compute_rolling_features(df, window=5)

    # Step 6: Rate-of-change
    df = compute_rate_of_change(df)

    # Step 7: Assemble
    out, feature_names = assemble_features(df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "daily_features_v3.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature list: {feature_names}")

    # Quick quality check
    print("\n[QC] NaN rates per feature:")
    for f in feature_names:
        nan_rate = out[f].isna().mean()
        if nan_rate > 0:
            print(f"  {f:30s}: {nan_rate:.1%} NaN")

    return out, feature_names


if __name__ == "__main__":
    main()

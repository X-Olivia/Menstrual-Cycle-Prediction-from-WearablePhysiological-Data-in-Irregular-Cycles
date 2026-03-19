"""
build_features_v4.py — Daily features for menstrual prediction (v4).

Produces daily_features_v4.csv from:
  - Base: processed_data/sleep.csv (daily wearable aggregates; must exist — see README).
  - mcPHASES raw: respiratory_rate_summary, sleep_score, sleep.csv, hormones_and_selfreport,
    computed_temperature, resting_heart_rate.
  - Cycle: processed_dataset/cycle_cleaned_ov.csv (new_workspace).

v4 fixes: per-cycle-early z-norm, day_in_cycle_frac from hist_cycle_len, RHR median,
boundary cycles removed, nightly_temperature_std. Output: processed_data/v4/daily_features_v4.csv.

Usage (from new_workspace):
  python record/build_features_v4.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths (new_workspace) ─────────────────────────────────────────────────────

NEW_WS = Path(__file__).resolve().parent.parent
WORKSPACE = str(NEW_WS)
MCPHASES_DIR = os.path.join(
    WORKSPACE,
    "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-"
    "and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0",
)
EXISTING_SLEEP = os.path.join(WORKSPACE, "processed_dataset", "daily_features", "sleep.csv")
CYCLE_CSV = os.path.join(WORKSPACE, "processed_dataset", "cycle_cleaned_ov.csv")
OUTPUT_DIR = os.path.join(WORKSPACE, "processed_dataset", "daily_features")

KEY = ["id", "study_interval", "day_in_study", "small_group_key"]
CYCLE_GROUP = ["id", "study_interval", "small_group_key"]

RAW_WEARABLE = [
    "rmssd_mean", "lf_mean", "hf_mean", "lf_hf_ratio",
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "wt_mean", "wt_std", "wt_min", "wt_max",
    "nightly_temperature", "resting_hr",
]

RAW_NEW = [
    "full_sleep_br", "deep_sleep_br",
    "sleep_score", "deep_sleep_min", "restlessness",
    "nightly_temperature_std",
]

ALL_RAW = RAW_WEARABLE + RAW_NEW
BASELINE_RELATIVE = {"wt_mean", "wt_std", "wt_min", "wt_max"}

SYMPTOM_COLS = ["cramps", "bloating", "sorebreasts", "moodswing"]
SYMPTOM_MAP = {
    "Not at all": 0, "Very Low/Little": 1, "Low": 2,
    "Moderate": 3, "High": 4, "Very High": 5,
}

DELTA_SOURCES = [
    "wt_mean", "nightly_temperature", "rmssd_mean",
    "hf_mean", "hr_mean", "full_sleep_br",
]

ROLLING_Z_SOURCES = [f"{c}_z" for c in RAW_WEARABLE] + [
    "full_sleep_br_z", "deep_sleep_br_z", "nightly_temperature_std_z",
]


def load_base():
    df = pd.read_csv(EXISTING_SLEEP)
    for feat in RAW_WEARABLE:
        flag = f"{feat}_missing"
        if flag in df.columns:
            df.loc[df[flag] == 1, feat] = np.nan
    drop_cols = [c for c in df.columns if c.endswith("_z")]
    drop_cols += ["wt_shift_7v3", "temp_shift_7v3"]
    drop_cols += [c for c in df.columns if c.endswith("_missing")]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[base] Loaded {len(df)} rows, {len(df.columns)} cols")
    return df


def load_rrs():
    path = os.path.join(MCPHASES_DIR, "respiratory_rate_summary.csv")
    rrs = pd.read_csv(path)
    rrs_daily = (
        rrs.groupby(["id", "study_interval", "day_in_study"])
        .agg(full_sleep_br=("full_sleep_breathing_rate", "mean"),
             deep_sleep_br=("deep_sleep_breathing_rate", "mean"))
        .reset_index()
    )
    print(f"[RRS] {len(rrs_daily)} daily rows")
    return rrs_daily


def load_sleep_architecture():
    ss = pd.read_csv(os.path.join(MCPHASES_DIR, "sleep_score.csv"))
    ss_daily = (
        ss.groupby(["id", "study_interval", "day_in_study"])
        .agg(sleep_score=("overall_score", "mean"),
             deep_sleep_min=("deep_sleep_in_minutes", "mean"),
             restlessness=("restlessness", "mean"))
        .reset_index()
    )
    sl = pd.read_csv(os.path.join(MCPHASES_DIR, "sleep.csv"))
    sl = sl[sl["mainsleep"] == True].copy()
    sl = sl.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    sl_daily = (
        sl.groupby(["id", "study_interval", "day_in_study"])
        .agg(minutesasleep=("minutesasleep", "mean"),
             sleep_efficiency=("efficiency", "mean"))
        .reset_index()
    )
    merged = ss_daily.merge(sl_daily, on=["id", "study_interval", "day_in_study"], how="outer")
    print(f"[Sleep] {len(merged)} daily rows")
    return merged


def load_symptoms():
    hr = pd.read_csv(os.path.join(MCPHASES_DIR, "hormones_and_selfreport.csv"))
    out = hr[["id", "study_interval", "day_in_study"]].copy()
    for col in SYMPTOM_COLS:
        out[col] = hr[col].map(SYMPTOM_MAP)
    print(f"[Symptoms] {len(out)} rows, valid rates: "
          + ", ".join(f"{c}={out[c].notna().mean():.0%}" for c in SYMPTOM_COLS))
    return out


def reload_rhr_median(df):
    path = os.path.join(MCPHASES_DIR, "resting_heart_rate.csv")
    rhr = pd.read_csv(path)
    rhr = rhr[rhr["value"] > 0]
    rhr_median = (
        rhr.groupby(["id", "study_interval", "day_in_study"])["value"]
        .median()
        .reset_index()
        .rename(columns={"value": "resting_hr_median"})
    )
    df = df.drop(columns=["resting_hr"], errors="ignore")
    df = df.merge(rhr_median, on=["id", "study_interval", "day_in_study"], how="left")
    df = df.rename(columns={"resting_hr_median": "resting_hr"})
    n_valid = df["resting_hr"].notna().sum()
    print(f"[RHR] Reloaded with median aggregation: {n_valid}/{len(df)} rows have values")
    return df


def load_nightly_temp_std():
    path = os.path.join(MCPHASES_DIR, "computed_temperature.csv")
    ct = pd.read_csv(path)
    ct_std = (
        ct.groupby(["id", "study_interval", "sleep_end_day_in_study"])
        .agg(nightly_temperature_std=("baseline_relative_nightly_standard_deviation", "median"))
        .reset_index()
        .rename(columns={"sleep_end_day_in_study": "day_in_study"})
    )
    print(f"[TempStd] {len(ct_std)} daily rows, "
          f"valid: {ct_std['nightly_temperature_std'].notna().mean():.0%}")
    return ct_std


def remove_boundary_cycles(df):
    cycle_data = pd.read_csv(CYCLE_CSV)
    ext = cycle_data["small_group_key"].str.extract(r"_cycle(\d+)$", expand=False)
    cycle_data["_cycle_num"] = ext.astype(int)
    last_cycles = (
        cycle_data.groupby(["id", "study_interval"])["_cycle_num"]
        .max()
        .reset_index()
    )
    last_cycles["small_group_key"] = (
        last_cycles["id"].astype(str) + "_"
        + last_cycles["study_interval"].astype(str)
        + "_cycle" + last_cycles["_cycle_num"].astype(str)
    )
    boundary_keys = set(last_cycles["small_group_key"].values)
    before = len(df)
    n_cycles_before = df["small_group_key"].nunique()
    df = df[~df["small_group_key"].isin(boundary_keys)].copy().reset_index(drop=True)
    n_removed = before - len(df)
    n_cycles_after = df["small_group_key"].nunique()
    print(f"[boundary] Removed {n_removed} rows from {n_cycles_before - n_cycles_after} "
          f"last cycles → {len(df)} rows, {n_cycles_after} cycles remain")
    return df


def fix_day_in_cycle_frac(df):
    denom = df["hist_cycle_len_mean"].clip(lower=15.0)
    df["day_in_cycle_frac"] = (df["day_in_cycle"] / denom).astype(np.float32)
    print(f"[frac] day_in_cycle_frac recomputed: "
          f"range [{df['day_in_cycle_frac'].min():.2f}, {df['day_in_cycle_frac'].max():.2f}], "
          f"mean {df['day_in_cycle_frac'].mean():.2f}")
    return df


def merge_all(df, rrs, sleep_arch, symptoms, temp_std):
    join_key = ["id", "study_interval", "day_in_study"]
    df = df.merge(rrs, on=join_key, how="left")
    df = df.merge(sleep_arch, on=join_key, how="left")
    df = df.merge(symptoms, on=join_key, how="left")
    df = df.merge(temp_std, on=join_key, how="left")
    print(f"[merge] {len(df)} rows, {len(df.columns)} cols after merge")
    return df


def compute_biphasic_shift(df):
    for raw_col, shift_col in [("wt_mean", "wt_shift_7v3"),
                                ("nightly_temperature", "temp_shift_7v3")]:
        df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
        grouped = df.groupby(CYCLE_GROUP, sort=False)[raw_col]
        recent_3 = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
        prior_3_9 = grouped.transform(lambda s: s.shift(3).rolling(4, min_periods=1).mean())
        df[shift_col] = recent_3 - prior_3_9
    print("[shift] Biphasic shift features recomputed (within-cycle)")
    return df


def interpolate_within_cycle(df, limit=3):
    interp_cols = [c for c in ALL_RAW if c in df.columns]
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    for col in interp_cols:
        df[col] = df.groupby(CYCLE_GROUP)[col].transform(
            lambda s: s.interpolate(method="linear", limit=limit, limit_direction="both")
        )
    n_still_nan = df[interp_cols].isna().sum().sum()
    print(f"[interp] Within-cycle interpolation done, {n_still_nan} NaN remaining")
    return df


def per_cycle_early_z_normalize(df, k_early=5, min_coverage=0.4, clip=5.0, eps=1e-8):
    z_cols = [c for c in ALL_RAW if c in df.columns]
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    subj_stats = {}
    for col in z_cols:
        stats = df.groupby("id")[col].agg(["mean", "std"])
        subj_stats[col] = stats
    n_cycle_based = 0
    n_fallback = 0
    for col in z_cols:
        is_bl = col in BASELINE_RELATIVE
        z_arr = np.full(len(df), 0.0, dtype=np.float32)
        for _, grp in df.groupby(CYCLE_GROUP):
            idx = grp.index.values
            x = grp[col].values.astype(np.float64)
            n = len(x)
            early = x[:k_early]
            valid = early[~np.isnan(early)]
            if len(valid) >= max(1, int(k_early * min_coverage)):
                mu = float(np.nanmean(valid))
                sigma = float(np.nanstd(valid))
                n_cycle_based += 1
                if is_bl:
                    z_arr[idx] = np.where(np.isnan(x), 0.0, np.clip(x - mu, -clip, clip))
                elif sigma > eps:
                    z_arr[idx] = np.where(np.isnan(x), 0.0, np.clip((x - mu) / sigma, -clip, clip))
                k = min(k_early, n)
                z_arr[idx[:k]] = 0.0
            else:
                subj_id = grp["id"].iloc[0]
                s_mu = subj_stats[col].loc[subj_id, "mean"]
                s_std = subj_stats[col].loc[subj_id, "std"]
                n_fallback += 1
                if is_bl:
                    z_arr[idx] = np.where(np.isnan(x), 0.0,
                                          np.clip(x - s_mu, -clip, clip) if not np.isnan(s_mu) else 0.0)
                elif not np.isnan(s_std) and s_std > eps:
                    z_arr[idx] = np.where(np.isnan(x), 0.0,
                                          np.clip((x - s_mu) / s_std, -clip, clip))
        df[f"{col}_z"] = z_arr
    total = n_cycle_based + n_fallback
    print(f"[z-norm] Per-cycle-early z-normalization for {len(z_cols)} features "
          f"(cycle-based: {n_cycle_based}/{total}, fallback: {n_fallback}/{total})")
    return df


def _ols_slope(arr):
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


def compute_rate_of_change(df):
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])
    for col in DELTA_SOURCES:
        if col not in df.columns:
            continue
        df[f"delta_{col}_1d"] = df.groupby(CYCLE_GROUP)[col].diff(1)
    print(f"[delta] Computed {len(DELTA_SOURCES)} rate-of-change features")
    return df


def assemble_features(df):
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
    all_features = feat_a + feat_b + feat_c + feat_d + feat_e + feat_f + feat_g
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    present = [f for f in all_features if f in df.columns]
    out = df[KEY + present].copy()
    print(f"[assemble] {len(present)} model features + {len(KEY)} keys = {len(out.columns)} cols")
    return out, present


def main():
    print("=" * 60)
    print("  build_features_v4.py (new_workspace → daily_features_v4.csv)")
    print("=" * 60)
    print(f"  Base sleep: {EXISTING_SLEEP}")
    print(f"  Cycle:      {CYCLE_CSV}")
    print(f"  Output:     {OUTPUT_DIR}/daily_features_v4.csv")

    if not os.path.isfile(EXISTING_SLEEP):
        raise SystemExit(
            f"Base file not found: {EXISTING_SLEEP}\n"
            "Produce processed_data/2/sleep.csv first (e.g. run main_workspace data_process/daily_data_2.ipynb "
            "or copy from main_workspace processed_data/2/sleep.csv)."
        )
    if not os.path.isfile(CYCLE_CSV):
        raise SystemExit(f"Cycle CSV not found: {CYCLE_CSV}. Run data_clean.py then ovulation_labels.py.")
    if not os.path.isdir(MCPHASES_DIR):
        raise SystemExit(f"mcPHASES dir not found: {MCPHASES_DIR}")

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
    out, feature_names = assemble_features(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "daily_features_v4.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Features: {len(feature_names)}")
    for f in feature_names:
        nan_rate = out[f].isna().mean()
        if nan_rate > 0:
            print(f"  {f:30s}: {nan_rate:.1%} NaN")
    return out, feature_names


if __name__ == "__main__":
    main()

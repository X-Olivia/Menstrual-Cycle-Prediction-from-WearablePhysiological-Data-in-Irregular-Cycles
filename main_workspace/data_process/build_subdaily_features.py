"""
build_subdaily_features.py — 从高频原始数据提取子日级特征
===========================================================
从 5 秒 HR、1 分钟腕温、5 分钟 HRV 数据中提取夜间/昼夜节律特征。

输出:
  - processed_data/v7/subdaily_features.csv
  - 每行 = (id, study_interval, day_in_study) 的子日级特征
"""

import os
import numpy as np
import pandas as pd

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBDATASET = os.path.join(WORKSPACE, "subdataset")
OUTPUT_DIR = os.path.join(WORKSPACE, "processed_data", "v7")

HR_FILE = os.path.join(SUBDATASET, "heart_rate_cycle.csv")
WT_FILE = os.path.join(SUBDATASET, "wrist_temperature_cycle.csv")
HRV_FILE = os.path.join(SUBDATASET, "heart_rate_variability_details_cycle.csv")
CT_FILE = os.path.join(SUBDATASET, "computed_temperature_cycle.csv")

JOIN_KEY = ["id", "study_interval", "day_in_study"]


def _ts_to_min_vec(ts_series):
    """Vectorized conversion of 'HH:MM:SS' Series to minutes since midnight."""
    parts = ts_series.str.split(":", expand=True).astype(float)
    return parts[0] * 60 + parts[1] + parts[2] / 60


def load_sleep_bounds():
    """Load per-night sleep boundaries from computed_temperature_cycle.csv."""
    ct = pd.read_csv(CT_FILE)
    ct = ct.drop_duplicates(
        subset=["id", "study_interval", "sleep_end_day_in_study"], keep="first"
    )
    ct["sleep_start_min"] = _ts_to_min_vec(ct["sleep_start_timestamp"])
    ct["sleep_end_min"] = _ts_to_min_vec(ct["sleep_end_timestamp"])
    ct["sleep_start_abs"] = ct["sleep_start_day_in_study"] * 1440 + ct["sleep_start_min"]
    ct["sleep_end_abs"] = ct["sleep_end_day_in_study"] * 1440 + ct["sleep_end_min"]
    ct["sleep_duration_min"] = ct["sleep_end_abs"] - ct["sleep_start_abs"]
    ct = ct[ct["sleep_duration_min"] > 60].copy()

    out = ct[["id", "study_interval", "day_in_study",
              "sleep_start_day_in_study", "sleep_start_min", "sleep_end_min",
              "sleep_start_abs", "sleep_end_abs", "sleep_duration_min"]].copy()
    print(f"[sleep_bounds] {len(out)} valid nights, {out['id'].nunique()} subjects")
    return out


def _build_expanded_sleep_lookup(sleep_bounds):
    """Create expanded sleep bounds with entries for both start-day and wake-day HR data."""
    sb = sleep_bounds.copy()

    hr_day_vals = sb["day_in_study"].values.copy()
    attr_vals = sb["day_in_study"].values.copy()
    sb1 = pd.DataFrame({
        "id": sb["id"].values,
        "study_interval": sb["study_interval"].values,
        "hr_day": hr_day_vals,
        "attributed_day": attr_vals,
        "sleep_start_abs": sb["sleep_start_abs"].values,
        "sleep_end_abs": sb["sleep_end_abs"].values,
        "sleep_duration_min": sb["sleep_duration_min"].values,
    })

    cross = sb[sb["sleep_start_day_in_study"] != sb["day_in_study"]].copy()
    if len(cross) > 0:
        sb2 = pd.DataFrame({
            "id": cross["id"].values,
            "study_interval": cross["study_interval"].values,
            "hr_day": cross["sleep_start_day_in_study"].astype(int).values,
            "attributed_day": cross["day_in_study"].values,
            "sleep_start_abs": cross["sleep_start_abs"].values,
            "sleep_end_abs": cross["sleep_end_abs"].values,
            "sleep_duration_min": cross["sleep_duration_min"].values,
        })
        expanded = pd.concat([sb1, sb2], ignore_index=True)
    else:
        expanded = sb1

    return expanded[["id", "study_interval", "hr_day", "attributed_day",
                     "sleep_start_abs", "sleep_end_abs", "sleep_duration_min"]]


# ── HR Feature Computation ───────────────────────────────────────────────────

def _hr_night_features(bpm_arr, sleep_duration_min):
    """Compute features from one night's sorted sleep HR array."""
    n = len(bpm_arr)
    if n < 10:
        return None

    win = max(6, min(n // 3, 360))
    rolling = pd.Series(bpm_arr).rolling(win, min_periods=max(3, win // 2), center=True).mean()
    valid = rolling.dropna()
    if len(valid) == 0:
        return None

    nadir_val = float(valid.min())
    nadir_idx = int(valid.idxmin())

    seg = max(1, min(n // 6, 360))
    onset_hr = float(np.mean(bpm_arr[:seg]))
    wake_hr = float(np.mean(bpm_arr[-seg:]))

    return {
        "hr_nocturnal_nadir": nadir_val,
        "hr_nadir_timing_frac": nadir_idx / max(1, n - 1),
        "hr_onset_mean": onset_hr,
        "hr_wake_mean": wake_hr,
        "hr_onset_to_nadir": onset_hr - nadir_val,
        "hr_wake_surge": wake_hr - nadir_val,
        "hr_nocturnal_iqr": float(np.percentile(bpm_arr, 75) - np.percentile(bpm_arr, 25)),
        "hr_nocturnal_range": float(bpm_arr.max() - bpm_arr.min()),
    }


def process_hr_features(sleep_bounds):
    """Extract nocturnal HR features from 5-second heart rate data (vectorized)."""
    print("[HR] Building sleep window lookup...")
    expanded = _build_expanded_sleep_lookup(sleep_bounds)

    print("[HR] Loading heart_rate_cycle.csv in chunks...")
    sleep_parts = []
    day_hr_parts = []

    for i, chunk in enumerate(pd.read_csv(
        HR_FILE, chunksize=5_000_000,
        usecols=["id", "study_interval", "day_in_study", "timestamp", "bpm", "confidence"],
        dtype={"id": "int16", "study_interval": "int16", "day_in_study": "int16",
               "bpm": "int16", "confidence": "float32"},
    )):
        chunk = chunk[(chunk["confidence"] >= 0.5) & (chunk["bpm"] >= 30) & (chunk["bpm"] <= 220)]
        chunk["ts_min"] = _ts_to_min_vec(chunk["timestamp"])
        chunk["abs_min"] = chunk["day_in_study"].astype("float64") * 1440 + chunk["ts_min"]
        chunk = chunk.drop(columns=["timestamp", "confidence"])

        # Sleep window matching
        merged = chunk.merge(
            expanded,
            left_on=["id", "study_interval", "day_in_study"],
            right_on=["id", "study_interval", "hr_day"],
            how="inner",
        )
        sleep_hr = merged[
            (merged["abs_min"] >= merged["sleep_start_abs"])
            & (merged["abs_min"] <= merged["sleep_end_abs"])
        ][["id", "study_interval", "attributed_day", "abs_min", "bpm"]].copy()
        sleep_parts.append(sleep_hr)

        # Daytime HR (10:00 - 22:00) for circadian amplitude
        daytime = chunk[(chunk["ts_min"] >= 600) & (chunk["ts_min"] <= 1320)]
        if len(daytime) > 0:
            day_stats = (
                daytime.groupby(["id", "study_interval", "day_in_study"])["bpm"]
                .agg(day_hr_p90=lambda x: np.percentile(x, 90))
                .reset_index()
            )
            day_hr_parts.append(day_stats)

        print(f"  chunk {i}: {len(chunk)} valid, {len(sleep_hr)} sleep", flush=True)

    # Aggregate sleep HR
    print("[HR] Computing per-night features...")
    all_sleep = pd.concat(sleep_parts, ignore_index=True)
    all_sleep = all_sleep.rename(columns={"attributed_day": "day_in_study"})
    all_sleep = all_sleep.sort_values(["id", "study_interval", "day_in_study", "abs_min"])

    results = []
    for (rid, si, dis), grp in all_sleep.groupby(["id", "study_interval", "day_in_study"]):
        dur = sleep_bounds.loc[
            (sleep_bounds["id"] == rid) & (sleep_bounds["study_interval"] == si)
            & (sleep_bounds["day_in_study"] == dis), "sleep_duration_min"
        ]
        dur_val = float(dur.iloc[0]) if len(dur) > 0 else 0
        feats = _hr_night_features(grp["bpm"].values.astype(np.float64), dur_val)
        if feats:
            feats["id"] = int(rid)
            feats["study_interval"] = int(si)
            feats["day_in_study"] = int(dis)
            results.append(feats)

    hr_df = pd.DataFrame(results)

    # Add circadian amplitude from daytime stats
    if day_hr_parts:
        day_hr_df = pd.concat(day_hr_parts, ignore_index=True)
        day_hr_agg = day_hr_df.groupby(["id", "study_interval", "day_in_study"])["day_hr_p90"].max().reset_index()
        hr_df = hr_df.merge(day_hr_agg, on=JOIN_KEY, how="left")
        hr_df["hr_circadian_amplitude"] = hr_df["day_hr_p90"] - hr_df["hr_nocturnal_nadir"]
        hr_df = hr_df.drop(columns=["day_hr_p90"])

    print(f"[HR] {len(hr_df)} nights with HR features, "
          f"{len([c for c in hr_df.columns if c not in JOIN_KEY])} features")
    return hr_df


# ── WT Feature Computation ───────────────────────────────────────────────────

def _wt_night_features(temp_arr):
    """Compute features from one night's sorted wrist temperature array."""
    n = len(temp_arr)
    if n < 10:
        return None

    start_idx = int(n * 0.2)
    end_idx = int(n * 0.8)
    plateau = temp_arr[start_idx:end_idx]

    t_max = np.nanmax(temp_arr)
    t_min = np.nanmin(temp_arr)
    threshold_90 = t_min + 0.9 * (t_max - t_min) if (t_max - t_min) > 0.01 else t_max
    rise_idx = np.where(temp_arr >= threshold_90)[0]
    rise_frac = float(rise_idx[0] / max(1, n - 1)) if len(rise_idx) > 0 else 1.0

    seg = max(1, n // 10)
    pre_wake = np.nanmean(temp_arr[-seg:])
    pre_pre = np.nanmean(temp_arr[-2 * seg:-seg]) if n >= 2 * seg else pre_wake

    return {
        "wt_nocturnal_plateau": float(np.nanmean(plateau)) if len(plateau) > 0 else np.nan,
        "wt_rise_time_frac": rise_frac,
        "wt_nocturnal_auc": float(np.nanmean(temp_arr)),
        "wt_pre_wake_drop": float(pre_wake - pre_pre),
        "wt_nocturnal_range_sub": float(t_max - t_min),
        "wt_nocturnal_std_sub": float(np.nanstd(temp_arr)),
    }


def process_wt_features(sleep_bounds):
    """Extract nocturnal wrist temperature features from 1-minute data."""
    print("[WT] Loading wrist_temperature_cycle.csv...")
    expanded = _build_expanded_sleep_lookup(sleep_bounds)

    wt = pd.read_csv(
        WT_FILE,
        usecols=["id", "study_interval", "day_in_study", "timestamp", "temperature_diff_from_baseline"],
        dtype={"id": "int16", "study_interval": "int16", "day_in_study": "int16",
               "temperature_diff_from_baseline": "float32"},
    )
    wt = wt[wt["temperature_diff_from_baseline"].abs() <= 5].copy()
    wt["ts_min"] = _ts_to_min_vec(wt["timestamp"])
    wt["abs_min"] = wt["day_in_study"].astype("float64") * 1440 + wt["ts_min"]
    wt = wt.drop(columns=["timestamp"])
    print(f"[WT] {len(wt)} valid readings")

    merged = wt.merge(
        expanded,
        left_on=["id", "study_interval", "day_in_study"],
        right_on=["id", "study_interval", "hr_day"],
        how="inner",
    )
    sleep_wt = merged[
        (merged["abs_min"] >= merged["sleep_start_abs"])
        & (merged["abs_min"] <= merged["sleep_end_abs"])
    ].copy()
    sleep_wt = sleep_wt.drop(columns=["day_in_study", "hr_day"], errors="ignore")
    sleep_wt = sleep_wt.rename(columns={"attributed_day": "day_in_study"})
    sleep_wt = sleep_wt.sort_values(["id", "study_interval", "day_in_study", "abs_min"])
    print(f"[WT] {len(sleep_wt)} readings in sleep windows")

    results = []
    for (rid, si, dis), grp in sleep_wt.groupby(["id", "study_interval", "day_in_study"]):
        feats = _wt_night_features(grp["temperature_diff_from_baseline"].values)
        if feats:
            feats["id"] = int(rid)
            feats["study_interval"] = int(si)
            feats["day_in_study"] = int(dis)
            results.append(feats)

    df = pd.DataFrame(results)
    print(f"[WT] {len(df)} nights with WT features")
    return df


# ── HRV Feature Computation ──────────────────────────────────────────────────

def process_hrv_features(sleep_bounds):
    """Extract temporal HRV features from 5-minute data."""
    print("[HRV] Loading heart_rate_variability_details_cycle.csv...")
    expanded = _build_expanded_sleep_lookup(sleep_bounds)

    hrv = pd.read_csv(HRV_FILE)
    hrv = hrv[
        (hrv["coverage"] >= 0.6) & (hrv["rmssd"] > 0)
        & (hrv["low_frequency"] > 0) & (hrv["high_frequency"] > 0)
    ].copy()
    hrv["ts_min"] = _ts_to_min_vec(hrv["timestamp"])
    hrv["abs_min"] = hrv["day_in_study"].astype("float64") * 1440 + hrv["ts_min"]
    hrv["lf_hf"] = hrv["low_frequency"] / hrv["high_frequency"]
    print(f"[HRV] {len(hrv)} valid readings")

    merged = hrv.merge(
        expanded,
        left_on=["id", "study_interval", "day_in_study"],
        right_on=["id", "study_interval", "hr_day"],
        how="inner",
    )
    sleep_hrv = merged[
        (merged["abs_min"] >= merged["sleep_start_abs"])
        & (merged["abs_min"] <= merged["sleep_end_abs"])
    ].copy()
    sleep_hrv = sleep_hrv.drop(columns=["day_in_study", "hr_day"], errors="ignore")
    sleep_hrv = sleep_hrv.rename(columns={"attributed_day": "day_in_study"})
    sleep_hrv = sleep_hrv.sort_values(["id", "study_interval", "day_in_study", "abs_min"])
    print(f"[HRV] {len(sleep_hrv)} readings in sleep windows")

    results = []
    for (rid, si, dis), grp in sleep_hrv.groupby(["id", "study_interval", "day_in_study"]):
        n = len(grp)
        if n < 4:
            continue
        mid = n // 2
        early = grp.iloc[:mid]
        late = grp.iloc[mid:]
        rmssd_vals = grp["rmssd"].values.astype(np.float64)

        slope = np.nan
        if n >= 3:
            x = np.arange(n, dtype=np.float64)
            xc = x - x.mean()
            denom = np.dot(xc, xc)
            if denom > 1e-12:
                slope = float(np.dot(xc, rmssd_vals - rmssd_vals.mean()) / denom)

        results.append({
            "id": int(rid), "study_interval": int(si), "day_in_study": int(dis),
            "hrv_early_night": float(early["rmssd"].mean()),
            "hrv_late_night": float(late["rmssd"].mean()),
            "hrv_night_slope": slope,
            "lf_hf_early_vs_late": float(early["lf_hf"].mean()) - float(late["lf_hf"].mean()),
            "hrv_nocturnal_range": float(rmssd_vals.max() - rmssd_vals.min()),
        })

    df = pd.DataFrame(results)
    print(f"[HRV] {len(df)} nights with HRV features")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  build_subdaily_features.py")
    print("  (Sub-daily features from high-frequency signals)")
    print("=" * 60)

    sleep_bounds = load_sleep_bounds()

    hr_feats = process_hr_features(sleep_bounds)
    wt_feats = process_wt_features(sleep_bounds)
    hrv_feats = process_hrv_features(sleep_bounds)

    out = hr_feats.merge(wt_feats, on=JOIN_KEY, how="outer")
    out = out.merge(hrv_feats, on=JOIN_KEY, how="outer")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "subdaily_features.csv")
    out.to_csv(out_path, index=False)

    feat_cols = [c for c in out.columns if c not in JOIN_KEY]
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}, Features: {len(feat_cols)}")

    print("\n[QC] NaN rates & stats:")
    for f in feat_cols:
        s = out[f]
        nan_pct = s.isna().mean()
        valid = s.dropna()
        if len(valid) > 0:
            print(f"  {f:30s}: NaN={nan_pct:.1%}, mean={valid.mean():.3f}, "
                  f"std={valid.std():.3f}")
        else:
            print(f"  {f:30s}: NaN={nan_pct:.1%} (all missing)")

    return out


if __name__ == "__main__":
    main()

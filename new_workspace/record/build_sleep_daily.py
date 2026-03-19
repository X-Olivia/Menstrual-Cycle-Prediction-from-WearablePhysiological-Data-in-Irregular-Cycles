"""
Build daily sleep-window features (processed_data/sleep.csv).

Equivalent to main_workspace data_process/daily_data_2.ipynb. Reads cycle anchor and
wearable signals from new_workspace (processed_dataset/cycle_cleaned_ov.csv,
processed_dataset/signals/*.csv), computes sleep/morning/evening/full window aggregates,
cycle position (day_in_cycle, hist_cycle_len, days_remaining_prior), biphasic shift,
missing fill, and per-cycle-early z-normalization. Writes processed_data/2/sleep.csv
(and full.csv, morning.csv, evening.csv, index.csv) for use by build_features_v4.py.

Run after: data_clean.py, ovulation_labels.py, wearable_signals.py.

Usage:
  cd new_workspace && python record/build_sleep_daily.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

NEW_WS = Path(__file__).resolve().parent.parent
SIGNALS_DIR = NEW_WS / "processed_dataset" / "signals"
CYCLE_CSV = NEW_WS / "processed_dataset" / "cycle_cleaned_ov.csv"
OUT_DIR = NEW_WS / "processed_dataset" / "daily_features" 
KEY = ["id", "study_interval", "day_in_study"]
CYCLE_LEN_PRIOR = 28.0
SHIFT_SHORT, SHIFT_LONG = 3, 7
INTERP_LIMIT = 3
K_CYCLE_EARLY, K_LONG = 5, 28
EPS, Z_CLIP = 1e-6, 5.0
POPULATION_CYCLE_MEAN, POPULATION_CYCLE_STD = 25.0, 5.0


def ts_to_min(ts):
    if pd.isna(ts):
        return np.nan
    p = str(ts).strip().split(":")
    h = int(p[0]) if len(p) > 0 else 0
    m = int(float(p[1])) if len(p) > 1 else 0
    s = float(p[2]) if len(p) > 2 else 0
    return h * 60 + m + s / 60


def agg_hrv(df):
    g = df.groupby(KEY, as_index=False)
    lf = g["low_frequency"].mean()
    hf = g["high_frequency"].mean()
    out = g.agg(rmssd_mean=("rmssd", "mean")).merge(lf.rename(columns={"low_frequency": "lf_mean"}))
    out = out.merge(hf.rename(columns={"high_frequency": "hf_mean"}))
    out["lf_hf_ratio"] = out["lf_mean"] / (out["hf_mean"].replace(0, np.nan))
    return out[[*KEY, "rmssd_mean", "lf_mean", "hf_mean", "lf_hf_ratio"]]


def agg_hr_window(path, window, sleep_bounds_df, cycle_days_df, ts_to_min_fn):
    sb = sleep_bounds_df
    agg_list = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk = chunk[(chunk["bpm"] >= 30) & (chunk["bpm"] <= 220) & (chunk["confidence"] >= 0.5)]
        chunk["timestamp_min"] = chunk["timestamp"].map(ts_to_min_fn)
        chunk = chunk.merge(cycle_days_df, on=KEY, how="inner")
        if window == "full":
            use = chunk[KEY + ["bpm"]]
        else:
            chunk = chunk.merge(sb, on=KEY, how="left")
            if window == "sleep":
                same = chunk.loc[chunk["timestamp_min"] <= chunk["sleep_end_min"], KEY + ["bpm"]]
                prev = chunk.drop(columns=["sleep_start_min", "sleep_end_min"], errors="ignore").copy()
                prev["wake_day"] = prev["day_in_study"] + 1
                prev = prev.merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]].rename(
                        columns={"day_in_study": "wake_day"}
                    ),
                    on=["id", "study_interval", "wake_day"],
                    how="inner",
                )
                prev = prev.loc[prev["timestamp_min"] >= prev["sleep_start_min"]].copy()
                prev["day_in_study"] = prev["wake_day"]
                prev = prev[KEY + ["bpm"]]
                use = pd.concat([same, prev], ignore_index=True)
            elif window == "morning":
                use = chunk.loc[
                    (chunk["timestamp_min"] >= chunk["sleep_end_min"])
                    & (chunk["timestamp_min"] < chunk["sleep_end_min"] + 30),
                    KEY + ["bpm"],
                ]
            elif window == "evening":
                chunk = chunk.drop(columns=["sleep_start_min", "sleep_end_min"], errors="ignore")
                chunk["wake_day"] = chunk["day_in_study"] + 1
                chunk = chunk.merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]].rename(
                        columns={"day_in_study": "wake_day"}
                    ),
                    on=["id", "study_interval", "wake_day"],
                    how="inner",
                )
                start_prev = (chunk["sleep_start_min"] - 30 + 1440) % 1440
                m_prev = (chunk["day_in_study"] == chunk["wake_day"] - 1) & (
                    (
                        (chunk["sleep_start_min"] >= 30)
                        & (chunk["timestamp_min"] >= chunk["sleep_start_min"] - 30)
                        & (chunk["timestamp_min"] < chunk["sleep_start_min"])
                    )
                    | ((chunk["sleep_start_min"] < 30) & (chunk["timestamp_min"] >= start_prev))
                )
                use_prev = chunk.loc[m_prev, KEY + ["bpm"]].copy()
                use_prev["day_in_study"] = chunk.loc[m_prev, "wake_day"].values
                same_df = chunk[KEY + ["timestamp_min", "bpm"]].merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]],
                    on=KEY,
                    how="inner",
                )
                m_same = (same_df["sleep_start_min"] < 30) & (
                    same_df["timestamp_min"] < same_df["sleep_start_min"]
                )
                use_same = same_df.loc[m_same, KEY + ["bpm"]].copy()
                use = pd.concat([use_prev, use_same], ignore_index=True)
            else:
                use = chunk[KEY + ["bpm"]]
        if len(use) == 0:
            continue
        agg_list.append(
            use.groupby(KEY, as_index=False)["bpm"].agg(
                hr_mean="mean", hr_std="std", hr_min="min", hr_max="max"
            )
        )
    if not agg_list:
        return pd.DataFrame(columns=KEY + ["hr_mean", "hr_std", "hr_min", "hr_max"])
    return pd.concat(agg_list).groupby(KEY, as_index=False).agg(
        hr_mean=("hr_mean", "mean"),
        hr_std=("hr_std", "mean"),
        hr_min=("hr_min", "min"),
        hr_max=("hr_max", "max"),
    )


def agg_wt_window(path, window, sleep_bounds_df, cycle_days_df, ts_to_min_fn):
    sb = sleep_bounds_df
    col = "temperature_diff_from_baseline"
    agg_list = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        chunk = chunk[chunk[col].abs() <= 5]
        chunk["timestamp_min"] = chunk["timestamp"].map(ts_to_min_fn)
        chunk = chunk.merge(cycle_days_df, on=KEY, how="inner")
        if window == "full":
            use = chunk[KEY + [col]]
        else:
            chunk = chunk.merge(sb, on=KEY, how="left")
            if window == "sleep":
                same = chunk.loc[chunk["timestamp_min"] <= chunk["sleep_end_min"], KEY + [col]]
                prev = chunk.drop(columns=["sleep_start_min", "sleep_end_min"], errors="ignore").copy()
                prev["wake_day"] = prev["day_in_study"] + 1
                prev = prev.merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]].rename(
                        columns={"day_in_study": "wake_day"}
                    ),
                    on=["id", "study_interval", "wake_day"],
                    how="inner",
                )
                prev = prev.loc[prev["timestamp_min"] >= prev["sleep_start_min"]].copy()
                prev["day_in_study"] = prev["wake_day"]
                prev = prev[KEY + [col]]
                use = pd.concat([same, prev], ignore_index=True)
            elif window == "morning":
                use = chunk.loc[
                    (chunk["timestamp_min"] >= chunk["sleep_end_min"])
                    & (chunk["timestamp_min"] < chunk["sleep_end_min"] + 30),
                    KEY + [col],
                ]
            elif window == "evening":
                chunk = chunk.drop(columns=["sleep_start_min", "sleep_end_min"], errors="ignore")
                chunk["wake_day"] = chunk["day_in_study"] + 1
                chunk = chunk.merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]].rename(
                        columns={"day_in_study": "wake_day"}
                    ),
                    on=["id", "study_interval", "wake_day"],
                    how="inner",
                )
                start_prev = (chunk["sleep_start_min"] - 30 + 1440) % 1440
                m_prev = (chunk["day_in_study"] == chunk["wake_day"] - 1) & (
                    (
                        (chunk["sleep_start_min"] >= 30)
                        & (chunk["timestamp_min"] >= chunk["sleep_start_min"] - 30)
                        & (chunk["timestamp_min"] < chunk["sleep_start_min"])
                    )
                    | ((chunk["sleep_start_min"] < 30) & (chunk["timestamp_min"] >= start_prev))
                )
                use_prev = chunk.loc[m_prev, KEY + [col]].copy()
                use_prev["day_in_study"] = chunk.loc[m_prev, "wake_day"].values
                same_df = chunk[KEY + ["timestamp_min", col]].merge(
                    sb[["id", "study_interval", "day_in_study", "sleep_start_min"]],
                    on=KEY,
                    how="inner",
                )
                m_same = (same_df["sleep_start_min"] < 30) & (
                    same_df["timestamp_min"] < same_df["sleep_start_min"]
                )
                use_same = same_df.loc[m_same, KEY + [col]].copy()
                use = pd.concat([use_prev, use_same], ignore_index=True)
            else:
                use = chunk[KEY + [col]]
        if len(use) == 0:
            continue
        agg_list.append(
            use.groupby(KEY, as_index=False)[col].agg(
                wt_mean="mean", wt_std="std", wt_min="min", wt_max="max"
            )
        )
    if not agg_list:
        return pd.DataFrame(columns=KEY + ["wt_mean", "wt_std", "wt_min", "wt_max"])
    return pd.concat(agg_list).groupby(KEY, as_index=False).agg(
        wt_mean=("wt_mean", "mean"),
        wt_std=("wt_std", "mean"),
        wt_min=("wt_min", "min"),
        wt_max=("wt_max", "max"),
    )


def add_biphasic_shift(df, col, out_col, short=3, long=7):
    df = df.sort_values(["id", "study_interval", "small_group_key", "day_in_study"])

    def _shift(x):
        recent = x.rolling(short, min_periods=1).mean()
        prior = x.rolling(long, min_periods=max(1, long // 2)).mean().shift(short)
        return (recent - prior).astype(np.float32)

    df[out_col] = df.groupby(["id", "study_interval"])[col].transform(_shift)
    return df


def _safe_z(val, mu, sig, eps, z_clip):
    if np.isnan(sig) or sig < eps:
        return 0.0
    return float(np.clip((val - mu) / sig, -z_clip, z_clip))


def rolling_z_v2(df, cols, k_early, k_long, eps, z_clip):
    out = df.sort_values(["id", "study_interval", "small_group_key", "day_in_study"]).copy()
    for col in cols:
        if col not in out.columns:
            continue
        z = np.zeros(len(out), dtype=np.float32)
        for (iid, sid, sgk), grp in out.groupby(["id", "study_interval", "small_group_key"]):
            x = grp[col].values
            idx = grp.index
            n = len(x)
            early_vals = x[:k_early]
            valid_early = early_vals[~np.isnan(early_vals)]
            use_early = len(valid_early) >= max(1, int(k_early * 0.4))
            if use_early:
                mu_early = np.nanmean(valid_early)
                sig_early = np.nanstd(valid_early)
            full_study = out[(out["id"] == iid) & (out["study_interval"] == sid)].sort_values(
                "day_in_study"
            )
            x_full = full_study[col].values
            idx_full = full_study.index.tolist()
            for i in range(n):
                global_i = idx_full.index(idx[i])
                if i < k_early and use_early:
                    z[idx[i]] = 0.0
                    continue
                if use_early:
                    z[idx[i]] = _safe_z(x[i], mu_early, sig_early, eps, z_clip)
                else:
                    hist = x_full[max(0, global_i - k_long) : global_i]
                    if len(hist) == 0:
                        z[idx[i]] = 0.0
                    else:
                        mu_r = np.nanmean(hist)
                        sig_r = np.nanstd(hist)
                        z[idx[i]] = _safe_z(x[i], mu_r, sig_r, eps, z_clip)
        out[col + "_z"] = z
    return out


def main():
    print("=" * 60)
    print("  build_sleep_daily.py (new_workspace → processed_data/2/sleep.csv)")
    print("=" * 60)
    print(f"  Cycle:  {CYCLE_CSV}")
    print(f"  Signals: {SIGNALS_DIR}")
    print(f"  Output: {OUT_DIR}")

    if not CYCLE_CSV.exists():
        raise SystemExit(
            f"Cycle CSV not found: {CYCLE_CSV}. Run data_clean.py then ovulation_labels.py."
        )
    if not SIGNALS_DIR.is_dir():
        raise SystemExit(f"Signals dir not found: {SIGNALS_DIR}. Run wearable_signals.py.")

    cycle = pd.read_csv(CYCLE_CSV)
    cycle_days = cycle[["id", "study_interval", "day_in_study", "small_group_key"]].drop_duplicates()
    print(f"cycle_days: {len(cycle_days)}")

    ct = pd.read_csv(SIGNALS_DIR / "computed_temperature_cycle.csv")
    ct_night = ct.groupby(["id", "study_interval", "sleep_end_day_in_study"], as_index=False).first()
    ct_night["sleep_start_min"] = ct_night["sleep_start_timestamp"].map(ts_to_min)
    ct_night["sleep_end_min"] = ct_night["sleep_end_timestamp"].map(ts_to_min)
    sleep_bounds = ct_night[
        ["id", "study_interval", "sleep_end_day_in_study", "sleep_start_min", "sleep_end_min"]
    ].rename(columns={"sleep_end_day_in_study": "day_in_study"})
    print(f"sleep_bounds: {len(sleep_bounds)}")

    hrv = pd.read_csv(SIGNALS_DIR / "heart_rate_variability_details_cycle.csv")
    n_hrv = len(hrv)
    hrv = hrv[
        (hrv["coverage"] >= 0.6)
        & (hrv["rmssd"] > 0)
        & (hrv["low_frequency"] > 0)
        & (hrv["high_frequency"] > 0)
    ]
    hrv["timestamp_min"] = hrv["timestamp"].map(ts_to_min)
    print(f"HRV: {100 * len(hrv) / n_hrv:.2f}% kept")

    hrv_merged = hrv.merge(sleep_bounds, on=KEY, how="left")
    hrv_full = agg_hrv(hrv)
    mask_same = hrv_merged["timestamp_min"] <= hrv_merged["sleep_end_min"]
    hrv_same = hrv_merged.loc[mask_same, hrv.columns]
    hrv_prev = hrv.copy()
    hrv_prev["wake_day"] = hrv_prev["day_in_study"] + 1
    hrv_prev = hrv_prev.merge(
        sleep_bounds.rename(columns={"day_in_study": "wake_day"})[
            ["id", "study_interval", "wake_day", "sleep_start_min"]
        ],
        on=["id", "study_interval", "wake_day"],
        how="inner",
    )
    hrv_prev = hrv_prev.loc[hrv_prev["timestamp_min"] >= hrv_prev["sleep_start_min"]].drop(
        columns=["sleep_start_min"], errors="ignore"
    )
    hrv_prev["day_in_study"] = hrv_prev["wake_day"]
    hrv_prev = hrv_prev[hrv.columns]
    hrv_sleep = agg_hrv(pd.concat([hrv_same, hrv_prev], ignore_index=True))
    hrv_morning = agg_hrv(
        hrv_merged.loc[
            (hrv_merged["timestamp_min"] >= hrv_merged["sleep_end_min"])
            & (hrv_merged["timestamp_min"] < hrv_merged["sleep_end_min"] + 30),
            hrv.columns,
        ]
    )
    sb_ev = sleep_bounds[["id", "study_interval", "day_in_study", "sleep_start_min"]].rename(
        columns={"day_in_study": "wake_day"}
    )
    hrv_ev_prev = hrv.copy()
    hrv_ev_prev["wake_day"] = hrv_ev_prev["day_in_study"] + 1
    hrv_ev_prev = hrv_ev_prev.merge(sb_ev, on=["id", "study_interval", "wake_day"], how="inner")
    start_prev = (hrv_ev_prev["sleep_start_min"] - 30 + 1440) % 1440
    mask_prev = (
        (hrv_ev_prev["sleep_start_min"] >= 30)
        & (hrv_ev_prev["timestamp_min"] >= hrv_ev_prev["sleep_start_min"] - 30)
        & (hrv_ev_prev["timestamp_min"] < hrv_ev_prev["sleep_start_min"])
    ) | ((hrv_ev_prev["sleep_start_min"] < 30) & (hrv_ev_prev["timestamp_min"] >= start_prev))
    hrv_ev_prev = hrv_ev_prev.loc[mask_prev].copy()
    hrv_ev_prev["day_in_study"] = hrv_ev_prev["wake_day"]
    hrv_ev_prev = hrv_ev_prev[hrv.columns]
    hrv_ev_same = hrv.merge(
        sleep_bounds[["id", "study_interval", "day_in_study", "sleep_start_min"]],
        on=KEY,
        how="inner",
    )
    hrv_ev_same = hrv_ev_same.loc[
        (hrv_ev_same["sleep_start_min"] < 30)
        & (hrv_ev_same["timestamp_min"] < hrv_ev_same["sleep_start_min"]),
        hrv.columns,
    ]
    hrv_evening = agg_hrv(pd.concat([hrv_ev_prev, hrv_ev_same], ignore_index=True))
    print(
        f"HRV daily: full {len(hrv_full)}, sleep {len(hrv_sleep)}, morning {len(hrv_morning)}, evening {len(hrv_evening)}"
    )

    ct_daily = ct_night[["id", "study_interval", "sleep_end_day_in_study", "nightly_temperature"]].rename(
        columns={"sleep_end_day_in_study": "day_in_study"}
    )
    rhr = pd.read_csv(SIGNALS_DIR / "resting_heart_rate_cycle.csv")[
        ["id", "study_interval", "day_in_study", "value"]
    ].rename(columns={"value": "resting_hr"})
    rhr = rhr.groupby(KEY, as_index=False)["resting_hr"].mean()
    hrv_by_w = {"full": hrv_full, "sleep": hrv_sleep, "morning": hrv_morning, "evening": hrv_evening}

    hr_path = SIGNALS_DIR / "heart_rate_cycle.csv"
    wt_path = SIGNALS_DIR / "wrist_temperature_cycle.csv"
    daily_by_window = {}
    for w in ["full", "sleep", "morning", "evening"]:
        hr_w = (
            agg_hr_window(hr_path, w, sleep_bounds, cycle_days, ts_to_min)
            if hr_path.exists()
            else pd.DataFrame(columns=KEY + ["hr_mean", "hr_std", "hr_min", "hr_max"])
        )
        wt_w = (
            agg_wt_window(wt_path, w, sleep_bounds, cycle_days, ts_to_min)
            if wt_path.exists()
            else pd.DataFrame(columns=KEY + ["wt_mean", "wt_std", "wt_min", "wt_max"])
        )
        base = (
            cycle_days.merge(hrv_by_w[w], on=KEY, how="left")
            .merge(hr_w, on=KEY, how="left")
            .merge(wt_w, on=KEY, how="left")
            .merge(ct_daily, on=KEY, how="left")
            .merge(rhr, on=KEY, how="left")
        )
        daily_by_window[w] = base
        print(f"  {w} rows {len(base)}")

    for w in daily_by_window:
        df = daily_by_window[w].sort_values(
            ["id", "study_interval", "small_group_key", "day_in_study"]
        )
        cycle_start = df.groupby(["id", "study_interval", "small_group_key"])[
            "day_in_study"
        ].transform("min")
        df["day_in_cycle"] = (df["day_in_study"] - cycle_start).astype(np.float32)
        df["day_in_cycle_frac"] = (df["day_in_cycle"] / CYCLE_LEN_PRIOR).astype(np.float32)
        daily_by_window[w] = df
    print("[P1] day_in_cycle + day_in_cycle_frac added")

    for w in daily_by_window:
        df = daily_by_window[w]
        if "wt_mean" in df.columns:
            df = add_biphasic_shift(df, "wt_mean", "wt_shift_7v3", SHIFT_SHORT, SHIFT_LONG)
        if "nightly_temperature" in df.columns:
            df = add_biphasic_shift(
                df, "nightly_temperature", "temp_shift_7v3", SHIFT_SHORT, SHIFT_LONG
            )
        daily_by_window[w] = df
    print("[P5] Biphasic shift (wt_shift_7v3, temp_shift_7v3) added")

    feat_cols = [
        "rmssd_mean",
        "lf_mean",
        "hf_mean",
        "lf_hf_ratio",
        "hr_mean",
        "hr_std",
        "hr_min",
        "hr_max",
        "wt_mean",
        "wt_std",
        "wt_min",
        "wt_max",
        "nightly_temperature",
        "resting_hr",
    ]
    for w, df in daily_by_window.items():
        out = df.sort_values(KEY + ["day_in_study"]).copy()
        for col in feat_cols:
            if col not in out.columns:
                continue
            miss = out[col].isna()
            out[col + "_missing"] = miss.astype(int)
            out[col] = (
                out.groupby(KEY)[col]
                .transform(
                    lambda x: x.interpolate(
                        method="linear", limit=INTERP_LIMIT, limit_direction="both"
                    )
                )
            )
            out.loc[out[col].isna(), col] = 0
        daily_by_window[w] = out
    print(f"[P3] Missing fill (interp limit={INTERP_LIMIT})")

    cycle_info = (
        cycle_days.groupby(["id", "study_interval", "small_group_key"])
        .agg(cycle_start=("day_in_study", "min"), cycle_len=("day_in_study", "count"))
        .reset_index()
        .sort_values(["id", "cycle_start"])
        .reset_index(drop=True)
    )
    hist_means, hist_stds = [], []
    for _, subj_df in cycle_info.groupby("id"):
        lens = subj_df["cycle_len"].values
        for i in range(len(lens)):
            if i == 0:
                hist_means.append(POPULATION_CYCLE_MEAN)
                hist_stds.append(POPULATION_CYCLE_STD)
            else:
                prev = lens[:i]
                hist_means.append(float(np.mean(prev)))
                hist_stds.append(
                    float(np.std(prev)) if len(prev) > 1 else POPULATION_CYCLE_STD
                )
    cycle_info["hist_cycle_len_mean"] = hist_means
    cycle_info["hist_cycle_len_std"] = hist_stds

    for w in daily_by_window:
        daily_by_window[w] = daily_by_window[w].merge(
            cycle_info[
                [
                    "id",
                    "study_interval",
                    "small_group_key",
                    "hist_cycle_len_mean",
                    "hist_cycle_len_std",
                ]
            ],
            on=["id", "study_interval", "small_group_key"],
            how="left",
        )
        daily_by_window[w]["hist_cycle_len_mean"] = (
            daily_by_window[w]["hist_cycle_len_mean"]
            .fillna(POPULATION_CYCLE_MEAN)
            .astype(np.float32)
        )
        daily_by_window[w]["hist_cycle_len_std"] = (
            daily_by_window[w]["hist_cycle_len_std"]
            .fillna(POPULATION_CYCLE_STD)
            .astype(np.float32)
        )
        dpr = (
            daily_by_window[w]["hist_cycle_len_mean"] - daily_by_window[w]["day_in_cycle"]
        ).astype(np.float64)
        daily_by_window[w]["days_remaining_prior"] = dpr.astype(np.float32)
        daily_by_window[w]["days_remaining_prior_log"] = (
            np.sign(dpr) * np.log1p(np.abs(dpr))
        ).astype(np.float32)
    print("[P6] hist_cycle_len + days_remaining_prior added")

    norm_cols = [c for c in feat_cols if not c.endswith("_missing")]
    for w in daily_by_window:
        daily_by_window[w] = rolling_z_v2(
            daily_by_window[w], norm_cols, K_CYCLE_EARLY, K_LONG, EPS, Z_CLIP
        )
    print(f"[P4] Z-norm (K_early={K_CYCLE_EARLY}, K_long={K_LONG})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_days.to_csv(OUT_DIR / "index.csv", index=False)
    for w, name in [("full", "full"), ("sleep", "sleep"), ("morning", "morning"), ("evening", "evening")]:
        daily_by_window[w].to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"\n[DONE] Saved: {list(OUT_DIR.glob('*.csv'))}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import pickle
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from protocol import (
    DEFAULT_HISTORY_CYCLE_LEN,
    DEFAULT_HISTORY_CYCLE_STD,
    FEATURE_SIGMA,
    LABEL_LUTEAL_MAX,
    LABEL_LUTEAL_MIN,
    OVULATION_PROBABILITY_THRESHOLD,
    QUALITY_MIN_OV_DAY,
    QUALITY_MIN_TEMP_SHIFT,
    QUALITY_POST_START_OFFSET,
    QUALITY_POST_WINDOW,
    QUALITY_PRE_WINDOW,
    REPORT_DAY_THRESHOLDS,
)

warnings.filterwarnings("ignore")

# Use new_workspace data only (no main_workspace dependency).
# File: new_workspace/record/experiment/multisignal_pipeline/data.py
# parents:
#   0 = multisignal_pipeline, 1 = experiment, 2 = record, 3 = new_workspace
NEW_WS = Path(__file__).resolve().parents[3]
PROCESSED = NEW_WS / "processed_dataset"
SIGNALS_DIR = PROCESSED / "signals"
CYCLE_OV_CSV = PROCESSED / "cycle_cleaned_ov.csv"
CYCLE_CSV = CYCLE_OV_CSV
WORKSPACE = str(NEW_WS)
LOAD_ALL_SIGNALS_CACHE = PROCESSED / "cache" / "multisignal_load_all_signals.pkl"


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def _pr(tag, ae, prefix="  "):
    ae = np.array(ae, dtype=float)
    if len(ae) == 0:
        return {}
    r = {
        "n": len(ae),
        "mae": float(np.mean(ae)),
        "acc_1d": float((ae <= REPORT_DAY_THRESHOLDS[0]).mean()),
        "acc_2d": float((ae <= REPORT_DAY_THRESHOLDS[1]).mean()),
        "acc_3d": float((ae <= REPORT_DAY_THRESHOLDS[2]).mean()),
        "acc_5d": float((ae <= REPORT_DAY_THRESHOLDS[3]).mean()),
    }
    print(
        f"{prefix}[{tag}] n={r['n']} MAE={r['mae']:.2f}"
        f" ±1d={r['acc_1d']:.1%} ±2d={r['acc_2d']:.1%}"
        f" ±3d={r['acc_3d']:.1%} ±5d={r['acc_5d']:.1%}"
    )
    return r


def _load_all_signals_source_files():
    return [
        CYCLE_OV_CSV,
        SIGNALS_DIR / "computed_temperature_cycle.csv",
        SIGNALS_DIR / "wrist_temperature_cycle.csv",
        SIGNALS_DIR / "resting_heart_rate_cycle.csv",
        SIGNALS_DIR / "heart_rate_variability_details_cycle.csv",
        SIGNALS_DIR / "heart_rate_cycle.csv",
    ]


def _snapshot_source_files(paths):
    return {
        str(path): {
            "mtime_ns": path.stat().st_mtime_ns,
            "size": path.stat().st_size,
        }
        for path in paths
    }


def _try_load_all_signals_cache():
    source_files = _load_all_signals_source_files()
    if not LOAD_ALL_SIGNALS_CACHE.exists():
        return None
    if not all(path.exists() for path in source_files):
        return None
    try:
        with LOAD_ALL_SIGNALS_CACHE.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if payload.get("source_snapshot") != _snapshot_source_files(source_files):
        return None
    print(f"  Using cached load_all_signals(): {LOAD_ALL_SIGNALS_CACHE}")
    return payload.get("result")


def _save_all_signals_cache(result):
    source_files = _load_all_signals_source_files()
    LOAD_ALL_SIGNALS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_snapshot": _snapshot_source_files(source_files),
        "result": result,
    }
    with LOAD_ALL_SIGNALS_CACHE.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt load_all_signals() cache: {LOAD_ALL_SIGNALS_CACHE}")


def load_all_signals():
    """Load and aggregate ALL available signals from new_workspace processed_dataset/signals."""
    cached = _try_load_all_signals_cache()
    if cached is not None:
        return cached

    print("  Loading cycle structure...")
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    sig_base = os.path.join(WORKSPACE, "processed_dataset", "signals")

    print("  Loading nightly temperature...")
    ct = pd.read_csv(os.path.join(sig_base, "computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    print("  Loading nocturnal wrist temperature...")
    wt = pd.read_csv(
        os.path.join(sig_base, "wrist_temperature_cycle.csv"),
        usecols=key + ["timestamp", "temperature_diff_from_baseline"],
    )
    wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
    noct_wt = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
    noct_temp_daily = (
        noct_wt.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
    )
    noct_temp_daily.rename(columns={"temperature_diff_from_baseline": "noct_temp"}, inplace=True)

    print("  Loading resting heart rate...")
    rhr = pd.read_csv(os.path.join(sig_base, "resting_heart_rate_cycle.csv"), usecols=key + ["value"])
    rhr_daily = rhr.groupby(key)["value"].mean().reset_index()
    rhr_daily.rename(columns={"value": "rhr"}, inplace=True)

    print("  Loading HRV details...")
    hrv = pd.read_csv(os.path.join(sig_base, "heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        rmssd_mean=("rmssd", "mean"),
        rmssd_std=("rmssd", "std"),
        lf_mean=("low_frequency", "mean"),
        hf_mean=("high_frequency", "mean"),
        hrv_coverage=("coverage", "mean"),
    ).reset_index()
    hrv_daily["lf_hf_ratio"] = hrv_daily["lf_mean"] / hrv_daily["hf_mean"].clip(lower=1)

    print("  Loading nocturnal HR (chunked, 0-6AM)...")
    hr_path = os.path.join(sig_base, "heart_rate_cycle.csv")
    hr_aggs = []
    for chunk in pd.read_csv(
        hr_path,
        chunksize=2_000_000,
        usecols=key + ["timestamp", "bpm", "confidence"],
    ):
        chunk["hour"] = pd.to_datetime(chunk["timestamp"]).dt.hour
        noct = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 6) & (chunk["confidence"] >= 1)]
        if len(noct) > 0:
            agg = noct.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
            hr_aggs.append(agg)
    hr_daily = pd.concat(hr_aggs).groupby(key).mean().reset_index()
    hr_daily.rename(
        columns={"mean": "noct_hr_mean", "std": "noct_hr_std", "min": "noct_hr_min"},
        inplace=True,
    )

    print("  Merging all signals...")
    merged = cc.merge(ct_daily, on=key, how="left")
    merged = merged.merge(noct_temp_daily, on=key, how="left")
    merged = merged.merge(rhr_daily, on=key, how="left")
    merged = merged.merge(hrv_daily, on=key, how="left")
    merged = merged.merge(hr_daily, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))

    signal_cols = [
        "nightly_temperature",
        "noct_temp",
        "rhr",
        "rmssd_mean",
        "rmssd_std",
        "lf_mean",
        "hf_mean",
        "lf_hf_ratio",
        "noct_hr_mean",
        "noct_hr_std",
        "noct_hr_min",
        "hrv_coverage",
    ]

    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        entry = {
            "dic": (grp["day_in_study"] - cs).values,
            "id": grp["id"].values[0],
            "cycle_len": n,
        }
        for sc in signal_cols:
            entry[sc] = grp[sc].values if sc in grp.columns else np.full(n, np.nan)
        cycle_series[sgk] = entry

    sgk_order = (
        merged.groupby("small_group_key")["day_in_study"].min().reset_index().rename(columns={"day_in_study": "start"})
    )
    sgk_order = sgk_order.merge(
        merged[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    subj_order = {}
    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        subj_order[uid] = sgks
        past_lens = []
        for sgk in sgks:
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = (
                    np.mean(past_lens) if past_lens else DEFAULT_HISTORY_CYCLE_LEN
                )
                cycle_series[sgk]["hist_cycle_std"] = (
                    np.std(past_lens) if len(past_lens) > 1 else DEFAULT_HISTORY_CYCLE_STD
                )
                past_lens.append(cycle_series[sgk]["cycle_len"])

    quality = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        raw = cycle_series[sgk]["nightly_temperature"]
        if np.isnan(raw).all():
            continue
        t = pd.Series(raw).interpolate(limit_direction="both").values
        ov = lh_ov_dict[sgk]
        n = len(t)
        if ov < QUALITY_MIN_OV_DAY or ov + QUALITY_POST_START_OFFSET >= n:
            continue
        pre = np.mean(t[max(0, ov - QUALITY_PRE_WINDOW) : ov])
        post = np.mean(
            t[ov + QUALITY_POST_START_OFFSET : min(n, ov + QUALITY_POST_START_OFFSET + QUALITY_POST_WINDOW)]
        )
        if post - pre >= QUALITY_MIN_TEMP_SHIFT:
            quality.add(sgk)

    labeled = [s for s in cycle_series if s in lh_ov_dict]
    print(f"  Cycles: {len(cycle_series)} | Labeled: {len(labeled)} | Quality: {len(quality)}")
    result = (lh_ov_dict, cycle_series, quality, subj_order, signal_cols)
    _save_all_signals_cache(result)
    return result


def _get_multi(data, sigs, sigma=FEATURE_SIGMA):
    """Get multiple cleaned signals stacked as (n_days, n_signals)."""
    arrays = []
    for sk in sigs:
        raw = data.get(sk)
        if raw is None or np.isnan(raw).all():
            return None
        arrays.append(_clean(raw, sigma=sigma))
    return np.column_stack(arrays)


def get_lh_ovulation_labels(cycle_csv: Path | None = None) -> pd.DataFrame:
    """
    LH-based ovulation ground truth labels.

    This is a direct, dependency-free copy of main_workspace/model/ovulation_detect.py:get_lh_ovulation_labels,
    but it reads from new_workspace/processed_dataset/cycle_cleaned_ov.csv.

    Returns a DataFrame with columns:
      small_group_key, ov_day_in_study, cs, ce, luteal_len, ov_dic, ...
    """
    cc = pd.read_csv(str(cycle_csv or CYCLE_CSV))
    ov = cc[cc["ovulation_prob_fused"] > OVULATION_PROBABILITY_THRESHOLD]
    lh_ov = (
        ov.groupby("small_group_key")
        .apply(lambda g: g.loc[g["ovulation_prob_fused"].idxmax()], include_groups=False)
        [["id", "study_interval", "day_in_study"]]
        .reset_index()
        .rename(columns={"day_in_study": "ov_day_in_study"})
    )
    cs = cc.groupby("small_group_key")["day_in_study"].min().reset_index().rename(columns={"day_in_study": "cs"})
    ce = cc.groupby("small_group_key")["day_in_study"].max().reset_index().rename(columns={"day_in_study": "ce"})
    lh_ov = lh_ov.merge(cs, on="small_group_key").merge(ce, on="small_group_key")
    lh_ov["luteal_len"] = lh_ov["ce"] - lh_ov["ov_day_in_study"]
    lh_ov = lh_ov[
        (lh_ov["luteal_len"] >= LABEL_LUTEAL_MIN)
        & (lh_ov["luteal_len"] <= LABEL_LUTEAL_MAX)
    ]
    lh_ov["ov_dic"] = lh_ov["ov_day_in_study"] - lh_ov["cs"]
    return lh_ov

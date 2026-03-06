"""
build_features_v5.py — 经期预测特征管线 v5
===========================================
基于 v4，包含以下改进：

1. z-normalization 前5天修复：不再置零，改用 per-subject z-score
   - 原问题：前5天被用作基线后全部置零，导致 21+ horizon 无穿戴信号
   - 修复后：前5天使用 per-subject 均值/标准差归一化，保留经期阶段信号

2. 新增上一周期长度特征：
   - prev_cycle_len: 上一周期的实际天数
   - prev_cycle_deviation: prev_cycle_len - hist_cycle_len_mean

输出:
  - processed_data/v5/daily_features_v5.csv
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
    interpolate_within_cycle, compute_rolling_features,
    compute_rate_of_change,
    WORKSPACE, KEY, CYCLE_GROUP, ALL_RAW, BASELINE_RELATIVE,
    RAW_WEARABLE, ROLLING_Z_SOURCES, DELTA_SOURCES, SYMPTOM_COLS,
)

OUTPUT_DIR = os.path.join(WORKSPACE, "processed_data", "v5")


def per_cycle_early_z_normalize_v5(df, k_early=5, min_coverage=0.4, clip=5.0, eps=1e-8):
    """v5: Same as v4 but early days use per-subject z-score instead of zero.

    Change from v4: line `z_arr[idx[:k]] = 0.0` replaced with per-subject
    z-normalization for the first k_early days, so the model sees wearable
    signals during the menstrual phase (which corresponds to 21+ horizon).
    """
    z_cols = [c for c in ALL_RAW if c in df.columns]
    df = df.sort_values(CYCLE_GROUP + ["day_in_study"])

    subj_stats = {}
    for col in z_cols:
        stats = df.groupby("id")[col].agg(["mean", "std"])
        subj_stats[col] = stats

    n_cycle_based = 0
    n_fallback = 0
    n_early_subject = 0

    for col in z_cols:
        is_bl = col in BASELINE_RELATIVE
        z_arr = np.full(len(df), 0.0, dtype=np.float32)

        for _, grp in df.groupby(CYCLE_GROUP):
            idx = grp.index.values
            x = grp[col].values.astype(np.float64)
            n = len(x)
            subj_id = grp["id"].iloc[0]

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

                # v5 FIX: use per-subject z-score for early days instead of zero
                k = min(k_early, n)
                s_mu = subj_stats[col].loc[subj_id, "mean"]
                s_std = subj_stats[col].loc[subj_id, "std"]
                x_early = x[:k]

                if is_bl:
                    if not np.isnan(s_mu):
                        z_arr[idx[:k]] = np.where(
                            np.isnan(x_early), 0.0,
                            np.clip(x_early - s_mu, -clip, clip).astype(np.float32),
                        )
                    else:
                        z_arr[idx[:k]] = 0.0
                elif not np.isnan(s_std) and s_std > eps:
                    z_arr[idx[:k]] = np.where(
                        np.isnan(x_early), 0.0,
                        np.clip((x_early - s_mu) / s_std, -clip, clip).astype(np.float32),
                    )
                else:
                    z_arr[idx[:k]] = 0.0
                n_early_subject += 1
            else:
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
    print(f"[z-norm-v5] Per-cycle-early z-normalization for {len(z_cols)} features "
          f"(cycle-based: {n_cycle_based}/{total}, fallback: {n_fallback}/{total})")
    print(f"[z-norm-v5] Early days: {n_early_subject} cycle×feature pairs use per-subject z-score (was zero in v4)")
    return df


def add_prev_cycle_features(df):
    """Add previous cycle length and deviation features."""
    cycle_lens = (
        df.groupby(CYCLE_GROUP)["day_in_study"]
        .count()
        .reset_index()
        .rename(columns={"day_in_study": "cycle_len"})
    )

    cycle_lens["_cycle_num"] = (
        cycle_lens["small_group_key"]
        .str.extract(r"_cycle(\d+)$")[0]
        .astype(int)
    )
    cycle_lens = cycle_lens.sort_values(["id", "study_interval", "_cycle_num"])

    cycle_lens["prev_cycle_len"] = (
        cycle_lens.groupby(["id", "study_interval"])["cycle_len"]
        .shift(1)
    )

    df = df.merge(
        cycle_lens[["id", "study_interval", "small_group_key", "prev_cycle_len"]],
        on=["id", "study_interval", "small_group_key"],
        how="left",
    )

    df["prev_cycle_deviation"] = df["prev_cycle_len"] - df["hist_cycle_len_mean"]

    n_valid = df["prev_cycle_len"].notna().sum()
    print(f"[prev_cycle] Added prev_cycle_len: {n_valid}/{len(df)} rows have values "
          f"({n_valid/len(df)*100:.0f}%)")
    print(f"[prev_cycle] prev_cycle_deviation range: "
          f"{df['prev_cycle_deviation'].min():.1f} ~ {df['prev_cycle_deviation'].max():.1f}")

    df["prev_cycle_len"] = df["prev_cycle_len"].fillna(df["hist_cycle_len_mean"])
    df["prev_cycle_deviation"] = df["prev_cycle_deviation"].fillna(0.0)

    return df


def assemble_features_v5(df):
    """v5 assembly: v4 features + prev_cycle features."""
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
    feat_h = ["prev_cycle_len", "prev_cycle_deviation"]

    all_features = feat_a + feat_b + feat_c + feat_d + feat_e + feat_f + feat_g + feat_h
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"[WARN] Missing features: {missing}")
    present = [f for f in all_features if f in df.columns]
    out = df[KEY + present].copy()
    print(f"[assemble-v5] {len(present)} model features + {len(KEY)} keys = {len(out.columns)} cols")
    return out, present


def main():
    print("=" * 60)
    print("  build_features_v5.py")
    print("  (v5: early-days z-score fix + prev_cycle features)")
    print("=" * 60)

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

    # v5: fixed z-normalization (early days use per-subject, not zero)
    df = per_cycle_early_z_normalize_v5(df)

    df = compute_rolling_features(df, window=5)
    df = compute_rate_of_change(df)

    # v5: add previous cycle length features
    df = add_prev_cycle_features(df)

    out, feature_names = assemble_features_v5(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "daily_features_v5.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")
    print(f"  Rows: {len(out)}")
    print(f"  Features: {len(feature_names)}")

    # QC: check early-days z-scores are no longer zero
    early_mask = df["day_in_cycle"] < 5
    z_cols = [f"{c}_z" for c in RAW_WEARABLE] + ["full_sleep_br_z", "deep_sleep_br_z"]
    n_nonzero = 0
    n_total = 0
    for c in z_cols:
        if c in out.columns:
            vals = out.loc[early_mask, c]
            n_nonzero += (vals != 0).sum()
            n_total += len(vals)
    print(f"\n[QC] Early days (day_in_cycle < 5): {n_nonzero}/{n_total} "
          f"z-scores are non-zero ({n_nonzero/n_total*100:.0f}% vs 0% in v4)")

    print("\n[QC] NaN rates per feature:")
    for f in feature_names:
        nan_rate = out[f].isna().mean()
        if nan_rate > 0:
            print(f"  {f:30s}: {nan_rate:.1%} NaN")

    return out, feature_names


if __name__ == "__main__":
    main()

"""Data loading and label construction for LightGBM day-level regression."""
import re

import numpy as np
import pandas as pd

from .config import FEATURES_CSV, CYCLE_CSV, MAX_CYCLE_LEN


def load_data(features_csv=FEATURES_CSV, cycle_csv=CYCLE_CSV):
    """Load features + cycle metadata, construct days_until_next_menses label."""
    feat = pd.read_csv(features_csv)
    cycle = pd.read_csv(cycle_csv)

    key = ["id", "study_interval", "day_in_study", "small_group_key"]
    df = feat.merge(
        cycle[["id", "study_interval", "day_in_study", "small_group_key"]],
        on=key, how="inner",
    )

    # Label: days_until_next_menses = cycle_end + 1 - day_in_study
    cycle_group = ["id", "study_interval", "small_group_key"]
    cycle_end = (
        df.groupby(cycle_group)["day_in_study"].max()
        .reset_index()
        .rename(columns={"day_in_study": "cycle_end"})
    )
    df = df.merge(cycle_end, on=cycle_group, how="left")
    df["days_until_next_menses"] = (df["cycle_end"] + 1 - df["day_in_study"]).astype(float)
    df = df.drop(columns=["cycle_end"])

    # Filter long cycles
    cycle_len = df.groupby(cycle_group)["day_in_study"].transform("count")
    before = len(df)
    df = df[cycle_len <= MAX_CYCLE_LEN].copy()
    n_filtered = (before - len(df))
    if n_filtered:
        print(f"[data] Filtered {n_filtered} rows from cycles > {MAX_CYCLE_LEN} days")

    key_cols = {"id", "study_interval", "day_in_study", "small_group_key",
                "days_until_next_menses"}
    available = [c for c in df.columns if c not in key_cols]

    print(f"[data] {len(df)} rows, {df['id'].nunique()} subjects, "
          f"{df.groupby(cycle_group).ngroups} cycles")
    print(f"[data] Label range: {df['days_until_next_menses'].min():.0f} ~ "
          f"{df['days_until_next_menses'].max():.0f}")

    return df, available


def subject_split(df, test_ratio=0.15, seed=42):
    """Subject-level split: fixed test set + train/val."""
    rng = np.random.RandomState(seed)
    subjects = sorted(df["id"].unique())
    rng.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = set(subjects[:n_test])
    trainval_subjects = [s for s in subjects if s not in test_subjects]

    # 80/20 train/val from remaining
    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(trainval_subjects)
    n_val = max(1, int(len(trainval_subjects) * 0.2))
    val_subjects = set(trainval_subjects[:n_val])
    train_subjects = set(trainval_subjects[n_val:])

    print(f"[split] train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)} subjects")
    return train_subjects, val_subjects, test_subjects


def _cycle_num(sgk: str) -> int:
    """Extract cycle number from small_group_key like '12345_1_cycle3' -> 3."""
    m = re.search(r"_cycle(\d+)$", sgk)
    return int(m.group(1)) if m else 0


def cycle_split(df, val_ratio=0.2, seed=42):
    """Leave-last-cycle-out split: train on earlier cycles, test on last cycle.

    Returns three boolean masks (train_mask, val_mask, test_mask) instead of
    subject sets, since rows from the same subject can appear in both train
    and test.

    - Test: last cycle of every subject (that has >= 2 cycles)
    - Validation: second-to-last cycle from a random val_ratio of subjects
    - Train: everything else
    """
    rng = np.random.RandomState(seed)
    cycle_group = ["id", "study_interval", "small_group_key"]

    # Build per-cycle summary: one row per cycle
    cycles = (
        df.groupby(cycle_group)
        .size()
        .reset_index(name="_n_rows")
    )
    cycles["_cycle_num"] = cycles["small_group_key"].apply(_cycle_num)
    cycles = cycles.sort_values(["id", "study_interval", "_cycle_num"])

    # For each subject, find last and second-to-last cycle
    max_cycle = (
        cycles.groupby("id")["_cycle_num"]
        .max()
        .reset_index()
        .rename(columns={"_cycle_num": "_max_cycle"})
    )
    cycles = cycles.merge(max_cycle, on="id", how="left")

    # Subjects with >= 2 cycles can contribute to test
    multi_cycle_subjects = cycles.loc[
        cycles["_max_cycle"] > cycles.groupby("id")["_cycle_num"].transform("min"),
        "id"
    ].unique()

    # Test = last cycle for multi-cycle subjects
    test_keys = set(
        cycles.loc[
            (cycles["id"].isin(multi_cycle_subjects))
            & (cycles["_cycle_num"] == cycles["_max_cycle"]),
            "small_group_key"
        ].values
    )

    # Val: second-to-last cycle from a random subset of subjects
    val_subjects = list(multi_cycle_subjects)
    rng.shuffle(val_subjects)
    n_val = max(1, int(len(val_subjects) * val_ratio))
    val_subject_set = set(val_subjects[:n_val])

    second_last = (
        cycles.loc[
            (cycles["id"].isin(val_subject_set))
            & (cycles["_cycle_num"] == cycles["_max_cycle"] - 1),
            "small_group_key"
        ].values
    )
    val_keys = set(second_last)

    # Train = everything else
    train_mask = ~df["small_group_key"].isin(test_keys | val_keys)
    val_mask = df["small_group_key"].isin(val_keys)
    test_mask = df["small_group_key"].isin(test_keys)

    n_train_subj = df.loc[train_mask, "id"].nunique()
    n_val_subj = df.loc[val_mask, "id"].nunique()
    n_test_subj = df.loc[test_mask, "id"].nunique()
    n_train_cycles = df.loc[train_mask, "small_group_key"].nunique()
    n_test_cycles = df.loc[test_mask, "small_group_key"].nunique()

    print(f"[cycle_split] Train: {train_mask.sum()} rows ({n_train_cycles} cycles, {n_train_subj} subj), "
          f"Val: {val_mask.sum()} rows ({n_val_subj} subj), "
          f"Test: {test_mask.sum()} rows ({n_test_cycles} cycles, {n_test_subj} subj)")

    return train_mask, val_mask, test_mask

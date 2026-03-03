"""Data loading and label construction for LightGBM day-level regression."""
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

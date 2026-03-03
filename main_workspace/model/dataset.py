"""Data layer D1–D5: in-cycle sequences + dual-task labels and masks."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CYCLE_CSV, FULL_CSV, SLEEP_CSV, FEATURE_COLS, FILL_MISSING, INPUT_DIM, MAX_CYCLE_LEN


def load_and_merge(cycle_path, full_path):
    """D1+D2: Load cycle and daily feature file; keep only in-cycle days; align features.
    Use (id, study_interval, day_in_study, small_group_key) to identify a cycle so each row
    uniquely corresponds to one day in one cycle.

    P2: full_path should point to sleep.csv (overnight window) as primary input, aligned with
    Wang 2025 (Apple wrist temperature algorithm) and Hamidovic 2023 (sleep HRV).
    """
    cycle = pd.read_csv(cycle_path)
    daily = pd.read_csv(full_path)
    if "small_group_key" in daily.columns:
        key = ["id", "study_interval", "day_in_study", "small_group_key"]
    else:
        key = ["id", "study_interval", "day_in_study"]
    # Keep only columns that exist in the daily file (some new features may not yet be generated)
    avail_feats = [c for c in FEATURE_COLS if c in daily.columns]
    df = cycle.merge(daily[key + avail_feats], on=key, how="inner")
    df[avail_feats] = df[avail_feats].fillna(FILL_MISSING).astype(np.float32)
    # Pad missing feature columns with zeros so downstream code always sees INPUT_DIM columns
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    return df


def add_next_menses(df):
    """D3: For each cycle compute next_menses_day, yielding days_until_next_menses and mask_menses.
    next_menses_day = cycle end day + 1 (max(day_in_study)+1 within same small_group_key),
    i.e. next menses onset = last day of cycle + 1, no cross-cycle or cross study_interval.
    """
    cycle_end = (
        df.groupby(["id", "small_group_key"], sort=True)["day_in_study"]
        .max()
        .reset_index()
        .rename(columns={"day_in_study": "cycle_end"})
    )
    cycle_end["next_menses_day"] = cycle_end["cycle_end"] + 1
    df = df.merge(
        cycle_end[["id", "small_group_key", "next_menses_day"]],
        on=["id", "small_group_key"],
        how="left",
    )
    df["days_until_next_menses"] = (df["next_menses_day"] - df["day_in_study"]).astype(np.float32)
    df["mask_menses"] = df["next_menses_day"].notna()
    return df


def add_ovulation_labels(df):
    """D4: ovulation_label=ovulation_prob_fused; only days with ovulation label participate in BCE (per-day mask)."""
    df["ovulation_label"] = df["ovulation_prob_fused"].astype(np.float32)
    # Correct: only days with ovulation_prob_fused labeled contribute to loss, not whole cycle if any day has label
    df["ovulation_mask"] = df["ovulation_prob_fused"].notna()
    return df


def build_sequences(df, max_cycle_len=MAX_CYCLE_LEN):
    """D5: Group by (id, small_group_key), sort by day_in_study ascending; return list of sequences.
    Cycles longer than max_cycle_len days (oligomenorrhea outliers) are excluded from both
    training and evaluation to prevent skewed MAE.
    """
    out = []
    skipped = 0
    for (uid, key), g in df.groupby(["id", "small_group_key"], sort=True):
        g = g.sort_values("day_in_study")
        if len(g) > max_cycle_len:
            skipped += 1
            continue
        X = g[FEATURE_COLS].values
        y_m = g["days_until_next_menses"].values
        y_ov = g["ovulation_label"].values
        mask_m = g["mask_menses"].values
        mask_ov = g["ovulation_mask"].values
        out.append(
            {
                "id": uid,
                "X": X,
                "y_menses": y_m,
                "y_ovulation": y_ov,
                "mask_menses": mask_m,
                "mask_ovulation": mask_ov,
                "length": len(X),
            }
        )
    if skipped:
        print(f"[dataset] Filtered {skipped} cycle(s) with length > {max_cycle_len} days.")
    return out


def prepare_all_sequences(cycle_path=CYCLE_CSV, full_path=SLEEP_CSV):
    """Run D1–D5 and return list of sequences.
    P2: Default input is SLEEP_CSV (overnight window) instead of FULL_CSV.
    """
    df = load_and_merge(cycle_path, full_path)
    df = add_next_menses(df)
    df = add_ovulation_labels(df)
    return build_sequences(df)


class CycleSequenceDataset(Dataset):
    """Cycle sequence dataset; supports filtering by subject (train/val/test)."""

    def __init__(self, sequences, subject_set):
        self.samples = [s for s in sequences if s["id"] in subject_set]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "X": torch.from_numpy(s["X"]),
            "y_menses": torch.from_numpy(s["y_menses"]),
            "y_ovulation": torch.from_numpy(s["y_ovulation"]),
            "mask_menses": torch.from_numpy(s["mask_menses"]),
            "mask_ovulation": torch.from_numpy(s["mask_ovulation"]),
            "length": s["length"],
        }


def collate_cycle_sequences(batch):
    """Variable-length sequences: pad to same length; return lengths for pack_padded."""
    max_len = max(b["length"] for b in batch)
    pad = np.zeros((len(batch), max_len, INPUT_DIM), dtype=np.float32)
    y_m = np.zeros((len(batch), max_len), dtype=np.float32)
    y_ov = np.zeros((len(batch), max_len), dtype=np.float32)
    mask_m = np.zeros((len(batch), max_len), dtype=bool)
    mask_ov = np.zeros((len(batch), max_len), dtype=bool)
    lengths = []
    for i, b in enumerate(batch):
        T = b["length"]
        pad[i, :T] = b["X"].numpy()
        y_m[i, :T] = b["y_menses"].numpy()
        y_ov[i, :T] = b["y_ovulation"].numpy()
        mask_m[i, :T] = b["mask_menses"].numpy()
        mask_ov[i, :T] = b["mask_ovulation"].numpy()
        lengths.append(T)
    return {
        "X": torch.from_numpy(pad),
        "y_menses": torch.from_numpy(y_m),
        "y_ovulation": torch.from_numpy(y_ov),
        "mask_menses": torch.from_numpy(mask_m),
        "mask_ovulation": torch.from_numpy(mask_ov),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }

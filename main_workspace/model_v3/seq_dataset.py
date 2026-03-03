"""Sequence dataset for GRU model: cycle-level sequences with variable length."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    FEATURES_CSV, CYCLE_CSV, MAX_CYCLE_LEN,
    FEAT_WEARABLE_Z, FEAT_RESPIRATORY_Z, FEAT_CYCLE_PRIOR,
)

SEQ_FEATURES = FEAT_WEARABLE_Z + FEAT_RESPIRATORY_Z   # 16-dim temporal
STATIC_FEATURES = FEAT_CYCLE_PRIOR                     # 6-dim per-day static


class CycleSequenceDataset(Dataset):
    """Each sample is one cycle: variable-length sequence of daily features."""

    def __init__(self, cycles, augment=False):
        self.cycles = cycles
        self.augment = augment

    def __len__(self):
        return len(self.cycles)

    def __getitem__(self, idx):
        c = self.cycles[idx]
        seq = c["seq"].copy()
        static = c["static"].copy()
        label = c["label"].copy()

        if self.augment and len(seq) > 3:
            start = np.random.randint(0, max(1, len(seq) // 3))
            seq, static, label = seq[start:], static[start:], label[start:]

        seq = np.nan_to_num(seq, nan=0.0)
        static = np.nan_to_num(static, nan=0.0)

        return (
            torch.FloatTensor(seq),
            torch.FloatTensor(static),
            torch.FloatTensor(label),
        )


def collate_cycles(batch):
    """Pad variable-length cycles to max length in batch."""
    seqs, statics, labels = zip(*batch)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    B = len(batch)
    D_seq = seqs[0].shape[1]
    D_static = statics[0].shape[1]

    seq_pad = torch.zeros(B, max_len, D_seq)
    static_pad = torch.zeros(B, max_len, D_static)
    label_pad = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        L = lengths[i]
        seq_pad[i, :L] = seqs[i]
        static_pad[i, :L] = statics[i]
        label_pad[i, :L] = labels[i]
        mask[i, :L] = True

    return seq_pad, static_pad, label_pad, mask, torch.LongTensor(lengths)


def build_cycle_data(features_csv=FEATURES_CSV, cycle_csv=CYCLE_CSV):
    """Load data and organize into cycle-level sequences."""
    feat = pd.read_csv(features_csv)
    cycle = pd.read_csv(cycle_csv)

    key = ["id", "study_interval", "day_in_study", "small_group_key"]
    cycle_group = ["id", "study_interval", "small_group_key"]

    df = feat.merge(cycle[key], on=key, how="inner")

    cycle_end = (
        df.groupby(cycle_group)["day_in_study"]
        .max().reset_index().rename(columns={"day_in_study": "cycle_end"})
    )
    df = df.merge(cycle_end, on=cycle_group, how="left")
    df["days_until_next_menses"] = (df["cycle_end"] + 1 - df["day_in_study"]).astype(float)
    df.drop(columns=["cycle_end"], inplace=True)

    cycle_len = df.groupby(cycle_group)["day_in_study"].transform("count")
    before = len(df)
    df = df[cycle_len <= MAX_CYCLE_LEN].copy()
    print(f"[seq-data] Filtered {before - len(df)} rows from cycles > {MAX_CYCLE_LEN} days")

    available_seq = [f for f in SEQ_FEATURES if f in df.columns]
    available_static = [f for f in STATIC_FEATURES if f in df.columns]
    print(f"[seq-data] Seq features: {len(available_seq)}/{len(SEQ_FEATURES)}, "
          f"Static features: {len(available_static)}/{len(STATIC_FEATURES)}")

    cycles = []
    for keys, grp in df.groupby(cycle_group):
        grp = grp.sort_values("day_in_study")
        cycles.append({
            "seq": grp[available_seq].values.astype(np.float32),
            "static": grp[available_static].values.astype(np.float32),
            "label": grp["days_until_next_menses"].values.astype(np.float32),
            "id": keys[0],
        })

    subjects = set(c["id"] for c in cycles)
    print(f"[seq-data] {len(cycles)} cycles, {len(subjects)} subjects")
    return cycles, available_seq, available_static

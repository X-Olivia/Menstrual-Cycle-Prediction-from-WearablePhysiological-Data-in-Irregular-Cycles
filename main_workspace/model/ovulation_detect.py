"""Ovulation detection & two-stage menstrual prediction.

Core utilities:
  - get_lh_ovulation_labels: LH-based ovulation ground truth
  - detect_ovulation_from_probs: causal detection from classifier probabilities
  - causal_temp_shift_detect: 3-over-6 coverline rule
  - compute_personal_luteal_from_lh: per-subject luteal lengths
  - two_stage_predict / two_stage_predict_leakage_free: hybrid prediction
"""
import re
import numpy as np
import pandas as pd

from .config import CYCLE_CSV


# ---------------------------------------------------------------------------
# LH-based ovulation labels
# ---------------------------------------------------------------------------

def get_lh_ovulation_labels():
    """Get LH-based ovulation labels with reasonable luteal lengths."""
    cc = pd.read_csv(CYCLE_CSV)
    ov = cc[cc["ovulation_prob_fused"] > 0.5]
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
    lh_ov = lh_ov[(lh_ov["luteal_len"] >= 8) & (lh_ov["luteal_len"] <= 20)]
    lh_ov["ov_dic"] = lh_ov["ov_day_in_study"] - lh_ov["cs"]
    return lh_ov


# ---------------------------------------------------------------------------
# Causal ovulation detection from classifier probabilities
# ---------------------------------------------------------------------------

def detect_ovulation_from_probs(cycle_df, probs, strategy="cumulative"):
    """Causal ovulation detection from daily P(post-ov) probabilities.

    Strategies:
    - 'threshold': first 2 consecutive days with prob > 0.5
    - 'cumulative': cumulative evidence score, trigger when > threshold
    - 'bayesian': combine cycle-position prior with signal posterior
    """
    n = len(cycle_df)
    if n < 5:
        return None

    days = cycle_df["day_in_cycle"].values
    hist_len = cycle_df["hist_cycle_len_mean"].values[0] if "hist_cycle_len_mean" in cycle_df.columns else 28

    if strategy == "threshold":
        consec = 0
        for i in range(n):
            if probs[i] > 0.5:
                consec += 1
                if consec >= 2:
                    return int(days[i] - 1)
            else:
                consec = 0
        return None

    elif strategy == "cumulative":
        score = 0.0
        decay = 0.85
        trigger = 1.5
        for i in range(n):
            evidence = probs[i] - 0.5
            score = score * decay + evidence
            if score > trigger and days[i] >= 8:
                return int(days[i] - 2)
        return None

    elif strategy == "bayesian":
        expected_ov = max(10, hist_len - 14)
        for i in range(n):
            d = days[i]
            prior = 1.0 / (1.0 + np.exp(-0.5 * (d - expected_ov)))
            posterior = prior * probs[i]
            if posterior > 0.4 and d >= 8:
                return int(d)
        return None

    return None

# ---------------------------------------------------------------------------
# Stage 1 — Causal ovulation detection from raw nightly temperature
# ---------------------------------------------------------------------------

TEMP_CSV = None  # set lazily

_COMP_TEMP_REL = "subdataset/computed_temperature_cycle.csv"


def _load_nightly_temp(workspace: str) -> pd.DataFrame:
    """Load raw nightly temperature per (id, study_interval, day_in_study)."""
    import os
    path = os.path.join(workspace, _COMP_TEMP_REL)
    ct = pd.read_csv(path)
    key = ["id", "study_interval", "day_in_study"]
    ct = ct[key + ["nightly_temperature"]].copy()
    ct = ct.sort_values(key)
    return ct


def causal_temp_shift_detect(
    temps: np.ndarray,
    baseline_window: int = 6,
    confirm_window: int = 3,
    min_shift: float = 0.15,
    min_baseline_valid: int = 4,
    min_confirm_valid: int = 3,
):
    """Causal 3-over-6 coverline rule on a single cycle's nightly temperatures.

    For each day d (starting from baseline_window + confirm_window),
    compare the mean of the most recent `confirm_window` days to the mean
    of the preceding `baseline_window` days.  If the difference >= min_shift,
    ovulation is detected at day (d - confirm_window + 1).

    Returns
    -------
    detect_day : int or None
        Index in `temps` where shift is first detected (the confirmation day).
    ov_day_est : int or None
        Estimated ovulation day = detect_day - confirm_window (shift onset).
    """
    n = len(temps)
    start = baseline_window + confirm_window
    for d in range(start, n):
        recent = temps[d - confirm_window: d]
        baseline = temps[d - confirm_window - baseline_window: d - confirm_window]

        valid_recent = recent[~np.isnan(recent)]
        valid_baseline = baseline[~np.isnan(baseline)]

        if len(valid_recent) < min_confirm_valid or len(valid_baseline) < min_baseline_valid:
            continue

        if np.mean(valid_recent) - np.mean(valid_baseline) >= min_shift:
            ov_day_est = d - confirm_window
            return d, ov_day_est

    return None, None


def detect_ovulation_all_cycles(df_daily: pd.DataFrame, temp_df: pd.DataFrame,
                                min_shift: float = 0.15):
    """Run causal ovulation detection for every cycle in df_daily.

    Parameters
    ----------
    df_daily : DataFrame with columns [id, study_interval, day_in_study, small_group_key, day_in_cycle]
    temp_df  : DataFrame from _load_nightly_temp (raw temps)

    Returns
    -------
    ov_info : dict  {small_group_key -> {'detect_day_in_cycle': int, 'ov_day_in_cycle': int}}
    """
    key = ["id", "study_interval", "day_in_study"]
    merged = df_daily[key + ["small_group_key", "day_in_cycle"]].merge(
        temp_df, on=key, how="left"
    )

    cycle_group = ["id", "study_interval", "small_group_key"]
    ov_info = {}

    for sgk, grp in merged.sort_values(key).groupby("small_group_key"):
        temps = grp["nightly_temperature"].values
        day_in_cycle = grp["day_in_cycle"].values

        detect_idx, ov_idx = causal_temp_shift_detect(temps, min_shift=min_shift)

        if detect_idx is not None and ov_idx is not None:
            ov_info[sgk] = {
                "detect_day_in_cycle": int(day_in_cycle[detect_idx]),
                "ov_day_in_cycle": int(day_in_cycle[ov_idx]),
            }
    return ov_info


# ---------------------------------------------------------------------------
# Stage 2 — Personal luteal length from PRIOR completed cycles
# ---------------------------------------------------------------------------

def compute_personal_luteal_from_lh(cycle_csv: str = None):
    """Compute per-subject average luteal length from LH-based ovulation labels.

    Uses cycle_clean_2.csv: finds ovulation day (max ovulation_prob_fused > 0.5),
    computes luteal_len = cycle_end - ov_day + 1 for each cycle.

    Returns dict {subject_id -> list of luteal lengths (one per cycle)}.
    """
    cycle_csv = cycle_csv or CYCLE_CSV
    cc = pd.read_csv(cycle_csv)

    if "ovulation_prob_fused" not in cc.columns:
        return {}

    ov_candidates = cc[cc["ovulation_prob_fused"] > 0.5].copy()
    ov_day = (
        ov_candidates.groupby("small_group_key")
        .apply(lambda g: g.loc[g["ovulation_prob_fused"].idxmax(), "day_in_study"],
               include_groups=False)
        .reset_index()
        .rename(columns={0: "ov_day"})
    )

    cycle_end = (
        cc.groupby("small_group_key")["day_in_study"]
        .max().reset_index()
        .rename(columns={"day_in_study": "cycle_end"})
    )

    merged = ov_day.merge(cycle_end, on="small_group_key")
    merged["luteal_len"] = merged["cycle_end"] - merged["ov_day"] + 1

    reasonable = merged[(merged["luteal_len"] >= 8) & (merged["luteal_len"] <= 20)]

    subj_map = cc[["small_group_key", "id"]].drop_duplicates()
    reasonable = reasonable.merge(subj_map, on="small_group_key")

    result = {}
    for uid, grp in reasonable.groupby("id"):
        result[uid] = grp["luteal_len"].tolist()

    return result


# ---------------------------------------------------------------------------
# Two-stage predictor
# ---------------------------------------------------------------------------

def two_stage_predict(
    df: pd.DataFrame,
    ov_info: dict,
    personal_luteal: dict,
    population_luteal_mean: float = 14.0,
):
    """Generate predictions for every row in df using the two-stage approach.

    For rows where ovulation has been detected (causally), use:
        pred = personal_luteal_mean - days_since_ov_detection
    For rows before detection, fall back to:
        pred = hist_cycle_len_mean - day_in_cycle  (= days_remaining_prior)

    Parameters
    ----------
    df : DataFrame with columns [small_group_key, day_in_cycle, days_remaining_prior, id]
    ov_info : {sgk -> {detect_day_in_cycle, ov_day_in_cycle}}
    personal_luteal : {subject_id -> [luteal_len, ...]}
    population_luteal_mean : fallback when no personal data

    Returns
    -------
    pred : np.ndarray of shape (len(df),)
    stage : np.ndarray of 'calendar' or 'ovulation' strings
    """
    pred = np.full(len(df), np.nan)
    stage = np.full(len(df), "calendar", dtype=object)

    for i, row in enumerate(df.itertuples()):
        sgk = row.small_group_key
        dic = row.day_in_cycle
        uid = row.id

        if sgk in ov_info:
            detect_day = ov_info[sgk]["detect_day_in_cycle"]
            ov_day = ov_info[sgk]["ov_day_in_cycle"]

            if dic >= detect_day:
                days_since_ov = dic - ov_day
                luteal_lens = personal_luteal.get(uid, [])
                if luteal_lens:
                    avg_lut = np.mean(luteal_lens)
                else:
                    avg_lut = population_luteal_mean
                pred[i] = max(1.0, avg_lut - days_since_ov)
                stage[i] = "ovulation"
                continue

        pred[i] = max(1.0, row.days_remaining_prior)

    return pred, stage


def two_stage_predict_leakage_free(
    df: pd.DataFrame,
    ov_info: dict,
    personal_luteal_all: dict,
    test_cycle_keys: set,
    population_luteal_mean: float = 14.0,
):
    """Same as two_stage_predict but excludes test cycle from personal luteal computation.

    For each subject, personal_luteal_mean is computed using only cycles NOT in test_cycle_keys.
    """
    pred = np.full(len(df), np.nan)
    stage = np.full(len(df), "calendar", dtype=object)

    sgk_arr = df["small_group_key"].values
    dic_arr = df["day_in_cycle"].values
    uid_arr = df["id"].values
    drp_arr = df["days_remaining_prior"].values

    sgk_to_uid = dict(zip(df["small_group_key"], df["id"]))
    subj_cycle_keys = {}
    for sgk, uid in sgk_to_uid.items():
        subj_cycle_keys.setdefault(uid, set()).add(sgk)

    for i in range(len(df)):
        sgk = sgk_arr[i]
        dic = dic_arr[i]
        uid = uid_arr[i]

        if sgk in ov_info:
            detect_day = ov_info[sgk]["detect_day_in_cycle"]
            ov_day = ov_info[sgk]["ov_day_in_cycle"]

            if dic >= detect_day:
                days_since_ov = dic - ov_day
                all_lut = personal_luteal_all.get(uid, [])
                avg_lut = np.mean(all_lut) if all_lut else population_luteal_mean
                pred[i] = max(1.0, avg_lut - days_since_ov)
                stage[i] = "ovulation"
                continue

        pred[i] = max(1.0, drp_arr[i])

    return pred, stage

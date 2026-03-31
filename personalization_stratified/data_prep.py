from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "multisignal_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from data import load_all_signals

from cld_metrics import is_high_vol_cld_strict, mean_abs_cld, median_abs_cld


@dataclass
class SubjectMeta:
    uid: str
    cycle_lengths: List[int]
    cycle_mean: float
    cycle_cv: float
    cycle_std: float
    median_abs_cld: float
    mean_abs_cld: float
    mean_shift_abs: float
    is_irregular: bool
    is_irregular_cld_strict: bool
    test_sgk: str | None
    n_history_before_test: int


def _safe_cv(vals: List[int]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr))
    if mean_v <= 1e-8:
        return std_v, float("nan")
    return std_v, std_v / mean_v


def build_subject_meta(
    subj_order: Dict[str, List[str]],
    cycle_series: Dict[str, dict],
    lh_dict: Dict[str, int],
    irregular_cv_threshold: float = 0.15,
) -> Dict[str, SubjectMeta]:
    out: Dict[str, SubjectMeta] = {}
    for uid, sgks in subj_order.items():
        valid = [s for s in sgks if s in cycle_series]
        clens = [int(cycle_series[s]["cycle_len"]) for s in valid]
        cstd, ccv = _safe_cv(clens)
        med_cld = median_abs_cld(clens)
        mean_cld = mean_abs_cld(clens)
        test_sgk = None
        n_hist = 0
        for s in reversed(valid):
            if s in lh_dict:
                test_sgk = s
                n_hist = valid.index(s)
                break
        out[uid] = SubjectMeta(
            uid=str(uid),
            cycle_lengths=clens,
            cycle_mean=float(np.mean(clens)) if clens else float("nan"),
            cycle_cv=ccv,
            cycle_std=cstd,
            median_abs_cld=med_cld,
            mean_abs_cld=mean_cld,
            mean_shift_abs=abs(float(np.mean(clens)) - 28.0) if clens else float("nan"),
            is_irregular=bool(np.isfinite(ccv) and ccv > irregular_cv_threshold),
            is_irregular_cld_strict=is_high_vol_cld_strict(med_cld, threshold_days=9.0),
            test_sgk=test_sgk,
            n_history_before_test=n_hist,
        )
    return out


def make_subject_meta_df(meta: Dict[str, SubjectMeta]) -> pd.DataFrame:
    rows = []
    for uid, m in meta.items():
        rows.append(
            {
                "id": uid,
                "n_cycles": len(m.cycle_lengths),
                "cycle_mean": m.cycle_mean,
                "cycle_std": m.cycle_std,
                "cycle_cv": m.cycle_cv,
                "median_abs_cld": m.median_abs_cld,
                "mean_abs_cld": m.mean_abs_cld,
                "mean_shift_abs": m.mean_shift_abs,
                "is_irregular": int(m.is_irregular),
                "is_irregular_cld_strict": int(m.is_irregular_cld_strict),
                "test_sgk": m.test_sgk,
                "n_history_before_test": m.n_history_before_test,
            }
        )
    return pd.DataFrame(rows)


def load_core_data():
    lh, cs, _, so, _ = load_all_signals()
    return lh, cs, so

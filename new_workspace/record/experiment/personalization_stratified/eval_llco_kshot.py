from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data_prep import build_subject_meta, load_core_data, make_subject_meta_df
from models_cyclelevel import (
    ANCHORS_ALL,
    estimate_b1_history_bias,
    global_cycle_mean,
    predict_menses_start_b1,
    predict_menses_start_b2,
)


PRE_ANCHORS = {-7, -3, -1}
POST_ANCHORS = {2, 5, 10}


def build_detector_dict(lh_dict: Dict[str, int], mode: str) -> Dict[str, int]:
    if mode == "oracle":
        return dict(lh_dict)
    raise ValueError(f"Unsupported detector mode: {mode}")


def estimate_pop_luteal_len(
    lh_dict: Dict[str, int],
    cycle_series: Dict[str, dict],
) -> float:
    vals = []
    for sgk, ov in lh_dict.items():
        if sgk not in cycle_series:
            continue
        clen = float(cycle_series[sgk]["cycle_len"])
        lut = clen - float(ov)
        if 8.0 <= lut <= 22.0:
            vals.append(lut)
    if not vals:
        return 13.0
    return float(np.mean(np.asarray(vals)))


def evaluate_llco_kshot(
    lh_dict: Dict[str, int],
    cycle_series: Dict[str, dict],
    subj_order: Dict[str, List[str]],
    cv_threshold: float,
    shots: List[int],
    detector_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    det_dict = build_detector_dict(lh_dict, mode=detector_mode)
    pop_cycle_len = global_cycle_mean(cycle_series)
    pop_luteal_len = estimate_pop_luteal_len(lh_dict, cycle_series)
    meta = build_subject_meta(subj_order, cycle_series, lh_dict, irregular_cv_threshold=cv_threshold)
    meta_df = make_subject_meta_df(meta)
    eval_meta = meta_df[meta_df["test_sgk"].notna()].copy()
    shift_thr = float(eval_meta["mean_shift_abs"].median()) if not eval_meta.empty else 0.0
    vol_thr = float(eval_meta["cycle_cv"].median()) if not eval_meta.empty else 0.0
    cld_thr = float(eval_meta["median_abs_cld"].median()) if not eval_meta.empty else 0.0

    def _stratum_cv(mean_shift_abs: float, cycle_cv: float) -> str:
        shift_high = bool(np.isfinite(mean_shift_abs) and mean_shift_abs >= shift_thr)
        vol_high = bool(np.isfinite(cycle_cv) and cycle_cv >= vol_thr)
        if (not shift_high) and (not vol_high):
            return "low_shift_low_vol"
        if shift_high and (not vol_high):
            return "high_shift_low_vol"
        if (not shift_high) and vol_high:
            return "low_shift_high_vol"
        return "high_shift_high_vol"

    def _stratum_cld(mean_shift_abs: float, median_cld: float) -> str:
        shift_high = bool(np.isfinite(mean_shift_abs) and mean_shift_abs >= shift_thr)
        vol_high = bool(np.isfinite(median_cld) and median_cld >= cld_thr)
        if (not shift_high) and (not vol_high):
            return "low_shift_low_cld"
        if shift_high and (not vol_high):
            return "high_shift_low_cld"
        if (not shift_high) and vol_high:
            return "low_shift_high_cld"
        return "high_shift_high_cld"

    rows = []
    for uid, m in meta.items():
        if m.test_sgk is None:
            continue
        sgks = [s for s in subj_order[uid] if s in cycle_series]
        test_idx = sgks.index(m.test_sgk)
        history_all = sgks[:test_idx]
        test_sgk = m.test_sgk
        actual = float(cycle_series[test_sgk]["cycle_len"])
        ov_true = lh_dict.get(test_sgk)
        ov_est = det_dict.get(test_sgk)
        if ov_true is None:
            continue
        for shot in shots:
            hist_k = history_all[-shot:] if shot > 0 else []
            hist_clen_k = [int(cycle_series[s]["cycle_len"]) for s in hist_k if s in cycle_series]
            bias_k = estimate_b1_history_bias(
                history_sgks=hist_k,
                cycle_series=cycle_series,
                lh_dict=lh_dict,
                det_dict=det_dict,
                pop_cycle_len=pop_cycle_len,
                pop_luteal_len=pop_luteal_len,
            )
            for anchor_offset in ANCHORS_ALL:
                anchor_day = ov_true + anchor_offset
                if not (0 <= anchor_day < actual):
                    continue
                true_remaining = actual - anchor_day

                pred_b1_start = predict_menses_start_b1(
                    ov_est=ov_est,
                    anchor_day=anchor_day,
                    pop_cycle_len=pop_cycle_len,
                    pop_luteal_len=pop_luteal_len,
                )
                pred_b2_start = predict_menses_start_b2(
                    ov_est=ov_est,
                    anchor_day=anchor_day,
                    hist_cycle_lengths=hist_clen_k,
                    pop_cycle_len=pop_cycle_len,
                    pop_luteal_len=pop_luteal_len,
                )
                pred_b1_remaining = pred_b1_start - anchor_day
                pred_b2_remaining = pred_b2_start - anchor_day
                pred_m3_remaining = pred_b1_remaining - bias_k

                for model_name, pred_remaining in [
                    ("B1_population_only", pred_b1_remaining),
                    ("B2_personalized_cycle", pred_b2_remaining),
                    ("M3_global_plus_bias", pred_m3_remaining),
                ]:
                    err = float(pred_remaining - true_remaining)
                    rows.append(
                        {
                            "id": uid,
                            "test_sgk": test_sgk,
                            "is_irregular": int(m.is_irregular),
                            "is_irregular_cld_strict": int(m.is_irregular_cld_strict),
                            "cycle_mean": m.cycle_mean,
                            "cycle_cv": m.cycle_cv,
                            "median_abs_cld": m.median_abs_cld,
                            "mean_abs_cld": m.mean_abs_cld,
                            "mean_shift_abs": m.mean_shift_abs,
                            "shift_threshold": shift_thr,
                            "vol_threshold": vol_thr,
                            "cld_threshold": cld_thr,
                            "irregular_2d_stratum": _stratum_cv(m.mean_shift_abs, m.cycle_cv),
                            "irregular_2d_stratum_cld": _stratum_cld(m.mean_shift_abs, m.median_abs_cld),
                            "shot_k": int(shot),
                            "model": model_name,
                            "anchor_offset": int(anchor_offset),
                            "anchor_group": (
                                "pre"
                                if int(anchor_offset) in PRE_ANCHORS
                                else ("post" if int(anchor_offset) in POST_ANCHORS else "other")
                            ),
                            "signed_err": err,
                            "abs_err": abs(err),
                            "acc_3d": float(abs(err) < 3.5),
                            "pipeline": "oracle_ovulation",
                        }
                    )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, meta_df
    agg_df = (
        long_df.groupby(
            [
                "id",
                "test_sgk",
                "is_irregular",
                "cycle_mean",
                "cycle_cv",
                "median_abs_cld",
                "mean_shift_abs",
                "irregular_2d_stratum",
                "irregular_2d_stratum_cld",
                "shot_k",
                "model",
            ],
            as_index=False,
        )
        .agg(mae=("abs_err", "mean"), acc_3d=("acc_3d", "mean"), n_anchors=("abs_err", "size"))
    )
    return long_df, agg_df


def main():
    parser = argparse.ArgumentParser(description="LLCO + k-shot personalization evaluation")
    parser.add_argument("--cv-threshold", type=float, default=0.15)
    parser.add_argument("--shots", type=str, default="0,1,2,3")
    parser.add_argument("--detector", type=str, default="oracle", choices=["oracle"])
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    args = parser.parse_args()

    shots = [int(x.strip()) for x in args.shots.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lh_dict, cycle_series, subj_order = load_core_data()
    long_df, agg_or_meta = evaluate_llco_kshot(
        lh_dict=lh_dict,
        cycle_series=cycle_series,
        subj_order=subj_order,
        cv_threshold=args.cv_threshold,
        shots=shots,
        detector_mode=args.detector,
    )
    if long_df.empty:
        print("No evaluable rows generated.")
        return
    agg_df = agg_or_meta
    long_path = out_dir / "llco_kshot_long.csv"
    agg_path = out_dir / "llco_kshot_agg.csv"
    long_df.to_csv(long_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"[DONE] long rows: {len(long_df)} -> {long_path}")
    print(f"[DONE] agg rows:  {len(agg_df)} -> {agg_path}")
    print(
        long_df.groupby(["shot_k", "model"])["abs_err"]
        .mean()
        .reset_index()
        .sort_values(["shot_k", "model"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

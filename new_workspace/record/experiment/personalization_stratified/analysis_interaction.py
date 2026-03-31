from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mean(x: pd.Series) -> float:
    arr = x.dropna().values
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def build_gain_table_from_long(long_df: pd.DataFrame) -> pd.DataFrame:
    use_df = long_df[long_df["anchor_group"].isin(["pre", "post"])].copy()
    agg = (
        use_df.groupby(
            [
                "id",
                "test_sgk",
                "irregular_2d_stratum",
                "cycle_cv",
                "mean_shift_abs",
                "shot_k",
                "anchor_group",
                "model",
            ],
            as_index=False,
        )
        .agg(mae=("abs_err", "mean"), acc_3d=("acc_3d", "mean"), n=("abs_err", "size"))
    )
    piv = agg.pivot_table(
        index=[
            "id",
            "test_sgk",
            "irregular_2d_stratum",
            "cycle_cv",
            "mean_shift_abs",
            "shot_k",
            "anchor_group",
        ],
        columns="model",
        values="mae",
        aggfunc="mean",
    ).reset_index()
    piv["gain_b2_vs_b1"] = piv["B1_population_only"] - piv["B2_personalized_cycle"]
    piv["gain_m3_vs_b1"] = piv["B1_population_only"] - piv["M3_global_plus_bias"]
    return piv


def summarize_gain_by_stratum_phase(gain_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shot in sorted(gain_df["shot_k"].unique()):
        for phase in ["pre", "post"]:
            d = gain_df[(gain_df["shot_k"] == shot) & (gain_df["anchor_group"] == phase)]
            for s in [
                "low_shift_low_vol",
                "high_shift_low_vol",
                "low_shift_high_vol",
                "high_shift_high_vol",
            ]:
                ds = d[d["irregular_2d_stratum"] == s]
                rows.append(
                    {
                        "shot_k": int(shot),
                        "phase": phase,
                        "irregular_2d_stratum": s,
                        "gain_b2_vs_b1": _safe_mean(ds["gain_b2_vs_b1"]),
                        "gain_m3_vs_b1": _safe_mean(ds["gain_m3_vs_b1"]),
                        "n_subjects": int(ds["id"].nunique()),
                        "n_rows": int(len(ds)),
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="2D irregular stratified gain analysis (pre/post)")
    parser.add_argument(
        "--long-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "llco_kshot_long.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    args = parser.parse_args()

    long_df = pd.read_csv(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gain_df = build_gain_table_from_long(long_df)
    gain_2d_prepost_df = summarize_gain_by_stratum_phase(gain_df)

    gain_path = out_dir / "interaction_gain_by_subject_prepost.csv"
    gain_2d_path = out_dir / "gain_2d_strata_prepost_summary.csv"
    gain_df.to_csv(gain_path, index=False)
    gain_2d_prepost_df.to_csv(gain_2d_path, index=False)

    print(f"[DONE] gain: {gain_path}")
    print(f"[DONE] 2d pre/post summary: {gain_2d_path}")
    print("\n[2D pre/post quick view]")
    print(gain_2d_prepost_df.to_string(index=False))


if __name__ == "__main__":
    main()

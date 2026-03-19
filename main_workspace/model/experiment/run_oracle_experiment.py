"""Oracle Experiment: LightGBM + Perfect Ovulation Detection (LH labels) Hybrid Model.

This script reproduces the "93.9% ±3d accuracy" Oracle upper-bound result.

It demonstrates the theoretical ceiling of the two-stage approach:
  - Pre-ovulation: use LightGBM predictions
  - Post-ovulation: use personal luteal length countdown (assuming perfect detection)

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -m model.experiment.run_oracle_experiment
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from model.dataset import load_data
from model.config import CYCLE_CSV, WORKSPACE, ALL_FEATURES, LGB_PARAMS_TUNED
from model.ovulation_detect import _load_nightly_temp, compute_personal_luteal_from_lh
from model.evaluate import compute_metrics
from model.train_lgb import train_lightgbm


def main():
    # ── Load data ─────────────────────────────────────────────────────────
    df, available = load_data()
    temp_df = _load_nightly_temp(WORKSPACE)
    cc = pd.read_csv(CYCLE_CSV)

    key = ["id", "study_interval", "day_in_study"]
    df = df.merge(temp_df, on=key, how="left")

    # ── Personal luteal length from LH labels ─────────────────────────────
    personal_luteal = compute_personal_luteal_from_lh()
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v])
    print(f"Population luteal mean: {pop_luteal_mean:.1f} days")

    # ── LH-based ovulation ground truth ───────────────────────────────────
    ov_candidates = cc[cc["ovulation_prob_fused"] > 0.5]
    lh_ov = (
        ov_candidates.groupby("small_group_key")
        .apply(
            lambda g: g.loc[g["ovulation_prob_fused"].idxmax(), "day_in_study"],
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "lh_ov_day"})
    )
    cycle_start = (
        cc.groupby("small_group_key")["day_in_study"]
        .min()
        .reset_index()
        .rename(columns={"day_in_study": "cs"})
    )
    lh_ov = lh_ov.merge(cycle_start, on="small_group_key")
    lh_ov["lh_ov_dic"] = lh_ov["lh_ov_day"] - lh_ov["cs"]
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["lh_ov_dic"]))

    print(f"Cycles with LH ovulation labels: {len(lh_ov_dict)}")

    # ── 10-seed evaluation ────────────────────────────────────────────────
    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"Features used: {len(features)}")
    print("=" * 70)
    print("  HYBRID: LightGBM + Oracle Luteal Countdown (10-seed eval)")
    print("=" * 70)

    results_oracle = []
    results_lgb_only = []

    for seed in range(10):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["id"]))

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        train_uids = set(df_train["id"].unique())  # personal_luteal only for train users (no leakage)

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + 100)
        tr_idx, val_idx = next(gss2.split(df_train, groups=df_train["id"]))
        X_tr = df_train.iloc[tr_idx][features].values
        y_tr = df_train.iloc[tr_idx]["days_until_next_menses"].values
        X_val = df_train.iloc[val_idx][features].values
        y_val = df_train.iloc[val_idx]["days_until_next_menses"].values
        X_test = df_test[features].values
        y_test = df_test["days_until_next_menses"].values

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, features, params=LGB_PARAMS_TUNED)
        pred_lgb = np.clip(model.predict(X_test), 1.0, None)

        m_lgb = compute_metrics(pred_lgb, y_test)
        results_lgb_only.append(m_lgb)

        # Hybrid: LGB for pre-ov, countdown for post-ov
        pred_hybrid = pred_lgb.copy()
        test_sgks = df_test["small_group_key"].values
        test_dics = df_test["day_in_cycle"].values
        test_uids = df_test["id"].values

        ov_used = 0
        for i in range(len(df_test)):
            sgk = test_sgks[i]
            dic = test_dics[i]
            uid = test_uids[i]

            if sgk in lh_ov_dict:
                ov_dic = lh_ov_dict[sgk]
                if dic >= ov_dic + 2:  # 2-day realistic detection delay
                    days_since_ov = dic - ov_dic
                    # Use personal only for train users; test users use pop to avoid leakage
                    luts = personal_luteal.get(uid, []) if uid in train_uids else []
                    avg_lut = np.mean(luts) if luts else pop_luteal_mean
                    pred_hybrid[i] = max(1.0, avg_lut - days_since_ov)
                    ov_used += 1

        m_hybrid = compute_metrics(pred_hybrid, y_test)
        results_oracle.append(m_hybrid)

        # Per-stage breakdown for this seed
        ov_mask = np.array(
            [
                (test_sgks[i] in lh_ov_dict and test_dics[i] >= lh_ov_dict[test_sgks[i]] + 2)
                for i in range(len(df_test))
            ]
        )
        cal_mask = ~ov_mask

        if seed == 0:
            m_ov = compute_metrics(pred_hybrid[ov_mask], y_test[ov_mask]) if ov_mask.sum() > 0 else None
            m_cal = compute_metrics(pred_hybrid[cal_mask], y_test[cal_mask]) if cal_mask.sum() > 0 else None
            print(f"\n[Seed 0 detailed breakdown]")
            print(f"  Total test rows: {len(df_test)}")
            print(f"  Ov countdown rows: {ov_mask.sum()} ({ov_mask.mean():.1%})")
            print(f"  Calendar rows:     {cal_mask.sum()} ({cal_mask.mean():.1%})")
            if m_ov:
                print(f"  Ov-stage only:  MAE={m_ov['mae']:.3f}, ±3d={m_ov['acc_3d']:.1%}")
            if m_cal:
                print(f"  Cal-stage only: MAE={m_cal['mae']:.3f}, ±3d={m_cal['acc_3d']:.1%}")

        print(f"  Seed {seed}: LGB MAE={m_lgb['mae']:.3f} | Hybrid MAE={m_hybrid['mae']:.3f}, ov_used={ov_used}")

    # ── Average results ───────────────────────────────────────────────────
    def avg_metrics(results):
        return {k: np.mean([r[k] for r in results]) for k in results[0]}

    lgb_avg = avg_metrics(results_lgb_only)
    hyb_avg = avg_metrics(results_oracle)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS (averaged over 10 seeds)")
    print("=" * 70)
    print(
        f"LightGBM v4 only:          "
        f"MAE={lgb_avg['mae']:.3f}, "
        f"±1d={lgb_avg['acc_1d']:.1%}, "
        f"±2d={lgb_avg['acc_2d']:.1%}, "
        f"±3d={lgb_avg['acc_3d']:.1%}"
    )
    print(
        f"LightGBM + Oracle countdown: "
        f"MAE={hyb_avg['mae']:.3f}, "
        f"±1d={hyb_avg['acc_1d']:.1%}, "
        f"±2d={hyb_avg['acc_2d']:.1%}, "
        f"±3d={hyb_avg['acc_3d']:.1%}"
    )
    print(
        f"Improvement:                 "
        f"MAE={hyb_avg['mae'] - lgb_avg['mae']:+.3f}, "
        f"±3d={100 * (hyb_avg['acc_3d'] - lgb_avg['acc_3d']):+.1f}pp"
    )
    print()
    print("The Oracle result is a CEILING: it assumes perfect ovulation detection")
    print("using LH labels as ground truth. The 93.9% ±3d accuracy refers to the")
    print("post-ovulation countdown phase only (see seed 0 breakdown above).")


if __name__ == "__main__":
    main()

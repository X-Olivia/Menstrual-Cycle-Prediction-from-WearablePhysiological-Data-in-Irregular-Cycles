"""Oracle + 黄体倒计时实验（经期预测）。

比较三种策略/模型：
1) LightGBM only：对全周期特征做回归，直接预测 `days_until_next_menses`。
2) Detected-ovulation hybrid：先从可穿戴信号预测排卵日（由排卵分类器输出的 `ov_prob`
   经过阈值/累积/贝叶斯等策略得到检测排卵日）。若当前天满足 `day_in_cycle >= ov_day + 2`，
   则对“排卵后阶段”使用黄体倒计时规则修正预测；否则保持 LightGBM 输出。
3) Oracle hybrid：排卵日使用 LH 的真值（Ground Truth）作为“Oracle”；其余逻辑同上。

Oracle 的核心思想：在排卵后，用“LH 真值排卵日 +（个人或人群平均）黄体期长度”推断到下一次月经还剩几天，
从而消除“排卵检测误差”对预测的影响。

默认数据读取 `new_workspace`：
  - `processed_dataset/cycle_cleaned_ov.csv`
  - `processed_dataset/signals/*.csv`（温度/心率/HRV/腕温等已按天聚合的数据）
  - `processed_dataset/daily_features/daily_features_v4.csv`（若不存在则回退到 `main_workspace`）

运行：
  python record/oracle_luteal_countdown_experiment.py

输出（仅打印到终端，不写入结果文件）包含：
  - 数据规模、可用于排卵分类的标注数量与比例
  - 黄体期估计的人群均值（用于无个人黄体期时的回退）
  - 排卵检测器的 LOSO AUC
  - 三种排卵检测策略（`threshold`/`cumulative`/`bayesian`）的排卵日检测评估
    （检测到的周期数、预测相对偏差均值、±3d/±5d 覆盖率）
  - 10 次（按受试者分组的）评估下的经期预测结果：LightGBM only / 检测-黄体倒计时 / Oracle-黄体倒计时
    的 MAE 与 ±3d 指标
  - 以 LH 排卵日为锚点的误差拆分（Pre：ov-7/ov-3/ov-1；Post：ov+2/ov+5/ov+10）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Paths: new_workspace data ─────────────────────────────────────────
NEW_WS = Path(__file__).resolve().parent.parent
MAIN_WS = NEW_WS.parent / "main_workspace"
PROCESSED = NEW_WS / "processed_dataset"
SIGNALS_DIR = PROCESSED / "signals"
CYCLE_OV_CSV = PROCESSED / "cycle_cleaned_ov.csv"
FEATURES_V4_NEW = NEW_WS / "processed_dataset" / "daily_features" / "daily_features_v4.csv"

sys.path.insert(0, str(MAIN_WS))
import model.config as config
config.CYCLE_CSV = str(CYCLE_OV_CSV)
config.WORKSPACE = str(NEW_WS)
if FEATURES_V4_NEW.exists():
    config.FEATURES_CSV = str(FEATURES_V4_NEW)
else:
    config.FEATURES_CSV = os.path.join(config.WORKSPACE, "processed_data", "v4", "daily_features_v4.csv")
    if not os.path.isfile(config.FEATURES_CSV):
        config.FEATURES_CSV = os.path.join(MAIN_WS, "processed_data", "v4", "daily_features_v4.csv")

from model.dataset import load_data
from model.config import CYCLE_CSV, WORKSPACE, ALL_FEATURES, LGB_PARAMS_TUNED
from model.ovulation_detect import (
    compute_personal_luteal_from_lh,
    get_lh_ovulation_labels,
    detect_ovulation_from_probs,
)
from model.evaluate import compute_metrics
from model.train_lgb import train_lightgbm


# ======================================================================
# Load wearable from new_workspace processed_dataset/signals
# ======================================================================

def load_wearable_daily(signals_dir: Path):
    """Load and aggregate wearable signals to day level from signals_dir (e.g. processed_dataset/signals)."""
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(signals_dir / "computed_temperature_cycle.csv")
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    rhr = pd.read_csv(signals_dir / "resting_heart_rate_cycle.csv")
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)

    hr = pd.read_csv(signals_dir / "heart_rate_cycle.csv")
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std", "hr_min"]

    hrv = pd.read_csv(signals_dir / "heart_rate_variability_details_cycle.csv")
    hrv_daily = hrv.groupby(key).agg(
        {"rmssd": "mean", "low_frequency": "mean", "high_frequency": "mean"}
    ).reset_index()
    hrv_daily.columns = key + ["rmssd_mean", "lf_mean", "hf_mean"]

    wt = pd.read_csv(signals_dir / "wrist_temperature_cycle.csv")
    wt_daily = wt.groupby(key)["temperature_diff_from_baseline"].agg(["mean", "max"]).reset_index()
    wt_daily.columns = key + ["wt_mean", "wt_max"]

    return ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily


# ======================================================================
# Causal features for ovulation classifier
# ======================================================================

RAW_SIGNALS = [
    "nightly_temperature", "hr_min", "hr_std", "lf_mean",
    "rmssd_mean", "wt_max", "hr_mean", "wt_mean", "hf_mean",
]


def build_ov_detection_features(base_df):
    """Build causal rolling features per cycle for ovulation detection."""
    base_df = base_df.sort_values(["small_group_key", "day_in_study"])
    chunks = []
    for sgk, grp in base_df.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study").copy()
        for sig in RAW_SIGNALS:
            if sig not in grp.columns:
                continue
            vals = grp[sig].values
            s = pd.Series(vals)
            rm3 = s.rolling(3, min_periods=2).mean().values
            rm7 = s.rolling(7, min_periods=4).mean().values
            grp[f"{sig}_rm3"] = rm3
            grp[f"{sig}_rm7"] = rm7
            grp[f"{sig}_svl"] = rm3 - rm7
            grp[f"{sig}_d1"] = s.diff().values
            grp[f"{sig}_d3"] = s.diff(3).values
        chunks.append(grp)
    return pd.concat(chunks, ignore_index=True)


def get_ov_feature_cols():
    cols = []
    for sig in RAW_SIGNALS:
        cols.extend([f"{sig}_rm3", f"{sig}_rm7", f"{sig}_svl", f"{sig}_d1", f"{sig}_d3"])
    cols.append("cycle_frac")
    return cols


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("  Oracle + Luteal Countdown Experiment (new_workspace data)")
    print("=" * 70)
    print(f"  Cycle CSV:    {CYCLE_OV_CSV}")
    print(f"  Signals dir:  {SIGNALS_DIR}")
    print(f"  Features CSV: {config.FEATURES_CSV}")

    if not CYCLE_OV_CSV.exists():
        raise SystemExit(f"Cycle CSV not found: {CYCLE_OV_CSV}. Run data_clean.py then ovulation_labels.py.")
    if not SIGNALS_DIR.is_dir():
        raise SystemExit(f"Signals dir not found: {SIGNALS_DIR}. Run wearable_signals.py.")

    df, available = load_data()
    print(f"Main dataset: {len(df)} rows, {df['id'].nunique()} subjects")

    ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily = load_wearable_daily(SIGNALS_DIR)
    key = ["id", "study_interval", "day_in_study"]
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily, wt_daily]:
        df = df.merge(src, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    print(f"Cycles with LH ovulation labels: {len(lh_ov_dict)}")

    personal_luteal = compute_personal_luteal_from_lh(cycle_csv=str(CYCLE_OV_CSV))
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v]) if personal_luteal else 14.0
    print(f"Population luteal mean: {pop_luteal_mean:.1f} days")

    df["is_post_ov"] = np.nan
    for i, row in df.iterrows():
        sgk = row["small_group_key"]
        if sgk in lh_ov_dict:
            df.at[i, "is_post_ov"] = 1.0 if row["day_in_cycle"] > lh_ov_dict[sgk] else 0.0

    labeled = df.dropna(subset=["is_post_ov"])
    print(f"Labeled rows for ov-classifier: {len(labeled)} ({labeled['is_post_ov'].mean():.1%} post-ov)")

    df["cycle_frac"] = df["day_in_cycle"] / df["hist_cycle_len_mean"].clip(lower=20)
    labeled = df.dropna(subset=["is_post_ov"]).copy()
    print("Building causal detection features...")
    labeled = build_ov_detection_features(labeled)
    feat_cols = get_ov_feature_cols()

    valid = labeled.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    print(f"Valid rows after feature filtering: {len(valid)}")

    X_ov = valid[feat_cols].fillna(0).values
    y_ov = valid["is_post_ov"].values.astype(int)
    groups_ov = valid["id"].values

    print("\nTraining ovulation classifier (LOSO)...")
    logo = LeaveOneGroupOut()
    valid["ov_prob"] = np.nan
    for train_idx, test_idx in logo.split(X_ov, y_ov, groups_ov):
        clf = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_ov[train_idx], y_ov[train_idx])
        prob = clf.predict_proba(X_ov[test_idx])[:, 1]
        valid.iloc[test_idx, valid.columns.get_loc("ov_prob")] = prob

    has_prob = valid["ov_prob"].notna()
    if has_prob.sum() > 0 and y_ov[has_prob].std() > 0:
        auc = roc_auc_score(y_ov[has_prob], valid.loc[has_prob, "ov_prob"])
        print(f"Ovulation classifier LOSO AUC: {auc:.3f}")

    # Anchor days relative to LH ground-truth ovulation day (ov_true).
    # We report metrics separately for pre-ovulation vs post-ovulation anchors.
    # Pre:  ov-7, ov-3, ov-1
    # Post: ov+2, ov+5, ov+10
    anchors_pre = [-7, -3, -1]
    anchors_post = [2, 5, 10]
    anchors_all = anchors_pre + anchors_post

    def _metrics_from_abs_err(ae: np.ndarray) -> dict:
        # Mirrors model.evaluate.compute_metrics' threshold convention:
        # acc_1d uses err < 1.5, acc_2d uses err < 2.5, acc_3d uses err < 3.5.
        if len(ae) == 0:
            return {"n": 0, "mae": float("nan"), "acc_1d": float("nan"), "acc_2d": float("nan"), "acc_3d": float("nan")}
        ae = np.asarray(ae, dtype=np.float64)
        return {
            "n": int(len(ae)),
            "mae": float(ae.mean()),
            "acc_1d": float((ae < 1.5).mean()),
            "acc_2d": float((ae < 2.5).mean()),
            "acc_3d": float((ae < 3.5).mean()),
        }

    strategies = ["threshold", "cumulative", "bayesian"]
    all_detected = {}
    for strat in strategies:
        detected_ov = {}
        for sgk, cyc in valid[has_prob].groupby("small_group_key"):
            cyc = cyc.sort_values("day_in_cycle")
            probs = cyc["ov_prob"].values
            ov_dic = detect_ovulation_from_probs(cyc, probs, strategy=strat)
            if ov_dic is not None:
                detected_ov[sgk] = ov_dic
        all_detected[strat] = detected_ov
        errors = [detected_ov[sgk] - lh_ov_dict[sgk] for sgk in detected_ov if sgk in lh_ov_dict]
        err = np.array(errors) if errors else np.array([])
        n_lab = sum(1 for sgk in valid["small_group_key"].unique() if sgk in lh_ov_dict)
        if len(err) > 0:
            print(f"  [{strat}] Detected: {len(detected_ov)}/{n_lab} cycles  | Mean offset: {err.mean():.1f}d  | ±3d: {(np.abs(err) <= 3).mean():.1%}  | ±5d: {(np.abs(err) <= 5).mean():.1%}")
        else:
            print(f"  [{strat}] Detected: {len(detected_ov)}/{n_lab} cycles")

    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"\n{'='*70}")
    print(f"  Hybrid Model: 10-seed evaluation (features: {len(features)})")
    print(f"{'='*70}")

    for strat in strategies:
        detected_ov = all_detected[strat]
        results_hybrid, results_lgb, results_oracle = [], [], []
        # Collect anchor-day absolute errors across seeds, then pool to report.
        anchor_errs = {
            "lgb": {k: [] for k in anchors_all},
            "hyb": {k: [] for k in anchors_all},
            "ora": {k: [] for k in anchors_all},
        }

        for seed in range(10):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            train_idx, test_idx = next(gss.split(df, groups=df["id"]))
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx].copy()
            train_uids = set(df_train["id"].unique())

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
            results_lgb.append(compute_metrics(pred_lgb, y_test))

            sgks = df_test["small_group_key"].values
            dics = df_test["day_in_cycle"].values
            uids = df_test["id"].values

            pred_det = pred_lgb.copy()
            for i in range(len(df_test)):
                sgk, dic, uid = sgks[i], dics[i], uids[i]
                if sgk in detected_ov:
                    ov_dic = detected_ov[sgk]
                    if dic >= ov_dic + 2:
                        days_since = dic - ov_dic
                        luts = personal_luteal.get(uid, []) if uid in train_uids else []
                        avg_lut = np.mean(luts) if luts else pop_luteal_mean
                        pred_det[i] = max(1.0, avg_lut - days_since)
            results_hybrid.append(compute_metrics(pred_det, y_test))

            pred_ora = pred_lgb.copy()
            for i in range(len(df_test)):
                sgk, dic, uid = sgks[i], dics[i], uids[i]
                if sgk in lh_ov_dict:
                    ov_dic = lh_ov_dict[sgk]
                    if dic >= ov_dic + 2:
                        days_since = dic - ov_dic
                        luts = personal_luteal.get(uid, []) if uid in train_uids else []
                        avg_lut = np.mean(luts) if luts else pop_luteal_mean
                        pred_ora[i] = max(1.0, avg_lut - days_since)
            results_oracle.append(compute_metrics(pred_ora, y_test))

            # Anchor-day evaluation (pre vs post) using LH ovulation day.
            # rel = day_in_cycle - ov_true (both in "day-in-cycle index").
            # We only count rows whose sgk has LH ovulation label.
            ov_arr = np.array([lh_ov_dict.get(sgk, np.nan) for sgk in sgks], dtype=float)
            valid = ~np.isnan(ov_arr)
            rel_round = np.full(len(dics), 999999, dtype=int)
            rel_round[valid] = np.rint(np.asarray(dics)[valid] - ov_arr[valid]).astype(int)

            abs_err_lgb = np.abs(pred_lgb - y_test)
            abs_err_hyb = np.abs(pred_det - y_test)
            abs_err_ora = np.abs(pred_ora - y_test)

            for k in anchors_all:
                mask_k = valid & (rel_round == k)
                if mask_k.sum() == 0:
                    continue
                anchor_errs["lgb"][k].append(abs_err_lgb[mask_k])
                anchor_errs["hyb"][k].append(abs_err_hyb[mask_k])
                anchor_errs["ora"][k].append(abs_err_ora[mask_k])

        def avg(results):
            return {k: np.mean([r[k] for r in results]) for k in results[0]}

        lgb_a, hyb_a, ora_a = avg(results_lgb), avg(results_hybrid), avg(results_oracle)
        print(f"\n--- Strategy: {strat} ---")
        print(f"  LightGBM only:        MAE={lgb_a['mae']:.3f}  ±3d={lgb_a['acc_3d']:.1%}")
        print(f"  Detected-ov hybrid:   MAE={hyb_a['mae']:.3f}  ±3d={hyb_a['acc_3d']:.1%}")
        print(f"  Oracle hybrid:        MAE={ora_a['mae']:.3f}  ±3d={ora_a['acc_3d']:.1%}")
        print(f"  Oracle vs LGB:        MAE={ora_a['mae']-lgb_a['mae']:+.3f}  ±3d={100*(ora_a['acc_3d']-lgb_a['acc_3d']):+.1f}pp")

        # Print anchor-day breakdown (pooled across seeds).
        print("  Anchor-day breakdown (pooled abs error across seeds)")
        print("    Pre anchors:  ov-7, ov-3, ov-1")
        for k in anchors_pre:
            ae_lgb = np.concatenate(anchor_errs["lgb"][k]) if anchor_errs["lgb"][k] else np.array([])
            ae_hyb = np.concatenate(anchor_errs["hyb"][k]) if anchor_errs["hyb"][k] else np.array([])
            ae_ora = np.concatenate(anchor_errs["ora"][k]) if anchor_errs["ora"][k] else np.array([])
            m_lgb = _metrics_from_abs_err(ae_lgb)
            m_hyb = _metrics_from_abs_err(ae_hyb)
            m_ora = _metrics_from_abs_err(ae_ora)
            print(
                f"    ov{k:>+3d}: "
                f"LGB n={m_lgb['n']:>4d} MAE={m_lgb['mae']:.3f} ±3d={m_lgb['acc_3d']:.1%} | "
                f"Hybrid n={m_hyb['n']:>4d} MAE={m_hyb['mae']:.3f} ±3d={m_hyb['acc_3d']:.1%} | "
                f"Oracle n={m_ora['n']:>4d} MAE={m_ora['mae']:.3f} ±3d={m_ora['acc_3d']:.1%}"
            )
        print("    Post anchors: ov+2, ov+5, ov+10")
        for k in anchors_post:
            ae_lgb = np.concatenate(anchor_errs["lgb"][k]) if anchor_errs["lgb"][k] else np.array([])
            ae_hyb = np.concatenate(anchor_errs["hyb"][k]) if anchor_errs["hyb"][k] else np.array([])
            ae_ora = np.concatenate(anchor_errs["ora"][k]) if anchor_errs["ora"][k] else np.array([])
            m_lgb = _metrics_from_abs_err(ae_lgb)
            m_hyb = _metrics_from_abs_err(ae_hyb)
            m_ora = _metrics_from_abs_err(ae_ora)
            print(
                f"    ov{k:>+3d}: "
                f"LGB n={m_lgb['n']:>4d} MAE={m_lgb['mae']:.3f} ±3d={m_lgb['acc_3d']:.1%} | "
                f"Hybrid n={m_hyb['n']:>4d} MAE={m_hyb['mae']:.3f} ±3d={m_hyb['acc_3d']:.1%} | "
                f"Oracle n={m_ora['n']:>4d} MAE={m_ora['mae']:.3f} ±3d={m_ora['acc_3d']:.1%}"
            )

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

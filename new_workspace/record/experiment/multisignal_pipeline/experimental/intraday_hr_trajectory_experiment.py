"""
Intraday HR trajectory vs daily scalar — prefix-aligned phase proxy (LOSOCV).

独立实验：不改动主线 detectors。从 heart_rate_cycle.csv 流式读取，按日历日将
confidence>=1 的 BPM 样本落入固定时间箱（默认 48×30min），得到「日内连续轨迹」
特征；与「当日 BPM 标量均值」对照。标签与同管线相位分类一致：周期内相对日 d 末
是否已过真实排卵日 y = 1[ov_dic <= d]。受试者级 Leave-One-Subject-Out + ROC-AUC。

用法（在 multisignal_pipeline 目录下）：
  python experimental/intraday_hr_trajectory_experiment.py
  python experimental/intraday_hr_trajectory_experiment.py --max-chunks 20   # 调试用
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

# multisignal_pipeline 根目录
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data import CYCLE_CSV, SIGNALS_DIR, get_lh_ovulation_labels  # noqa: E402


def _minutes_of_day(ts_series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts_series, errors="coerce")
    return (
        t.dt.hour.to_numpy(dtype=float) * 60.0
        + t.dt.minute.to_numpy(dtype=float)
        + t.dt.second.to_numpy(dtype=float) / 60.0
    )


def _aggregate_hr_intraday(
    hr_path: Path,
    n_bins: int,
    chunk_size: int,
    max_chunks: int | None,
    min_conf: int = 1,
):
    """Returns dict (id, study_interval, day_in_study) -> (sum_vec, cnt_vec, total_bpm, total_n)."""
    bin_width = 1440.0 / n_bins
    store: dict[tuple, dict] = {}

    def _get(k):
        if k not in store:
            store[k] = {
                "sum": np.zeros(n_bins, dtype=np.float64),
                "cnt": np.zeros(n_bins, dtype=np.int64),
                "tot_bpm": 0.0,
                "tot_n": 0,
            }
        return store[k]

    reader = pd.read_csv(
        hr_path,
        usecols=["id", "study_interval", "day_in_study", "timestamp", "bpm", "confidence"],
        chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader):
        if max_chunks is not None and i >= max_chunks:
            break
        chunk = chunk.loc[chunk["confidence"] >= min_conf]
        if chunk.empty:
            continue
        mins = _minutes_of_day(chunk["timestamp"])
        chunk = chunk.assign(_m=mins)
        chunk = chunk.loc[~chunk["_m"].isna()]
        if chunk.empty:
            continue
        chunk["_bi"] = np.clip((chunk["_m"] // bin_width).astype(np.int64), 0, n_bins - 1)
        for key, sub in chunk.groupby(["id", "study_interval", "day_in_study"], sort=False):
            k = (int(key[0]), int(key[1]), int(key[2]))
            o = _get(k)
            bi = sub["_bi"].to_numpy(dtype=np.int64)
            w = sub["bpm"].to_numpy(dtype=float)
            sums = np.bincount(bi, weights=w, minlength=n_bins)
            cnts = np.bincount(bi, minlength=n_bins)
            o["sum"] += sums
            o["cnt"] += cnts
            o["tot_bpm"] += float(w.sum())
            o["tot_n"] += int(len(sub))
        if (i + 1) % 10 == 0:
            print(f"    HR chunks processed: {i + 1} | keys: {len(store)}", flush=True)

    out = {}
    for k, o in store.items():
        vec = np.divide(
            o["sum"],
            np.maximum(o["cnt"], 1),
            out=np.full(n_bins, np.nan),
            where=o["cnt"] > 0,
        )
        scal = o["tot_bpm"] / o["tot_n"] if o["tot_n"] > 0 else np.nan
        out[k] = (vec, scal)
    return out


def _build_labeled_rows(
    hr_feats: dict,
    lh_map: dict[str, int],
    min_cycle_days: int = 10,
    min_bin_coverage: float = 0.25,
):
    cc = pd.read_csv(CYCLE_CSV, usecols=["small_group_key", "id", "study_interval", "day_in_study"])
    X_traj, X_scal, y, groups = [], [], [], []
    n_bins = next(iter(hr_feats.values()))[0].shape[0] if hr_feats else 48

    for sgk, grp in cc.groupby("small_group_key"):
        if sgk not in lh_map:
            continue
        grp = grp.sort_values("day_in_study")
        if len(grp) < min_cycle_days:
            continue
        ov = int(lh_map[sgk])
        uid = int(grp["id"].iloc[0])
        for rel_i, row in enumerate(grp.itertuples(index=False)):
            day_abs = int(row.day_in_study)
            key = (uid, int(row.study_interval), day_abs)
            if key not in hr_feats:
                continue
            vec, scal = hr_feats[key]
            filled = np.sum(~np.isnan(vec))
            if filled < max(1, int(min_bin_coverage * n_bins)):
                continue
            X_traj.append(vec)
            X_scal.append([scal])
            y.append(int(ov <= rel_i))
            groups.append(uid)

    if not X_traj:
        return None, None, None, None
    X_traj = np.vstack(X_traj)
    X_scal = np.vstack(X_scal)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=int)
    return X_traj, X_scal, y, groups


def _loso_auc(X, y, groups):
    logo = LeaveOneGroupOut()
    ys, ps = [], []
    for tr, te in logo.split(X, y, groups):
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="mean")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=10,
                        min_samples_leaf=20,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(X[tr], y[tr])
        ys.append(y[te])
        ps.append(pipe.predict_proba(X[te])[:, 1])
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    if len(np.unique(y_all)) < 2:
        return float("nan")
    return float(roc_auc_score(y_all, p_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbins", type=int, default=48, help="24h 分箱数（默认 48≈30min/箱）")
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--max-chunks", type=int, default=None, help="仅处理前 N 个 CSV 块（调试）")
    ap.add_argument("--min-bin-coverage", type=float, default=0.25, help="该日至少多少比例箱非空才入集")
    args = ap.parse_args()

    hr_path = SIGNALS_DIR / "heart_rate_cycle.csv"
    if not hr_path.exists():
        raise SystemExit(f"未找到 {hr_path}")

    print("=" * 72)
    print("  Intraday HR trajectory vs scalar — LOSO ROC-AUC (phase proxy)")
    print("=" * 72)
    print(f"  Cycle CSV: {CYCLE_CSV}")
    print(f"  HR CSV:    {hr_path}")
    print(f"  Bins: {args.nbins} | chunk_size={args.chunk_size} | max_chunks={args.max_chunks}")

    lh = get_lh_ovulation_labels()
    lh_map = dict(zip(lh["small_group_key"], lh["ov_dic"]))
    print(f"  LH-labeled cycles: {len(lh_map)}")

    print("  Streaming HR → intraday bins (may take several minutes on full file)...")
    hr_feats = _aggregate_hr_intraday(
        hr_path,
        n_bins=args.nbins,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )
    print(f"  Unique (id, study_interval, day) HR keys: {len(hr_feats)}")

    built = _build_labeled_rows(
        hr_feats,
        lh_map,
        min_bin_coverage=args.min_bin_coverage,
    )
    if built[0] is None:
        raise SystemExit("没有有效样本；检查 max-chunks 是否过小或覆盖率阈值过高。")
    X_traj, X_scal, y, groups = built
    print(f"  Labeled rows: {len(y)} | subjects: {len(np.unique(groups))} | post-ov rate: {y.mean():.3f}")

    auc_traj = _loso_auc(X_traj, y, groups)
    auc_scal = _loso_auc(X_scal, y, groups)
    print("\n  --- Results (Leave-One-Subject-Out, pooled predictions) ---")
    print(f"  Intraday trajectory ({args.nbins} bins)  ROC-AUC: {auc_traj:.4f}")
    print(f"  Daily scalar (mean BPM)              ROC-AUC: {auc_scal:.4f}")
    print(f"  Delta (trajectory - scalar):          {auc_traj - auc_scal:+.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

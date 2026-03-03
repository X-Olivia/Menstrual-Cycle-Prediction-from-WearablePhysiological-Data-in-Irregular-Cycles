"""检查 processed_data/2/full.csv 中 z 分数列：范围、均值、异常值。在 main_workspace 下运行: python data_process/check_full_csv.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from model.config import FULL_CSV, FEATURE_COLS

def main():
    path = ROOT / "processed_data" / "2" / "full.csv"
    if not path.exists():
        path = Path(FULL_CSV)
    df = pd.read_csv(path)
    z_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not z_cols:
        z_cols = [c for c in df.columns if c.endswith("_z")]
    X = df[z_cols].astype(float)
    X = np.nan_to_num(X.values, nan=0.0)

    print("=== 生成数据检查: full.csv (z 分数列) ===\n")
    print(f"行数: {len(df)}, z 列数: {len(z_cols)}")
    print(f"z 列: {z_cols}\n")

    print("各列: min, max, mean, std, |值|>5 的个数")
    print("-" * 60)
    for j, col in enumerate(z_cols):
        v = X[:, j]
        n_bad = np.sum(np.abs(v) > 5)
        print(f"  {col:28s}  min={v.min():8.3f}  max={v.max():8.3f}  mean={np.mean(v):7.3f}  std={np.std(v):6.3f}  |x|>5: {n_bad}")

    mean_abs = np.mean(np.abs(X))
    print("-" * 60)
    print(f"  Feature mean |x| (probe 用): {mean_abs:.4f}  (正常 z 期望约 0.5~1)")
    if mean_abs > 100:
        print("  [异常] 存在极大值，需检查 daily_data_2 中 z 计算与裁剪")
    else:
        print("  [正常] z 尺度在合理范围")

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count or inf_count:
        print(f"  NaN 数: {nan_count}, Inf 数: {inf_count}")
    print()

if __name__ == "__main__":
    main()

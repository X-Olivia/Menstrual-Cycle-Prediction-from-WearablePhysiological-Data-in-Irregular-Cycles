"""A/B comparison: v3 (per-subject z-norm) vs v4 (per-cycle-early z-norm).
Run with 10-seed multi-seed evaluation on the same 22-feature set."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v3.config import FEATURES_CSV, FEATURES_V4_CSV, ALL_FEATURES
from model_v3.robust_eval import run_multi_seed

N_SEEDS = 10

print("=" * 70)
print("  A/B Test: v3 (per-subject z-norm) vs v4 (per-cycle-early z-norm)")
print("  Features: 22 (ablation-validated optimal)")
print("=" * 70)

print("\n" + "#" * 70)
print("  [A] v3 baseline (per-subject z-normalization)")
print("#" * 70)
r_v3 = run_multi_seed(n_seeds=N_SEEDS, features_csv=FEATURES_CSV, feature_list=ALL_FEATURES)

print("\n" + "#" * 70)
print("  [B] v4 (per-cycle-early z-normalization)")
print("#" * 70)
r_v4 = run_multi_seed(n_seeds=N_SEEDS, features_csv=FEATURES_V4_CSV, feature_list=ALL_FEATURES)

print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
print(f"  {'Metric':25s} {'v3 (per-subj)':>20s} {'v4 (per-cycle)':>20s} {'Δ':>10s}")
print(f"  {'-' * 75}")

delta_mae = r_v4["test_mae_mean"] - r_v3["test_mae_mean"]
delta_acc3 = r_v4["test_acc3_mean"] - r_v3["test_acc3_mean"]

print(f"  {'Test MAE':25s} "
      f"{r_v3['test_mae_mean']:.3f} ± {r_v3['test_mae_std']:.3f}      "
      f"{r_v4['test_mae_mean']:.3f} ± {r_v4['test_mae_std']:.3f}      "
      f"{delta_mae:+.3f}")
print(f"  {'Test ±3d Acc':25s} "
      f"{r_v3['test_acc3_mean']:.3f}              "
      f"{r_v4['test_acc3_mean']:.3f}              "
      f"{delta_acc3:+.3f}")
print(f"  {'Val MAE':25s} "
      f"{r_v3['val_mae_mean']:.3f}              "
      f"{r_v4['val_mae_mean']:.3f}              "
      f"{r_v4['val_mae_mean'] - r_v3['val_mae_mean']:+.3f}")

if delta_mae < 0:
    print(f"\n  >>> v4 WINS: MAE improved by {-delta_mae:.3f} days <<<")
elif delta_mae > 0:
    print(f"\n  >>> v3 better by {delta_mae:.3f} days MAE <<<")
else:
    print(f"\n  >>> TIE <<<")

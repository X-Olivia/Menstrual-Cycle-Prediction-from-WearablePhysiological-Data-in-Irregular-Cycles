"""Additional test: v4 normalization with rolling features."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v3.config import FEATURES_V4_CSV, ALL_FEATURES, ALL_FEATURES_ROLLING
from model_v3.robust_eval import run_multi_seed

N_SEEDS = 10

print("=" * 70)
print("  v4 + rolling features (per-cycle-early z-norm + 70 features)")
print("=" * 70)

r = run_multi_seed(
    n_seeds=N_SEEDS,
    features_csv=FEATURES_V4_CSV,
    feature_list=ALL_FEATURES_ROLLING,
)

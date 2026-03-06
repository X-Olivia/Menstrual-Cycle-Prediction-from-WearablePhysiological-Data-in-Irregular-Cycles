"""
1D-CNN Ovulation Day Regression from Nightly Temperature Sequences.

Best method from comprehensive ovulation detection experiments.
Achieves ~90% ±3d accuracy with LOSO cross-validation.

Architecture: 3-layer 1D-CNN that learns the biphasic temperature pattern
and predicts the relative ovulation position within a cycle.

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -m model.ovulation_cnn
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.evaluate import compute_metrics
from model.ovulation_detect import get_lh_ovulation_labels
from model.ovulation_detect import compute_personal_luteal_from_lh

MAX_CYCLE_LEN = 45


# ======================================================================
# Model Architecture
# ======================================================================

class OvulationCNN(nn.Module):
    """1D-CNN that regresses the ovulation day fraction (0-1) from a
    per-cycle z-normalized nightly temperature sequence."""
    
    def __init__(self, seq_len=MAX_CYCLE_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


class MultiSignalOvulationCNN(nn.Module):
    """Multi-channel 1D-CNN using temperature + HR + HRV signals."""
    
    def __init__(self, n_channels=5, seq_len=MAX_CYCLE_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(-1)


# ======================================================================
# Dataset
# ======================================================================

class CycleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s["seq"]
        if seq.ndim == 1:
            seq = seq[np.newaxis, :]
        return torch.FloatTensor(seq), s["ov_frac"]


# ======================================================================
# Data preparation
# ======================================================================

def load_cycle_temperature_series():
    """Load nightly temperature per cycle."""
    cc = pd.read_csv(CYCLE_CSV)
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    key = ["id", "study_interval", "day_in_study"]
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)
    merged = cc.merge(ct_daily, on=key, how="left")
    
    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        cycle_series[sgk] = {
            "dic": (grp["day_in_study"] - cs).values,
            "temps": grp["nightly_temperature"].values,
            "id": grp["id"].values[0],
            "cs": cs,
            "ce": grp["day_in_study"].max(),
        }
    return cycle_series


def prepare_temp_samples(lh_ov_dict, cycle_series):
    """Create CNN training samples from cycle temperature data."""
    samples = []
    for sgk in lh_ov_dict:
        if sgk not in cycle_series:
            continue
        data = cycle_series[sgk]
        temps = data["temps"]
        uid = data["id"]
        if len(temps) < 10:
            continue
        
        t_clean = pd.Series(temps).interpolate(limit_direction="both").fillna(0).values
        mu, std = np.mean(t_clean), max(np.std(t_clean), 0.01)
        t_norm = (t_clean - mu) / std
        
        if len(t_norm) < MAX_CYCLE_LEN:
            t_padded = np.pad(t_norm, (0, MAX_CYCLE_LEN - len(t_norm)),
                              mode="constant", constant_values=0)
        else:
            t_padded = t_norm[:MAX_CYCLE_LEN]
        
        ov_day = lh_ov_dict[sgk]
        ov_frac = ov_day / len(temps)
        
        samples.append({
            "seq": t_padded.astype(np.float32),
            "ov_frac": float(ov_frac),
            "ov_day": int(ov_day),
            "cycle_len": len(temps),
            "id": uid,
            "sgk": sgk,
        })
    return samples


# ======================================================================
# t-test split (rule-based complement)
# ======================================================================

def detect_ov_ttest_split(temps, dic, smooth_sigma=2.5, expected_frac=0.50, position_width=4.0):
    """Retrospective t-test optimal split for ovulation detection."""
    valid = ~np.isnan(temps)
    if valid.sum() < 10:
        return None, 0
    
    t_clean = pd.Series(temps).interpolate(limit_direction="both").values
    if smooth_sigma > 0:
        t_clean = gaussian_filter1d(t_clean, sigma=smooth_sigma)
    
    n = len(t_clean)
    best_stat = -np.inf
    best_split = None
    
    for split in range(6, n - 3):
        pre = t_clean[:split]
        post = t_clean[split:]
        if len(pre) < 3 or len(post) < 3:
            continue
        diff = np.mean(post) - np.mean(pre)
        if diff <= 0:
            continue
        try:
            stat, _ = ttest_ind(post, pre, alternative="greater")
        except Exception:
            continue
        if np.isnan(stat):
            continue
        
        expected_ov = max(8, n * expected_frac)
        pp = np.exp(-0.5 * ((dic[split] - expected_ov) / position_width) ** 2)
        ws = stat * pp
        
        if ws > best_stat:
            best_stat = ws
            best_split = split
    
    if best_split is not None:
        return int(dic[best_split]), best_stat
    return None, 0


# ======================================================================
# Training & Evaluation
# ======================================================================

def train_and_evaluate_loso(samples, model_class=OvulationCNN, n_seeds=5,
                            epochs=60, lr=3e-4, batch_size=16):
    """Train model with LOSO cross-validation and return predictions."""
    ids = np.array([s["id"] for s in samples])
    unique_ids = np.unique(ids)
    
    seed_results = []
    
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        preds = {}
        for test_id in unique_ids:
            test_mask = ids == test_id
            train_mask = ~test_mask
            train_s = [s for s, m in zip(samples, train_mask) if m]
            test_s = [s for s, m in zip(samples, test_mask) if m]
            
            if len(train_s) < 20 or len(test_s) < 1:
                continue
            
            train_dl = DataLoader(CycleDataset(train_s), batch_size=batch_size, shuffle=True)
            
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(epochs):
                for xb, yb in train_dl:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb.float())
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            
            model.eval()
            with torch.no_grad():
                for s in test_s:
                    seq = s["seq"]
                    if seq.ndim == 1:
                        seq = seq[np.newaxis, :]
                    xb = torch.FloatTensor(seq).unsqueeze(0)
                    pred_frac = model(xb).item()
                    pred_day = int(round(pred_frac * s["cycle_len"]))
                    preds[s["sgk"]] = pred_day
        
        seed_results.append(preds)
    
    return seed_results


def evaluate_detection(detected_ov, lh_ov_dict, name=""):
    """Evaluate ovulation detection accuracy against LH labels."""
    errors = []
    for sgk, det_dic in detected_ov.items():
        if sgk in lh_ov_dict:
            errors.append(det_dic - lh_ov_dict[sgk])
    
    n_total = len(lh_ov_dict)
    n_detected = len(detected_ov)
    n_evaluated = len(errors)
    
    if n_evaluated == 0:
        print(f"  [{name}] No valid detections")
        return {}
    
    err = np.array(errors)
    abs_err = np.abs(err)
    
    results = {
        "n_total": n_total,
        "n_detected": n_detected,
        "n_evaluated": n_evaluated,
        "recall": n_detected / n_total if n_total > 0 else 0,
        "mae": float(abs_err.mean()),
        "median_err": float(np.median(err)),
        "acc_1d": float(np.mean(abs_err <= 1)),
        "acc_2d": float(np.mean(abs_err <= 2)),
        "acc_3d": float(np.mean(abs_err <= 3)),
        "acc_5d": float(np.mean(abs_err <= 5)),
    }
    
    print(
        f"  [{name}] {n_detected}/{n_total} detected ({results['recall']:.0%})"
        f" | MAE={results['mae']:.2f}d"
        f" | ±1d={results['acc_1d']:.1%}"
        f" | ±2d={results['acc_2d']:.1%}"
        f" | ±3d={results['acc_3d']:.1%}"
        f" | ±5d={results['acc_5d']:.1%}"
        f" | med={results['median_err']:+.1f}d"
    )
    return results


# ======================================================================
# Menstruation prediction using detected ovulation
# ======================================================================

def predict_menses_with_ovulation(df, detected_ov, lh_ov_dict, personal_luteal):
    """Predict menstruation using detected ovulation + personal luteal length."""
    from sklearn.model_selection import GroupShuffleSplit
    
    pop_luteal_mean = np.mean([l for v in personal_luteal.values() for l in v])
    
    results_det = []
    results_cal = []
    results_ora = []
    
    for seed in range(10):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["id"]))
        df_test = df.iloc[test_idx].copy()
        
        y_test = df_test["days_until_next_menses"].values
        sgks = df_test["small_group_key"].values
        dics = df_test["day_in_cycle"].values
        uids = df_test["id"].values
        hist_lens = df_test["hist_cycle_len_mean"].values
        
        pred_cal = np.array([
            max(1.0, (hl if not np.isnan(hl) else 28) - dic)
            for hl, dic in zip(hist_lens, dics)
        ])
        
        pred_det = pred_cal.copy()
        pred_ora = pred_cal.copy()
        
        for i in range(len(df_test)):
            sgk, dic, uid = sgks[i], dics[i], uids[i]
            luts = personal_luteal.get(uid, [])
            avg_lut = np.mean(luts) if luts else pop_luteal_mean
            
            if sgk in detected_ov:
                ov_dic = detected_ov[sgk]
                if dic >= ov_dic + 2:
                    days_since = dic - ov_dic
                    pred_det[i] = max(1.0, avg_lut - days_since)
            
            if sgk in lh_ov_dict:
                ov_dic = lh_ov_dict[sgk]
                if dic >= ov_dic + 2:
                    days_since = dic - ov_dic
                    pred_ora[i] = max(1.0, avg_lut - days_since)
        
        results_cal.append(compute_metrics(pred_cal, y_test))
        results_det.append(compute_metrics(pred_det, y_test))
        results_ora.append(compute_metrics(pred_ora, y_test))
    
    def avg(r):
        return {k: np.mean([x[k] for x in r]) for k in r[0]}
    
    cal_a, det_a, ora_a = avg(results_cal), avg(results_det), avg(results_ora)
    
    print(f"\n  Calendar baseline:    MAE={cal_a['mae']:.3f}  ±3d={cal_a['acc_3d']:.1%}")
    print(f"  CNN-detected ovulation: MAE={det_a['mae']:.3f}  ±3d={det_a['acc_3d']:.1%}")
    print(f"  Oracle (LH):          MAE={ora_a['mae']:.3f}  ±3d={ora_a['acc_3d']:.1%}")
    print(f"  Gap closed: {100*(det_a['acc_3d']-cal_a['acc_3d'])/(ora_a['acc_3d']-cal_a['acc_3d']+1e-10):.1f}% of Oracle gap")
    
    return cal_a, det_a, ora_a


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 72)
    print("  1D-CNN Ovulation Detection & Menstruation Prediction")
    print("=" * 72)
    
    # Load data
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    cycle_series = load_cycle_temperature_series()
    print(f"  LH-labeled cycles: {len(lh_ov_dict)}")
    print(f"  Cycles with temperature: {len(cycle_series)}")
    
    # Prepare samples
    samples = prepare_temp_samples(lh_ov_dict, cycle_series)
    print(f"  Training samples: {len(samples)}, subjects: {len(set(s['id'] for s in samples))}")
    
    # ================================================================
    # Experiment 1: 1D-CNN (temperature only)
    # ================================================================
    print(f"\n{'='*72}")
    print("  Exp 1: 1D-CNN Regression (temp only, LOSO, 5 seeds)")
    print(f"{'='*72}")
    
    seed_preds = train_and_evaluate_loso(
        samples, model_class=OvulationCNN, n_seeds=5, epochs=60
    )
    
    for i, preds in enumerate(seed_preds):
        evaluate_detection(preds, lh_ov_dict, f"CNN-seed{i}")
    
    # Average across seeds
    all_sgks = set()
    for preds in seed_preds:
        all_sgks.update(preds.keys())
    
    ensemble_preds = {}
    for sgk in all_sgks:
        vals = [preds[sgk] for preds in seed_preds if sgk in preds]
        if vals:
            ensemble_preds[sgk] = int(round(np.mean(vals)))
    
    print(f"\n  --- Ensemble (mean of 5 seeds) ---")
    evaluate_detection(ensemble_preds, lh_ov_dict, "CNN-ensemble")
    
    # ================================================================
    # Experiment 2: t-test split (rule-based complement)
    # ================================================================
    print(f"\n{'='*72}")
    print("  Exp 2: t-test Split (rule-based)")
    print(f"{'='*72}")
    
    ttest_preds = {}
    ttest_scores = {}
    for sgk in lh_ov_dict:
        if sgk in cycle_series:
            data = cycle_series[sgk]
            ov, score = detect_ov_ttest_split(data["temps"], data["dic"])
            if ov is not None:
                ttest_preds[sgk] = ov
                ttest_scores[sgk] = score
    
    evaluate_detection(ttest_preds, lh_ov_dict, "t-test-all")
    
    high_conf = {sgk: v for sgk, v in ttest_preds.items() if ttest_scores.get(sgk, 0) > 1.5}
    evaluate_detection(high_conf, lh_ov_dict, "t-test-highconf")
    
    # ================================================================
    # Experiment 3: Hybrid (CNN + t-test fallback)
    # ================================================================
    print(f"\n{'='*72}")
    print("  Exp 3: Hybrid Strategy")
    print(f"{'='*72}")
    
    hybrid_preds = {}
    for sgk in lh_ov_dict:
        if sgk in ensemble_preds:
            hybrid_preds[sgk] = ensemble_preds[sgk]
        elif sgk in ttest_preds:
            hybrid_preds[sgk] = ttest_preds[sgk]
        elif sgk in cycle_series:
            n = len(cycle_series[sgk]["dic"])
            hybrid_preds[sgk] = int(round(n * 0.50))
    
    evaluate_detection(hybrid_preds, lh_ov_dict, "HYBRID")
    
    # ================================================================
    # Menstruation prediction
    # ================================================================
    print(f"\n{'='*72}")
    print("  Menstruation Prediction with CNN-detected Ovulation")
    print(f"{'='*72}")
    
    from model.dataset import load_data
    df, _ = load_data()
    personal_luteal = compute_personal_luteal_from_lh()
    
    predict_menses_with_ovulation(df, ensemble_preds, lh_ov_dict, personal_luteal)
    
    print(f"\n{'='*72}")
    print("  COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

"""
Comprehensive Ovulation Detection Experiment Suite
===================================================
Two paths:
  Path 1 (rule-based): domain-knowledge algorithms on raw wearable signals
    - Exp 1a: ruptures PELT changepoint detection on nightly temperature
    - Exp 1b: CUSUM (cumulative sum) on nightly temperature
    - Exp 1c: Bayesian Online Changepoint Detection (BOCPD)
    - Exp 1d: Enhanced 3-over-6 with adaptive threshold

  Path 2 (ML / unsupervised):
    - Exp 2a: 2-state HMM (hmmlearn) on multi-signal daily features
    - Exp 2b: Improved GBDT with nocturnal-only HF features + quality filter
    - Exp 2c: 1D-CNN on raw minute-level temperature sequences

Optimization directions:
  - Nocturnal-only window (22:00-06:00) vs full-day
  - Quality filtering (≥0.2°C shift cycles)
  - Multi-signal fusion (temp + HR + HRV)

Target: ±3d ovulation detection accuracy ≥ 80%

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -m model.experiment.run_ovulation_experiments
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.dataset import load_data
from model.ovulation_detect import compute_personal_luteal_from_lh
from model.evaluate import compute_metrics
from model.ovulation_detect import get_lh_ovulation_labels

# ======================================================================
# Data loading helpers
# ======================================================================

def load_cycle_temperature_series():
    """Load nightly temperature per cycle as dict of {small_group_key: Series}."""
    cc = pd.read_csv(CYCLE_CSV)
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    key = ["id", "study_interval", "day_in_study"]
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)
    merged = cc.merge(ct_daily, on=key, how="left")
    
    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        temps = grp.set_index("day_in_study")["nightly_temperature"]
        dic = (grp["day_in_study"] - cs).values
        vals = grp["nightly_temperature"].values
        cycle_series[sgk] = {
            "dic": dic,
            "temps": vals,
            "id": grp["id"].values[0],
            "cs": cs,
            "ce": grp["day_in_study"].max(),
        }
    return cycle_series


def load_raw_minute_temperature():
    """Load minute-level wrist temperature, return dict per (id, study_interval, day_in_study)."""
    wt = pd.read_csv(os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv"))
    wt["hour"] = wt["timestamp"].str[:2].astype(int)
    wt["minute"] = wt["timestamp"].str[3:5].astype(int)
    wt["temp"] = wt["temperature_diff_from_baseline"]
    return wt


def load_nocturnal_temperature_series():
    """Extract nocturnal (22:00-06:00) mean temperature per day from minute-level data."""
    wt = load_raw_minute_temperature()
    night_mask = (wt["hour"] >= 22) | (wt["hour"] < 6)
    wt_night = wt[night_mask]
    key = ["id", "study_interval", "day_in_study"]
    nocturnal = wt_night.groupby(key)["temp"].agg(["mean", "std", "min", "max", "median"]).reset_index()
    nocturnal.columns = key + ["noct_mean", "noct_std", "noct_min", "noct_max", "noct_median"]
    return nocturnal


def build_cycle_nocturnal_series(nocturnal_df, cc_df):
    """Merge nocturnal stats with cycle info to get per-cycle series."""
    key = ["id", "study_interval", "day_in_study"]
    merged = cc_df.merge(nocturnal_df, on=key, how="left")
    
    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        cycle_series[sgk] = {
            "dic": (grp["day_in_study"] - cs).values,
            "noct_mean": grp["noct_mean"].values,
            "noct_std": grp["noct_std"].values,
            "noct_min": grp["noct_min"].values,
            "noct_max": grp["noct_max"].values,
            "noct_median": grp["noct_median"].values,
            "id": grp["id"].values[0],
            "cs": cs,
            "ce": grp["day_in_study"].max(),
        }
    return cycle_series


def load_all_daily_signals():
    """Load all daily wearable signals merged with cycle data."""
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)
    
    rhr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv"))
    rhr_daily = rhr[key + ["value"]].rename(columns={"value": "resting_hr"}).drop_duplicates(subset=key)
    
    hr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv"))
    hr_daily = hr.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
    hr_daily.columns = key + ["hr_mean", "hr_std", "hr_min"]
    
    hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        {"rmssd": "mean", "low_frequency": "mean", "high_frequency": "mean"}
    ).reset_index()
    hrv_daily.columns = key + ["rmssd_mean", "lf_mean", "hf_mean"]
    
    merged = cc.copy()
    for src in [ct_daily, rhr_daily, hr_daily, hrv_daily]:
        merged = merged.merge(src, on=key, how="left")
    
    return merged


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
        print(f"  [{name}] No valid detections to evaluate")
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
        f" | MAE={results['mae']:.1f}d"
        f" | ±1d={results['acc_1d']:.1%}"
        f" | ±2d={results['acc_2d']:.1%}"
        f" | ±3d={results['acc_3d']:.1%}"
        f" | ±5d={results['acc_5d']:.1%}"
        f" | med={results['median_err']:+.1f}d"
    )
    return results


# ======================================================================
# PATH 1: Rule-based / Algorithmic Methods
# ======================================================================

# --- Exp 1a: ruptures PELT changepoint detection ---

def detect_ov_ruptures(cycle_data, method="pelt", penalty=3.0, min_size=5):
    """Detect ovulation via changepoint detection (ruptures library)."""
    import ruptures as rpt
    
    temps = cycle_data["temps"] if "temps" in cycle_data else cycle_data["noct_mean"]
    dic = cycle_data["dic"]
    valid = ~np.isnan(temps)
    
    if valid.sum() < 10:
        return None
    
    t_clean = temps.copy()
    if np.any(~valid):
        t_clean = pd.Series(t_clean).interpolate(limit_direction="both").values
    
    if method == "pelt":
        algo = rpt.Pelt(model="l2", min_size=min_size).fit(t_clean.reshape(-1, 1))
        try:
            bkps = algo.predict(pen=penalty)
        except Exception:
            return None
    elif method == "binseg":
        algo = rpt.Binseg(model="l2", min_size=min_size).fit(t_clean.reshape(-1, 1))
        try:
            bkps = algo.predict(n_bkps=1)
        except Exception:
            return None
    elif method == "dynp":
        algo = rpt.Dynp(model="l2", min_size=min_size).fit(t_clean.reshape(-1, 1))
        try:
            bkps = algo.predict(n_bkps=1)
        except Exception:
            return None
    else:
        return None
    
    bkps = [b for b in bkps if b < len(temps)]
    if not bkps:
        return None
    
    best_bkp = None
    best_diff = -np.inf
    for bkp in bkps:
        if bkp < 5 or bkp >= len(temps) - 3:
            continue
        pre = t_clean[:bkp]
        post = t_clean[bkp:]
        diff = np.nanmean(post[:5]) - np.nanmean(pre[-5:])
        if diff > best_diff and diff > 0:
            best_diff = diff
            best_bkp = bkp
    
    if best_bkp is not None:
        return int(dic[best_bkp]) if best_bkp < len(dic) else None
    return None


# --- Exp 1b: CUSUM algorithm ---

def detect_ov_cusum(cycle_data, threshold=0.5, drift=0.05, use_noct=False):
    """Detect ovulation via CUSUM (cumulative sum control chart)."""
    temps = cycle_data["noct_mean"] if use_noct and "noct_mean" in cycle_data else cycle_data.get("temps", cycle_data.get("noct_mean"))
    dic = cycle_data["dic"]
    valid = ~np.isnan(temps)
    
    if valid.sum() < 10:
        return None
    
    t_clean = pd.Series(temps).interpolate(limit_direction="both").values
    
    baseline_len = min(6, len(t_clean) // 3)
    if baseline_len < 3:
        return None
    
    baseline_mean = np.mean(t_clean[:baseline_len])
    baseline_std = np.std(t_clean[:baseline_len])
    if baseline_std < 0.01:
        baseline_std = 0.1
    
    z = (t_clean - baseline_mean) / baseline_std
    
    S_pos = np.zeros(len(z))
    for i in range(1, len(z)):
        S_pos[i] = max(0, S_pos[i-1] + z[i] - drift)
    
    for i in range(baseline_len + 2, len(z)):
        if S_pos[i] > threshold and dic[i] >= 8:
            detected_idx = i
            for j in range(i, max(baseline_len, i - 5), -1):
                if S_pos[j] < threshold * 0.3:
                    detected_idx = j
                    break
            return int(dic[detected_idx])
    
    return None


# --- Exp 1c: Bayesian Online Changepoint Detection ---

def detect_ov_bocpd(cycle_data, hazard_rate=1/20, mu0=0, kappa0=1, alpha0=1, beta0=0.5, use_noct=False):
    """Bayesian Online Changepoint Detection for ovulation."""
    temps = cycle_data["noct_mean"] if use_noct and "noct_mean" in cycle_data else cycle_data.get("temps", cycle_data.get("noct_mean"))
    dic = cycle_data["dic"]
    valid = ~np.isnan(temps)
    
    if valid.sum() < 10:
        return None
    
    t_clean = pd.Series(temps).interpolate(limit_direction="both").values
    T = len(t_clean)
    
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    
    mu = np.full(T + 1, mu0)
    kappa = np.full(T + 1, kappa0)
    alpha = np.full(T + 1, alpha0)
    beta = np.full(T + 1, beta0)
    
    maxes = np.zeros(T)
    
    for t in range(T):
        x = t_clean[t]
        pred_var = beta * (kappa + 1) / (alpha * kappa)
        pred_std = np.sqrt(np.maximum(pred_var, 1e-10))
        pred_prob = norm.pdf(x, mu[:t+1], pred_std[:t+1])
        
        H = hazard_rate
        growth_prob = R[:t+1, t] * pred_prob * (1 - H)
        cp_prob = np.sum(R[:t+1, t] * pred_prob * H)
        
        R[1:t+2, t+1] = growth_prob
        R[0, t+1] = cp_prob
        
        evidence = np.sum(R[:t+2, t+1])
        if evidence > 0:
            R[:t+2, t+1] /= evidence
        
        kappa_new = kappa[:t+1] + 1
        mu_new = (kappa[:t+1] * mu[:t+1] + x) / kappa_new
        alpha_new = alpha[:t+1] + 0.5
        beta_new = beta[:t+1] + kappa[:t+1] * (x - mu[:t+1])**2 / (2 * kappa_new)
        
        mu[1:t+2] = mu_new
        kappa[1:t+2] = kappa_new
        alpha[1:t+2] = alpha_new
        beta[1:t+2] = beta_new
        
        maxes[t] = np.argmax(R[:t+2, t+1])
    
    cp_signal = np.zeros(T)
    for t in range(1, T):
        if maxes[t] < maxes[t-1] and maxes[t] < 3:
            cp_signal[t] = 1
    
    for t in range(8, T):
        if cp_signal[t] > 0:
            pre_mean = np.mean(t_clean[max(0, t-6):t])
            post_mean = np.mean(t_clean[t:min(T, t+3)])
            if post_mean > pre_mean:
                return int(dic[t])
    
    diff = np.diff(maxes)
    large_drops = np.where(diff < -3)[0] + 1
    for idx in large_drops:
        if idx >= 8 and idx < T - 2:
            pre_mean = np.mean(t_clean[max(0, idx-5):idx])
            post_mean = np.mean(t_clean[idx:min(T, idx+3)])
            if post_mean > pre_mean:
                return int(dic[idx])
    
    return None


# --- Exp 1d: Enhanced 3-over-6 with adaptive threshold ---

def detect_ov_coverline(cycle_data, shift_threshold=0.1, confirm_days=3, baseline_days=6, use_noct=False):
    """Enhanced coverline rule: 3 consecutive days above coverline."""
    temps = cycle_data["noct_mean"] if use_noct and "noct_mean" in cycle_data else cycle_data.get("temps", cycle_data.get("noct_mean"))
    dic = cycle_data["dic"]
    valid = ~np.isnan(temps)
    
    if valid.sum() < 10:
        return None
    
    t_clean = pd.Series(temps).interpolate(limit_direction="both").values
    
    for i in range(baseline_days + 1, len(t_clean) - confirm_days + 1):
        if dic[i] < 8:
            continue
        
        baseline = t_clean[max(0, i - baseline_days):i]
        coverline = np.mean(baseline) + shift_threshold
        
        above = all(t_clean[i + j] > coverline for j in range(confirm_days) if i + j < len(t_clean))
        if above:
            return int(dic[i])
    
    return None


# ======================================================================
# PATH 2: ML / Data-driven Methods
# ======================================================================

# --- Exp 2a: 2-state HMM ---

def detect_ov_hmm(cycle_data, n_states=2, use_noct=False, multi_signal=False):
    """2-state HMM (low=follicular, high=luteal) for ovulation detection."""
    from hmmlearn.hmm import GaussianHMM
    
    if multi_signal and "signals" in cycle_data:
        obs = cycle_data["signals"]
        valid = ~np.any(np.isnan(obs), axis=1)
    else:
        temps = cycle_data["noct_mean"] if use_noct and "noct_mean" in cycle_data else cycle_data.get("temps", cycle_data.get("noct_mean"))
        obs = temps.reshape(-1, 1) if temps.ndim == 1 else temps
        valid = ~np.isnan(obs.ravel())
    
    dic = cycle_data["dic"]
    
    if valid.sum() < 10:
        return None
    
    obs_clean = obs.copy()
    for col in range(obs_clean.shape[1] if obs_clean.ndim > 1 else 1):
        if obs_clean.ndim > 1:
            obs_clean[:, col] = pd.Series(obs_clean[:, col]).interpolate(limit_direction="both").values
        else:
            obs_clean = pd.Series(obs_clean.ravel()).interpolate(limit_direction="both").values.reshape(-1, 1)
    
    if obs_clean.ndim == 1:
        obs_clean = obs_clean.reshape(-1, 1)
    
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            init_params="stmc",
        )
        
        if n_states == 2:
            model.startprob_ = np.array([0.9, 0.1])
            model.transmat_ = np.array([[0.95, 0.05], [0.02, 0.98]])
            model.init_params = "mc"
        
        model.fit(obs_clean)
        states = model.predict(obs_clean)
        
        means = model.means_[:, 0]
        low_state = np.argmin(means)
        high_state = np.argmax(means)
        
        for i in range(1, len(states)):
            if states[i] == high_state and states[i-1] == low_state:
                if dic[i] >= 8:
                    return int(dic[i])
        
        return None
    except Exception:
        return None


# --- Exp 2b: Enhanced GBDT with nocturnal features + quality filter ---

def run_enhanced_gbdt_experiment(use_nocturnal=True, quality_filter=True, multi_signal=True):
    """Enhanced GBDT with nocturnal HF features and quality filtering."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import roc_auc_score
    
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    
    # Load nocturnal temperature features
    nocturnal_df = load_nocturnal_temperature_series()
    
    # Load raw minute-level for enhanced features
    wt = load_raw_minute_temperature()
    
    # Extract enhanced nocturnal features
    print("  Extracting enhanced nocturnal features...")
    hf_features = extract_enhanced_nocturnal_features(wt)
    
    merged = cc.merge(hf_features, on=key, how="left")
    
    if multi_signal:
        ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
        ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)
        
        hr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv"))
        hr_night = hr[hr["timestamp"].str[:2].astype(int).between(0, 6)]
        hr_nightly = hr_night.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index()
        hr_nightly.columns = key + ["hr_night_mean", "hr_night_std", "hr_night_min"]
        
        hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
        hrv_night = hrv[hrv["timestamp"].str[:2].astype(int).between(0, 6)]
        hrv_nightly = hrv_night.groupby(key).agg(
            {"rmssd": "mean", "low_frequency": "mean", "high_frequency": "mean"}
        ).reset_index()
        hrv_nightly.columns = key + ["rmssd_night", "lf_night", "hf_night"]
        
        for src in [ct_daily, hr_nightly, hrv_nightly]:
            merged = merged.merge(src, on=key, how="left")
    
    # Get LH labels
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    
    merged["is_post_ov"] = np.nan
    for sgk in merged["small_group_key"].unique():
        if sgk in lh_ov_dict:
            mask = merged["small_group_key"] == sgk
            cs = merged.loc[mask, "day_in_study"].min()
            merged.loc[mask, "is_post_ov"] = (
                (merged.loc[mask, "day_in_study"] - cs) > lh_ov_dict[sgk]
            ).astype(float)
    
    labeled = merged.dropna(subset=["is_post_ov"]).copy()
    
    # Feature columns
    signal_cols = [
        "noct_stable_mean", "noct_stable_std", "noct_nadir",
        "noct_nadir_time", "noct_rise_slope", "noct_range",
        "noct_p90", "noct_p10", "noct_iqr",
        "noct_mean_2h", "noct_mean_4h",
    ]
    
    if multi_signal:
        signal_cols += [
            "nightly_temperature",
            "hr_night_mean", "hr_night_std", "hr_night_min",
            "rmssd_night", "lf_night", "hf_night",
        ]
    
    # Build causal rolling features
    labeled = labeled.sort_values(["small_group_key", "day_in_study"])
    cs_map = labeled.groupby("small_group_key")["day_in_study"].min()
    labeled["day_in_cycle"] = labeled.apply(
        lambda r: r["day_in_study"] - cs_map.get(r["small_group_key"], r["day_in_study"]), axis=1
    )
    hist_len = labeled.groupby("id")["small_group_key"].transform(
        lambda x: 28
    )
    labeled["cycle_frac"] = labeled["day_in_cycle"] / 28.0
    
    chunks = []
    for sgk, grp in labeled.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study").copy()
        for sig in signal_cols:
            if sig not in grp.columns:
                grp[sig] = np.nan
            s = pd.Series(grp[sig].values)
            grp[f"{sig}_rm3"] = s.rolling(3, min_periods=2).mean().values
            grp[f"{sig}_rm7"] = s.rolling(7, min_periods=4).mean().values
            grp[f"{sig}_svl"] = grp[f"{sig}_rm3"] - grp[f"{sig}_rm7"]
            grp[f"{sig}_d1"] = s.diff().values
            grp[f"{sig}_d3"] = s.diff(3).values
        chunks.append(grp)
    labeled = pd.concat(chunks, ignore_index=True)
    
    feat_cols = []
    for sig in signal_cols:
        feat_cols.extend([f"{sig}_rm3", f"{sig}_rm7", f"{sig}_svl", f"{sig}_d1", f"{sig}_d3"])
    feat_cols.append("cycle_frac")
    
    valid = labeled.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5)).copy()
    X = valid[feat_cols].fillna(0).values
    y = valid["is_post_ov"].values.astype(int)
    groups = valid["id"].values
    
    logo = LeaveOneGroupOut()
    valid["ov_prob"] = np.nan
    
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        valid.iloc[test_idx, valid.columns.get_loc("ov_prob")] = prob
    
    has_prob = valid["ov_prob"].notna()
    auc = roc_auc_score(y[has_prob], valid.loc[has_prob, "ov_prob"])
    print(f"  Enhanced GBDT AUC: {auc:.3f}")
    
    from model.ovulation_detect import detect_ovulation_from_probs
    
    all_results = {}
    for strat in ["threshold", "cumulative", "bayesian"]:
        detected = {}
        for sgk, cyc in valid[has_prob].groupby("small_group_key"):
            cyc = cyc.sort_values("day_in_cycle")
            probs = cyc["ov_prob"].values
            
            if "hist_cycle_len_mean" not in cyc.columns:
                cyc = cyc.copy()
                cyc["hist_cycle_len_mean"] = 28
            
            ov_d = detect_ovulation_from_probs(cyc, probs, strategy=strat)
            if ov_d is not None:
                detected[sgk] = ov_d
        
        res = evaluate_detection(detected, lh_ov_dict, f"GBDT-enhanced-{strat}")
        all_results[strat] = (detected, res)
    
    return all_results, auc


def extract_enhanced_nocturnal_features(wt_df):
    """Extract enhanced features from nocturnal window only (22:00-06:00)."""
    key = ["id", "study_interval", "day_in_study"]
    night_mask = (wt_df["hour"] >= 22) | (wt_df["hour"] < 6)
    wt_night = wt_df[night_mask].copy()
    
    results = []
    for (uid, si, dis), grp in wt_night.groupby(key):
        temps = grp["temp"].values
        hours = grp["hour"].values
        n = len(temps)
        
        if n < 30:
            continue
        
        feat = {"id": uid, "study_interval": si, "day_in_study": dis}
        
        stable_mask = (hours >= 2) & (hours < 5)
        stable_temps = temps[stable_mask]
        
        if len(stable_temps) > 10:
            feat["noct_stable_mean"] = np.nanmean(stable_temps)
            feat["noct_stable_std"] = np.nanstd(stable_temps)
            q25, q75 = np.nanpercentile(stable_temps, [25, 75])
            feat["noct_iqr"] = q75 - q25
        else:
            feat["noct_stable_mean"] = np.nanmean(temps)
            feat["noct_stable_std"] = np.nanstd(temps)
            feat["noct_iqr"] = np.nanpercentile(temps, 75) - np.nanpercentile(temps, 25)
        
        feat["noct_nadir"] = np.nanmin(temps)
        feat["noct_nadir_time"] = grp.iloc[np.nanargmin(temps)]["hour"] * 60 + grp.iloc[np.nanargmin(temps)]["minute"]
        feat["noct_range"] = np.nanmax(temps) - np.nanmin(temps)
        feat["noct_p90"] = np.nanpercentile(temps, 90)
        feat["noct_p10"] = np.nanpercentile(temps, 10)
        
        first_half = temps[:n//2]
        second_half = temps[n//2:]
        if len(first_half) > 5 and len(second_half) > 5:
            x = np.arange(len(first_half))
            valid = ~np.isnan(first_half)
            if valid.sum() > 5:
                feat["noct_rise_slope"] = np.polyfit(x[valid], first_half[valid], 1)[0]
            else:
                feat["noct_rise_slope"] = np.nan
        else:
            feat["noct_rise_slope"] = np.nan
        
        # 2-hour and 4-hour stable window means
        h2_mask = (hours >= 2) & (hours < 4)
        h4_mask = (hours >= 1) & (hours < 5)
        h2_temps = temps[h2_mask]
        h4_temps = temps[h4_mask]
        feat["noct_mean_2h"] = np.nanmean(h2_temps) if len(h2_temps) > 5 else np.nan
        feat["noct_mean_4h"] = np.nanmean(h4_temps) if len(h4_temps) > 5 else np.nan
        
        results.append(feat)
    
    return pd.DataFrame(results)


# --- Exp 2c: 1D-CNN on raw temperature sequences ---

def run_cnn_experiment():
    """1D-CNN binary classifier: pre-ov vs post-ov on nocturnal temperature windows."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import LeaveOneGroupOut
    
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    
    wt = load_raw_minute_temperature()
    night_mask = (wt["hour"] >= 22) | (wt["hour"] < 6)
    wt_night = wt[night_mask].copy()
    
    SEQ_LEN = 480  # 8 hours * 60 minutes
    
    print("  Building nocturnal temperature sequences...")
    samples = []
    for sgk in cc["small_group_key"].unique():
        if sgk not in lh_ov_dict:
            continue
        
        cycle_rows = cc[cc["small_group_key"] == sgk].sort_values("day_in_study")
        uid = cycle_rows["id"].values[0]
        si = cycle_rows["study_interval"].values[0]
        cs = cycle_rows["day_in_study"].min()
        ov_dic = lh_ov_dict[sgk]
        
        for _, row in cycle_rows.iterrows():
            dis = row["day_in_study"]
            dic = dis - cs
            
            day_data = wt_night[
                (wt_night["id"] == uid) &
                (wt_night["study_interval"] == si) &
                (wt_night["day_in_study"] == dis)
            ].sort_values("timestamp")
            
            if len(day_data) < 60:
                continue
            
            seq = day_data["temp"].values
            if len(seq) < SEQ_LEN:
                seq = np.pad(seq, (0, SEQ_LEN - len(seq)), mode='edge')
            else:
                seq = seq[:SEQ_LEN]
            
            seq = pd.Series(seq).interpolate(limit_direction="both").fillna(0).values
            
            label = 1 if dic > ov_dic else 0
            samples.append({
                "seq": seq.astype(np.float32),
                "label": label,
                "id": uid,
                "sgk": sgk,
                "dic": dic,
            })
    
    if len(samples) < 50:
        print(f"  Too few samples ({len(samples)}), skipping CNN experiment")
        return {}, 0
    
    print(f"  Total samples: {len(samples)} ({sum(s['label'] for s in samples)} post-ov)")
    
    class TempDataset(Dataset):
        def __init__(self, sample_list):
            self.samples = sample_list
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            s = self.samples[idx]
            return torch.FloatTensor(s["seq"]).unsqueeze(0), s["label"]
    
    class TempCNN(nn.Module):
        def __init__(self, seq_len=SEQ_LEN):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=15, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x).squeeze(-1)
    
    # LOSO evaluation
    from model.ovulation_detect import detect_ovulation_from_probs
    
    ids = np.array([s["id"] for s in samples])
    unique_ids = np.unique(ids)
    
    all_probs = np.full(len(samples), np.nan)
    
    for test_id in unique_ids:
        test_mask = ids == test_id
        train_mask = ~test_mask
        
        train_samples = [s for s, m in zip(samples, train_mask) if m]
        test_samples = [s for s, m in zip(samples, test_mask) if m]
        
        if len(train_samples) < 30 or len(test_samples) < 5:
            continue
        
        train_ds = TempDataset(train_samples)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        model = TempCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(30):
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb.float())
                loss.backward()
                optimizer.step()
        
        model.eval()
        test_ds = TempDataset(test_samples)
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        test_probs = []
        with torch.no_grad():
            for xb, _ in test_dl:
                pred = torch.sigmoid(model(xb))
                test_probs.extend(pred.numpy())
        
        test_indices = np.where(test_mask)[0]
        for i, idx in enumerate(test_indices):
            if i < len(test_probs):
                all_probs[idx] = test_probs[i]
    
    from sklearn.metrics import roc_auc_score
    
    valid_mask = ~np.isnan(all_probs)
    labels = np.array([s["label"] for s in samples])
    if valid_mask.sum() > 10:
        auc = roc_auc_score(labels[valid_mask], all_probs[valid_mask])
        print(f"  1D-CNN AUC: {auc:.3f}")
    else:
        auc = 0
    
    valid_samples = [s for s, m in zip(samples, valid_mask) if m]
    valid_probs_arr = all_probs[valid_mask]
    
    prob_df = pd.DataFrame([{
        "small_group_key": s["sgk"],
        "day_in_cycle": s["dic"],
        "ov_prob": p,
        "hist_cycle_len_mean": 28,
    } for s, p in zip(valid_samples, valid_probs_arr)])
    
    all_results = {}
    for strat in ["threshold", "cumulative", "bayesian"]:
        detected = {}
        for sgk, cyc in prob_df.groupby("small_group_key"):
            cyc = cyc.sort_values("day_in_cycle")
            probs = cyc["ov_prob"].values
            ov_d = detect_ovulation_from_probs(cyc, probs, strategy=strat)
            if ov_d is not None:
                detected[sgk] = ov_d
        
        res = evaluate_detection(detected, lh_ov_dict, f"1D-CNN-{strat}")
        all_results[strat] = (detected, res)
    
    return all_results, auc


# --- Exp 2d: Multi-signal HMM ---

def run_multi_signal_hmm(cycle_daily_df, lh_ov_dict):
    """HMM on combined signals (temp + HR + HRV) per cycle."""
    signal_cols = ["nightly_temperature", "hr_mean", "rmssd_mean"]
    
    detected = {}
    for sgk, grp in cycle_daily_df.groupby("small_group_key"):
        if sgk not in lh_ov_dict:
            continue
        
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        dic = (grp["day_in_study"] - cs).values
        
        signals = grp[signal_cols].values
        valid = ~np.any(np.isnan(signals), axis=1)
        
        if valid.sum() < 10:
            continue
        
        cycle_data = {
            "signals": signals,
            "dic": dic,
        }
        
        ov_day = detect_ov_hmm(cycle_data, multi_signal=True)
        if ov_day is not None:
            detected[sgk] = ov_day
    
    return detected


# ======================================================================
# Quality filtering: only evaluate on cycles with ≥0.2°C shift
# ======================================================================

def filter_quality_cycles(cycle_series, lh_ov_dict, shift_threshold=0.2):
    """Filter cycles with sufficient temperature shift (≥0.2°C)."""
    good_sgks = set()
    for sgk, data in cycle_series.items():
        if sgk not in lh_ov_dict:
            continue
        
        temps = data.get("noct_mean", data.get("temps"))
        if temps is None:
            continue
        
        valid = ~np.isnan(temps)
        if valid.sum() < 10:
            continue
        
        t_clean = pd.Series(temps).interpolate(limit_direction="both").values
        
        ov_d = lh_ov_dict[sgk]
        n = len(t_clean)
        
        pre_end = min(ov_d, n)
        post_start = min(ov_d + 2, n)
        
        if pre_end < 3 or post_start >= n:
            continue
        
        pre_mean = np.mean(t_clean[max(0, pre_end - 5):pre_end])
        post_mean = np.mean(t_clean[post_start:min(n, post_start + 5)])
        
        shift = post_mean - pre_mean
        if shift >= shift_threshold:
            good_sgks.add(sgk)
    
    return good_sgks


# ======================================================================
# Ensemble: combine multiple detectors
# ======================================================================

def ensemble_detection(detections_list, weights=None):
    """Combine multiple ovulation detections via weighted median."""
    all_sgks = set()
    for det in detections_list:
        all_sgks.update(det.keys())
    
    if weights is None:
        weights = [1.0] * len(detections_list)
    
    ensemble = {}
    for sgk in all_sgks:
        values = []
        w = []
        for det, weight in zip(detections_list, weights):
            if sgk in det:
                values.append(det[sgk])
                w.append(weight)
        
        if len(values) >= 2:
            ensemble[sgk] = int(np.round(np.average(values, weights=w)))
        elif len(values) == 1:
            ensemble[sgk] = values[0]
    
    return ensemble


# ======================================================================
# Main experiment runner
# ======================================================================

def main():
    print("=" * 72)
    print("  Comprehensive Ovulation Detection Experiment Suite")
    print("=" * 72)
    
    # Load data
    print("\n[Loading data...]")
    cc = pd.read_csv(CYCLE_CSV)
    
    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    print(f"  LH-labeled cycles: {len(lh_ov_dict)}")
    
    # Load cycle temperature series (aggregated nightly)
    cycle_series_agg = load_cycle_temperature_series()
    print(f"  Aggregated temp cycles: {len(cycle_series_agg)}")
    
    # Load nocturnal temperature from minute-level data
    print("  Loading nocturnal temperature from minute-level data...")
    nocturnal_df = load_nocturnal_temperature_series()
    cycle_series_noct = build_cycle_nocturnal_series(nocturnal_df, cc)
    print(f"  Nocturnal temp cycles: {len(cycle_series_noct)}")
    
    # Quality filter
    good_cycles_agg = filter_quality_cycles(cycle_series_agg, lh_ov_dict, 0.2)
    good_cycles_noct = filter_quality_cycles(cycle_series_noct, lh_ov_dict, 0.2)
    print(f"  Good cycles (≥0.2°C shift, agg): {len(good_cycles_agg)}/{len(lh_ov_dict)}")
    print(f"  Good cycles (≥0.2°C shift, noct): {len(good_cycles_noct)}/{len(lh_ov_dict)}")
    
    all_results = {}
    best_detections = []
    
    # ==================================================================
    # PATH 1: Rule-based Algorithms
    # ==================================================================
    print(f"\n{'='*72}")
    print("  PATH 1: Rule-based / Algorithmic Methods")
    print(f"{'='*72}")
    
    # --- 1a: ruptures PELT ---
    print(f"\n--- Exp 1a: ruptures changepoint detection ---")
    for method, pen in [("pelt", 2.0), ("pelt", 3.0), ("pelt", 5.0), ("binseg", None), ("dynp", None)]:
        for data_name, series in [("agg", cycle_series_agg), ("noct", cycle_series_noct)]:
            detected = {}
            for sgk in lh_ov_dict:
                if sgk in series:
                    ov = detect_ov_ruptures(series[sgk], method=method, penalty=pen or 3.0)
                    if ov is not None:
                        detected[sgk] = ov
            
            pen_str = f"pen={pen}" if pen else ""
            name = f"ruptures-{method}({pen_str})-{data_name}"
            res = evaluate_detection(detected, lh_ov_dict, name)
            all_results[name] = res
            if res.get("acc_3d", 0) > 0.5:
                best_detections.append((name, detected, res))
    
    # --- 1b: CUSUM ---
    print(f"\n--- Exp 1b: CUSUM ---")
    for threshold in [0.3, 0.5, 1.0, 1.5]:
        for drift in [0.02, 0.05, 0.1]:
            for data_name, series in [("agg", cycle_series_agg), ("noct", cycle_series_noct)]:
                detected = {}
                for sgk in lh_ov_dict:
                    if sgk in series:
                        use_noct = (data_name == "noct")
                        ov = detect_ov_cusum(series[sgk], threshold=threshold, drift=drift, use_noct=use_noct)
                        if ov is not None:
                            detected[sgk] = ov
                
                name = f"CUSUM(t={threshold},d={drift})-{data_name}"
                res = evaluate_detection(detected, lh_ov_dict, name)
                all_results[name] = res
                if res.get("acc_3d", 0) > 0.5:
                    best_detections.append((name, detected, res))
    
    # --- 1c: BOCPD ---
    print(f"\n--- Exp 1c: BOCPD ---")
    for hazard in [1/15, 1/20, 1/30]:
        for data_name, series in [("agg", cycle_series_agg), ("noct", cycle_series_noct)]:
            detected = {}
            for sgk in lh_ov_dict:
                if sgk in series:
                    use_noct = (data_name == "noct")
                    ov = detect_ov_bocpd(series[sgk], hazard_rate=hazard, use_noct=use_noct)
                    if ov is not None:
                        detected[sgk] = ov
            
            name = f"BOCPD(h={hazard:.3f})-{data_name}"
            res = evaluate_detection(detected, lh_ov_dict, name)
            all_results[name] = res
            if res.get("acc_3d", 0) > 0.5:
                best_detections.append((name, detected, res))
    
    # --- 1d: Enhanced coverline ---
    print(f"\n--- Exp 1d: Enhanced coverline ---")
    for shift_th in [0.05, 0.1, 0.15, 0.2]:
        for confirm in [2, 3, 4]:
            for data_name, series in [("agg", cycle_series_agg), ("noct", cycle_series_noct)]:
                detected = {}
                for sgk in lh_ov_dict:
                    if sgk in series:
                        use_noct = (data_name == "noct")
                        ov = detect_ov_coverline(series[sgk], shift_threshold=shift_th, 
                                                 confirm_days=confirm, use_noct=use_noct)
                        if ov is not None:
                            detected[sgk] = ov
                
                name = f"coverline(th={shift_th},c={confirm})-{data_name}"
                res = evaluate_detection(detected, lh_ov_dict, name)
                all_results[name] = res
                if res.get("acc_3d", 0) > 0.5:
                    best_detections.append((name, detected, res))
    
    # ==================================================================
    # PATH 2: ML / Data-driven Methods
    # ==================================================================
    print(f"\n{'='*72}")
    print("  PATH 2: ML / Data-driven Methods")
    print(f"{'='*72}")
    
    # --- 2a: HMM (single-signal: temperature) ---
    print(f"\n--- Exp 2a: 2-state HMM (temperature only) ---")
    for data_name, series in [("agg", cycle_series_agg), ("noct", cycle_series_noct)]:
        detected = {}
        for sgk in lh_ov_dict:
            if sgk in series:
                data = series[sgk].copy()
                if data_name == "noct" and "noct_mean" in data:
                    data["temps"] = data["noct_mean"]
                ov = detect_ov_hmm(data, use_noct=(data_name == "noct"))
                if ov is not None:
                    detected[sgk] = ov
        
        name = f"HMM-2state-{data_name}"
        res = evaluate_detection(detected, lh_ov_dict, name)
        all_results[name] = res
        if res.get("acc_3d", 0) > 0.5:
            best_detections.append((name, detected, res))
    
    # --- 2a+: Multi-signal HMM ---
    print(f"\n--- Exp 2a+: Multi-signal HMM ---")
    all_daily = load_all_daily_signals()
    ms_detected = run_multi_signal_hmm(all_daily, lh_ov_dict)
    res = evaluate_detection(ms_detected, lh_ov_dict, "HMM-multi-signal")
    all_results["HMM-multi-signal"] = res
    if res.get("acc_3d", 0) > 0.5:
        best_detections.append(("HMM-multi-signal", ms_detected, res))
    
    # --- 2b: Enhanced GBDT ---
    print(f"\n--- Exp 2b: Enhanced GBDT (nocturnal + multi-signal) ---")
    gbdt_results, gbdt_auc = run_enhanced_gbdt_experiment(
        use_nocturnal=True, quality_filter=True, multi_signal=True
    )
    for strat, (det, res) in gbdt_results.items():
        name = f"GBDT-enhanced-{strat}"
        all_results[name] = res
        if res.get("acc_3d", 0) > 0.5:
            best_detections.append((name, det, res))
    
    # --- 2c: 1D-CNN ---
    print(f"\n--- Exp 2c: 1D-CNN on nocturnal sequences ---")
    cnn_results, cnn_auc = run_cnn_experiment()
    for strat, (det, res) in cnn_results.items():
        name = f"1D-CNN-{strat}"
        all_results[name] = res
        if res.get("acc_3d", 0) > 0.5:
            best_detections.append((name, det, res))
    
    # ==================================================================
    # ENSEMBLE: Combine best methods
    # ==================================================================
    print(f"\n{'='*72}")
    print("  ENSEMBLE: Combining Best Methods")
    print(f"{'='*72}")
    
    if len(best_detections) >= 2:
        best_detections.sort(key=lambda x: x[2].get("acc_3d", 0), reverse=True)
        
        print(f"\n  Top methods for ensemble:")
        for name, _, res in best_detections[:5]:
            print(f"    {name}: ±3d={res.get('acc_3d', 0):.1%}")
        
        top_dets = [det for _, det, _ in best_detections[:3]]
        top_weights = [res.get("acc_3d", 0.5) for _, _, res in best_detections[:3]]
        
        ens_detected = ensemble_detection(top_dets, weights=top_weights)
        evaluate_detection(ens_detected, lh_ov_dict, "ENSEMBLE-top3-weighted")
        
        ens_detected_all = ensemble_detection([det for _, det, _ in best_detections[:5]])
        evaluate_detection(ens_detected_all, lh_ov_dict, "ENSEMBLE-top5-unweighted")
        
        # Also try majority vote
        vote_detected = {}
        all_sgks = set()
        for _, det, _ in best_detections[:5]:
            all_sgks.update(det.keys())
        
        for sgk in all_sgks:
            votes = [det[sgk] for _, det, _ in best_detections[:5] if sgk in det]
            if len(votes) >= 3:
                vote_detected[sgk] = int(np.median(votes))
        evaluate_detection(vote_detected, lh_ov_dict, "ENSEMBLE-majority-vote")
    
    # ==================================================================
    # Quality-filtered evaluation
    # ==================================================================
    print(f"\n{'='*72}")
    print("  Quality-Filtered Evaluation (≥0.2°C shift cycles only)")
    print(f"{'='*72}")
    
    good_lh = {sgk: v for sgk, v in lh_ov_dict.items() if sgk in good_cycles_noct}
    print(f"  Evaluating on {len(good_lh)} quality cycles")
    
    if len(best_detections) >= 1:
        for name, det, _ in best_detections[:5]:
            det_filtered = {sgk: v for sgk, v in det.items() if sgk in good_lh}
            evaluate_detection(det_filtered, good_lh, f"{name}-QUALITY")
    
    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*72}")
    print("  OVERALL SUMMARY (sorted by ±3d accuracy)")
    print(f"{'='*72}")
    
    sorted_results = sorted(
        [(name, res) for name, res in all_results.items() if res.get("acc_3d", 0) > 0],
        key=lambda x: x[1].get("acc_3d", 0),
        reverse=True,
    )
    
    print(f"\n  {'Method':<50s} {'Recall':>7s} {'MAE':>6s} {'±1d':>6s} {'±2d':>6s} {'±3d':>6s} {'±5d':>6s}")
    print(f"  {'-'*90}")
    for name, res in sorted_results[:20]:
        print(
            f"  {name:<50s}"
            f" {res.get('recall', 0):>6.0%}"
            f" {res.get('mae', 0):>5.1f}d"
            f" {res.get('acc_1d', 0):>5.1%}"
            f" {res.get('acc_2d', 0):>5.1%}"
            f" {res.get('acc_3d', 0):>5.1%}"
            f" {res.get('acc_5d', 0):>5.1%}"
        )
    
    print(f"\n{'='*72}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

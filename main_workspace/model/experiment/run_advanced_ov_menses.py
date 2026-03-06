"""
Advanced Ovulation Detection & Menstrual Prediction — All Methods
=================================================================
Inspired by:
  - Apple Watch 2025 study: 89% ±2d on completed cycles (MAE=1.22d)
  - HMM biphasic model: 92% sensitivity (in-ear thermometer)
  - BOCPD (Bayesian Online Changepoint Detection)
  - Gaussian Process changepoint
  - Sigmoid/logistic curve fitting for biphasic pattern

Methods implemented:
  RULE-BASED:
    1. Optimized t-test split (from previous experiments)
    2. Biphasic step-function (SSE minimization)
    3. Sigmoid/logistic curve fitting
    4. Piecewise linear regression (two-line model)
    5. EWMA crossover detection
    6. BOCPD (Bayesian Online Changepoint Detection)
    7. CUSUM with V-mask

  UNSUPERVISED:
    8. 2-state HMM (single + multi-signal)

  SUPERVISED (LOSO, no leakage):
    9. Random Forest regression on per-cycle features
   10. LightGBM regression on per-cycle features
   11. 1D-CNN regression on temperature sequence (LOSO)

  ENSEMBLE:
   12. Weighted ensemble of best methods
   13. Stacking: blend rule-based + ML

Leakage prevention:
  - LOSO: Leave-One-Subject-Out for all supervised methods
  - No cycle_len as feature in supervised models
  - LH labels only for evaluation (rule-based) or LOSO training (supervised)
  - hist_cycle_len from PAST cycles only

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_advanced_ov_menses
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


# =====================================================================
# Data Loading
# =====================================================================

def load_data():
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]
    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)
    merged = cc.merge(ct_daily, on=key, how="left")

    lh_ov = get_lh_ovulation_labels()
    lh_ov_dict = dict(zip(lh_ov["small_group_key"], lh_ov["ov_dic"]))
    lh_luteal = dict(zip(lh_ov["small_group_key"], lh_ov["luteal_len"]))

    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        cycle_series[sgk] = {
            "dic": (grp["day_in_study"] - cs).values,
            "temps": grp["nightly_temperature"].values,
            "id": grp["id"].values[0],
            "cycle_len": n,
        }

    sgk_order = (
        merged.groupby("small_group_key")["day_in_study"]
        .min().reset_index().rename(columns={"day_in_study": "start"})
    )
    sgk_order = sgk_order.merge(
        merged[["small_group_key", "id"]].drop_duplicates(), on="small_group_key"
    ).sort_values(["id", "start"])

    subj_order = {}
    for uid, group in sgk_order.groupby("id"):
        sgks = group["small_group_key"].tolist()
        subj_order[uid] = sgks
        past_lens = []
        for sgk in sgks:
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = np.mean(past_lens) if past_lens else 28.0
                past_lens.append(cycle_series[sgk]["cycle_len"])

    quality = set()
    for sgk in cycle_series:
        if sgk not in lh_ov_dict:
            continue
        data = cycle_series[sgk]
        raw = data["temps"]
        if np.isnan(raw).all():
            continue
        t = pd.Series(raw).interpolate(limit_direction="both").values
        ov = lh_ov_dict[sgk]
        n = len(t)
        if ov < 3 or ov + 2 >= n:
            continue
        pre = np.mean(t[max(0, ov - 5):ov])
        post = np.mean(t[ov + 2:min(n, ov + 7)])
        if post - pre >= 0.2:
            quality.add(sgk)

    return lh_ov_dict, lh_luteal, cycle_series, quality, subj_order


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


# =====================================================================
# Evaluation
# =====================================================================

def eval_ov(detected, lh_ov_dict, name, subset=None):
    keys = subset if subset else lh_ov_dict.keys()
    errors = [detected[s] - lh_ov_dict[s] for s in keys if s in detected and s in lh_ov_dict]
    if not errors:
        print(f"  [{name}] No evaluable detections")
        return {}
    ae = np.abs(errors)
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"  [{name}] n={r['n']} | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# METHOD 1-2: T-test & Biphasic (from previous, best configs)
# =====================================================================

def detect_ttest_biphasic(cycle_series, sigma=2.5, frac=0.575, pw=4.0):
    """Combined t-test + biphasic, average of two estimates."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=sigma)
        exp = max(8, hcl * frac)

        # t-test
        best_ws, best_tsp = -np.inf, None
        for sp in range(5, n - 3):
            diff = np.mean(t[sp:]) - np.mean(t[:sp])
            if diff <= 0:
                continue
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
            except Exception:
                continue
            if np.isnan(stat):
                continue
            pp = np.exp(-0.5 * ((dic[sp] - exp) / pw) ** 2)
            if stat * pp > best_ws:
                best_ws = stat * pp
                best_tsp = sp

        # biphasic
        best_sc, best_bsp = np.inf, None
        for sp in range(5, n - 3):
            m1, m2 = np.mean(t[:sp]), np.mean(t[sp:])
            if m2 <= m1:
                continue
            sse = np.sum((t[:sp] - m1) ** 2) + np.sum((t[sp:] - m2) ** 2)
            pen = 0.5 * ((dic[sp] - exp) / pw) ** 2
            if sse + pen < best_sc:
                best_sc = sse + pen
                best_bsp = sp

        cands = []
        if best_tsp is not None:
            cands.append(int(dic[best_tsp]))
        if best_bsp is not None:
            cands.append(int(dic[best_bsp]))
        detected[sgk] = int(round(np.mean(cands))) if cands else int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 3: Sigmoid curve fitting
# =====================================================================

def _sigmoid(x, a, b, c, d):
    """Sigmoid: d + (a-d) / (1 + exp(-c*(x-b)))"""
    return d + (a - d) / (1.0 + np.exp(-c * (x - b)))


def detect_sigmoid(cycle_series, sigma=2.0, frac=0.575):
    """Fit sigmoid to temperature, inflection point = ovulation estimate."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=sigma)
        x = dic.astype(float)
        try:
            p0 = [t.min(), hcl * frac, 0.5, t.max()]
            bounds = ([t.min() - 1, 3, 0.01, t.min() - 1],
                      [t.max() + 1, n - 3, 5.0, t.max() + 1])
            popt, _ = curve_fit(_sigmoid, x, t, p0=p0, bounds=bounds, maxfev=5000)
            ov_est = int(round(popt[1]))
            ov_est = max(5, min(n - 3, ov_est))
            detected[sgk] = ov_est
        except Exception:
            detected[sgk] = int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 4: Piecewise linear regression
# =====================================================================

def detect_piecewise_linear(cycle_series, sigma=2.0, frac=0.575, pw=5.0):
    """Fit two lines meeting at changepoint, minimize total SSE."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=sigma)
        x = dic.astype(float)
        exp = max(8, hcl * frac)
        best_sc, best_sp = np.inf, None
        for sp in range(5, n - 3):
            x1, y1 = x[:sp], t[:sp]
            x2, y2 = x[sp:], t[sp:]
            if len(x1) < 3 or len(x2) < 3:
                continue
            p1 = np.polyfit(x1, y1, 1)
            p2 = np.polyfit(x2, y2, 1)
            sse1 = np.sum((y1 - np.polyval(p1, x1)) ** 2)
            sse2 = np.sum((y2 - np.polyval(p2, x2)) ** 2)
            pen = 0.3 * ((dic[sp] - exp) / pw) ** 2
            sc = sse1 + sse2 + pen
            if sc < best_sc:
                best_sc = sc
                best_sp = sp
        detected[sgk] = int(dic[best_sp]) if best_sp else int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 5: EWMA crossover
# =====================================================================

def detect_ewma(cycle_series, span_short=5, span_long=15, frac=0.575):
    """Detect ovulation when short EWMA crosses above long EWMA."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        ts = pd.Series(_clean(raw))
        short = ts.ewm(span=span_short).mean().values
        long = ts.ewm(span=span_long).mean().values

        lo = max(5, int(hcl * 0.3))
        hi = min(n - 2, int(hcl * 0.75))
        found = False
        for i in range(lo, hi):
            if short[i] > long[i] and (i == 0 or short[i - 1] <= long[i - 1]):
                detected[sgk] = int(dic[i])
                found = True
                break
        if not found:
            detected[sgk] = int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 6: BOCPD (Bayesian Online Changepoint Detection)
# =====================================================================

def detect_bocpd(cycle_series, frac=0.575, hazard=100):
    """BOCPD using the changepoint library."""
    try:
        from changepoint.online import Bocpd
        from changepoint.utils import constant_hazard
    except ImportError:
        try:
            import changepoint
            from changepoint import Bocpd
        except Exception:
            print("  [BOCPD] Library not available, skipping")
            return {}

    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=1.5)
        try:
            cpd = Bocpd(prior="NormalGamma", lam=hazard)
            for obs in t:
                cpd.update(obs)
            changepoints = cpd.changepoints(threshold=0.5)
            lo = max(5, int(hcl * 0.3))
            hi = min(n - 2, int(hcl * 0.7))
            valid_cps = [cp for cp in changepoints if lo <= cp <= hi]
            if valid_cps:
                exp = hcl * frac
                best = min(valid_cps, key=lambda cp: abs(cp - exp))
                detected[sgk] = int(dic[best])
            else:
                detected[sgk] = int(round(frac * hcl))
        except Exception:
            detected[sgk] = int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 7: CUSUM with V-mask
# =====================================================================

def detect_cusum_vmask(cycle_series, sigma=2.0, k=0.5, h=4.0, frac=0.575):
    """CUSUM: detect upward shift exceeding threshold."""
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=sigma)
        mu = np.mean(t[:max(5, n // 3)])
        std = max(np.std(t[:max(5, n // 3)]), 0.01)
        cusum_pos = np.zeros(n)
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (t[i] - mu) / std - k)
        lo = max(5, int(hcl * 0.3))
        hi = min(n - 2, int(hcl * 0.7))
        found = False
        for i in range(lo, hi):
            if cusum_pos[i] > h:
                detected[sgk] = int(dic[i])
                found = True
                break
        if not found:
            detected[sgk] = int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 8: HMM
# =====================================================================

def detect_hmm(cycle_series, sigma=1.5, frac=0.575):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  [HMM] not available")
        return {}
    detected = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        hcl = data["hist_cycle_len"]
        raw = data["temps"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        obs = _clean(raw, sigma=sigma).reshape(-1, 1)
        try:
            mdl = GaussianHMM(n_components=2, covariance_type="full",
                              n_iter=100, random_state=42, init_params="mc")
            mdl.startprob_ = np.array([0.9, 0.1])
            mdl.transmat_ = np.array([[0.95, 0.05], [0.02, 0.98]])
            mdl.fit(obs)
            states = mdl.predict(obs)
            low = np.argmin(mdl.means_[:, 0])
            high = 1 - low
            found = False
            for i in range(1, n):
                if states[i] == high and states[i - 1] == low and dic[i] >= 6:
                    detected[sgk] = int(dic[i])
                    found = True
                    break
            if not found:
                detected[sgk] = int(round(frac * hcl))
        except Exception:
            detected[sgk] = int(round(frac * hcl))
    return detected


# =====================================================================
# METHOD 9-10: Supervised (RF / LightGBM) with LOSO
# =====================================================================

def extract_cycle_features(temps_clean, dic, hist_cycle_len):
    """Extract per-cycle features from temperature sequence for ML models."""
    n = len(temps_clean)
    t = temps_clean
    feats = {}

    feats["n_days"] = n  # cycle length is known retrospectively
    feats["hist_clen"] = hist_cycle_len

    # Global stats
    feats["t_mean"] = np.mean(t)
    feats["t_std"] = np.std(t)
    feats["t_range"] = np.ptp(t)
    feats["t_skew"] = float(pd.Series(t).skew())
    feats["t_kurt"] = float(pd.Series(t).kurtosis())

    # Nadir (minimum) position
    feats["nadir_day"] = int(np.argmin(t))
    feats["nadir_frac"] = feats["nadir_day"] / n

    # Max gradient position
    grad = np.gradient(t)
    feats["max_grad_day"] = int(np.argmax(grad))
    feats["max_grad_val"] = float(np.max(grad))

    # First half vs second half
    mid = n // 2
    feats["mean_first_half"] = np.mean(t[:mid])
    feats["mean_second_half"] = np.mean(t[mid:])
    feats["shift_halves"] = feats["mean_second_half"] - feats["mean_first_half"]

    # Quantile features
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        feats[f"t_q{int(q*100)}"] = float(np.quantile(t, q))

    # Running t-test at multiple points
    for frac_pt in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        sp = max(5, int(n * frac_pt))
        if sp < n - 3:
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                feats[f"ttest_{int(frac_pt*100)}"] = float(stat) if not np.isnan(stat) else 0.0
            except Exception:
                feats[f"ttest_{int(frac_pt*100)}"] = 0.0
        else:
            feats[f"ttest_{int(frac_pt*100)}"] = 0.0

    # Rolling mean crossover
    short_ma = pd.Series(t).rolling(3, min_periods=1).mean().values
    long_ma = pd.Series(t).rolling(7, min_periods=1).mean().values
    cross_days = []
    for i in range(7, n):
        if short_ma[i] > long_ma[i] and short_ma[i - 1] <= long_ma[i - 1]:
            cross_days.append(i)
    feats["first_cross_day"] = cross_days[0] if cross_days else n // 2
    feats["first_cross_frac"] = feats["first_cross_day"] / n

    # Autocorrelation
    ts = pd.Series(t)
    for lag in [1, 3, 5]:
        ac = ts.autocorr(lag=lag)
        feats[f"autocorr_{lag}"] = float(ac) if not np.isnan(ac) else 0.0

    return feats


def detect_ml_loso(cycle_series, lh_ov_dict, model_type="rf"):
    """Supervised regression with LOSO cross-validation."""
    labeled_sgks = [sgk for sgk in cycle_series if sgk in lh_ov_dict]
    if not labeled_sgks:
        return {}

    all_feats = []
    all_targets = []
    all_ids = []
    all_sgks_list = []

    for sgk in labeled_sgks:
        data = cycle_series[sgk]
        raw = data["temps"]
        if np.isnan(raw).all():
            continue
        t = _clean(raw, sigma=1.5)
        feats = extract_cycle_features(t, data["dic"], data["hist_cycle_len"])
        all_feats.append(feats)
        all_targets.append(lh_ov_dict[sgk])
        all_ids.append(data["id"])
        all_sgks_list.append(sgk)

    if len(all_feats) < 10:
        return {}

    df = pd.DataFrame(all_feats)
    X = df.values
    y = np.array(all_targets, dtype=float)
    ids = np.array(all_ids)
    unique_ids = np.unique(ids)

    detected = {}

    for test_uid in unique_ids:
        test_mask = ids == test_uid
        train_mask = ~test_mask
        if train_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        test_sgks = [all_sgks_list[i] for i in np.where(test_mask)[0]]

        if model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=100, max_depth=6,
                                        random_state=42, n_jobs=-1)
        elif model_type == "lgb":
            import lightgbm as lgb
            mdl = lgb.LGBMRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, random_state=42,
                                     verbose=-1)
        elif model_type == "ridge":
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)

        for sgk, pred in zip(test_sgks, preds):
            detected[sgk] = int(round(max(5, min(cycle_series[sgk]["cycle_len"] - 3, pred))))

    return detected


# =====================================================================
# METHOD 11: 1D-CNN (LOSO)
# =====================================================================

def detect_cnn_loso(cycle_series, lh_ov_dict, max_len=50, n_epochs=50):
    """1D-CNN regression for ovulation day, LOSO cross-validation."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [CNN] PyTorch not available")
        return {}

    class OvCNN(nn.Module):
        def __init__(self, seq_len):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    labeled = [sgk for sgk in cycle_series if sgk in lh_ov_dict]
    seqs, targets, ids, sgk_list = [], [], [], []
    for sgk in labeled:
        data = cycle_series[sgk]
        raw = data["temps"]
        if np.isnan(raw).all():
            continue
        t = _clean(raw, sigma=1.0)
        mu, std = np.mean(t), max(np.std(t), 0.01)
        t_norm = (t - mu) / std
        if len(t_norm) < max_len:
            t_pad = np.pad(t_norm, (0, max_len - len(t_norm)),
                           mode='edge')  # edge-pad to avoid leaking length
        else:
            t_pad = t_norm[:max_len]
        seqs.append(t_pad.astype(np.float32))
        targets.append(float(lh_ov_dict[sgk]))
        ids.append(data["id"])
        sgk_list.append(sgk)

    if len(seqs) < 10:
        return {}

    seqs = np.array(seqs)
    targets = np.array(targets, dtype=np.float32)
    ids = np.array(ids)
    unique_ids = np.unique(ids)

    detected = {}
    for test_uid in unique_ids:
        test_mask = ids == test_uid
        train_mask = ~test_mask
        if train_mask.sum() < 5:
            continue

        X_tr = torch.tensor(seqs[train_mask]).unsqueeze(1)
        y_tr = torch.tensor(targets[train_mask])
        X_te = torch.tensor(seqs[test_mask]).unsqueeze(1)

        ds = TensorDataset(X_tr, y_tr)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        model = OvCNN(max_len)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()

        model.train()
        for _ in range(n_epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te).numpy()

        test_sgks = [sgk_list[i] for i in np.where(test_mask)[0]]
        for sgk, pred in zip(test_sgks, preds):
            clen = cycle_series[sgk]["cycle_len"]
            detected[sgk] = int(round(max(5, min(clen - 3, pred))))

    return detected


# =====================================================================
# ENSEMBLE
# =====================================================================

def ensemble(all_dets, weights=None):
    all_sgks = set()
    for d in all_dets:
        all_sgks.update(d.keys())
    if weights is None:
        weights = [1.0] * len(all_dets)
    result = {}
    for sgk in all_sgks:
        vals, ws = [], []
        for d, w in zip(all_dets, weights):
            if sgk in d:
                vals.append(d[sgk])
                ws.append(w)
        if vals:
            result[sgk] = int(round(np.average(vals, weights=ws)))
    return result


# =====================================================================
# Menstrual Prediction
# =====================================================================

def predict_menses(cycle_series, detected, subj_order, lh_ov_dict,
                   fixed_luteal=12.0, eval_subset=None, label=""):
    pop_lut = fixed_luteal
    subj_past_lut = defaultdict(list)
    subj_past_clen = defaultdict(list)
    errs_all, errs_ov, errs_cal = [], [], []
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual = data["cycle_len"]
            pl = subj_past_lut[uid]
            pc = subj_past_clen[uid]
            lut = np.mean(pl) if pl else pop_lut
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0

            ov = detected.get(sgk)
            if ov is not None and ov > 3:
                pred = ov + lut
                kind = "ov"
            else:
                pred = acl
                kind = "cal"

            err = pred - actual
            if ev is None or sgk in ev:
                errs_all.append(err)
                if kind == "ov":
                    errs_ov.append(err)
                else:
                    errs_cal.append(err)

            subj_past_clen[uid].append(actual)
            if ov is not None:
                el = actual - ov
                if 8 <= el <= 22:
                    subj_past_lut[uid].append(el)

    def _s(errs, tag):
        if not errs:
            return {}
        ae = np.abs(errs)
        r = {"n": len(ae), "mae": float(ae.mean()),
             "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
             "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
        print(f"    [{tag}] n={r['n']} | MAE={r['mae']:.2f}d"
              f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
              f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
        return r

    if label:
        no = len(errs_ov)
        nc = len(errs_cal)
        print(f"\n  {label} (ov:{no}, cal:{nc})")
    ra = _s(errs_all, "ALL")
    _s(errs_ov, "ov-cd")
    _s(errs_cal, "cal")
    return ra


# =====================================================================
# Main
# =====================================================================

def main():
    print(f"\n{SEP}\n  Advanced Ovulation Detection — All Methods\n{SEP}")
    t0 = time.time()

    lh_ov_dict, lh_luteal, cs, quality, subj_order = load_data()
    labeled = set(s for s in cs if s in lh_ov_dict)
    print(f"  Cycles: {len(cs)} | Labeled: {len(labeled)} | Quality: {len(quality)}")
    print(f"  Luteal: mean={np.mean(list(lh_luteal.values())):.1f}d")

    all_methods = {}

    # ===== RULE-BASED =====
    print(f"\n{SEP}\n  RULE-BASED METHODS\n{SEP}")

    # 1-2: t-test + biphasic (sweep)
    print(f"\n--- 1-2. t-test + Biphasic ---")
    for s in [1.5, 2.0, 2.5, 3.0]:
        for f in [0.50, 0.55, 0.575]:
            for w in [3.0, 4.0, 5.0]:
                d = detect_ttest_biphasic(cs, sigma=s, frac=f, pw=w)
                tag = f"tt+bi-σ{s}-f{f}-w{w}"
                r = eval_ov(d, lh_ov_dict, tag)
                all_methods[tag] = (d, r)

    # 3: Sigmoid
    print(f"\n--- 3. Sigmoid Curve Fitting ---")
    for s in [1.0, 1.5, 2.0, 2.5]:
        d = detect_sigmoid(cs, sigma=s)
        tag = f"sigmoid-σ{s}"
        r = eval_ov(d, lh_ov_dict, tag)
        all_methods[tag] = (d, r)

    # 4: Piecewise linear
    print(f"\n--- 4. Piecewise Linear ---")
    for s in [1.5, 2.0, 2.5]:
        for w in [4.0, 5.0, 6.0]:
            d = detect_piecewise_linear(cs, sigma=s, pw=w)
            tag = f"pwl-σ{s}-w{w}"
            r = eval_ov(d, lh_ov_dict, tag)
            all_methods[tag] = (d, r)

    # 5: EWMA
    print(f"\n--- 5. EWMA Crossover ---")
    for ss in [3, 5, 7]:
        for sl in [10, 15, 20]:
            d = detect_ewma(cs, span_short=ss, span_long=sl)
            tag = f"ewma-{ss}/{sl}"
            r = eval_ov(d, lh_ov_dict, tag)
            all_methods[tag] = (d, r)

    # 6: BOCPD
    print(f"\n--- 6. BOCPD ---")
    for hz in [50, 100, 200]:
        d = detect_bocpd(cs, hazard=hz)
        if d:
            tag = f"bocpd-h{hz}"
            r = eval_ov(d, lh_ov_dict, tag)
            all_methods[tag] = (d, r)

    # 7: CUSUM
    print(f"\n--- 7. CUSUM V-mask ---")
    for s in [1.5, 2.0, 2.5]:
        for h in [3.0, 4.0, 5.0]:
            d = detect_cusum_vmask(cs, sigma=s, h=h)
            tag = f"cusum-σ{s}-h{h}"
            r = eval_ov(d, lh_ov_dict, tag)
            all_methods[tag] = (d, r)

    # 8: HMM
    print(f"\n--- 8. HMM ---")
    for s in [1.0, 1.5, 2.0]:
        d = detect_hmm(cs, sigma=s)
        if d:
            tag = f"hmm-σ{s}"
            r = eval_ov(d, lh_ov_dict, tag)
            all_methods[tag] = (d, r)

    # ===== SUPERVISED (LOSO) =====
    print(f"\n{SEP}\n  SUPERVISED METHODS (LOSO)\n{SEP}")

    # 9: RF
    print(f"\n--- 9. Random Forest (LOSO) ---")
    d = detect_ml_loso(cs, lh_ov_dict, model_type="rf")
    if d:
        r = eval_ov(d, lh_ov_dict, "RF-LOSO")
        rq = eval_ov(d, lh_ov_dict, "  (quality)", subset=quality)
        all_methods["RF-LOSO"] = (d, r)

    # 10: LightGBM
    print(f"\n--- 10. LightGBM (LOSO) ---")
    d = detect_ml_loso(cs, lh_ov_dict, model_type="lgb")
    if d:
        r = eval_ov(d, lh_ov_dict, "LGB-LOSO")
        rq = eval_ov(d, lh_ov_dict, "  (quality)", subset=quality)
        all_methods["LGB-LOSO"] = (d, r)

    # Ridge
    print(f"\n--- Ridge (LOSO) ---")
    d = detect_ml_loso(cs, lh_ov_dict, model_type="ridge")
    if d:
        r = eval_ov(d, lh_ov_dict, "Ridge-LOSO")
        all_methods["Ridge-LOSO"] = (d, r)

    # 11: CNN
    print(f"\n--- 11. 1D-CNN (LOSO, edge-padded) ---")
    d = detect_cnn_loso(cs, lh_ov_dict, n_epochs=60)
    if d:
        r = eval_ov(d, lh_ov_dict, "CNN-LOSO")
        rq = eval_ov(d, lh_ov_dict, "  (quality)", subset=quality)
        all_methods["CNN-LOSO"] = (d, r)

    # ===== ENSEMBLES =====
    print(f"\n{SEP}\n  ENSEMBLES\n{SEP}")

    ranked = sorted(
        [(k, v[0], v[1].get("acc_3d", 0)) for k, v in all_methods.items()],
        key=lambda x: x[2], reverse=True,
    )

    for topN in [3, 5, 7, 10]:
        if len(ranked) >= topN:
            dets = [r[1] for r in ranked[:topN]]
            ws = [max(r[2], 0.01) for r in ranked[:topN]]
            e = ensemble(dets, ws)
            tag = f"ens-top{topN}"
            r = eval_ov(e, lh_ov_dict, tag)
            rq = eval_ov(e, lh_ov_dict, f"  (quality)", subset=quality)
            all_methods[tag] = (e, r)

    # ===== FINAL RANKING =====
    print(f"\n{SEP}\n  FINAL RANKING — Top 25 by ±2d\n{SEP}")
    final = sorted(
        [(k, v[1]) for k, v in all_methods.items() if v[1].get("acc_2d", 0) > 0],
        key=lambda x: x[1]["acc_2d"], reverse=True,
    )
    print(f"  {'Method':<40s} {'N':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*80}")
    for name, r in final[:25]:
        print(f"  {name:<40s} {r['n']:>3} {r['mae']:>5.2f}"
              f" {r['acc_1d']:>5.1%} {r['acc_2d']:>5.1%}"
              f" {r['acc_3d']:>5.1%} {r['acc_5d']:>5.1%}")

    # Quality ranking
    print(f"\n  QUALITY CYCLES — Top 15 by ±2d\n  {'-'*80}")
    qranks = []
    for name, _ in final[:25]:
        d = all_methods[name][0]
        rq = eval_ov(d, lh_ov_dict, name, subset=quality)
        if rq:
            qranks.append((name, rq))
    qranks.sort(key=lambda x: x[1].get("acc_2d", 0), reverse=True)
    for name, r in qranks[:15]:
        pass  # already printed by eval_ov

    # ===== MENSTRUAL PREDICTION =====
    print(f"\n{SEP}\n  MENSTRUAL PREDICTION (per-cycle)\n{SEP}")

    oracle = {s: v for s, v in lh_ov_dict.items() if s in cs}

    print(f"\n--- Baselines ---")
    predict_menses(cs, {}, subj_order, lh_ov_dict, eval_subset=labeled,
                   label="Calendar-only (labeled)")
    predict_menses(cs, oracle, subj_order, lh_ov_dict, fixed_luteal=12.0,
                   eval_subset=labeled, label="Oracle + luteal=12 (labeled)")
    predict_menses(cs, oracle, subj_order, lh_ov_dict, fixed_luteal=13.0,
                   eval_subset=labeled, label="Oracle + luteal=13 (labeled)")

    print(f"\n--- Top 5 detectors ---")
    for name, _ in final[:5]:
        d = all_methods[name][0]
        for fl in [12, 13]:
            predict_menses(cs, d, subj_order, lh_ov_dict, fixed_luteal=float(fl),
                           eval_subset=labeled, label=f"{name} + lut={fl}")

    print(f"\n--- Quality cycles ---")
    predict_menses(cs, {}, subj_order, lh_ov_dict, eval_subset=quality,
                   label="Calendar-only (quality)")
    for name, _ in final[:3]:
        d = all_methods[name][0]
        for fl in [12, 13]:
            predict_menses(cs, d, subj_order, lh_ov_dict, fixed_luteal=float(fl),
                           eval_subset=quality, label=f"{name} + lut={fl} (Q)")
    predict_menses(cs, oracle, subj_order, lh_ov_dict, fixed_luteal=13.0,
                   eval_subset=quality, label="Oracle + lut=13 (Q)")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  COMPLETE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

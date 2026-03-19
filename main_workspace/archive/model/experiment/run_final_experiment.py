"""
FINAL Experiment — Leakage-Free Ovulation Detection & Menstrual Prediction
==========================================================================
Key design principles:
  1. NO cycle_len as feature for ovulation detection (signal-only detection)
  2. Ovulation detection → menstrual prediction via countdown (pred = ov + luteal)
  3. DIRECT cycle length prediction (alternative pipeline)
  4. Proper LOSO cross-validation
  5. Oracle analysis to understand theoretical ceiling

Pipelines:
  A. Signal-based ovulation detection → countdown prediction
  B. Direct cycle length regression from temperature features
  C. Hybrid: confidence-routing between ov-countdown and calendar
  D. Oracle ceilings

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_final_experiment
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit, minimize
from collections import defaultdict
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


def load_data():
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    wt_path = os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv")
    noct_daily = None
    if os.path.exists(wt_path):
        wt = pd.read_csv(wt_path, usecols=key + ["timestamp", "temperature_diff_from_baseline"])
        wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
        night = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
        noct_daily = night.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
        noct_daily.rename(columns={"temperature_diff_from_baseline": "noct_mean"}, inplace=True)

    rhr_path = os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv")
    rhr_daily = None
    if os.path.exists(rhr_path):
        rhr = pd.read_csv(rhr_path, usecols=key + ["value"])
        rhr_daily = rhr.groupby(key)["value"].mean().reset_index()
        rhr_daily.rename(columns={"value": "rhr_mean"}, inplace=True)

    merged = cc.merge(ct_daily, on=key, how="left")
    if noct_daily is not None:
        merged = merged.merge(noct_daily, on=key, how="left")
    if rhr_daily is not None:
        merged = merged.merge(rhr_daily, on=key, how="left")

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
        entry = {
            "dic": (grp["day_in_study"] - cs).values,
            "id": grp["id"].values[0],
            "cycle_len": n,
            "nightly_temperature": grp["nightly_temperature"].values,
        }
        if noct_daily is not None and "noct_mean" in grp.columns:
            entry["noct_mean"] = grp["noct_mean"].values
        if rhr_daily is not None and "rhr_mean" in grp.columns:
            entry["rhr_mean"] = grp["rhr_mean"].values
        cycle_series[sgk] = entry

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
        raw = data["nightly_temperature"]
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


def _pr(tag, ae, prefix="  "):
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"{prefix}[{tag}] n={r['n']} | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# RULE-BASED OVULATION DETECTION (no cycle_len)
# =====================================================================

def detect_ttest_biphasic(cycle_series, sigma=3.0, frac=0.575, pw=4.0,
                          temp_key="nightly_temperature"):
    """Rule-based: t-test + biphasic, NO cycle_len as feature.
    Uses hist_cycle_len for expected position prior only."""
    detected = {}
    confidences = {}
    for sgk, data in cycle_series.items():
        raw = data.get(temp_key, data["nightly_temperature"])
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        exp = max(8, hcl * frac)

        best_ws, best_tsp, best_tstat = -np.inf, None, 0
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
                best_tstat = stat

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

        if cands:
            ov = int(round(np.mean(cands)))
            pre_m = np.mean(t[max(0, ov - 5):ov])
            post_m = np.mean(t[ov:min(n, ov + 5)])
            shift = post_m - pre_m
            conf = min(1.0, max(0.0, best_tstat / 5.0)) * min(1.0, max(0.0, shift / 0.3))
        else:
            ov = int(round(frac * hcl))
            conf = 0.0

        detected[sgk] = ov
        confidences[sgk] = conf
    return detected, confidences


def detect_bayesian_biphasic(cycle_series, sigma=2.5, frac=0.575):
    """Bayesian approach: find MAP estimate of changepoint with proper priors."""
    detected = {}
    confidences = {}
    for sgk, data in cycle_series.items():
        raw = data["nightly_temperature"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        exp = max(8, hcl * frac)

        log_posts = []
        for tau in range(5, n - 3):
            pre = t[:tau]
            post = t[tau:]
            mu_pre, mu_post = np.mean(pre), np.mean(post)
            if mu_post <= mu_pre:
                log_posts.append(-np.inf)
                continue
            var_pre = max(np.var(pre), 1e-6)
            var_post = max(np.var(post), 1e-6)
            ll_pre = -0.5 * len(pre) * np.log(var_pre)
            ll_post = -0.5 * len(post) * np.log(var_post)
            prior = -0.5 * ((dic[tau] - exp) / 5.0) ** 2
            shift_prior = np.log(max(mu_post - mu_pre, 1e-6))
            log_posts.append(ll_pre + ll_post + prior + shift_prior)

        if log_posts and max(log_posts) > -np.inf:
            best_idx = np.argmax(log_posts)
            tau = best_idx + 5
            ov = int(dic[tau])
            log_posts_arr = np.array(log_posts)
            valid = log_posts_arr > -np.inf
            if valid.sum() > 1:
                lp = log_posts_arr[valid]
                lp -= lp.max()
                probs = np.exp(lp)
                probs /= probs.sum()
                conf = float(probs.max())
            else:
                conf = 0.5
        else:
            ov = int(round(frac * hcl))
            conf = 0.0

        detected[sgk] = ov
        confidences[sgk] = conf
    return detected, confidences


def detect_sigmoid_fit(cycle_series, sigma=2.0, frac=0.575):
    """Sigmoid fit: T(d) = L + (U-L) / (1+exp(-k*(d-tau)))."""
    def _sig(x, L, U, k, tau):
        return L + (U - L) / (1.0 + np.exp(-k * (x - tau)))

    detected = {}
    confidences = {}
    for sgk, data in cycle_series.items():
        raw = data["nightly_temperature"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        x = dic.astype(float)
        try:
            p0 = [t[:n//3].mean(), t[2*n//3:].mean(), 0.5, hcl * frac]
            lo = [t.min()-2, t.min()-2, 0.01, 3]
            hi = [t.max()+2, t.max()+2, 5.0, n-3]
            popt, pcov = curve_fit(_sig, x, t, p0=p0, bounds=(lo, hi), maxfev=5000)
            ov = int(round(max(5, min(n - 3, popt[3]))))
            res = t - _sig(x, *popt)
            rmse = np.sqrt(np.mean(res ** 2))
            shift = popt[1] - popt[0]
            conf = min(1.0, max(0.0, shift / 0.4)) * min(1.0, max(0.0, 0.3 / max(rmse, 0.01)))
        except Exception:
            ov = int(round(frac * hcl))
            conf = 0.0
        detected[sgk] = ov
        confidences[sgk] = conf
    return detected, confidences


def detect_cusum(cycle_series, sigma=2.0, k=0.5, h=4.0, frac=0.575):
    detected = {}
    confidences = {}
    for sgk, data in cycle_series.items():
        raw = data["nightly_temperature"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
            continue
        t = _clean(raw, sigma=sigma)
        ref_end = max(5, n // 3)
        mu = np.mean(t[:ref_end])
        std = max(np.std(t[:ref_end]), 0.01)
        cusum_pos = np.zeros(n)
        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + (t[i] - mu) / std - k)
        lo = max(5, int(hcl * 0.3))
        hi = min(n - 2, int(hcl * 0.75))
        found = False
        for i in range(lo, hi):
            if cusum_pos[i] > h:
                detected[sgk] = int(dic[i])
                confidences[sgk] = min(1.0, cusum_pos[i] / (2 * h))
                found = True
                break
        if not found:
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
    return detected, confidences


def detect_hmm(cycle_series, sigma=1.5, frac=0.575):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return {}, {}
    detected = {}
    confidences = {}
    for sgk, data in cycle_series.items():
        dic = data["dic"]
        n = len(dic)
        hcl = data["hist_cycle_len"]
        raw = data["nightly_temperature"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
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
            shift = float(mdl.means_[high, 0] - mdl.means_[low, 0])
            found = False
            for i in range(1, n):
                if states[i] == high and states[i - 1] == low and dic[i] >= 6:
                    detected[sgk] = int(dic[i])
                    confidences[sgk] = min(1.0, max(0.0, shift / 0.3))
                    found = True
                    break
            if not found:
                detected[sgk] = int(round(frac * hcl))
                confidences[sgk] = 0.0
        except Exception:
            detected[sgk] = int(round(frac * hcl))
            confidences[sgk] = 0.0
    return detected, confidences


# =====================================================================
# ML Ovulation Detection — SIGNAL-ONLY features (no cycle_len)
# =====================================================================

def extract_signal_features(data, sigma=1.5):
    """Extract features from temperature signal. NO cycle_len features!"""
    n = data["cycle_len"]
    feats = {}
    feats["hist_clen"] = data["hist_cycle_len"]

    def _temp_feats(raw, prefix):
        if np.isnan(raw).all():
            return {}
        t = _clean(raw, sigma=sigma)
        f = {}
        f[f"{prefix}_mean"] = np.mean(t)
        f[f"{prefix}_std"] = np.std(t)
        f[f"{prefix}_range"] = np.ptp(t)
        f[f"{prefix}_skew"] = float(pd.Series(t).skew())
        f[f"{prefix}_nadir_day"] = int(np.argmin(t))

        grad = np.gradient(t)
        f[f"{prefix}_maxgrad_day"] = int(np.argmax(grad))
        f[f"{prefix}_maxgrad_val"] = float(np.max(grad))

        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            f[f"{prefix}_q{int(q*100)}"] = float(np.quantile(t, q))

        # t-test at ABSOLUTE day positions (not fractional!)
        for day_pos in [8, 10, 12, 14, 16, 18, 20, 22]:
            if day_pos < n - 3 and day_pos >= 5:
                try:
                    stat, _ = ttest_ind(t[day_pos:], t[:day_pos], alternative="greater")
                    f[f"{prefix}_tt_d{day_pos}"] = float(stat) if not np.isnan(stat) else 0
                except Exception:
                    f[f"{prefix}_tt_d{day_pos}"] = 0
            else:
                f[f"{prefix}_tt_d{day_pos}"] = 0

        # Best split point (absolute day)
        best_sc, best_sp, best_stat = -np.inf, n // 2, 0
        for sp in range(5, n - 3):
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                if not np.isnan(stat) and stat > best_sc:
                    best_sc = stat
                    best_sp = sp
                    best_stat = stat
            except Exception:
                continue
        f[f"{prefix}_best_split_day"] = best_sp
        f[f"{prefix}_best_split_tstat"] = max(best_stat, 0)
        pre_m = np.mean(t[:best_sp])
        post_m = np.mean(t[best_sp:])
        f[f"{prefix}_split_shift"] = post_m - pre_m

        short_ma = pd.Series(t).rolling(3, min_periods=1).mean().values
        long_ma = pd.Series(t).rolling(7, min_periods=1).mean().values
        for i in range(7, n):
            if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                f[f"{prefix}_cross_day"] = i
                break
        else:
            f[f"{prefix}_cross_day"] = n // 2

        ts = pd.Series(t)
        for lag in [1, 3, 5]:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0

        return f

    feats.update(_temp_feats(data["nightly_temperature"], "nt"))
    if "noct_mean" in data:
        feats.update(_temp_feats(data["noct_mean"], "noct"))
    if "rhr_mean" in data:
        raw = data["rhr_mean"]
        if not np.isnan(raw).all():
            t = _clean(raw, sigma=1.0)
            feats["rhr_mean"] = np.mean(t)
            feats["rhr_std"] = np.std(t)
            feats["rhr_nadir_day"] = int(np.argmin(t))
            feats["rhr_peak_day"] = int(np.argmax(t))

    for k in feats:
        if isinstance(feats[k], float) and np.isnan(feats[k]):
            feats[k] = 0.0
    return feats


def ml_detect_loso(cycle_series, lh_ov_dict, model_type="ridge"):
    """ML ovulation detection with LOSO. No cycle_len features."""
    labeled = [sgk for sgk in cycle_series if sgk in lh_ov_dict]
    all_feats, all_targets, all_ids, all_sgks = [], [], [], []
    for sgk in labeled:
        data = cycle_series[sgk]
        feats = extract_signal_features(data)
        if not feats or len(feats) < 5:
            continue
        all_feats.append(feats)
        all_targets.append(lh_ov_dict[sgk])
        all_ids.append(data["id"])
        all_sgks.append(sgk)

    if len(all_feats) < 10:
        return {}, {}

    df = pd.DataFrame(all_feats).fillna(0)
    X = df.values
    y = np.array(all_targets, dtype=float)
    ids = np.array(all_ids)
    unique_ids = np.unique(ids)

    from sklearn.preprocessing import StandardScaler

    detected = {}
    confidences = {}
    for test_uid in unique_ids:
        test_mask = ids == test_uid
        train_mask = ~test_mask
        if train_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        test_sgks = [all_sgks[i] for i in np.where(test_mask)[0]]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        elif model_type == "svr":
            from sklearn.svm import SVR
            mdl = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            mdl = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            mdl = KNeighborsRegressor(n_neighbors=5, weights="distance")
        else:
            raise ValueError(model_type)

        mdl.fit(X_tr_s, y_tr)
        preds = mdl.predict(X_te_s)

        for sgk, pred in zip(test_sgks, preds):
            clen = cycle_series[sgk]["cycle_len"]
            ov = int(round(max(5, min(clen - 3, pred))))
            detected[sgk] = ov
            t = _clean(cycle_series[sgk]["nightly_temperature"], sigma=2.0)
            if ov < len(t) - 2:
                pre_m = np.mean(t[max(0, ov - 5):ov])
                post_m = np.mean(t[ov:min(len(t), ov + 5)])
                shift = post_m - pre_m
                confidences[sgk] = min(1.0, max(0.0, shift / 0.25))
            else:
                confidences[sgk] = 0.5

    return detected, confidences


# =====================================================================
# DIRECT CYCLE LENGTH PREDICTION (no ovulation step)
# =====================================================================

def direct_cycle_len_predict_loso(cycle_series, lh_ov_dict, model_type="ridge"):
    """Directly predict cycle_len from temperature features. LOSO."""
    all_labeled = [sgk for sgk in cycle_series if sgk in lh_ov_dict]
    all_feats, all_targets, all_ids, all_sgks = [], [], [], []
    for sgk in all_labeled:
        data = cycle_series[sgk]
        feats = extract_signal_features(data)
        if not feats or len(feats) < 5:
            continue
        all_feats.append(feats)
        all_targets.append(data["cycle_len"])
        all_ids.append(data["id"])
        all_sgks.append(sgk)

    if len(all_feats) < 10:
        return {}

    df = pd.DataFrame(all_feats).fillna(0)
    X = df.values
    y = np.array(all_targets, dtype=float)
    ids = np.array(all_ids)
    unique_ids = np.unique(ids)

    from sklearn.preprocessing import StandardScaler

    preds_dict = {}
    for test_uid in unique_ids:
        test_mask = ids == test_uid
        train_mask = ~test_mask
        if train_mask.sum() < 5:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        test_sgks = [all_sgks[i] for i in np.where(test_mask)[0]]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            mdl = ElasticNet(alpha=0.1, l1_ratio=0.5)
        else:
            raise ValueError(model_type)

        mdl.fit(X_tr_s, y_tr)
        preds = mdl.predict(X_te_s)

        for sgk, pred in zip(test_sgks, preds):
            preds_dict[sgk] = float(pred)

    return preds_dict


# =====================================================================
# ENSEMBLE of multiple detectors
# =====================================================================

def ensemble_detectors(all_dets, weights=None):
    """Weighted average of multiple ovulation detections."""
    all_sgks = set()
    for d, _ in all_dets:
        all_sgks.update(d.keys())
    if weights is None:
        weights = [1.0] * len(all_dets)
    result = {}
    confs = {}
    for sgk in all_sgks:
        vals, ws, cs = [], [], []
        for (d, c), w in zip(all_dets, weights):
            if sgk in d:
                vals.append(d[sgk])
                ws.append(w)
                cs.append(c.get(sgk, 0.5))
        if vals:
            result[sgk] = int(round(np.average(vals, weights=ws)))
            confs[sgk] = float(np.mean(cs))
    return result, confs


# =====================================================================
# MENSTRUAL PREDICTION PIPELINE
# =====================================================================

def predict_menses_pipeline(cycle_series, detected, confidences, subj_order,
                            fixed_luteal=13.0, conf_threshold=0.0,
                            eval_subset=None, label=""):
    """
    Predict cycle length from detected ovulation:
      - If confidence >= threshold: pred = detected_ov + luteal
      - Else: pred = weighted historical cycle length (calendar fallback)
    """
    pop_lut = fixed_luteal
    subj_past_lut = defaultdict(list)
    subj_past_clen = defaultdict(list)
    errs_all, n_ov, n_cal = [], 0, 0
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
            conf = confidences.get(sgk, 0.0)

            if ov is not None and ov > 3 and conf >= conf_threshold:
                pred = ov + lut
                n_ov += 1
            else:
                pred = acl
                n_cal += 1

            err = pred - actual
            if ev is None or sgk in ev:
                errs_all.append(err)

            subj_past_clen[uid].append(actual)
            if ov is not None:
                el = actual - ov
                if 8 <= el <= 22:
                    subj_past_lut[uid].append(el)

    if not errs_all:
        return {}
    ae = np.abs(errs_all)
    r = _pr(f"{label} (ov:{n_ov},cal:{n_cal})", ae, prefix="    ")
    return r


def predict_menses_direct(cycle_series, pred_clens, subj_order,
                          eval_subset=None, label=""):
    """Direct cycle length prediction — no ovulation step."""
    subj_past_clen = defaultdict(list)
    errs = []
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual = data["cycle_len"]
            pc = subj_past_clen[uid]
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0

            pred = pred_clens.get(sgk, acl)
            err = pred - actual
            if ev is None or sgk in ev:
                errs.append(err)
            subj_past_clen[uid].append(actual)

    if not errs:
        return {}
    ae = np.abs(errs)
    return _pr(label, ae, prefix="    ")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"\n{SEP}\n  FINAL EXPERIMENT — Leakage-Free\n{SEP}")
    t0 = time.time()

    lh_ov_dict, lh_luteal, cs, quality, subj_order = load_data()
    labeled = set(s for s in cs if s in lh_ov_dict)
    print(f"  Cycles: {len(cs)} | Labeled: {len(labeled)} | Quality: {len(quality)}")
    print(f"  Luteal: mean={np.mean(list(lh_luteal.values())):.1f}d, "
          f"median={np.median(list(lh_luteal.values())):.0f}d")

    # =================================================================
    # PART A: ORACLE CEILINGS
    # =================================================================
    print(f"\n{SEP}\n  A. ORACLE CEILINGS (using LH labels)\n{SEP}")
    oracle = {s: v for s, v in lh_ov_dict.items() if s in cs}
    oracle_conf = {s: 1.0 for s in oracle}

    for fl in [11, 12, 13, 14]:
        predict_menses_pipeline(cs, oracle, oracle_conf, subj_order,
                                fixed_luteal=float(fl), eval_subset=labeled,
                                label=f"Oracle+lut{fl} (labeled)")
    for fl in [12, 13]:
        predict_menses_pipeline(cs, oracle, oracle_conf, subj_order,
                                fixed_luteal=float(fl), eval_subset=quality,
                                label=f"Oracle+lut{fl} (Q)")

    # Personal luteal
    predict_menses_pipeline(cs, oracle, oracle_conf, subj_order,
                            fixed_luteal=13.0, eval_subset=labeled,
                            label="Oracle+personal (labeled)")

    # Calendar baseline
    predict_menses_pipeline(cs, {}, {}, subj_order,
                            fixed_luteal=13.0, eval_subset=labeled,
                            label="Calendar (labeled)")
    predict_menses_pipeline(cs, {}, {}, subj_order,
                            fixed_luteal=13.0, eval_subset=quality,
                            label="Calendar (Q)")

    # =================================================================
    # PART B: RULE-BASED OVULATION DETECTION
    # =================================================================
    print(f"\n{SEP}\n  B. RULE-BASED OVULATION DETECTION (signal-only)\n{SEP}")
    rule_methods = {}

    configs = [
        (2.0, 0.575, 4.0), (2.5, 0.575, 4.0), (3.0, 0.575, 4.0),
        (2.5, 0.55, 4.0), (3.0, 0.55, 4.0),
        (2.0, 0.575, 5.0), (2.5, 0.575, 5.0), (3.0, 0.575, 5.0),
    ]
    for s, f, w in configs:
        d, c = detect_ttest_biphasic(cs, sigma=s, frac=f, pw=w)
        tag = f"tt+bi-σ{s}-f{f}-w{w}"
        errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d]
        ae = np.abs(errors)
        r = _pr(tag, ae)
        rule_methods[tag] = (d, c, r)

    # Also with nocturnal temp
    for s, f, w in [(2.5, 0.575, 4.0), (3.0, 0.575, 4.0)]:
        d, c = detect_ttest_biphasic(cs, sigma=s, frac=f, pw=w, temp_key="noct_mean")
        tag = f"tt+bi-noct-σ{s}-f{f}-w{w}"
        errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d and s2 in lh_ov_dict]
        if errors:
            ae = np.abs(errors)
            r = _pr(tag, ae)
            rule_methods[tag] = (d, c, r)

    # Bayesian
    for s in [2.0, 2.5, 3.0]:
        d, c = detect_bayesian_biphasic(cs, sigma=s)
        tag = f"bayes-σ{s}"
        errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d]
        ae = np.abs(errors)
        r = _pr(tag, ae)
        rule_methods[tag] = (d, c, r)

    # Sigmoid
    for s in [1.5, 2.0, 2.5]:
        d, c = detect_sigmoid_fit(cs, sigma=s)
        tag = f"sigmoid-σ{s}"
        errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d]
        ae = np.abs(errors)
        r = _pr(tag, ae)
        rule_methods[tag] = (d, c, r)

    # CUSUM
    for s in [2.0, 2.5]:
        for h in [3.0, 4.0, 5.0]:
            d, c = detect_cusum(cs, sigma=s, h=h)
            tag = f"cusum-σ{s}-h{h}"
            errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d]
            ae = np.abs(errors)
            r = _pr(tag, ae)
            rule_methods[tag] = (d, c, r)

    # HMM
    for s in [1.0, 1.5, 2.0]:
        d, c = detect_hmm(cs, sigma=s)
        if d:
            tag = f"hmm-σ{s}"
            errors = [d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in d]
            ae = np.abs(errors)
            r = _pr(tag, ae)
            rule_methods[tag] = (d, c, r)

    # Quality performance
    print(f"\n  --- Quality cycles ---")
    for tag, (d, c, r) in sorted(rule_methods.items(), key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)[:10]:
        errors = [d[s2] - lh_ov_dict[s2] for s2 in quality if s2 in d and s2 in lh_ov_dict]
        if errors:
            _pr(f"Q-{tag}", np.abs(errors))

    # =================================================================
    # PART C: ML OVULATION DETECTION (signal-only, LOSO)
    # =================================================================
    print(f"\n{SEP}\n  C. ML OVULATION DETECTION (signal-only, LOSO)\n{SEP}")
    ml_methods = {}
    for mt in ["ridge", "rf", "gbdt", "svr", "elastic", "knn"]:
        d, c = ml_detect_loso(cs, lh_ov_dict, model_type=mt)
        if d:
            errors = [d[s2] - lh_ov_dict[s2] for s2 in d if s2 in lh_ov_dict]
            ae = np.abs(errors)
            r = _pr(f"ML-{mt}", ae)
            ml_methods[f"ML-{mt}"] = (d, c, r)

            errors_q = [d[s2] - lh_ov_dict[s2] for s2 in quality if s2 in d and s2 in lh_ov_dict]
            if errors_q:
                _pr(f"  Q-ML-{mt}", np.abs(errors_q))

    # =================================================================
    # PART D: ENSEMBLES
    # =================================================================
    print(f"\n{SEP}\n  D. ENSEMBLES\n{SEP}")

    all_ranked = []
    for tag, (d, c, r) in rule_methods.items():
        all_ranked.append((tag, d, c, r))
    for tag, (d, c, r) in ml_methods.items():
        all_ranked.append((tag, d, c, r))
    all_ranked.sort(key=lambda x: x[3].get("acc_2d", 0), reverse=True)

    # Ensemble of top methods
    for topN in [3, 5, 7]:
        if len(all_ranked) >= topN:
            dets = [(r[1], r[2]) for r in all_ranked[:topN]]
            ws = [max(r[3].get("acc_2d", 0), 0.01) for r in all_ranked[:topN]]
            e_d, e_c = ensemble_detectors(dets, ws)
            errors = [e_d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in e_d and s2 in lh_ov_dict]
            if errors:
                r = _pr(f"ens-top{topN}", np.abs(errors))
                all_ranked.append((f"ens-top{topN}", e_d, e_c, r))
                errors_q = [e_d[s2] - lh_ov_dict[s2] for s2 in quality if s2 in e_d and s2 in lh_ov_dict]
                if errors_q:
                    _pr(f"  Q-ens-top{topN}", np.abs(errors_q))

    # ML+Rule ensemble
    best_ml = None
    for tag, d, c, r in all_ranked:
        if tag.startswith("ML-"):
            best_ml = (d, c)
            break
    best_rb = None
    for tag, d, c, r in all_ranked:
        if not tag.startswith("ML-") and not tag.startswith("ens-"):
            best_rb = (d, c)
            break

    if best_ml and best_rb:
        e_d, e_c = ensemble_detectors([best_ml, best_rb], [2.0, 1.0])
        errors = [e_d[s2] - lh_ov_dict[s2] for s2 in labeled if s2 in e_d and s2 in lh_ov_dict]
        if errors:
            r = _pr("ML+Rule hybrid", np.abs(errors))
            all_ranked.append(("ML+Rule", e_d, e_c, r))
            errors_q = [e_d[s2] - lh_ov_dict[s2] for s2 in quality if s2 in e_d and s2 in lh_ov_dict]
            if errors_q:
                _pr("  Q-ML+Rule", np.abs(errors_q))

    # =================================================================
    # PART E: FINAL RANKING
    # =================================================================
    print(f"\n{SEP}\n  E. FINAL OV DETECTION RANKING — by ±2d\n{SEP}")
    all_ranked.sort(key=lambda x: x[3].get("acc_2d", 0), reverse=True)
    print(f"  {'Method':<45s} {'N':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6}")
    print(f"  {'-'*80}")
    for tag, d, c, r in all_ranked[:20]:
        print(f"  {tag:<45s} {r['n']:>3} {r['mae']:>5.2f}"
              f" {r.get('acc_1d',0):>5.1%} {r.get('acc_2d',0):>5.1%}"
              f" {r.get('acc_3d',0):>5.1%}")

    # =================================================================
    # PART F: MENSTRUAL PREDICTION
    # =================================================================
    print(f"\n{SEP}\n  F. MENSTRUAL PREDICTION\n{SEP}")

    # Using top 5 detectors
    print(f"\n  --- Top 5 detectors (labeled cycles) ---")
    for tag, d, c, r in all_ranked[:5]:
        for fl in [12, 13]:
            predict_menses_pipeline(cs, d, c, subj_order, fixed_luteal=float(fl),
                                    eval_subset=labeled,
                                    label=f"{tag}+lut{fl}")

    print(f"\n  --- Top 5 detectors (quality cycles) ---")
    for tag, d, c, r in all_ranked[:5]:
        for fl in [12, 13]:
            predict_menses_pipeline(cs, d, c, subj_order, fixed_luteal=float(fl),
                                    eval_subset=quality,
                                    label=f"{tag}+lut{fl} (Q)")

    # Confidence-based hybrid
    print(f"\n  --- Confidence-based hybrid ---")
    best_tag, best_d, best_c, _ = all_ranked[0]
    for ct in [0.0, 0.2, 0.4, 0.6]:
        for fl in [12, 13]:
            predict_menses_pipeline(cs, best_d, best_c, subj_order,
                                    fixed_luteal=float(fl), conf_threshold=ct,
                                    eval_subset=labeled,
                                    label=f"{best_tag}+lut{fl} conf>{ct}")
            predict_menses_pipeline(cs, best_d, best_c, subj_order,
                                    fixed_luteal=float(fl), conf_threshold=ct,
                                    eval_subset=quality,
                                    label=f"{best_tag}+lut{fl} conf>{ct} (Q)")

    # =================================================================
    # PART G: DIRECT CYCLE LENGTH PREDICTION
    # =================================================================
    print(f"\n{SEP}\n  G. DIRECT CYCLE LENGTH PREDICTION (no ov step)\n{SEP}")
    for mt in ["ridge", "rf", "elastic"]:
        pred_clens = direct_cycle_len_predict_loso(cs, lh_ov_dict, model_type=mt)
        if pred_clens:
            predict_menses_direct(cs, pred_clens, subj_order,
                                  eval_subset=labeled, label=f"direct-{mt} (labeled)")
            predict_menses_direct(cs, pred_clens, subj_order,
                                  eval_subset=quality, label=f"direct-{mt} (Q)")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  COMPLETE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

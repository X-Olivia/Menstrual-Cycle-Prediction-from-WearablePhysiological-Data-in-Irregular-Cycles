"""
Advanced Ovulation Detection v2 — Ablation + Multi-signal + Nocturnal
=====================================================================
Key improvements:
  1. Ablation: calendar-only features vs temperature features vs both
  2. Use nocturnal minute-level temperature (higher SNR)
  3. Multi-signal: add HR, HRV, RHR
  4. Confidence-based routing for menstrual prediction
  5. More ML models: SVR, ElasticNet, KNN
  6. Stacking ensemble (meta-learner)

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_advanced_ov_menses_v2
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from collections import defaultdict
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


def load_data():
    """Load all cycle data with nocturnal temperature and multi-signal."""
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    wt_path = os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv")
    has_wt = os.path.exists(wt_path)
    noct_daily = None
    if has_wt:
        wt = pd.read_csv(wt_path, usecols=key + ["timestamp", "temperature_diff_from_baseline"])
        wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
        night = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
        noct_daily = night.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
        noct_daily.rename(columns={"temperature_diff_from_baseline": "noct_mean"}, inplace=True)
        print(f"  Nocturnal temperature loaded: {len(noct_daily)} day-rows")

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

    signal_cols = ["nightly_temperature"]
    if noct_daily is not None:
        signal_cols.append("noct_mean")
    if rhr_daily is not None:
        signal_cols.append("rhr_mean")
    print(f"  Available signals: {signal_cols}")

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
        }
        for sc in signal_cols:
            entry[sc] = grp[sc].values if sc in grp.columns else np.full(n, np.nan)
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

    return lh_ov_dict, lh_luteal, cycle_series, quality, subj_order, signal_cols


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def eval_ov(detected, lh_ov_dict, name, subset=None, quiet=False):
    keys = subset if subset else lh_ov_dict.keys()
    errors = [detected[s] - lh_ov_dict[s] for s in keys if s in detected and s in lh_ov_dict]
    if not errors:
        if not quiet:
            print(f"  [{name}] No evaluable detections")
        return {}
    ae = np.abs(errors)
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    if not quiet:
        print(f"  [{name}] n={r['n']} | MAE={r['mae']:.2f}d"
              f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
              f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# Feature extraction (ABLATION-aware)
# =====================================================================

def extract_features(data, mode="full", sigma=1.5):
    """
    mode='cal':  only calendar features (hist_cycle_len, n_days)
    mode='temp': only temperature features
    mode='full': all features
    mode='temp_only_no_clen': temperature without n_days
    mode='multi': temp + hr + hrv + rhr
    mode='noct':  nocturnal temperature features
    mode='noct_multi': nocturnal temp + hr + hrv + rhr
    """
    feats = {}
    n = data["cycle_len"]

    # Calendar features
    if mode in ("cal", "full"):
        feats["n_days"] = n
        feats["hist_clen"] = data["hist_cycle_len"]

    # Temperature features
    def _temp_feats(raw, prefix="t", sig=sigma):
        if np.isnan(raw).all():
            return {}
        t = _clean(raw, sigma=sig)
        f = {}
        f[f"{prefix}_mean"] = np.mean(t)
        f[f"{prefix}_std"] = np.std(t)
        f[f"{prefix}_range"] = np.ptp(t)
        f[f"{prefix}_skew"] = float(pd.Series(t).skew())
        f[f"{prefix}_kurt"] = float(pd.Series(t).kurtosis())
        f[f"{prefix}_nadir_day"] = int(np.argmin(t))
        f[f"{prefix}_nadir_frac"] = f[f"{prefix}_nadir_day"] / n

        grad = np.gradient(t)
        f[f"{prefix}_maxgrad_day"] = int(np.argmax(grad))
        f[f"{prefix}_maxgrad_val"] = float(np.max(grad))
        f[f"{prefix}_maxgrad_frac"] = f[f"{prefix}_maxgrad_day"] / n

        mid = n // 2
        f[f"{prefix}_mean_h1"] = np.mean(t[:mid])
        f[f"{prefix}_mean_h2"] = np.mean(t[mid:])
        f[f"{prefix}_shift_h"] = f[f"{prefix}_mean_h2"] - f[f"{prefix}_mean_h1"]

        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            f[f"{prefix}_q{int(q*100)}"] = float(np.quantile(t, q))

        for frac_pt in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
            sp = max(5, int(n * frac_pt))
            if sp < n - 3:
                try:
                    stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                    f[f"{prefix}_tt_{int(frac_pt*100)}"] = float(stat) if not np.isnan(stat) else 0
                except Exception:
                    f[f"{prefix}_tt_{int(frac_pt*100)}"] = 0
            else:
                f[f"{prefix}_tt_{int(frac_pt*100)}"] = 0

        short_ma = pd.Series(t).rolling(3, min_periods=1).mean().values
        long_ma = pd.Series(t).rolling(7, min_periods=1).mean().values
        for i in range(7, n):
            if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                f[f"{prefix}_cross_day"] = i
                f[f"{prefix}_cross_frac"] = i / n
                break
        else:
            f[f"{prefix}_cross_day"] = n // 2
            f[f"{prefix}_cross_frac"] = 0.5

        ts = pd.Series(t)
        for lag in [1, 3, 5]:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0

        # Biphasic split quality
        best_score = -np.inf
        best_sp = n // 2
        for sp in range(5, n - 3):
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                if not np.isnan(stat) and stat > best_score:
                    best_score = stat
                    best_sp = sp
            except Exception:
                continue
        f[f"{prefix}_best_split"] = best_sp
        f[f"{prefix}_best_split_frac"] = best_sp / n
        f[f"{prefix}_best_split_tstat"] = max(best_score, 0)
        pre_m = np.mean(t[:best_sp])
        post_m = np.mean(t[best_sp:])
        f[f"{prefix}_split_shift"] = post_m - pre_m

        return f

    def _signal_feats(raw, prefix, sig=1.0):
        if np.isnan(raw).all():
            return {}
        t = _clean(raw, sigma=sig)
        f = {}
        f[f"{prefix}_mean"] = np.mean(t)
        f[f"{prefix}_std"] = np.std(t)
        f[f"{prefix}_range"] = np.ptp(t)
        f[f"{prefix}_nadir_frac"] = np.argmin(t) / n
        f[f"{prefix}_peak_frac"] = np.argmax(t) / n
        mid = n // 2
        f[f"{prefix}_shift_h"] = np.mean(t[mid:]) - np.mean(t[:mid])

        grad = np.gradient(t)
        f[f"{prefix}_maxgrad_frac"] = np.argmax(np.abs(grad)) / n

        for q in [0.25, 0.5, 0.75]:
            f[f"{prefix}_q{int(q*100)}"] = float(np.quantile(t, q))

        return f

    if mode in ("temp", "full", "temp_only_no_clen"):
        feats.update(_temp_feats(data["nightly_temperature"], "nt"))

    if mode in ("noct", "noct_multi") and "noct_mean" in data:
        feats.update(_temp_feats(data["noct_mean"], "noct"))

    if mode in ("multi", "noct_multi", "full"):
        if "rhr_mean" in data:
            feats.update(_signal_feats(data["rhr_mean"], "rhr"))

    if mode == "temp_only_no_clen":
        pass
    elif mode in ("noct", "noct_multi"):
        feats["hist_clen"] = data["hist_cycle_len"]
        feats["n_days"] = n

    for k in feats:
        if isinstance(feats[k], float) and np.isnan(feats[k]):
            feats[k] = 0.0

    return feats


# =====================================================================
# ML Training (LOSO)
# =====================================================================

def train_eval_loso(cycle_series, lh_ov_dict, mode="full", model_type="ridge",
                    verbose=True, tag_prefix=""):
    labeled = [sgk for sgk in cycle_series if sgk in lh_ov_dict]
    if not labeled:
        return {}, {}

    all_feats, all_targets, all_ids, all_sgks = [], [], [], []
    for sgk in labeled:
        data = cycle_series[sgk]
        feats = extract_features(data, mode=mode)
        if not feats:
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
        elif model_type == "lasso":
            from sklearn.linear_model import Lasso
            mdl = Lasso(alpha=0.1)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            mdl = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == "svr":
            from sklearn.svm import SVR
            mdl = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            mdl = KNeighborsRegressor(n_neighbors=5, weights="distance")
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            mdl = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=42)
        elif model_type == "lgb":
            import lightgbm as lgb
            mdl = lgb.LGBMRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, verbose=-1, random_state=42)
        elif model_type == "xgb":
            from sklearn.ensemble import GradientBoostingRegressor
            mdl = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                            learning_rate=0.03, subsample=0.8,
                                            random_state=42)
        else:
            raise ValueError(model_type)

        mdl.fit(X_tr_s, y_tr)
        preds = mdl.predict(X_te_s)

        for sgk, pred in zip(test_sgks, preds):
            clen = cycle_series[sgk]["cycle_len"]
            detected[sgk] = int(round(max(5, min(clen - 3, pred))))

    tag = f"{tag_prefix}{model_type}({mode})"
    r_all = eval_ov(detected, lh_ov_dict, tag, quiet=not verbose)
    return detected, r_all


# =====================================================================
# Rule-based best (from v1)
# =====================================================================

def detect_ttest_biphasic(cycle_series, sigma=3.0, frac=0.575, pw=4.0, temp_key="nightly_temperature"):
    detected = {}
    for sgk, data in cycle_series.items():
        raw = data[temp_key] if temp_key in data else data["nightly_temperature"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            continue
        t = _clean(raw, sigma=sigma)
        exp = max(8, hcl * frac)

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
# Stacking Ensemble
# =====================================================================

def stacking_ensemble(cycle_series, lh_ov_dict, quality):
    """Meta-learner: stack predictions from multiple base models."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    base_configs = [
        ("ridge", "full"), ("ridge", "temp"), ("ridge", "cal"),
        ("rf", "full"), ("gbdt", "full"),
        ("svr", "full"), ("elastic", "full"),
        ("knn", "full"),
    ]

    labeled = [s for s in cycle_series if s in lh_ov_dict]
    sgk2idx = {s: i for i, s in enumerate(labeled)}
    n = len(labeled)

    base_preds = np.zeros((n, len(base_configs)))
    base_preds[:] = np.nan

    for j, (mt, mode) in enumerate(base_configs):
        det, _ = train_eval_loso(cycle_series, lh_ov_dict, mode=mode,
                                 model_type=mt, verbose=False)
        for sgk, pred in det.items():
            if sgk in sgk2idx:
                base_preds[sgk2idx[sgk], j] = pred

    ids = np.array([cycle_series[s]["id"] for s in labeled])
    y = np.array([lh_ov_dict[s] for s in labeled], dtype=float)
    unique_ids = np.unique(ids)

    valid = ~np.isnan(base_preds).any(axis=1)
    detected = {}

    for test_uid in unique_ids:
        test_mask = (ids == test_uid) & valid
        train_mask = (ids != test_uid) & valid
        if train_mask.sum() < 5 or test_mask.sum() == 0:
            continue

        X_tr = base_preds[train_mask]
        y_tr = y[train_mask]
        X_te = base_preds[test_mask]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        meta = Ridge(alpha=1.0)
        meta.fit(X_tr_s, y_tr)
        preds = meta.predict(X_te_s)

        test_sgks = [labeled[i] for i in np.where(test_mask)[0]]
        for sgk, pred in zip(test_sgks, preds):
            clen = cycle_series[sgk]["cycle_len"]
            detected[sgk] = int(round(max(5, min(clen - 3, pred))))

    return detected


# =====================================================================
# Menstrual Prediction
# =====================================================================

def predict_menses(cycle_series, detected, subj_order, lh_ov_dict,
                   fixed_luteal=13.0, eval_subset=None,
                   use_personal_luteal=False, label=""):
    pop_lut = fixed_luteal
    subj_past_lut = defaultdict(list)
    subj_past_clen = defaultdict(list)
    errs_all = []
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual = data["cycle_len"]
            pl = subj_past_lut[uid]
            pc = subj_past_clen[uid]

            if use_personal_luteal and pl:
                lut = np.mean(pl)
            else:
                lut = pop_lut

            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0

            ov = detected.get(sgk)
            if ov is not None and ov > 3:
                pred = ov + lut
            else:
                pred = acl

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
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"    [{label}] n={r['n']} | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# Confidence-based Hybrid Prediction
# =====================================================================

def hybrid_predict_menses(cycle_series, detected_dict, subj_order, lh_ov_dict,
                          fixed_luteal=13.0, conf_threshold=0.0,
                          eval_subset=None, label=""):
    """
    detected_dict: {sgk: (ov_day, confidence)}
    Use ovulation-based prediction only if confidence > threshold.
    """
    pop_lut = fixed_luteal
    subj_past_lut = defaultdict(list)
    subj_past_clen = defaultdict(list)
    errs_all = []
    n_ov, n_cal = 0, 0
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

            ov_info = detected_dict.get(sgk)
            if ov_info is not None:
                ov, conf = ov_info
                if conf >= conf_threshold and ov > 3:
                    pred = ov + lut
                    n_ov += 1
                else:
                    pred = acl
                    n_cal += 1
            else:
                pred = acl
                n_cal += 1

            err = pred - actual
            if ev is None or sgk in ev:
                errs_all.append(err)

            subj_past_clen[uid].append(actual)
            if ov_info is not None:
                ov = ov_info[0]
                el = actual - ov
                if 8 <= el <= 22:
                    subj_past_lut[uid].append(el)

    if not errs_all:
        return {}
    ae = np.abs(errs_all)
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"    [{label}] (ov:{n_ov}, cal:{n_cal}) n={r['n']} | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


# =====================================================================
# Main
# =====================================================================

def main():
    print(f"\n{SEP}\n  Advanced Ovulation Detection v2 — Ablation + Multi-signal\n{SEP}")
    t0 = time.time()

    lh_ov_dict, lh_luteal, cs, quality, subj_order, sig_cols = load_data()
    labeled = set(s for s in cs if s in lh_ov_dict)
    print(f"  Cycles: {len(cs)} | Labeled: {len(labeled)} | Quality: {len(quality)}")

    # ===== PART A: ABLATION STUDY =====
    print(f"\n{SEP}\n  A. ABLATION: What drives ovulation detection?\n{SEP}")

    modes = [
        ("cal", "Calendar only (n_days + hist_clen)"),
        ("temp", "Temp features only (no calendar)"),
        ("temp_only_no_clen", "Temp features without n_days"),
        ("full", "Full (calendar + temp)"),
    ]
    if "noct_mean" in sig_cols:
        modes.append(("noct", "Nocturnal temp + calendar"))
        modes.append(("noct_multi", "Nocturnal temp + multi-signal + calendar"))
    modes.append(("multi", "Multi-signal (temp + HR/HRV/RHR)"))

    models = ["ridge", "rf", "gbdt", "svr", "knn", "elastic"]
    best_results = []

    for mode, desc in modes:
        print(f"\n  --- {desc} ---")
        for mt in models:
            det, r = train_eval_loso(cs, lh_ov_dict, mode=mode, model_type=mt,
                                     verbose=True, tag_prefix="")
            if r:
                best_results.append((f"{mt}({mode})", det, r))

    # ===== PART B: Quality cycle performance =====
    print(f"\n{SEP}\n  B. Quality Cycle Performance — Top Models\n{SEP}")
    best_results.sort(key=lambda x: x[2].get("acc_2d", 0), reverse=True)
    for name, det, r in best_results[:20]:
        rq = eval_ov(det, lh_ov_dict, f"Q-{name}", subset=quality)

    # ===== PART C: Stacking Ensemble =====
    print(f"\n{SEP}\n  C. Stacking Ensemble\n{SEP}")
    det_stack = stacking_ensemble(cs, lh_ov_dict, quality)
    r_stack = eval_ov(det_stack, lh_ov_dict, "Stacking")
    rq_stack = eval_ov(det_stack, lh_ov_dict, "Stacking(Q)", subset=quality)

    # Add rule-based for comparison
    print(f"\n  --- Rule-based (best config) ---")
    det_rb = detect_ttest_biphasic(cs, sigma=3.0, frac=0.575, pw=4.0)
    eval_ov(det_rb, lh_ov_dict, "tt+bi (all)")
    eval_ov(det_rb, lh_ov_dict, "tt+bi (Q)", subset=quality)
    if "noct_mean" in sig_cols:
        det_rb_noct = detect_ttest_biphasic(cs, sigma=3.0, frac=0.575, pw=4.0, temp_key="noct_mean")
        eval_ov(det_rb_noct, lh_ov_dict, "tt+bi-noct (all)")
        eval_ov(det_rb_noct, lh_ov_dict, "tt+bi-noct (Q)", subset=quality)

    # Weighted average ensemble of top ML + best rule-based
    print(f"\n  --- Hybrid: ML + Rule-based ensemble ---")
    top_ml = best_results[0] if best_results else None
    if top_ml:
        hybrid_ens = {}
        for sgk in set(list(top_ml[1].keys()) + list(det_rb.keys())):
            vals = []
            ws = []
            if sgk in top_ml[1]:
                vals.append(top_ml[1][sgk])
                ws.append(2.0)
            if sgk in det_rb:
                vals.append(det_rb[sgk])
                ws.append(1.0)
            if vals:
                hybrid_ens[sgk] = int(round(np.average(vals, weights=ws)))
        eval_ov(hybrid_ens, lh_ov_dict, "ML+RB ensemble (all)")
        eval_ov(hybrid_ens, lh_ov_dict, "ML+RB ensemble (Q)", subset=quality)

    # ===== PART D: Summary =====
    print(f"\n{SEP}\n  D. FINAL RANKING — by ±2d (all labeled)\n{SEP}")
    all_entries = best_results.copy()
    if det_stack:
        all_entries.append(("Stacking", det_stack, r_stack or {}))
    if top_ml and hybrid_ens:
        re = eval_ov(hybrid_ens, lh_ov_dict, "ML+RB", quiet=True)
        if re:
            all_entries.append(("ML+RB", hybrid_ens, re))

    all_entries.sort(key=lambda x: x[2].get("acc_2d", 0), reverse=True)
    print(f"  {'Method':<40s} {'N':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*80}")
    for name, det, r in all_entries[:20]:
        if r:
            print(f"  {name:<40s} {r['n']:>3} {r['mae']:>5.2f}"
                  f" {r.get('acc_1d',0):>5.1%} {r.get('acc_2d',0):>5.1%}"
                  f" {r.get('acc_3d',0):>5.1%} {r.get('acc_5d',0):>5.1%}")

    # Quality
    print(f"\n  QUALITY — by ±2d\n  {'-'*80}")
    q_ents = []
    for name, det, r in all_entries[:15]:
        rq = eval_ov(det, lh_ov_dict, name, subset=quality, quiet=True)
        if rq:
            q_ents.append((name, rq))
    q_ents.sort(key=lambda x: x[1].get("acc_2d", 0), reverse=True)
    for name, r in q_ents[:10]:
        print(f"  {name:<40s} {r['n']:>3} {r['mae']:>5.2f}"
              f" {r.get('acc_1d',0):>5.1%} {r.get('acc_2d',0):>5.1%}"
              f" {r.get('acc_3d',0):>5.1%} {r.get('acc_5d',0):>5.1%}")

    # ===== PART E: MENSTRUAL PREDICTION =====
    print(f"\n{SEP}\n  E. MENSTRUAL PREDICTION\n{SEP}")

    oracle = {s: v for s, v in lh_ov_dict.items() if s in cs}

    print(f"\n  --- Baselines ---")
    predict_menses(cs, {}, subj_order, lh_ov_dict, eval_subset=labeled,
                   label="Calendar (labeled)")
    predict_menses(cs, oracle, subj_order, lh_ov_dict, fixed_luteal=13.0,
                   eval_subset=labeled, label="Oracle+lut13 (labeled)")
    predict_menses(cs, {}, subj_order, lh_ov_dict, eval_subset=quality,
                   label="Calendar (Q)")
    predict_menses(cs, oracle, subj_order, lh_ov_dict, fixed_luteal=13.0,
                   eval_subset=quality, label="Oracle+lut13 (Q)")

    print(f"\n  --- Top detectors (labeled) ---")
    for name, det, r in all_entries[:5]:
        for fl in [12, 13]:
            predict_menses(cs, det, subj_order, lh_ov_dict, fixed_luteal=float(fl),
                           eval_subset=labeled, label=f"{name}+lut{fl}")
        predict_menses(cs, det, subj_order, lh_ov_dict,
                       use_personal_luteal=True, eval_subset=labeled,
                       label=f"{name}+personal")

    print(f"\n  --- Top detectors (quality) ---")
    for name, det, r in all_entries[:5]:
        for fl in [12, 13]:
            predict_menses(cs, det, subj_order, lh_ov_dict, fixed_luteal=float(fl),
                           eval_subset=quality, label=f"{name}+lut{fl} (Q)")

    # ===== PART F: Confidence-based hybrid for menstrual =====
    print(f"\n{SEP}\n  F. Confidence-based Hybrid Prediction\n{SEP}")

    best_det_name, best_det, _ = all_entries[0]
    from sklearn.linear_model import Ridge as _R

    def _build_conf(det, cs, lh_ov_dict):
        """Build confidence = abs(detected - calendar_estimate) inverse."""
        confs = {}
        for sgk in det:
            ov = det[sgk]
            data = cs[sgk]
            hcl = data["hist_cycle_len"]
            expected = hcl * 0.575
            residual = abs(ov - expected)
            conf = 1.0 / (1.0 + residual)
            if sgk in lh_ov_dict:
                raw = data["nightly_temperature"]
                if not np.isnan(raw).all():
                    t = _clean(raw, sigma=2.0)
                    n = len(t)
                    if ov >= 3 and ov < n - 2:
                        pre = np.mean(t[max(0, ov-5):ov])
                        post = np.mean(t[ov:min(n, ov+5)])
                        shift = post - pre
                        if shift > 0.15:
                            conf = max(conf, 0.7)
                        if shift > 0.25:
                            conf = max(conf, 0.9)
            confs[sgk] = (ov, conf)
        return confs

    det_conf = _build_conf(best_det, cs, lh_ov_dict)
    for ct in [0.0, 0.3, 0.5, 0.7, 0.9]:
        hybrid_predict_menses(cs, det_conf, subj_order, lh_ov_dict,
                              fixed_luteal=13.0, conf_threshold=ct,
                              eval_subset=labeled,
                              label=f"conf>{ct} {best_det_name}+lut13 (labeled)")
        hybrid_predict_menses(cs, det_conf, subj_order, lh_ov_dict,
                              fixed_luteal=13.0, conf_threshold=ct,
                              eval_subset=quality,
                              label=f"conf>{ct} {best_det_name}+lut13 (Q)")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  COMPLETE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

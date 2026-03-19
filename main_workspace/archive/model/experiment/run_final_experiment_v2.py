"""
FINAL Experiment v2 — Maximum-effort Ovulation Detection & Menses Prediction
=============================================================================
Improvements over v1:
  1. Relative features: deviation from hist_expected_ov
  2. Wavelet denoising before feature extraction
  3. Multi-resolution features (raw + smoothed at multiple scales)
  4. AR model coefficients as features
  5. Stacking meta-learner: rule-based + ML predictions combined
  6. Per-subject adaptive: use subject's own history more
  7. Direct cycle length prediction with better features
  8. Try regime-switching: model temp as two regimes (low/high)

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_final_experiment_v2
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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
        past_lens, past_ovs = [], []
        for sgk in sgks:
            if sgk in cycle_series:
                cycle_series[sgk]["hist_cycle_len"] = np.mean(past_lens) if past_lens else 28.0
                cycle_series[sgk]["hist_cycle_std"] = np.std(past_lens) if len(past_lens) > 1 else 4.0
                cycle_series[sgk]["hist_n_cycles"] = len(past_lens)
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
# Enhanced Feature Extraction
# =====================================================================

def wavelet_denoise(arr, wavelet='db4', level=2):
    """Wavelet denoising."""
    try:
        import pywt
        coeffs = pywt.wavedec(arr, wavelet, level=level)
        threshold = np.sqrt(2 * np.log(len(arr))) * np.median(np.abs(coeffs[-1])) / 0.6745
        coeffs_t = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs_t, wavelet)[:len(arr)]
    except ImportError:
        return arr


def extract_enhanced_features(data, sigma=1.5):
    """Enhanced features: multi-resolution, relative, AR, wavelet."""
    n = data["cycle_len"]
    hcl = data["hist_cycle_len"]
    hcs = data.get("hist_cycle_std", 4.0)
    hn = data.get("hist_n_cycles", 0)
    exp_ov = hcl * 0.575
    feats = {}

    feats["hist_clen"] = hcl
    feats["hist_cstd"] = hcs
    feats["hist_n"] = hn

    def _temp_feats(raw, prefix, sigs=[1.0, 2.0, 3.0]):
        if np.isnan(raw).all():
            return {}
        f = {}
        t0 = _clean(raw, sigma=0)

        # Wavelet denoised
        tw = wavelet_denoise(t0)
        f[f"{prefix}_wav_mean"] = np.mean(tw)
        f[f"{prefix}_wav_std"] = np.std(tw)

        for sig in sigs:
            t = _clean(raw, sigma=sig)
            sfx = f"s{sig}"

            f[f"{prefix}_{sfx}_mean"] = np.mean(t)
            f[f"{prefix}_{sfx}_std"] = np.std(t)
            f[f"{prefix}_{sfx}_range"] = np.ptp(t)
            f[f"{prefix}_{sfx}_skew"] = float(pd.Series(t).skew())

            nadir = int(np.argmin(t))
            f[f"{prefix}_{sfx}_nadir_day"] = nadir
            f[f"{prefix}_{sfx}_nadir_dev"] = nadir - exp_ov

            grad = np.gradient(t)
            mgd = int(np.argmax(grad))
            f[f"{prefix}_{sfx}_mgd"] = mgd
            f[f"{prefix}_{sfx}_mgd_dev"] = mgd - exp_ov
            f[f"{prefix}_{sfx}_mgv"] = float(np.max(grad))

            # Absolute-day t-tests
            for d in [8, 10, 12, 14, 16, 18, 20]:
                if 5 <= d < n - 3:
                    try:
                        stat, _ = ttest_ind(t[d:], t[:d], alternative="greater")
                        f[f"{prefix}_{sfx}_tt{d}"] = float(stat) if not np.isnan(stat) else 0
                    except Exception:
                        f[f"{prefix}_{sfx}_tt{d}"] = 0
                else:
                    f[f"{prefix}_{sfx}_tt{d}"] = 0

            # Best split
            best_sc, best_sp = -np.inf, n // 2
            for sp in range(5, n - 3):
                try:
                    stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                    if not np.isnan(stat) and stat > best_sc:
                        best_sc = stat
                        best_sp = sp
                except Exception:
                    continue
            f[f"{prefix}_{sfx}_bs"] = best_sp
            f[f"{prefix}_{sfx}_bs_dev"] = best_sp - exp_ov
            f[f"{prefix}_{sfx}_bs_tstat"] = max(best_sc, 0)
            pre_m = np.mean(t[:best_sp])
            post_m = np.mean(t[best_sp:])
            f[f"{prefix}_{sfx}_shift"] = post_m - pre_m

            # Segment means (relative to expected ovulation)
            eov = int(max(5, min(n - 5, exp_ov)))
            if eov >= 5:
                f[f"{prefix}_{sfx}_pre5"] = np.mean(t[max(0, eov - 5):eov])
            if eov + 5 <= n:
                f[f"{prefix}_{sfx}_post5"] = np.mean(t[eov:eov + 5])
            if eov >= 5 and eov + 5 <= n:
                f[f"{prefix}_{sfx}_seg_shift"] = f.get(f"{prefix}_{sfx}_post5", 0) - f.get(f"{prefix}_{sfx}_pre5", 0)

            # Quantiles
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                f[f"{prefix}_{sfx}_q{int(q * 100)}"] = float(np.quantile(t, q))

        # AR(2) coefficients
        t1 = _clean(raw, sigma=1.5)
        if len(t1) > 5:
            try:
                from numpy.linalg import lstsq
                Y = t1[2:]
                X_ar = np.column_stack([t1[1:-1], t1[:-2], np.ones(len(Y))])
                coefs, _, _, _ = lstsq(X_ar, Y, rcond=None)
                f[f"{prefix}_ar1"] = float(coefs[0])
                f[f"{prefix}_ar2"] = float(coefs[1])
                f[f"{prefix}_ar_bias"] = float(coefs[2])
            except Exception:
                f[f"{prefix}_ar1"] = 0
                f[f"{prefix}_ar2"] = 0
                f[f"{prefix}_ar_bias"] = 0

        # Rolling window features
        t2 = _clean(raw, sigma=2.0)
        short_ma = pd.Series(t2).rolling(3, min_periods=1).mean().values
        long_ma = pd.Series(t2).rolling(7, min_periods=1).mean().values
        cross_day = None
        for i in range(7, n):
            if short_ma[i] > long_ma[i] and short_ma[i - 1] <= long_ma[i - 1]:
                cross_day = i
                break
        f[f"{prefix}_cross_day"] = cross_day if cross_day else n // 2
        f[f"{prefix}_cross_dev"] = (cross_day if cross_day else n // 2) - exp_ov

        # Autocorrelation
        ts = pd.Series(t1)
        for lag in [1, 2, 3, 5, 7]:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0

        return f

    feats.update(_temp_feats(data["nightly_temperature"], "nt"))
    if "noct_mean" in data:
        feats.update(_temp_feats(data["noct_mean"], "noct", sigs=[1.0, 2.0]))

    if "rhr_mean" in data:
        raw = data["rhr_mean"]
        if not np.isnan(raw).all():
            t = _clean(raw, sigma=1.0)
            feats["rhr_mean"] = np.mean(t)
            feats["rhr_std"] = np.std(t)
            feats["rhr_nadir"] = int(np.argmin(t))
            feats["rhr_nadir_dev"] = int(np.argmin(t)) - exp_ov

    for k in feats:
        if isinstance(feats[k], float) and (np.isnan(feats[k]) or np.isinf(feats[k])):
            feats[k] = 0.0
    return feats


# =====================================================================
# Rule-based detection methods
# =====================================================================

def detect_ttest_biphasic(cs, sigma=3.0, frac=0.575, pw=4.0, tk="nightly_temperature"):
    detected, confs = {}, {}
    for sgk, data in cs.items():
        raw = data.get(tk, data["nightly_temperature"])
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]
        if n < 12 or np.isnan(raw).all():
            detected[sgk] = int(round(frac * hcl))
            confs[sgk] = 0.0
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
            except:
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
            conf = min(1.0, max(0, best_tstat / 5)) * min(1.0, max(0, shift / 0.3))
        else:
            ov = int(round(frac * hcl))
            conf = 0.0
        detected[sgk] = ov
        confs[sgk] = conf
    return detected, confs


# =====================================================================
# ML Detection (LOSO, signal-only)
# =====================================================================

def ml_detect_loso(cs, lh_ov_dict, model_type="ridge", feat_fn=extract_enhanced_features):
    labeled = [sgk for sgk in cs if sgk in lh_ov_dict]
    all_feats, all_tgt, all_ids, all_sgks = [], [], [], []
    for sgk in labeled:
        feats = feat_fn(cs[sgk])
        if not feats or len(feats) < 5:
            continue
        all_feats.append(feats)
        all_tgt.append(lh_ov_dict[sgk])
        all_ids.append(cs[sgk]["id"])
        all_sgks.append(sgk)

    if len(all_feats) < 10:
        return {}, {}

    df = pd.DataFrame(all_feats).fillna(0)
    X = df.values
    y = np.array(all_tgt, dtype=float)
    ids = np.array(all_ids)
    uniq = np.unique(ids)

    from sklearn.preprocessing import StandardScaler
    det, confs = {}, {}

    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X[tr])
        X_te_s = sc.transform(X[te])
        y_tr = y[tr]

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        elif model_type == "svr":
            from sklearn.svm import SVR
            m = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == "huber":
            from sklearn.linear_model import HuberRegressor
            m = HuberRegressor(epsilon=1.35)
        elif model_type == "lasso":
            from sklearn.linear_model import Lasso
            m = Lasso(alpha=0.1)
        elif model_type == "bayridge":
            from sklearn.linear_model import BayesianRidge
            m = BayesianRidge()
        else:
            raise ValueError(model_type)

        m.fit(X_tr_s, y_tr)
        preds = m.predict(X_te_s)
        test_sgks = [all_sgks[i] for i in np.where(te)[0]]

        for sgk, pred in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            ov = int(round(max(5, min(clen - 3, pred))))
            det[sgk] = ov
            t = _clean(cs[sgk]["nightly_temperature"], sigma=2.0)
            nn = len(t)
            if ov < nn - 2:
                pre = np.mean(t[max(0, ov - 5):ov])
                post = np.mean(t[ov:min(nn, ov + 5)])
                confs[sgk] = min(1.0, max(0, (post - pre) / 0.25))
            else:
                confs[sgk] = 0.5

    return det, confs


# =====================================================================
# STACKING META-LEARNER
# =====================================================================

def stacking_detect(cs, lh_ov_dict, quality):
    """Meta-learner: combine rule-based + multiple ML predictions."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge as _R
    from sklearn.ensemble import RandomForestRegressor

    rb_configs = [
        (2.0, 0.575, 4.0), (2.5, 0.575, 4.0), (3.0, 0.575, 4.0),
        (2.0, 0.575, 5.0), (3.0, 0.55, 4.0),
    ]
    ml_types = ["ridge", "elastic", "svr", "rf", "bayridge"]

    labeled = [s for s in cs if s in lh_ov_dict]
    sgk2idx = {s: i for i, s in enumerate(labeled)}
    N = len(labeled)

    # Base predictions
    rb_preds = np.full((N, len(rb_configs)), np.nan)
    for j, (s, f, w) in enumerate(rb_configs):
        d, _ = detect_ttest_biphasic(cs, sigma=s, frac=f, pw=w)
        for sgk, ov in d.items():
            if sgk in sgk2idx:
                rb_preds[sgk2idx[sgk], j] = ov

    ml_preds = np.full((N, len(ml_types)), np.nan)
    for j, mt in enumerate(ml_types):
        d, _ = ml_detect_loso(cs, lh_ov_dict, model_type=mt)
        for sgk, ov in d.items():
            if sgk in sgk2idx:
                ml_preds[sgk2idx[sgk], j] = ov

    base = np.hstack([rb_preds, ml_preds])
    # Add hist_clen as meta-feature
    hist = np.array([cs[s]["hist_cycle_len"] for s in labeled]).reshape(-1, 1)
    meta_X = np.hstack([base, hist])

    y = np.array([lh_ov_dict[s] for s in labeled], dtype=float)
    ids = np.array([cs[s]["id"] for s in labeled])
    uniq = np.unique(ids)

    # Fill NaN with column mean
    for col in range(meta_X.shape[1]):
        mask = np.isnan(meta_X[:, col])
        if mask.any():
            meta_X[mask, col] = np.nanmean(meta_X[:, col])

    det = {}
    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue

        sc = StandardScaler()
        X_tr = sc.fit_transform(meta_X[tr])
        X_te = sc.transform(meta_X[te])

        m = _R(alpha=0.5)
        m.fit(X_tr, y[tr])
        preds = m.predict(X_te)

        test_sgks = [labeled[i] for i in np.where(te)[0]]
        for sgk, pred in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, pred))))

    return det


# =====================================================================
# Direct Cycle Length Prediction
# =====================================================================

def direct_clen_predict(cs, lh_ov_dict, model_type="ridge"):
    """Predict cycle_len directly (for menstrual prediction)."""
    all_labeled = [s for s in cs if s in lh_ov_dict]
    all_feats, all_tgt, all_ids, all_sgks = [], [], [], []
    for sgk in all_labeled:
        feats = extract_enhanced_features(cs[sgk])
        if not feats or len(feats) < 5:
            continue
        all_feats.append(feats)
        all_tgt.append(cs[sgk]["cycle_len"])
        all_ids.append(cs[sgk]["id"])
        all_sgks.append(sgk)

    if len(all_feats) < 10:
        return {}

    df = pd.DataFrame(all_feats).fillna(0)
    X = df.values
    y = np.array(all_tgt, dtype=float)
    ids = np.array(all_ids)
    uniq = np.unique(ids)

    from sklearn.preprocessing import StandardScaler
    preds = {}

    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        elif model_type == "bayridge":
            from sklearn.linear_model import BayesianRidge
            m = BayesianRidge()
        else:
            raise ValueError(model_type)

        m.fit(X_tr, y[tr])
        ps = m.predict(X_te)
        test_sgks = [all_sgks[i] for i in np.where(te)[0]]
        for sgk, p in zip(test_sgks, ps):
            preds[sgk] = float(p)

    return preds


# =====================================================================
# Menstrual Prediction Pipeline
# =====================================================================

def predict_menses(cs, detected, confs, subj_order, lh_ov_dict,
                   fixed_lut=13.0, conf_thresh=0.0, eval_subset=None, label=""):
    pop_lut = fixed_lut
    subj_plut = defaultdict(list)
    subj_pclen = defaultdict(list)
    errs = []
    n_ov, n_cal = 0, 0
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            data = cs[sgk]
            actual = data["cycle_len"]
            pl = subj_plut[uid]
            pc = subj_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0

            ov = detected.get(sgk)
            conf = confs.get(sgk, 0.0)
            if ov is not None and ov > 3 and conf >= conf_thresh:
                pred = ov + lut
                n_ov += 1
            else:
                pred = acl
                n_cal += 1

            err = pred - actual
            if ev is None or sgk in ev:
                errs.append(err)
            subj_pclen[uid].append(actual)
            if ov is not None:
                el = actual - ov
                if 8 <= el <= 22:
                    subj_plut[uid].append(el)

    if not errs:
        return {}
    ae = np.abs(errs)
    return _pr(f"{label} (ov:{n_ov},cal:{n_cal})", ae, prefix="    ")


def predict_menses_direct(cs, pred_clens, subj_order, eval_subset=None, label=""):
    subj_pclen = defaultdict(list)
    errs = []
    ev = set(eval_subset) if eval_subset else None
    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual = cs[sgk]["cycle_len"]
            pc = subj_pclen[uid]
            acl = np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc)))) if pc else 28.0
            pred = pred_clens.get(sgk, acl)
            err = pred - actual
            if ev is None or sgk in ev:
                errs.append(err)
            subj_pclen[uid].append(actual)
    if not errs:
        return {}
    return _pr(label, np.abs(errs), prefix="    ")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"\n{SEP}\n  FINAL v2 — Maximum-effort Detection & Prediction\n{SEP}")
    t0 = time.time()

    lh, lut, cs, quality, so = load_data()
    labeled = set(s for s in cs if s in lh)
    print(f"  Cycles: {len(cs)} | Labeled: {len(labeled)} | Quality: {len(quality)}")

    oracle = {s: v for s, v in lh.items() if s in cs}
    oconf = {s: 1.0 for s in oracle}

    # =================================================================
    # A: Oracle ceiling
    # =================================================================
    print(f"\n{SEP}\n  A. ORACLE CEILINGS\n{SEP}")
    for fl in [12, 13, 14]:
        predict_menses(cs, oracle, oconf, so, lh, fixed_lut=float(fl),
                       eval_subset=labeled, label=f"Oracle+lut{fl} (lab)")
        predict_menses(cs, oracle, oconf, so, lh, fixed_lut=float(fl),
                       eval_subset=quality, label=f"Oracle+lut{fl} (Q)")
    predict_menses(cs, {}, {}, so, lh, eval_subset=labeled, label="Calendar (lab)")
    predict_menses(cs, {}, {}, so, lh, eval_subset=quality, label="Calendar (Q)")

    # =================================================================
    # B: Ovulation Detection — Rule-based
    # =================================================================
    print(f"\n{SEP}\n  B. RULE-BASED OVULATION DETECTION\n{SEP}")
    all_methods = {}
    for s in [2.0, 2.5, 3.0]:
        for f in [0.55, 0.575]:
            for w in [3.5, 4.0, 5.0]:
                d, c = detect_ttest_biphasic(cs, sigma=s, frac=f, pw=w)
                tag = f"rb-σ{s}-f{f}-w{w}"
                errs = [d[s2] - lh[s2] for s2 in labeled if s2 in d and s2 in lh]
                r = _pr(tag, np.abs(errs))
                all_methods[tag] = (d, c, r)

    # =================================================================
    # C: ML Detection (LOSO, enhanced features)
    # =================================================================
    print(f"\n{SEP}\n  C. ML DETECTION (enhanced features, LOSO)\n{SEP}")
    for mt in ["ridge", "elastic", "svr", "rf", "gbdt", "huber", "bayridge", "lasso"]:
        d, c = ml_detect_loso(cs, lh, model_type=mt)
        if d:
            errs = [d[s2] - lh[s2] for s2 in d if s2 in lh]
            r = _pr(f"ML-{mt}", np.abs(errs))
            all_methods[f"ML-{mt}"] = (d, c, r)

    # =================================================================
    # D: Stacking
    # =================================================================
    print(f"\n{SEP}\n  D. STACKING META-LEARNER\n{SEP}")
    ds = stacking_detect(cs, lh, quality)
    if ds:
        errs = [ds[s2] - lh[s2] for s2 in ds if s2 in lh]
        r = _pr("stacking", np.abs(errs))
        all_methods["stacking"] = (ds, {s: 0.5 for s in ds}, r)

    # =================================================================
    # E: Ensembles
    # =================================================================
    print(f"\n{SEP}\n  E. ENSEMBLES\n{SEP}")
    ranked = sorted(all_methods.items(), key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)

    for topN in [3, 5, 7]:
        if len(ranked) >= topN:
            items = ranked[:topN]
            all_sgks = set()
            for _, (d, c, r) in items:
                all_sgks.update(d.keys())
            ws = [max(v[2].get("acc_2d", 0), 0.01) for _, v in items]
            ed = {}
            ec = {}
            for sgk in all_sgks:
                vals, ww, cc = [], [], []
                for (_, (d, c, r)), w in zip(items, ws):
                    if sgk in d:
                        vals.append(d[sgk])
                        ww.append(w)
                        cc.append(c.get(sgk, 0.5))
                if vals:
                    ed[sgk] = int(round(np.average(vals, weights=ww)))
                    ec[sgk] = float(np.mean(cc))
            errs = [ed[s2] - lh[s2] for s2 in labeled if s2 in ed and s2 in lh]
            if errs:
                r = _pr(f"ens-top{topN}", np.abs(errs))
                all_methods[f"ens-top{topN}"] = (ed, ec, r)

    # =================================================================
    # F: FINAL RANKING
    # =================================================================
    print(f"\n{SEP}\n  F. OVULATION DETECTION RANKING\n{SEP}")
    ranked = sorted(all_methods.items(), key=lambda x: x[1][2].get("acc_2d", 0), reverse=True)
    print(f"  {'Method':<45s} {'N':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
    print(f"  {'-'*85}")
    for tag, (d, c, r) in ranked[:20]:
        print(f"  {tag:<45s} {r['n']:>3} {r['mae']:>5.2f}"
              f" {r.get('acc_1d',0):>5.1%} {r.get('acc_2d',0):>5.1%}"
              f" {r.get('acc_3d',0):>5.1%} {r.get('acc_5d',0):>5.1%}")

    print(f"\n  QUALITY CYCLES\n  {'-'*85}")
    q_res = []
    for tag, (d, c, r) in ranked[:15]:
        errs = [d[s] - lh[s] for s in quality if s in d and s in lh]
        if errs:
            rq = _pr(f"Q-{tag}", np.abs(errs))
            q_res.append((tag, rq))
    q_res.sort(key=lambda x: x[1].get("acc_2d", 0), reverse=True)

    # =================================================================
    # G: MENSTRUAL PREDICTION
    # =================================================================
    print(f"\n{SEP}\n  G. MENSTRUAL PREDICTION\n{SEP}")

    # Top detectors
    print(f"\n  --- Top 5 (labeled) ---")
    for tag, (d, c, r) in ranked[:5]:
        for fl in [12, 13]:
            predict_menses(cs, d, c, so, lh, fixed_lut=float(fl),
                           eval_subset=labeled, label=f"{tag}+lut{fl}")

    print(f"\n  --- Top 5 (quality) ---")
    for tag, (d, c, r) in ranked[:5]:
        for fl in [12, 13]:
            predict_menses(cs, d, c, so, lh, fixed_lut=float(fl),
                           eval_subset=quality, label=f"{tag}+lut{fl} (Q)")

    # Confidence-based hybrid
    print(f"\n  --- Confidence hybrid (best detector) ---")
    best_tag, (best_d, best_c, _) = ranked[0]
    for ct in [0.0, 0.2, 0.4, 0.6]:
        for fl in [12, 13]:
            predict_menses(cs, best_d, best_c, so, lh, fixed_lut=float(fl),
                           conf_thresh=ct, eval_subset=labeled,
                           label=f"conf>{ct}+lut{fl}")
            predict_menses(cs, best_d, best_c, so, lh, fixed_lut=float(fl),
                           conf_thresh=ct, eval_subset=quality,
                           label=f"conf>{ct}+lut{fl} (Q)")

    # =================================================================
    # H: DIRECT CYCLE LENGTH PREDICTION
    # =================================================================
    print(f"\n{SEP}\n  H. DIRECT CYCLE LENGTH PREDICTION\n{SEP}")
    for mt in ["ridge", "rf", "elastic", "gbdt", "bayridge"]:
        pc = direct_clen_predict(cs, lh, model_type=mt)
        if pc:
            predict_menses_direct(cs, pc, so, eval_subset=labeled,
                                  label=f"direct-{mt} (lab)")
            predict_menses_direct(cs, pc, so, eval_subset=quality,
                                  label=f"direct-{mt} (Q)")

    # Blend: direct + ov-countdown
    print(f"\n  --- Blend: best direct + best ov-countdown ---")
    best_direct_mt = "ridge"
    pc_best = direct_clen_predict(cs, lh, model_type=best_direct_mt)
    best_d, best_c = ranked[0][1][0], ranked[0][1][1]
    for alpha in [0.3, 0.5, 0.7]:
        blended = {}
        for sgk in cs:
            ov = best_d.get(sgk)
            conf = best_c.get(sgk, 0.0)
            direct_pred = pc_best.get(sgk)
            hcl = cs[sgk]["hist_cycle_len"]
            if ov is not None and direct_pred is not None:
                ov_pred = ov + 13.0
                blended[sgk] = alpha * ov_pred + (1 - alpha) * direct_pred
            elif direct_pred is not None:
                blended[sgk] = direct_pred
            else:
                blended[sgk] = hcl
        predict_menses_direct(cs, blended, so, eval_subset=labeled,
                              label=f"blend(α={alpha}) (lab)")
        predict_menses_direct(cs, blended, so, eval_subset=quality,
                              label=f"blend(α={alpha}) (Q)")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  COMPLETE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

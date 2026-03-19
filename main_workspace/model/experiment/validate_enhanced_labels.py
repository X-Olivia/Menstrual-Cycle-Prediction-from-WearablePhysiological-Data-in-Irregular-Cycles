"""
Validate Enhanced Ovulation Labels (95 → 122 cycles)
=====================================================
Compare key algorithms on original vs enhanced label sets.
Runs a representative subset of algorithms from run_multisignal_ov_v2.

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.validate_enhanced_labels
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from collections import defaultdict
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels, get_enhanced_ovulation_labels

SEP = "=" * 76


def _clean(arr, sigma=0):
    s = pd.Series(arr).interpolate(limit_direction="both")
    out = s.fillna(s.mean() if s.notna().any() else 0).values
    if sigma > 0:
        out = gaussian_filter1d(out, sigma=sigma)
    return out


def _pr(tag, ae, prefix="  "):
    ae = np.array(ae, dtype=float)
    if len(ae) == 0:
        return {}
    r = {"n": len(ae), "mae": float(np.mean(ae)),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"{prefix}[{tag}] n={r['n']} MAE={r['mae']:.2f}"
          f" ±1d={r['acc_1d']:.1%} ±2d={r['acc_2d']:.1%}"
          f" ±3d={r['acc_3d']:.1%} ±5d={r['acc_5d']:.1%}")
    return r


def load_signals_with_labels(use_enhanced=False):
    """Load all signals and specified label set."""
    cc = pd.read_csv(CYCLE_CSV)
    key = ["id", "study_interval", "day_in_study"]

    ct = pd.read_csv(os.path.join(WORKSPACE, "subdataset/computed_temperature_cycle.csv"))
    ct_daily = ct[key + ["nightly_temperature"]].drop_duplicates(subset=key)

    wt = pd.read_csv(os.path.join(WORKSPACE, "subdataset/wrist_temperature_cycle.csv"),
                      usecols=key + ["timestamp", "temperature_diff_from_baseline"])
    wt["hour"] = pd.to_datetime(wt["timestamp"], format="%H:%M:%S").dt.hour
    noct_wt = wt[(wt["hour"] >= 0) & (wt["hour"] <= 6)]
    noct_temp_daily = noct_wt.groupby(key)["temperature_diff_from_baseline"].mean().reset_index()
    noct_temp_daily.rename(columns={"temperature_diff_from_baseline": "noct_temp"}, inplace=True)

    rhr = pd.read_csv(os.path.join(WORKSPACE, "subdataset/resting_heart_rate_cycle.csv"),
                       usecols=key + ["value"])
    rhr_daily = rhr.groupby(key)["value"].mean().reset_index()
    rhr_daily.rename(columns={"value": "rhr"}, inplace=True)

    hrv = pd.read_csv(os.path.join(WORKSPACE, "subdataset/heart_rate_variability_details_cycle.csv"))
    hrv_daily = hrv.groupby(key).agg(
        rmssd_mean=("rmssd", "mean"), rmssd_std=("rmssd", "std"),
        lf_mean=("low_frequency", "mean"), hf_mean=("high_frequency", "mean"),
        hrv_coverage=("coverage", "mean"),
    ).reset_index()
    hrv_daily["lf_hf_ratio"] = hrv_daily["lf_mean"] / hrv_daily["hf_mean"].clip(lower=1)

    hr_path = os.path.join(WORKSPACE, "subdataset/heart_rate_cycle.csv")
    hr_aggs = []
    for chunk in pd.read_csv(hr_path, chunksize=2_000_000,
                              usecols=key + ["timestamp", "bpm", "confidence"]):
        chunk["hour"] = pd.to_datetime(chunk["timestamp"]).dt.hour
        noct = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 6) & (chunk["confidence"] >= 1)]
        if len(noct) > 0:
            hr_aggs.append(noct.groupby(key)["bpm"].agg(["mean", "std", "min"]).reset_index())
    hr_daily = pd.concat(hr_aggs).groupby(key).mean().reset_index()
    hr_daily.rename(columns={"mean": "noct_hr_mean", "std": "noct_hr_std",
                              "min": "noct_hr_min"}, inplace=True)

    merged = cc.merge(ct_daily, on=key, how="left")
    merged = merged.merge(noct_temp_daily, on=key, how="left")
    merged = merged.merge(rhr_daily, on=key, how="left")
    merged = merged.merge(hrv_daily, on=key, how="left")
    merged = merged.merge(hr_daily, on=key, how="left")

    if use_enhanced:
        lh_df = get_enhanced_ovulation_labels()
    else:
        lh_df = get_lh_ovulation_labels()
    lh_dict = dict(zip(lh_df["small_group_key"], lh_df["ov_dic"]))

    signal_cols = ["nightly_temperature", "noct_temp", "rhr",
                   "rmssd_mean", "rmssd_std", "lf_mean", "hf_mean", "lf_hf_ratio",
                   "noct_hr_mean", "noct_hr_std", "noct_hr_min", "hrv_coverage"]

    cycle_series = {}
    for sgk, grp in merged.groupby("small_group_key"):
        grp = grp.sort_values("day_in_study")
        cs = grp["day_in_study"].min()
        n = len(grp)
        if n < 10:
            continue
        entry = {"dic": (grp["day_in_study"] - cs).values, "id": grp["id"].values[0],
                 "cycle_len": n}
        for sc in signal_cols:
            entry[sc] = grp[sc].values if sc in grp.columns else np.full(n, np.nan)
        cycle_series[sgk] = entry

    sgk_order = (merged.groupby("small_group_key")["day_in_study"]
                 .min().reset_index().rename(columns={"day_in_study": "start"}))
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
                cycle_series[sgk]["hist_cycle_std"] = np.std(past_lens) if len(past_lens) > 1 else 4.0
                past_lens.append(cycle_series[sgk]["cycle_len"])

    quality = set()
    for sgk in cycle_series:
        if sgk not in lh_dict:
            continue
        raw = cycle_series[sgk]["nightly_temperature"]
        if np.isnan(raw).all():
            continue
        t = pd.Series(raw).interpolate(limit_direction="both").values
        ov = lh_dict[sgk]
        n = len(t)
        if ov < 3 or ov + 2 >= n:
            continue
        pre = np.mean(t[max(0, ov - 5):ov])
        post = np.mean(t[ov + 2:min(n, ov + 7)])
        if post - pre >= 0.2:
            quality.add(sgk)

    labeled = [s for s in cycle_series if s in lh_dict]
    return lh_dict, cycle_series, quality, subj_order, signal_cols, labeled


# =====================================================================
# Representative algorithms (from run_multisignal_ov_v2.py)
# =====================================================================

def detect_ttest_optimal(cs, sig_key, sigma=2.0, invert=False, pw=4.0, frac=0.575):
    det, conf = {}, {}
    for sgk, data in cs.items():
        raw = data.get(sig_key)
        hcl = data["hist_cycle_len"]
        n = data["cycle_len"]
        if raw is None or np.isnan(raw).all() or n < 12:
            continue
        s = _clean(raw, sigma=sigma)
        if invert:
            s = -s
        expected = max(7, min(n - 5, int(hcl * frac)))
        best_t, best_d = -1e9, expected
        for d in range(5, n - 3):
            pre = s[max(0, d - 5):d]
            post = s[d:min(n, d + 5)]
            if len(pre) < 2 or len(post) < 2:
                continue
            tval, _ = ttest_ind(post, pre, equal_var=False)
            gw = np.exp(-((d - expected) ** 2) / (2 * pw ** 2))
            score = tval * gw
            if score > best_t:
                best_t = score
                best_d = d
        det[sgk] = best_d
        conf[sgk] = max(0, min(1, best_t / 5.0))
    return det, conf


def ml_detect_loso(cs, lh, model_type="gbdt"):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor

    model_map = {
        "ridge": lambda: Ridge(alpha=10),
        "elastic": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "svr": lambda: SVR(kernel="rbf", C=10, epsilon=1.0),
        "rf": lambda: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "gbdt": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                   learning_rate=0.05, random_state=42),
        "bayridge": lambda: BayesianRidge(),
        "knn": lambda: KNeighborsRegressor(n_neighbors=5),
        "huber": lambda: HuberRegressor(epsilon=1.5, max_iter=200),
    }

    try:
        from xgboost import XGBRegressor
        model_map["xgb"] = lambda: XGBRegressor(n_estimators=200, max_depth=3,
                                                  learning_rate=0.05, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        model_map["lgbm"] = lambda: LGBMRegressor(n_estimators=200, max_depth=3,
                                                    learning_rate=0.05, verbose=-1)
    except ImportError:
        pass

    if model_type not in model_map:
        return {}, {}

    sig_keys = ["nightly_temperature", "noct_temp", "rhr",
                "rmssd_mean", "rmssd_std", "noct_hr_mean", "noct_hr_std"]

    def _extract(data):
        feats = {}
        hcl = data["hist_cycle_len"]
        feats["hist_clen"] = hcl
        feats["hist_cstd"] = data.get("hist_cycle_std", 4.0)
        n = data["cycle_len"]
        exp_ov = hcl * 0.575

        for sk in sig_keys:
            raw = data.get(sk)
            if raw is None or np.isnan(raw).all():
                continue
            t = _clean(raw, sigma=1.5)
            prefix = sk[:4]
            feats[f"{prefix}_mean"] = float(np.mean(t))
            feats[f"{prefix}_std"] = float(np.std(t))
            feats[f"{prefix}_range"] = float(np.ptp(t))
            half = n // 2
            feats[f"{prefix}_h1_mean"] = float(np.mean(t[:half]))
            feats[f"{prefix}_h2_mean"] = float(np.mean(t[half:]))
            feats[f"{prefix}_h_diff"] = feats[f"{prefix}_h2_mean"] - feats[f"{prefix}_h1_mean"]
            feats[f"{prefix}_nadir"] = float(np.argmin(t)) / max(n - 1, 1)
            best_t, best_d = -1e9, n // 2
            for d in range(5, n - 3):
                pre = t[max(0, d - 4):d]
                post = t[d:min(n, d + 4)]
                if len(pre) < 2 or len(post) < 2:
                    continue
                tv, _ = ttest_ind(post, pre, equal_var=False)
                if sk in ("rmssd_mean", "rmssd_std"):
                    tv = -tv
                if tv > best_t:
                    best_t = tv
                    best_d = d
            feats[f"{prefix}_bs"] = best_d
            feats[f"{prefix}_bs_n"] = best_d / max(n - 1, 1)
        for k in list(feats):
            if not np.isfinite(feats[k]):
                feats[k] = 0.0
        return feats

    sgks_labeled = [s for s in cs if s in lh]
    if not sgks_labeled:
        return {}, {}

    feats_all = []
    targets = []
    uids = []
    valid_sgks = []
    for s in sgks_labeled:
        f = _extract(cs[s])
        if not f:
            continue
        feats_all.append(f)
        targets.append(lh[s])
        uids.append(cs[s]["id"])
        valid_sgks.append(s)

    if len(feats_all) < 10:
        return {}, {}

    df_feat = pd.DataFrame(feats_all).fillna(0)
    X = df_feat.values.astype(float)
    y = np.array(targets, dtype=float)
    uid_arr = np.array(uids)
    unique_uids = np.unique(uid_arr)

    det, confs = {}, {}
    for uid in unique_uids:
        te = uid_arr == uid
        tr = ~te
        if tr.sum() < 5 or te.sum() == 0:
            continue
        model = model_map[model_type]()
        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])
        for i, idx in enumerate(np.where(te)[0]):
            s = valid_sgks[idx]
            n = cs[s]["cycle_len"]
            p = int(np.clip(np.round(preds[i]), 5, n - 3))
            det[s] = p
            confs[s] = 0.7
    return det, confs


def predict_menses(cs, det, conf, so, lh, fl=13.0, eval_subset=None, label=""):
    lut_bank = defaultdict(list)
    for s, ov in det.items():
        if s in lh:
            uid = cs[s]["id"]
            cl = cs[s]["cycle_len"]
            lut_bank[uid].append(cl - ov)

    errors = []
    for uid, sgks in so.items():
        past_lut = []
        past_lens = []
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual_len = cs[sgk]["cycle_len"]

            if sgk in det and conf.get(sgk, 0) > 0.3:
                my_lut = np.mean(past_lut) if past_lut else fl
                pred = det[sgk] + my_lut
            elif past_lens:
                pred = np.mean(past_lens)
            else:
                pred = 28.0

            if eval_subset is None or sgk in eval_subset:
                if sgk in lh:
                    errors.append(abs(pred - actual_len))

            if sgk in lh:
                true_lut = actual_len - lh[sgk]
                past_lut.append(true_lut)
            past_lens.append(actual_len)

    if errors:
        ae = np.array(errors)
        print(f"  [{label}] n={len(ae)} MAE={np.mean(ae):.2f}"
              f" ±1d={float((ae<=1).mean()):.1%} ±2d={float((ae<=2).mean()):.1%}"
              f" ±3d={float((ae<=3).mean()):.1%}")


def main():
    t0 = time.time()

    for label_name, use_enh in [("ORIGINAL (95)", False), ("ENHANCED (122)", True)]:
        print(f"\n{'#' * 76}")
        print(f"  LABEL SET: {label_name}")
        print(f"{'#' * 76}")

        print(f"\n  Loading signals...")
        lh, cs, quality, so, sig_cols, labeled = load_signals_with_labels(use_enhanced=use_enh)
        labeled_set = set(labeled)
        print(f"  Labeled: {len(labeled)} | Quality: {len(quality)}")

        all_results = []

        # A. Key rule-based algorithms
        print(f"\n{SEP}\n  A. RULE-BASED (Key methods)\n{SEP}")
        for sig, inv, sigma in [
            ("nightly_temperature", False, 2.0),
            ("noct_temp", False, 2.0),
            ("rhr", False, 1.5),
            ("rmssd_mean", True, 1.5),
            ("noct_hr_mean", False, 1.5),
        ]:
            tag = f"ttest-{sig[:8]}-σ{sigma}"
            d, c = detect_ttest_optimal(cs, sig, sigma=sigma, invert=inv)
            errs = [abs(d[s] - lh[s]) for s in d if s in labeled_set]
            if errs:
                r = _pr(tag, errs)
                all_results.append((tag, d, c, r))

        # B. Key ML models (LOSO)
        print(f"\n{SEP}\n  B. ML (LOSO)\n{SEP}")
        for mt in ["ridge", "rf", "gbdt", "xgb", "lgbm"]:
            tag = f"ML-{mt}"
            d, c = ml_detect_loso(cs, lh, model_type=mt)
            if d:
                errs = [abs(d[s] - lh[s]) for s in d if s in lh]
                if errs:
                    r = _pr(tag, errs)
                    all_results.append((tag, d, c, r))

        # C. Final ranking
        print(f"\n{SEP}\n  RANKING — ALL LABELED (n={len(labeled)})\n{SEP}")
        all_results.sort(key=lambda x: x[3].get("acc_2d", 0), reverse=True)
        print(f"  {'Method':<35s} {'n':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
        print(f"  {'-'*72}")
        for tag, d, c, r in all_results:
            print(f"  {tag:<35s} {r['n']:>3} {r['mae']:>5.2f}"
                  f" {r['acc_1d']:>5.1%} {r['acc_2d']:>5.1%}"
                  f" {r['acc_3d']:>5.1%} {r['acc_5d']:>5.1%}")

        # Quality subset
        print(f"\n  QUALITY SUBSET (n={len(quality)})")
        print(f"  {'Method':<35s} {'n':>3} {'MAE':>5} {'±1d':>6} {'±2d':>6} {'±3d':>6} {'±5d':>6}")
        print(f"  {'-'*72}")
        for tag, d, c, r in all_results:
            errs_q = [abs(d[s] - lh[s]) for s in d if s in quality and s in lh]
            if errs_q:
                ae = np.array(errs_q)
                print(f"  {tag:<35s} {len(ae):>3} {np.mean(ae):>5.2f}"
                      f" {(ae<=1).mean():>5.1%} {(ae<=2).mean():>5.1%}"
                      f" {(ae<=3).mean():>5.1%} {(ae<=5).mean():>5.1%}")

        # D. Menstrual prediction with top detector
        print(f"\n{SEP}\n  MENSTRUAL PREDICTION (top detector)\n{SEP}")
        if all_results:
            best_tag, best_d, best_c, _ = all_results[0]
            for fl in [12, 13, 14]:
                predict_menses(cs, best_d, best_c, so, lh, fl=float(fl),
                               eval_subset=labeled_set, label=f"{best_tag}+lut{fl}")

        # Oracle & calendar baselines
        oracle = {s: v for s, v in lh.items() if s in cs}
        oconf = {s: 1.0 for s in oracle}
        predict_menses(cs, oracle, oconf, so, lh, fl=13.0,
                       eval_subset=labeled_set, label="Oracle+lut13")
        predict_menses(cs, {}, {}, so, lh,
                       eval_subset=labeled_set, label="Calendar-only")

    elapsed = time.time() - t0
    print(f"\n{SEP}\n  DONE ({elapsed:.0f}s)\n{SEP}")


if __name__ == "__main__":
    main()

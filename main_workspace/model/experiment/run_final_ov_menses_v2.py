"""
Final Pipeline v5 — Corrected with actual luteal=12d
=====================================================
Key finding from v4: actual luteal phase = 12.0 ± 1.9 days (NOT 14!)
Using correct population luteal dramatically improves menstrual prediction.

Also:
  - Best ov detection on quality cycles: σ1.5-f0.5-w5.0 → ±3d=84.4%, MAE=2.2d
  - Confidence-based routing: only countdown when shift is clear
  - Iterative personal luteal: calibrate from past detected ov + actual cycle_len

Usage:
    cd /Users/xujing/FYP/main_workspace
    python -u -m model.experiment.run_final_ov_menses_v2
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from model.config import CYCLE_CSV, WORKSPACE
from model.ovulation_detect import get_lh_ovulation_labels

SEP = "=" * 76


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
                cycle_series[sgk]["hist_cycle_lens"] = list(past_lens)
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


def detect_with_confidence(cycle_series, smooth_sigma=2.5, frac_prior=0.575,
                           prior_width=4.0):
    """Biphasic + t-test ensemble, returns {sgk: (ov_day, confidence, shift)}."""
    result = {}
    for sgk, data in cycle_series.items():
        raw = data["temps"]
        dic = data["dic"]
        n = len(raw)
        hcl = data["hist_cycle_len"]

        if n < 12 or np.isnan(raw).all():
            result[sgk] = (int(round(frac_prior * hcl)), 0.0, 0.0)
            continue

        t = _clean(raw, sigma=smooth_sigma)
        expected = max(8, hcl * frac_prior)

        best_tstat, best_tsp = -np.inf, None
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
            pp = np.exp(-0.5 * ((dic[sp] - expected) / prior_width) ** 2)
            ws = stat * pp
            if ws > best_tstat:
                best_tstat = ws
                best_tsp = sp

        best_sse_sc, best_bsp = np.inf, None
        for sp in range(5, n - 3):
            mu1, mu2 = np.mean(t[:sp]), np.mean(t[sp:])
            if mu2 <= mu1:
                continue
            sse = np.sum((t[:sp] - mu1) ** 2) + np.sum((t[sp:] - mu2) ** 2)
            pen = 0.5 * ((dic[sp] - expected) / prior_width) ** 2
            sc = sse + pen
            if sc < best_sse_sc:
                best_sse_sc = sc
                best_bsp = sp

        candidates = []
        if best_tsp is not None:
            shift_t = np.mean(t[best_tsp:]) - np.mean(t[:best_tsp])
            candidates.append((int(dic[best_tsp]), best_tstat, shift_t))
        if best_bsp is not None:
            shift_b = np.mean(t[best_bsp:]) - np.mean(t[:best_bsp])
            candidates.append((int(dic[best_bsp]), shift_b * 10, shift_b))

        if candidates:
            ov_day = int(round(np.mean([c[0] for c in candidates])))
            conf = max(c[1] for c in candidates)
            shift = max(c[2] for c in candidates)
            result[sgk] = (ov_day, conf, shift)
        else:
            result[sgk] = (int(round(frac_prior * hcl)), 0.0, 0.0)

    return result


def eval_ov(detected, lh_ov_dict, name, subset=None):
    keys = subset if subset else lh_ov_dict.keys()
    errors = [detected[s][0] - lh_ov_dict[s]
              for s in keys if s in detected and s in lh_ov_dict]
    if not errors:
        return {}
    ae = np.abs(errors)
    r = {"n": len(ae), "mae": float(ae.mean()),
         "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
         "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
    print(f"  [{name}] n={r['n']}"
          f" | MAE={r['mae']:.2f}d"
          f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
          f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
    return r


def predict_menses(cycle_series, detections, subj_order, lh_ov_dict,
                   conf_threshold=0.0, shift_threshold=0.0,
                   fixed_luteal=12.0, use_personal_luteal=False,
                   label="", eval_subset=None):
    """Per-cycle menstrual prediction with configurable routing."""
    pop_luteal = fixed_luteal
    subj_past_luteal = defaultdict(list)
    subj_past_clens = defaultdict(list)

    errs_all, errs_ov, errs_cal = [], [], []

    eval_set = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cycle_series:
                continue
            data = cycle_series[sgk]
            actual = data["cycle_len"]

            past_luts = subj_past_luteal[uid]
            past_clens = subj_past_clens[uid]
            luteal = np.mean(past_luts) if (use_personal_luteal and past_luts) else pop_luteal

            if past_clens:
                w = np.exp(np.linspace(-1, 0, len(past_clens)))
                avg_clen = np.average(past_clens, weights=w)
            else:
                avg_clen = 28.0

            det = detections.get(sgk)
            use_ov = False
            if det is not None:
                ov_day, conf, shift = det
                if conf >= conf_threshold and shift >= shift_threshold:
                    pred = ov_day + luteal
                    use_ov = True

            if not use_ov:
                pred = avg_clen

            err = pred - actual

            should_record = eval_set is None or sgk in eval_set
            if should_record:
                errs_all.append(err)
                if use_ov:
                    errs_ov.append(err)
                else:
                    errs_cal.append(err)

            subj_past_clens[uid].append(actual)
            if det is not None:
                el = actual - det[0]
                if 8 <= el <= 22:
                    subj_past_luteal[uid].append(el)

    def _s(errs, tag):
        if not errs:
            return {}
        ae = np.abs(errs)
        r = {"n": len(ae), "mae": float(ae.mean()),
             "acc_1d": float((ae <= 1).mean()), "acc_2d": float((ae <= 2).mean()),
             "acc_3d": float((ae <= 3).mean()), "acc_5d": float((ae <= 5).mean())}
        print(f"    [{tag}] n={r['n']}"
              f" | MAE={r['mae']:.2f}d"
              f" | ±1d={r['acc_1d']:.1%} | ±2d={r['acc_2d']:.1%}"
              f" | ±3d={r['acc_3d']:.1%} | ±5d={r['acc_5d']:.1%}")
        return r

    if label:
        n_ov = len(errs_ov)
        n_cal = len(errs_cal)
        print(f"\n  {label}  (ov:{n_ov}, cal:{n_cal})")
    ra = _s(errs_all, "ALL")
    ro = _s(errs_ov, "ov-countdown")
    rc = _s(errs_cal, "cal-fallback")
    return ra, ro, rc


def main():
    print(f"\n{SEP}\n  Final Pipeline v5 — Corrected Luteal + Quality Focus\n{SEP}")

    lh_ov_dict, lh_luteal, cs, quality, subj_order = load_data()
    n_labeled = sum(1 for s in cs if s in lh_ov_dict)
    lut_vals = [lh_luteal[s] for s in lh_luteal if s in cs]
    print(f"  Cycles: {len(cs)} | Labeled: {n_labeled} | Quality: {len(quality)}")
    print(f"  Actual luteal: mean={np.mean(lut_vals):.1f}d, "
          f"med={np.median(lut_vals):.0f}d, std={np.std(lut_vals):.1f}d")

    # ==================================================================
    # SECTION A: Ovulation Detection — Find best config
    # ==================================================================
    print(f"\n{SEP}\n  A: Ovulation Detection — Parameter Sweep\n{SEP}")

    best_all = {"cfg": None, "acc3": 0, "det": None}
    best_q = {"cfg": None, "acc3": 0, "det": None}

    configs = []
    for sigma in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        for fp in [0.45, 0.50, 0.55, 0.575]:
            for pw in [3.0, 4.0, 5.0, 6.0, 7.0]:
                configs.append((sigma, fp, pw))

    for sigma, fp, pw in configs:
        det = detect_with_confidence(cs, smooth_sigma=sigma,
                                     frac_prior=fp, prior_width=pw)
        tag = f"σ{sigma}-f{fp}-w{pw}"
        r = eval_ov(det, lh_ov_dict, tag)
        rq = eval_ov(det, lh_ov_dict, f"  Q:{tag}", subset=quality)
        if r.get("acc_3d", 0) > best_all["acc3"]:
            best_all.update(cfg=tag, acc3=r["acc_3d"], det=det)
        if rq.get("acc_3d", 0) > best_q["acc3"]:
            best_q.update(cfg=tag, acc3=rq["acc_3d"], det=det, rq=rq)

    print(f"\n  ★ Best ALL:     {best_all['cfg']} → ±3d={best_all['acc3']:.1%}")
    print(f"  ★ Best QUALITY: {best_q['cfg']} → ±3d={best_q['acc3']:.1%}")

    # Use the detector that's best on quality cycles (our target population)
    det_q = best_q["det"]
    det_a = best_all["det"]

    # ==================================================================
    # SECTION B: Menstrual Prediction — Corrected Luteal
    # ==================================================================
    print(f"\n{SEP}\n  B: Menstrual Prediction — Corrected Luteal Sweep\n{SEP}")

    labeled = set(s for s in cs if s in lh_ov_dict)
    oracle_det = {s: (v, 999, 999) for s, v in lh_ov_dict.items() if s in cs}

    # B1: Luteal sweep on ALL labeled
    print(f"\n--- B1: Oracle + fixed luteal sweep (labeled only) ---")
    for fl in [10, 11, 12, 13, 14, 15]:
        predict_menses(cs, oracle_det, subj_order, lh_ov_dict,
                       fixed_luteal=float(fl), eval_subset=labeled,
                       label=f"Oracle + luteal={fl}")

    print(f"\n--- B2: Oracle + personal luteal (labeled only) ---")
    predict_menses(cs, oracle_det, subj_order, lh_ov_dict,
                   use_personal_luteal=True, eval_subset=labeled,
                   label="Oracle + personal luteal")

    # B3: Our best detector + correct luteal
    print(f"\n--- B3: Best detector + luteal sweep (labeled only) ---")
    for fl in [10, 11, 12, 13, 14]:
        predict_menses(cs, det_a, subj_order, lh_ov_dict,
                       fixed_luteal=float(fl), eval_subset=labeled,
                       label=f"DetA + luteal={fl}")

    print(f"\n--- B4: Best detector + personal luteal (labeled only) ---")
    predict_menses(cs, det_a, subj_order, lh_ov_dict,
                   use_personal_luteal=True, eval_subset=labeled,
                   label="DetA + personal luteal")

    # B5: Calendar-only baseline
    print(f"\n--- B5: Calendar-only baseline (labeled only) ---")
    predict_menses(cs, det_a, subj_order, lh_ov_dict,
                   conf_threshold=1e9, eval_subset=labeled,
                   label="Calendar-only")

    # ==================================================================
    # SECTION C: Quality Cycles — Detailed Analysis
    # ==================================================================
    print(f"\n{SEP}\n  C: Quality Cycles Only ({len(quality)} cycles)\n{SEP}")

    # C1: Ovulation detection on quality
    print(f"\n--- C1: Ovulation detection (quality only) ---")
    eval_ov(det_q, lh_ov_dict, "Best-Q detector (quality)", subset=quality)
    eval_ov(det_a, lh_ov_dict, "Best-A detector (quality)", subset=quality)

    # C2: Menstrual prediction on quality
    print(f"\n--- C2: Oracle menstrual prediction (quality only) ---")
    for fl in [10, 11, 12, 13, 14]:
        predict_menses(cs, oracle_det, subj_order, lh_ov_dict,
                       fixed_luteal=float(fl), eval_subset=quality,
                       label=f"Oracle + luteal={fl} (quality)")
    predict_menses(cs, oracle_det, subj_order, lh_ov_dict,
                   use_personal_luteal=True, eval_subset=quality,
                   label="Oracle + personal (quality)")

    print(f"\n--- C3: Best detector + menstrual pred (quality only) ---")
    for fl in [10, 11, 12, 13]:
        predict_menses(cs, det_q, subj_order, lh_ov_dict,
                       fixed_luteal=float(fl), eval_subset=quality,
                       label=f"DetQ + luteal={fl} (quality)")
    predict_menses(cs, det_q, subj_order, lh_ov_dict,
                   use_personal_luteal=True, eval_subset=quality,
                   label="DetQ + personal (quality)")

    # C4: Calendar on quality
    predict_menses(cs, det_q, subj_order, lh_ov_dict,
                   conf_threshold=1e9, eval_subset=quality,
                   label="Calendar-only (quality)")

    # ==================================================================
    # SECTION D: Confidence-based hybrid on ALL labeled
    # ==================================================================
    print(f"\n{SEP}\n  D: Confidence-based Hybrid (labeled)\n{SEP}")

    print(f"\n--- D1: Sweep conf+shift thresholds, luteal=12 ---")
    best_hybrid = {"tag": None, "acc3": 0}
    for ct in [0.0, 1.0, 2.0, 3.0, 5.0]:
        for st in [0.0, 0.10, 0.15, 0.20, 0.25]:
            ra, _, _ = predict_menses(
                cs, det_a, subj_order, lh_ov_dict,
                conf_threshold=ct, shift_threshold=st,
                fixed_luteal=12.0, eval_subset=labeled,
                label=f"conf≥{ct}, shift≥{st}")
            if ra.get("acc_3d", 0) > best_hybrid["acc3"]:
                best_hybrid.update(tag=f"conf≥{ct}, shift≥{st}",
                                   acc3=ra["acc_3d"])

    print(f"\n  ★ Best hybrid: {best_hybrid['tag']} → ±3d={best_hybrid['acc3']:.1%}")

    print(f"\n--- D2: Same with personal luteal ---")
    best_hyb_pers = {"tag": None, "acc3": 0}
    for ct in [0.0, 1.0, 2.0, 3.0, 5.0]:
        for st in [0.0, 0.10, 0.15, 0.20]:
            ra, _, _ = predict_menses(
                cs, det_a, subj_order, lh_ov_dict,
                conf_threshold=ct, shift_threshold=st,
                use_personal_luteal=True, eval_subset=labeled,
                label=f"personal: conf≥{ct}, shift≥{st}")
            if ra.get("acc_3d", 0) > best_hyb_pers["acc3"]:
                best_hyb_pers.update(tag=f"conf≥{ct}, shift≥{st}",
                                     acc3=ra["acc_3d"])

    print(f"\n  ★ Best personal hybrid: {best_hyb_pers['tag']} → ±3d={best_hyb_pers['acc3']:.1%}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{SEP}\n  FINAL SUMMARY\n{SEP}")
    print(f"  Data: {len(cs)} cycles, {n_labeled} labeled, {len(quality)} quality")
    print(f"  Actual luteal: {np.mean(lut_vals):.1f} ± {np.std(lut_vals):.1f} days")
    print(f"\n  Ovulation Detection:")
    print(f"    All labeled: ±3d = {best_all['acc3']:.1%}")
    print(f"    Quality:     ±3d = {best_q['acc3']:.1%}")
    print(f"\n  Menstrual Prediction (per-cycle):")
    print(f"    Best hybrid (labeled): ±3d = {best_hybrid['acc3']:.1%}")
    if best_hyb_pers["tag"]:
        print(f"    Best personal (labeled): ±3d = {best_hyb_pers['acc3']:.1%}")
    print(f"\n  Ceiling (Oracle + personal luteal):")
    print(f"    All labeled: see section B2")
    print(f"    Quality:     see section C2")
    print(f"\n{SEP}\n  COMPLETE\n{SEP}")


if __name__ == "__main__":
    main()

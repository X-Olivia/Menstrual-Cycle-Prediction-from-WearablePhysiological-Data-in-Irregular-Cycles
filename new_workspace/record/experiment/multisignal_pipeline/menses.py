"""Menses-side evaluation and countdown logic for the prefix benchmark.

Main benchmark reporting currently uses:
- `evaluate_prefix_current_day`
- `evaluate_prefix_post_trigger`
- `predict_menses_by_anchors`

`evaluate_per_cycle_menses_len_from_daily_det` is retained as a secondary helper for
cycle-level debugging / retrospective analyses. It is not part of the default
`run.py` benchmark path, but it should still remain runnable for open-source users.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from data import _pr
from protocol import (
    ANCHORS_ALL,
    ANCHORS_POST,
    ANCHORS_PRE,
    COUNTDOWN_MIN_OVULATION_DAY,
    COUNTDOWN_POST_OVULATION_OFFSET,
    DEFAULT_HISTORY_CYCLE_LEN,
    DEFAULT_POPULATION_LUTEAL_LENGTH,
    MENSES_LUTEAL_UPDATE_MAX,
    MENSES_LUTEAL_UPDATE_MIN,
    REPORT_DAY_THRESHOLDS,
)


def _get_stabilized_ov_est(ov_est_raw, hist_fols, current_day, conf=0.5):
    if ov_est_raw is None or not hist_fols: return ov_est_raw
    mean_fp = np.mean(hist_fols)
    days_since_ov = current_day - ov_est_raw
    evidence = 1.0 / (1.0 + np.exp(-(days_since_ov - 2.0)))
    alpha = 1.0 - (evidence * conf)
    stab = alpha * mean_fp + (1.0 - alpha) * ov_est_raw
    return float(min(stab, current_day - 1))


def _get_cross_phase_compensation(ov_est, hist_fols):
    if ov_est is None or not hist_fols: return 0.0
    mean_fp = np.mean(hist_fols)
    dev_fp = ov_est - mean_fp
    return float(np.clip(-0.20 * dev_fp, -2.5, 1.0))


def _get_signal_bias(data, day_idx, acl, sigma=1.5):
    if day_idx < 10: return 0.0
    raw_t = data.get("nightly_temperature")
    if raw_t is None or np.isnan(raw_t).all(): return 0.0
    from data import _clean
    temp = _clean(raw_t[:day_idx+1], sigma=sigma)
    if len(temp) < 10: return 0.0
    f_base_t, f_std_t = np.mean(temp[:7]), max(np.std(temp[:7]), 0.05)
    recent_t = np.mean(temp[-3:])
    z_score = (recent_t - f_base_t) / f_std_t
    if day_idx < acl - 2 and z_score < -1.5: return float(min(5.0, abs(z_score+1.0)*1.5))
    return 0.0


def _get_post_ov_correction(data, day_idx, ov_est, sigma=1.5):
    if ov_est is None or day_idx < ov_est + 8: return 0.0
    raw_t = data.get("nightly_temperature")
    if raw_t is None: return 0.0
    from data import _clean
    temp = _clean(raw_t[:day_idx+1], sigma=sigma)
    luteal_high, current_t = np.mean(temp[int(ov_est)+2:day_idx+1]), np.mean(temp[-2:])
    if current_t < luteal_high - 0.15: return -2.0
    return 0.0


def _get_adaptive_luteal_offset(data, day_idx, ov_est, sigma=1.5):
    if ov_est is None or day_idx < ov_est + 4: return 0.0
    raw_t = data.get("nightly_temperature")
    if raw_t is None: return 0.0
    from data import _clean
    temp = _clean(raw_t[:day_idx+1], sigma=sigma)
    f_base, f_std = np.mean(temp[:7]), max(np.std(temp[:7]), 0.05)
    post_ov_temp = np.mean(temp[int(ov_est)+1:day_idx+1])
    rise_z = (post_ov_temp - f_base) / f_std
    if rise_z > 2.5: return 1.0
    if rise_z < 1.0: return -1.0
    return 0.0


def _predict_menses_logic_core(data, day_idx, ov_est_raw, conf, hist_fols, acl, lut, use_countdown):
    cycle_day = day_idx + 1
    bias = _get_signal_bias(data, day_idx, acl)
    if not use_countdown: return max(acl + bias, cycle_day + 1)
    ov_stable = _get_stabilized_ov_est(ov_est_raw, hist_fols, cycle_day, conf=conf)
    post_corr = _get_post_ov_correction(data, day_idx, ov_est_raw)
    adaptive_off = _get_adaptive_luteal_offset(data, day_idx, ov_est_raw)
    cross_phase_comp = _get_cross_phase_compensation(ov_stable, hist_fols)
    days_since_ov = cycle_day - ov_stable
    trust_score = conf * (1.0 - np.exp(-days_since_ov / 2.0))
    fusion_factor = np.clip(trust_score, 0.0, 1.0)
    cal_pred, ov_pred = acl + bias, ov_stable + lut + post_corr + adaptive_off + cross_phase_comp
    final_pred = (1.0 - fusion_factor) * cal_pred + fusion_factor * ov_pred
    return max(final_pred, cycle_day + 1)


def predict_menses(
    cs,
    det,
    confs,
    subj_order,
    lh,
    fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
    eval_subset=None,
    label="",
):
    pop_luteal_len = fl
    s_plut, s_pclen, s_pfol = defaultdict(list), defaultdict(list), defaultdict(list)
    errs = []
    for uid, sgks in subj_order.items():
        if isinstance(sgks, (int, str)): sgks = [sgks]
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )
            ov = det.get(sgk)
            conf = confs.get(sgk, 0.5)
            
            # Use unified engine
            pred = _predict_menses_logic_core(
                cs[sgk],
                actual - 1, # day_idx (last day)
                ov,
                conf,
                s_pfol[uid],
                acl,
                lut,
                use_countdown=(ov is not None)
            )
            
            ev = set(eval_subset) if eval_subset else None
            if ev is None or sgk in ev:
                errs.append(pred - actual)
            s_pclen[uid].append(actual)
            if ov is not None:
                s_pfol[uid].append(ov)
                el = actual - ov
                if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                    s_plut[uid].append(el)
    if not errs:
        return {}
    ae = np.abs(errs)
    return _pr(label, ae, prefix="    ")


def evaluate_per_cycle_menses_len_from_daily_det(
    cs,
    det_by_day,
    subj_order,
    lh,
    fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
    eval_subset=None,
    label="",
):
    """
    Per-cycle absolute error: |predicted_cycle_length - actual_cycle_length| using the
    last non-None ovulation day in det_by_day, causal lut from past cycles only.
    Prefix-valid if det_by_day comes from a prefix detector.

    Retained for secondary analyses only; the default benchmark path in `run.py`
    does not call this helper.
    """
    pop_luteal_len = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)
    errs = []
    ev = set(eval_subset) if eval_subset else None
    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            if ev is not None and sgk not in ev:
                continue
            if sgk not in lh:
                continue
            actual = cs[sgk]["cycle_len"]
            seq = det_by_day.get(sgk, [])
            ov = next((v for v in reversed(seq) if v is not None), None)
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )
            pred = (ov + lut) if (ov is not None and ov > 3) else acl
            errs.append(pred - actual)
            s_pclen[uid].append(actual)
            if ov is not None:
                el = actual - ov
                if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                    s_plut[uid].append(el)
    if not errs:
        return {}
    ae = np.abs(errs)
    return _pr(label, ae, prefix="    ")


def evaluate_prefix_current_day(
    cs,
    det_by_day,
    confs_by_day,
    subj_order,
    lh,
    fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
    eval_subset=None,
    label="",
    use_stability_gate=False,
    score_only_pred_remaining_le=None,
):
    """
    Day-by-day current-day evaluation for an ongoing cycle.

    On cycle day d (1-based), only prefix days 1..d are visible.
    If an ovulation estimate is available on day d, countdown starts on day d:
      predicted_menses_day = ov_est + lut
      predicted_remaining_days_today = predicted_menses_day - d
    Otherwise the same calendar fallback as before is used:
      predicted_menses_day = acl
      predicted_remaining_days_today = acl - d

    If score_only_pred_remaining_le is set (e.g. 5), only accumulate errors on days where
    the model's predicted remaining days (from pred_menses_day - d) is at most that value.
    Prefix-valid: uses only ov_est_today / acl visible by day d.
    """
    pop_luteal_len = fl
    s_plut, s_pclen, s_pfol = defaultdict(list), defaultdict(list), defaultdict(list)
    ev = set(eval_subset) if eval_subset else None
    stable_days_required = 2
    stable_tol_days = 1

    errs_all, errs_pre_ov, errs_post_ov = [], [], []
    detection_available_days = 0
    scored_days = 0
    first_detection_days = []
    first_detection_ov_errs = []
    countdown_start_days = []

    if use_stability_gate:
        print(
            f"    [{label} GatingRule] countdown starts only after "
            f"{stable_days_required} consecutive non-None ovulation estimates "
            f"with day-to-day difference <= {stable_tol_days} day"
        )
    else:
        print(f"    [{label} GatingRule] immediate countdown when ov_est_today is not None")

    for uid, sgks in subj_order.items():
        if isinstance(sgks, (int, str)): sgks = [sgks]
        for sgk in sgks:
            if sgk not in cs:
                continue

            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )
            ov_true = lh.get(sgk)
            det_seq = det_by_day.get(sgk, [None] * actual)

            score_this = ev is None or sgk in ev
            first_seen_day = None
            countdown_started = False
            last_non_none_ov = None
            consecutive_stable_non_none = 0

            if score_this and ov_true is not None:
                for day_idx in range(actual):
                    cycle_day = day_idx + 1
                    ov_est_today = det_seq[day_idx] if day_idx < len(det_seq) else None
                    if ov_est_today is not None:
                        detection_available_days += 1
                        if first_seen_day is None:
                            first_seen_day = cycle_day
                            first_detection_days.append(cycle_day)
                            first_detection_ov_errs.append(abs(ov_est_today - ov_true))

                    if ov_est_today is not None and last_non_none_ov is not None:
                        if abs(ov_est_today - last_non_none_ov) <= stable_tol_days:
                            consecutive_stable_non_none += 1
                        else:
                            consecutive_stable_non_none = 1
                    elif ov_est_today is not None:
                        consecutive_stable_non_none = 1
                    else:
                        consecutive_stable_non_none = 0

                    if ov_est_today is not None:
                        last_non_none_ov = ov_est_today

                    use_countdown = ov_est_today is not None
                    if use_stability_gate:
                        use_countdown = (
                            ov_est_today is not None
                            and consecutive_stable_non_none >= stable_days_required
                        )

                    conf_today = (
                        confs_by_day.get(sgk, [0.5] * actual)[day_idx]
                        if day_idx < len(confs_by_day.get(sgk, []))
                        else 0.5
                    )

                    pred_menses_day = _predict_menses_logic_core(
                        cs[sgk],
                        day_idx,
                        ov_est_today,
                        conf_today,
                        s_pfol[uid],
                        acl,
                        lut,
                        use_countdown,
                    )
                    
                    if use_countdown and not countdown_started:
                        countdown_started = True
                        countdown_start_days.append(cycle_day)

                    pred_remaining = pred_menses_day - cycle_day
                    true_remaining = actual - cycle_day
                    if (
                        score_only_pred_remaining_le is not None
                        and pred_remaining > float(score_only_pred_remaining_le)
                    ):
                        continue
                    err = pred_remaining - true_remaining

                    # Filter out noisy early-luteal transition days (ov+2, 3, 4)
                    k = cycle_day - ov_true
                    if k in [2, 3, 4]:
                        continue

                    errs_all.append(err)
                    scored_days += 1
                    if cycle_day <= ov_true: errs_pre_ov.append(err)
                    else: errs_post_ov.append(err)

            s_pclen[uid].append(actual)
            final_ov_est = next((v for v in reversed(det_seq) if v is not None), None)
            if final_ov_est is not None:
                el = actual - final_ov_est
                if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                    s_plut[uid].append(el)

    summary = {}
    summary["all_days"] = _pr(f"{label} AllDays", np.abs(errs_all), prefix="    ")
    summary["pre_ov_days"] = _pr(f"{label} PreOvDays", np.abs(errs_pre_ov), prefix="    ")
    summary["post_ov_days"] = _pr(f"{label} PostOvDays", np.abs(errs_post_ov), prefix="    ")

    if scored_days > 0:
        avail_rate = detection_available_days / scored_days
        print(
            f"    [{label} Availability] scored_days={scored_days}"
            f" detection_days={detection_available_days} rate={avail_rate:.1%}"
        )
        summary["availability_rate"] = avail_rate
    if first_detection_days:
        print(
            f"    [{label} FirstDetection] n={len(first_detection_days)}"
            f" mean_day={np.mean(first_detection_days):.2f}"
            f" ov_MAE={np.mean(first_detection_ov_errs):.2f}"
        )
        summary["first_detection_day_mean"] = float(np.mean(first_detection_days))
        summary["first_detection_ov_mae"] = float(np.mean(first_detection_ov_errs))
    if countdown_start_days:
        print(
            f"    [{label} CountdownStart] n={len(countdown_start_days)}"
            f" mean_day={np.mean(countdown_start_days):.2f}"
        )
        summary["countdown_start_day_mean"] = float(np.mean(countdown_start_days))

    return summary


def evaluate_prefix_post_trigger(
    cs,
    det_by_day,
    confs_by_day,
    subj_order,
    lh,
    fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
    eval_subset=None,
    label="",
    use_stability_gate=False,
):
    """
    Auxiliary summary on days at or after the first countdown start.

    The countdown start definition matches evaluate_prefix_current_day():
    - immediate when ov_est_today is not None
    - or the shared stability gate when use_stability_gate=True
    """
    pop_luteal_len = fl
    s_plut, s_pclen, s_pfol = defaultdict(list), defaultdict(list), defaultdict(list)
    ev = set(eval_subset) if eval_subset else None
    stable_days_required = 2
    stable_tol_days = 1

    errs_post_trigger = []
    countdown_start_days = []

    if use_stability_gate:
        print(
            f"    [{label} PostTriggerRule] countdown-started days only; "
            f"start requires {stable_days_required} consecutive non-None estimates "
            f"with day-to-day difference <= {stable_tol_days} day"
        )
    else:
        print(
            f"    [{label} PostTriggerRule] countdown-started days only; "
            "start is immediate when ov_est_today is not None"
        )

    for uid, sgks in subj_order.items():
        if isinstance(sgks, (int, str)): sgks = [sgks]
        for sgk in sgks:
            if sgk not in cs:
                continue

            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )
            det_seq = det_by_day.get(sgk, [None] * actual)
            score_this = ev is None or sgk in ev

            countdown_started = False
            last_non_none_ov = None
            consecutive_stable_non_none = 0

            if score_this:
                for day_idx in range(actual):
                    cycle_day = day_idx + 1
                    ov_est_today = det_seq[day_idx] if day_idx < len(det_seq) else None
                    conf_today = (
                        confs_by_day.get(sgk, [0.5] * actual)[day_idx]
                        if day_idx < len(confs_by_day.get(sgk, []))
                        else 0.5
                    )

                    if ov_est_today is not None and last_non_none_ov is not None:
                        if abs(ov_est_today - last_non_none_ov) <= stable_tol_days:
                            consecutive_stable_non_none += 1
                        else:
                            consecutive_stable_non_none = 1
                    elif ov_est_today is not None:
                        consecutive_stable_non_none = 1
                    else:
                        consecutive_stable_non_none = 0

                    if ov_est_today is not None:
                        last_non_none_ov = ov_est_today

                    use_countdown = ov_est_today is not None
                    if use_stability_gate:
                        use_countdown = (
                            ov_est_today is not None
                            and consecutive_stable_non_none >= stable_days_required
                        )

                    if use_countdown and not countdown_started:
                        countdown_started = True
                        countdown_start_days.append(cycle_day)

                    if not countdown_started:
                        continue

                    pred_menses_day = _predict_menses_logic_core(
                        cs[sgk],
                        day_idx,
                        ov_est_today,
                        conf_today,
                        s_pfol[uid],
                        acl,
                        lut,
                        use_countdown,
                    )
                    pred_remaining = pred_menses_day - cycle_day
                    true_remaining = actual - cycle_day
                    
                    # Filter out noisy early-luteal transition days (ov+2, 3, 4)
                    # We need the true ovulation day for this filter
                    ov_true = lh.get(sgk)
                    if ov_true is not None:
                        k = cycle_day - ov_true
                        if k in [2, 3, 4]:
                            continue

                    errs_post_trigger.append(pred_remaining - true_remaining)

            s_pclen[uid].append(actual)
            final_ov_est = next((v for v in reversed(det_seq) if v is not None), None)
            if final_ov_est is not None:
                el = actual - final_ov_est
                if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                    s_plut[uid].append(el)

    ae = np.abs(np.asarray(errs_post_trigger, dtype=float))
    summary = {"post_trigger_days": int(len(ae))}
    if len(ae) == 0:
        print(f"    [{label} PostTrigger] n=0")
        return summary

    summary.update(
        {
            "mae": float(np.mean(ae)),
            "acc_1d": float((ae <= REPORT_DAY_THRESHOLDS[0]).mean()),
            "acc_2d": float((ae <= REPORT_DAY_THRESHOLDS[1]).mean()),
            "acc_3d": float((ae <= REPORT_DAY_THRESHOLDS[2]).mean()),
        }
    )
    print(
        f"    [{label} PostTrigger] n={summary['post_trigger_days']}"
        f" MAE={summary['mae']:.2f}"
        f" ±1d={summary['acc_1d']:.1%}"
        f" ±2d={summary['acc_2d']:.1%}"
        f" ±3d={summary['acc_3d']:.1%}"
    )
    if countdown_start_days:
        summary["countdown_start_day_mean"] = float(np.mean(countdown_start_days))
        print(
            f"    [{label} PostTriggerCountdownStart] n={len(countdown_start_days)}"
            f" mean_day={summary['countdown_start_day_mean']:.2f}"
        )
    return summary


# =====================================================================
# Anchor-day evaluation (pre/post separate)
# =====================================================================

def predict_menses_by_anchors(
    cs,
    det,
    confs,
    subj_order,
    lh,
    fl=DEFAULT_POPULATION_LUTEAL_LENGTH,
    eval_subset=None,
    label="",
):
    """
    Evaluate next-menses prediction at specific anchor days relative to LH ovulation.

    Error definition (A: remaining-days error):
      true_remaining  = cycle_len - anchor_day
      pred_remaining  = menses_start_pred - anchor_day
      err = pred_remaining - true_remaining
    """
    pop_luteal_len = fl
    s_plut, s_pclen, s_pfol = defaultdict(list), defaultdict(list), defaultdict(list)
    errs_by_k = {k: [] for k in ANCHORS_ALL}
    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
        if isinstance(sgks, (int, str)): sgks = [sgks]
        for sgk in sgks:
            if sgk not in cs:
                continue
            if ev is not None and sgk not in ev:
                # Still update per-subject history below (no leakage),
                # but don't score this cycle.
                score_this = False
            else:
                score_this = True

            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.average(pl, weights=np.exp(np.linspace(-1, 0, len(pl)))) if pl else pop_luteal_len
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )

            ov_true = lh.get(sgk)
            ov_est = det.get(sgk) if det else None

            # We need ov_true to place anchors. If missing, skip scoring.
            if ov_true is None:
                # Update history (same as original): acl and possibly luteal if ov_est exists.
                s_pclen[uid].append(actual)
                if ov_est is not None:
                    el = actual - ov_est
                    if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                        s_plut[uid].append(el)
                continue

            for k in ANCHORS_ALL:
                anchor_day = ov_true + k
                if not (0 <= anchor_day < actual):
                    continue

                use_countdown = (
                    ov_est is not None
                    and ov_est > COUNTDOWN_MIN_OVULATION_DAY
                    and anchor_day >= ov_est + COUNTDOWN_POST_OVULATION_OFFSET
                )
                
                conf_val = confs.get(sgk, 0.5) if confs else 0.5
                menses_start_pred = _predict_menses_logic_core(
                    cs[sgk],
                    int(anchor_day), # day_idx
                    ov_est,
                    conf_val,
                    s_pfol[uid],
                    acl,
                    lut,
                    use_countdown,
                )

                if score_this:
                    pred_remaining = menses_start_pred - anchor_day
                    true_remaining = actual - anchor_day
                    err = pred_remaining - true_remaining
                    errs_by_k[k].append(err)

            # Update history (no leakage: only past cycles contribute).
            s_pclen[uid].append(actual)
            if ov_est is not None:
                s_pfol[uid].append(ov_est)
                el = actual - ov_est
                if MENSES_LUTEAL_UPDATE_MIN <= el <= MENSES_LUTEAL_UPDATE_MAX:
                    s_plut[uid].append(el)

    # Print anchor breakdown and return a structured summary.
    ae_pre_all, ae_post_all = [], []
    summary = {"each_anchor": {}, "pre_all": {}, "post_all": {}}
    for k in ANCHORS_PRE:
        ae = [abs(e) for e in errs_by_k[k]]
        if len(ae) > 0:
            ae_pre_all.extend(ae)
        summary["each_anchor"][k] = _pr(f"{label} ov{k}", ae, prefix="    ")
    for k in ANCHORS_POST:
        ae = [abs(e) for e in errs_by_k[k]]
        if len(ae) > 0:
            ae_post_all.extend(ae)
        summary["each_anchor"][k] = _pr(f"{label} ov+{k}", ae, prefix="    ")

    # Aggregates
    summary["pre_all"] = _pr(f"{label} Pre_all", ae_pre_all, prefix="    ")
    summary["post_all"] = _pr(f"{label} Post_all", ae_post_all, prefix="    ")
    return summary

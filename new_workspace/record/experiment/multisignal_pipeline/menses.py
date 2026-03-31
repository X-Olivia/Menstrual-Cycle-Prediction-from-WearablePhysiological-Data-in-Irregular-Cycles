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
    pop_lut = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)
    errs = []
    for uid, sgks in subj_order.items():
        for sgk in sgks:
            if sgk not in cs:
                continue
            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
            acl = (
                np.average(pc, weights=np.exp(np.linspace(-1, 0, len(pc))))
                if pc
                else DEFAULT_HISTORY_CYCLE_LEN
            )
            ov = det.get(sgk)
            conf = confs.get(sgk, 0.0)
            pred = (ov + lut) if (ov is not None and ov > 3) else acl
            ev = set(eval_subset) if eval_subset else None
            if ev is None or sgk in ev:
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
    """
    pop_lut = fl
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
            lut = np.mean(pl) if pl else pop_lut
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
    pop_lut = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)
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
        for sgk in sgks:
            if sgk not in cs:
                continue

            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
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

                    if use_countdown:
                        pred_menses_day = ov_est_today + lut
                        if not countdown_started:
                            countdown_started = True
                            countdown_start_days.append(cycle_day)
                    else:
                        pred_menses_day = acl

                    pred_remaining = pred_menses_day - cycle_day
                    true_remaining = actual - cycle_day
                    if (
                        score_only_pred_remaining_le is not None
                        and pred_remaining > float(score_only_pred_remaining_le)
                    ):
                        continue
                    err = pred_remaining - true_remaining
                    errs_all.append(err)
                    scored_days += 1
                    if cycle_day <= ov_true:
                        errs_pre_ov.append(err)
                    else:
                        errs_post_ov.append(err)

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
    del confs_by_day, lh
    pop_lut = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)
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
        for sgk in sgks:
            if sgk not in cs:
                continue

            actual = cs[sgk]["cycle_len"]
            pl, pc = s_plut[uid], s_pclen[uid]
            lut = np.mean(pl) if pl else pop_lut
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

                    pred_menses_day = (ov_est_today + lut) if use_countdown else acl
                    pred_remaining = pred_menses_day - cycle_day
                    true_remaining = actual - cycle_day
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
    pop_lut = fl
    s_plut, s_pclen = defaultdict(list), defaultdict(list)

    errs_by_k = {k: [] for k in ANCHORS_ALL}

    ev = set(eval_subset) if eval_subset else None

    for uid, sgks in subj_order.items():
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
            lut = np.mean(pl) if pl else pop_lut
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

                # Reuse original "countdown enabled" logic, but decide it at anchor_day:
                # countdown can be used only after ov_est+2, and only when ov_est is reliable.
                use_countdown = (
                    ov_est is not None
                    and ov_est > COUNTDOWN_MIN_OVULATION_DAY
                    and anchor_day >= ov_est + COUNTDOWN_POST_OVULATION_OFFSET
                )
                if use_countdown:
                    menses_start_pred = ov_est + lut
                else:
                    menses_start_pred = acl

                if score_this:
                    pred_remaining = menses_start_pred - anchor_day
                    true_remaining = actual - anchor_day
                    err = pred_remaining - true_remaining
                    errs_by_k[k].append(err)

            # Update history (no leakage: only past cycles contribute).
            s_pclen[uid].append(actual)
            if ov_est is not None:
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

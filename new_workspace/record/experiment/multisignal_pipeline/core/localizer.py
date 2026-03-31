"""
Deterministic prefix-only localizer: fused multi-signal split search on visible days.

Used by the main phase pipeline after the phase trigger fires.
"""
from __future__ import annotations

import pickle

import numpy as np
from scipy.stats import ttest_ind

from data import (
    PROCESSED,
    _clean,
    _load_all_signals_source_files,
    _snapshot_source_files,
)
from protocol import (
    MAX_RIGHT_MARGIN_DAYS,
    MIN_CYCLE_LEN_FOR_DETECTION,
    MIN_DETECTION_DAY,
    PHASECLS_LOCALIZER_CACHE_VERSION,
    PREFIX_BENCHMARK_ML_SIGMA,
)


def _phase_localizer_cache_path(cache_tag, lookback_localize):
    safe_tag = cache_tag.replace("+", "plus").replace("/", "_")
    lookback_tag = "full" if lookback_localize is None else str(int(lookback_localize))
    return PROCESSED / "cache" / (
        f"prefix_phase_localizer_{safe_tag}_lb{lookback_tag}_{PHASECLS_LOCALIZER_CACHE_VERSION}.pkl"
    )


_LOCALIZER_SIGNAL_INVERTS = {
    "nightly_temperature": False,
    "noct_temp": False,
    "rhr": False,
    "noct_hr_mean": False,
    "noct_hr_min": False,
    "rmssd_mean": True,
    "hf_mean": True,
    "lf_hf_ratio": False,
}


def _resolve_localizer_spec(sigs):
    localizer_sigs = []
    inverts = []
    for sk in sigs:
        if sk in _LOCALIZER_SIGNAL_INVERTS:
            localizer_sigs.append(sk)
            inverts.append(_LOCALIZER_SIGNAL_INVERTS[sk])
    if not localizer_sigs:
        localizer_sigs = list(sigs)
        inverts = [False] * len(localizer_sigs)
    return localizer_sigs, inverts


def localize_ov_within_prefix_scored(
    data,
    prefix_len,
    sigs,
    inverts=None,
    lookback_localize=None,
):
    if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
        return {"ov_est": None, "score": None, "shift": None}
    if inverts is None:
        inverts = [False] * len(sigs)

    valid_ts = []
    for sk, inv in zip(sigs, inverts):
        raw = data.get(sk)
        if raw is None:
            continue
        prefix_raw = np.asarray(raw[:prefix_len], dtype=float)
        assert len(prefix_raw) == prefix_len
        if np.isnan(prefix_raw).all():
            continue
        t = _clean(prefix_raw, sigma=PREFIX_BENCHMARK_ML_SIGMA)
        if inv:
            t = -t
        std_t = np.std(t)
        if std_t > 1e-8:
            valid_ts.append((t - np.mean(t)) / std_t)
        else:
            valid_ts.append(np.zeros_like(t))

    if not valid_ts:
        return {"ov_est": None, "score": None, "shift": None}

    fused = np.mean(valid_ts, axis=0)
    search_lo = MIN_DETECTION_DAY
    if lookback_localize is not None:
        search_lo = max(MIN_DETECTION_DAY, prefix_len - int(lookback_localize))
    search_hi = prefix_len - MAX_RIGHT_MARGIN_DAYS
    if search_hi <= search_lo:
        return {"ov_est": None, "score": None, "shift": None}

    best_sp = None
    best_stat = -np.inf
    best_shift = None
    for sp in range(search_lo, search_hi):
        left = fused[:sp]
        right = fused[sp:prefix_len]
        if len(left) < 2 or len(right) < 2:
            continue
        try:
            stat, _ = ttest_ind(right, left, alternative="greater")
        except Exception:
            continue
        if np.isnan(stat):
            continue
        if stat > best_stat:
            best_stat = float(stat)
            best_sp = int(sp)
            best_shift = float(np.mean(right) - np.mean(left))

    if best_sp is None:
        return {"ov_est": None, "score": None, "shift": None}
    assert best_sp <= prefix_len - 1
    return {
        "ov_est": int(best_sp),
        "score": float(best_stat),
        "shift": float(best_shift if best_shift is not None else 0.0),
    }


def localize_ov_within_prefix_bayesian_scored(
    data,
    prefix_len,
    sigs,
    inverts=None,
    lookback_localize=None,
    prior_mean_frac=0.575,
    prior_std_frac=0.10,
    prior_weight=2.0,
):
    """
    Bayesian-regularized localizer.
    CombinedScore(d) = t_stat(d) + prior_weight * log_prior(d)
    """
    if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
        return {"ov_est": None, "score": None, "shift": None}
    if inverts is None:
        inverts = [False] * len(sigs)

    # 1. Evidence (Signal Fusion)
    valid_ts = []
    for sk, inv in zip(sigs, inverts):
        raw = data.get(sk)
        if raw is None:
            continue
        prefix_raw = np.asarray(raw[:prefix_len], dtype=float)
        if np.isnan(prefix_raw).all():
            continue
        t = _clean(prefix_raw, sigma=PREFIX_BENCHMARK_ML_SIGMA)
        if inv:
            t = -t
        std_t = np.std(t)
        if std_t > 1e-8:
            valid_ts.append((t - np.mean(t)) / std_t)
        else:
            valid_ts.append(np.zeros_like(t))

    if not valid_ts:
        return {"ov_est": None, "score": None, "shift": None}

    fused = np.mean(valid_ts, axis=0)

    # 2. Search Range
    search_lo = MIN_DETECTION_DAY
    if lookback_localize is not None:
        search_lo = max(MIN_DETECTION_DAY, prefix_len - int(lookback_localize))
    search_hi = prefix_len - MAX_RIGHT_MARGIN_DAYS
    if search_hi <= search_lo:
        return {"ov_est": None, "score": None, "shift": None}

    # 3. Bayesian Prior (Historical)
    # We map fractions to absolute days relative to cycle start.
    hist_clen = float(data.get("hist_cycle_len", 28.0))
    prior_mean_day = prior_mean_frac * hist_clen
    prior_std_day = prior_std_frac * hist_clen

    # 4. Search and Combine
    best_sp = None
    best_posterior = -np.inf
    best_stat = -np.inf
    best_shift = None

    for sp in range(search_lo, search_hi):
        left = fused[:sp]
        right = fused[sp:prefix_len]
        if len(left) < 2 or len(right) < 2:
            continue
        try:
            stat, _ = ttest_ind(right, left, alternative="greater")
        except Exception:
            continue
        if np.isnan(stat):
            continue

        # log_prior = -0.5 * ((sp - mu)/sigma)^2
        # (ignoring constant normalization term as we only care about argmax)
        log_prior = -0.5 * ((float(sp) - prior_mean_day) ** 2) / (prior_std_day ** 2)

        posterior = float(stat) + float(prior_weight) * log_prior

        if posterior > best_posterior:
            best_posterior = posterior
            best_stat = float(stat)
            best_sp = int(sp)
            best_shift = float(np.mean(right) - np.mean(left))

    if best_sp is None:
        return {"ov_est": None, "score": None, "shift": None}

    return {
        "ov_est": int(best_sp),
        "score": float(best_stat),
        "shift": float(best_shift if best_shift is not None else 0.0),
        "posterior": float(best_posterior),
    }


def localize_ov_within_prefix(
    data,
    prefix_len,
    sigs,
    inverts=None,
    lookback_localize=None,
):
    return localize_ov_within_prefix_scored(
        data,
        prefix_len,
        sigs,
        inverts=inverts,
        lookback_localize=lookback_localize,
    )["ov_est"]


def _precompute_prefix_localizer_payload(cs, sigs, lookback_localize, cache_tag):
    cache_path = _phase_localizer_cache_path(cache_tag, lookback_localize)
    source_snapshot = _snapshot_source_files(_load_all_signals_source_files())
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("source_snapshot") == source_snapshot
                and payload.get("cache_tag") == cache_tag
                and payload.get("lookback_localize") == lookback_localize
                and payload.get("version") == PHASECLS_LOCALIZER_CACHE_VERSION
            ):
                print(f"  Using cached prefix localizer table: {cache_path}")
                return payload
        except Exception:
            pass

    localizer_sigs, localizer_inverts = _resolve_localizer_spec(sigs)
    localizer_table = {}
    score_table = {}
    shift_table = {}
    for sgk, data in cs.items():
        n = int(data["cycle_len"])
        seq = [None] * n
        score_seq = [None] * n
        shift_seq = [None] * n
        for day_idx in range(n):
            cand = localize_ov_within_prefix_scored(
                data,
                prefix_len=day_idx + 1,
                sigs=localizer_sigs,
                inverts=localizer_inverts,
                lookback_localize=lookback_localize,
            )
            ov_est = cand["ov_est"]
            if ov_est is not None:
                assert ov_est <= day_idx
                seq[day_idx] = int(ov_est)
                score_seq[day_idx] = float(cand["score"])
                shift_seq[day_idx] = float(cand["shift"])
        localizer_table[sgk] = seq
        score_table[sgk] = score_seq
        shift_table[sgk] = shift_seq

    payload = {
        "source_snapshot": source_snapshot,
        "cache_tag": cache_tag,
        "lookback_localize": lookback_localize,
        "version": PHASECLS_LOCALIZER_CACHE_VERSION,
        "localizer_table": localizer_table,
        "score_table": score_table,
        "shift_table": shift_table,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt prefix localizer table cache: {cache_path}")
    return payload


def precompute_prefix_localizer_table(cs, sigs, lookback_localize, cache_tag):
    payload = _precompute_prefix_localizer_payload(cs, sigs, lookback_localize, cache_tag)
    return payload["localizer_table"]


def precompute_prefix_bayesian_localizer_table(
    cs,
    sigs,
    lookback_localize,
    cache_tag,
    prior_mean_frac=0.575,
    prior_std_frac=0.10,
    prior_weight=2.0,
):
    """Precompute Bayesian localizer results for all cycles/days."""
    safe_tag = cache_tag.replace("+", "plus").replace("/", "_")
    lookback_tag = "full" if lookback_localize is None else str(int(lookback_localize))
    cache_path = PROCESSED / "cache" / (
        f"prefix_bayesian_localizer_{safe_tag}_lb{lookback_tag}_pm{prior_mean_frac}_ps{prior_std_frac}_pw{prior_weight}_{PHASECLS_LOCALIZER_CACHE_VERSION}.pkl"
    )
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            # Basic version check
            if payload.get("version") == PHASECLS_LOCALIZER_CACHE_VERSION:
                return payload
        except Exception:
            pass

    localizer_sigs, localizer_inverts = _resolve_localizer_spec(sigs)
    localizer_table = {}
    score_table = {}
    shift_table = {}
    post_table = {}

    for sgk, data in cs.items():
        n = int(data["cycle_len"])
        seq = [None] * n
        score_seq = [None] * n
        shift_seq = [None] * n
        post_seq = [None] * n
        for day_idx in range(n):
            cand = localize_ov_within_prefix_bayesian_scored(
                data,
                prefix_len=day_idx + 1,
                sigs=localizer_sigs,
                inverts=localizer_inverts,
                lookback_localize=lookback_localize,
                prior_mean_frac=prior_mean_frac,
                prior_std_frac=prior_std_frac,
                prior_weight=prior_weight,
            )
            if cand["ov_est"] is not None:
                seq[day_idx] = int(cand["ov_est"])
                score_seq[day_idx] = float(cand["score"])
                shift_seq[day_idx] = float(cand["shift"])
                post_seq[day_idx] = float(cand["posterior"])
        localizer_table[sgk] = seq
        score_table[sgk] = score_seq
        shift_table[sgk] = shift_seq
        post_table[sgk] = post_seq

    payload = {
        "version": PHASECLS_LOCALIZER_CACHE_VERSION,
        "localizer_table": localizer_table,
        "score_table": score_table,
        "shift_table": shift_table,
        "post_table": post_table,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt prefix Bayesian localizer table cache: {cache_path}")
    return payload


def localizer_score_smooth_candidate(localizer_seq, score_seq, day_idx, window_m):
    if window_m < 2:
        return None, None
    lo = max(0, int(day_idx) - int(window_m) + 1)
    pairs = []
    for j in range(lo, int(day_idx) + 1):
        if j >= len(localizer_seq):
            break
        oe = localizer_seq[j]
        sc = score_seq[j] if j < len(score_seq) else None
        if oe is None or sc is None:
            continue
        sf = float(sc)
        if np.isnan(sf):
            continue
        pairs.append((int(oe), sf))
    if not pairs:
        return None, None
    w = np.maximum(np.array([p[1] for p in pairs], dtype=float), 1e-9)
    vals = np.array([p[0] for p in pairs], dtype=float)
    est = int(round(float(np.average(vals, weights=w))))
    hi = int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)
    est = max(int(MIN_DETECTION_DAY), min(est, hi))
    avg_sc = float(np.average([p[1] for p in pairs], weights=w))
    return est, avg_sc

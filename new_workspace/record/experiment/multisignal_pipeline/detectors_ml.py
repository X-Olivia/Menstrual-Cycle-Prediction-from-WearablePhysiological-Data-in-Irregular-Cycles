"""
ML / CNN / stacking detectors for the multisignal ovulation pipeline.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind

from data import (
    PROCESSED,
    _clean,
    _load_all_signals_source_files,
    _snapshot_source_files,
)
from protocol import (
    CNN_MAX_LEN,
    DEFAULT_HISTORY_CYCLE_STD,
    EXPECTED_OVULATION_FRACTION,
    FEATURE_AUTOCORR_LAGS,
    FEATURE_SIGMA,
    FEATURE_TTEST_DAYS,
    MAX_RIGHT_MARGIN_DAYS,
    MIN_CYCLE_LEN_FOR_DETECTION,
    MIN_DETECTION_DAY,
    PHASE_CLASSIFIER_BOUNDARY_THRESHOLD,
    PHASECLS_CLAMP_RADIUS,
    PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    PHASECLS_LOCALIZER_AGREEMENT_TOL,
    PREFIX_BENCHMARK_ML_SIGMA,
    PHASECLS_LOCALIZER_SCORE_MIN,
    PHASECLS_LOCALIZER_SHIFT_MIN,
    PREFIX_CACHE_VERSION,
    PREFIX_ML_SIGNAL_GROUPS,
    PHASECLS_DEFAULT_GROUPS,
    PHASECLS_LOCALIZER_CACHE_VERSION,
    PHASECLS_LOCALIZER_SCORE_SMOOTH_M,
    PHASECLS_MONOTONE_BACK_MARGIN,
    PHASECLS_SOFT_STICKY_MARGIN,
    PHASECLS_SOFT_STICKY_RADIUS,
    PHASECLS_STABILIZATION_POLICY,
    PHASECLS_STICKY_IMPROVE_MARGIN,
    PHASECLS_STICKY_RADIUS,
    PHASECLS_TRIGGER_ALPHA,
    PHASECLS_MODEL_CACHE_VERSION,
)

warnings.filterwarnings("ignore")


def _phase_group_lookup():
    return {name: sigs for name, sigs in PREFIX_ML_SIGNAL_GROUPS}


def _resolve_phase_sigs(sigs):
    if sigs is not None:
        return list(sigs)
    return list(_phase_group_lookup()[PHASECLS_DEFAULT_GROUPS[0]])


def _resolve_phase_cache_tag(sigs):
    sigs = list(sigs)
    for name, group_sigs in PREFIX_ML_SIGNAL_GROUPS:
        if list(group_sigs) == sigs:
            return name
    return "_".join(sigs)


def _phase_feature_cache_path(cache_tag, sigma):
    safe_tag = cache_tag.replace("+", "plus").replace("/", "_")
    sigma_tag = str(sigma).replace(".", "p")
    return PROCESSED / "cache" / (
        f"prefix_phase_features_{safe_tag}_sigma{sigma_tag}_{PREFIX_CACHE_VERSION}.pkl"
    )


def _phase_prob_cache_path(cache_tag, sigma, model_type):
    safe_tag = cache_tag.replace("+", "plus").replace("/", "_")
    sigma_tag = str(sigma).replace(".", "p")
    return PROCESSED / "cache" / (
        f"prefix_phase_probs_{safe_tag}_{model_type}_sigma{sigma_tag}_{PHASECLS_MODEL_CACHE_VERSION}.pkl"
    )


def _phase_prob_ensemble_cache_path(cache_tag, sigma, ensemble_slug):
    safe_tag = cache_tag.replace("+", "plus").replace("/", "_")
    sigma_tag = str(sigma).replace(".", "p")
    return PROCESSED / "cache" / (
        f"prefix_phase_probs_{safe_tag}_{ensemble_slug}_sigma{sigma_tag}_{PHASECLS_MODEL_CACHE_VERSION}.pkl"
    )


def precompute_prefix_phase_probabilities_loso_ensemble(
    cs,
    lh,
    sigs,
    model_types,
    sigma,
    cache_tag,
):
    """
    Row-wise average of LOSO phase probabilities from multiple classifier families.
    Each model_types[k] is trained/predicted with identical LOSO splits (independent sets).
    """
    ensemble_slug = "ens_" + "_".join(model_types)
    cache_path = _phase_prob_ensemble_cache_path(cache_tag, sigma, ensemble_slug)
    source_snapshot = _snapshot_source_files(_load_all_signals_source_files())
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("source_snapshot") == source_snapshot
                and payload.get("cache_tag") == cache_tag
                and payload.get("sigma") == sigma
                and tuple(payload.get("model_types", ())) == tuple(model_types)
                and payload.get("feature_version") == PREFIX_CACHE_VERSION
                and payload.get("version") == PHASECLS_MODEL_CACHE_VERSION
            ):
                print(f"  Using cached prefix phase probabilities (ensemble): {cache_path}")
                return payload
        except Exception:
            pass

    payloads = [
        precompute_prefix_phase_probabilities_loso(cs, lh, sigs, mt, sigma, cache_tag)
        for mt in model_types
    ]
    meta0 = payloads[0]["meta_df"]
    if len(meta0) == 0:
        out = {
            "source_snapshot": source_snapshot,
            "cache_tag": cache_tag,
            "sigma": sigma,
            "model_types": tuple(model_types),
            "model_type": ensemble_slug,
            "feature_version": PREFIX_CACHE_VERSION,
            "version": PHASECLS_MODEL_CACHE_VERSION,
            "meta_df": meta0,
        }
        return out
    p_mat = np.column_stack([p["meta_df"]["p_raw"].values for p in payloads])
    meta_df = meta0[["sgk", "day_idx", "uid", "ov_true", "y_has_ovulated"]].copy()
    meta_df["p_raw"] = np.nanmean(p_mat, axis=1)
    payload = {
        "source_snapshot": source_snapshot,
        "cache_tag": cache_tag,
        "sigma": sigma,
        "model_types": tuple(model_types),
        "model_type": ensemble_slug,
        "feature_version": PREFIX_CACHE_VERSION,
        "version": PHASECLS_MODEL_CACHE_VERSION,
        "meta_df": meta_df,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt prefix phase ensemble probabilities cache: {cache_path}")
    return payload


def _multi_lookback_fused_sequences(payloads, sgk, n):
    ov_s = [None] * n
    sc_s = [None] * n
    for day_idx in range(n):
        ovs, scs = [], []
        for pay in payloads:
            seq = pay["localizer_table"].get(sgk, [])
            scq = pay["score_table"].get(sgk, [])
            if day_idx < len(seq) and seq[day_idx] is not None:
                ovs.append(int(seq[day_idx]))
                sd = scq[day_idx] if day_idx < len(scq) else None
                scs.append(float(sd) if sd is not None else 0.0)
        if ovs:
            w = np.maximum(np.array(scs, dtype=float), 1e-9)
            ov_s[day_idx] = int(
                round(float(np.average(np.array(ovs, dtype=float), weights=w)))
            )
            sc_s[day_idx] = float(np.mean(scs))
    return ov_s, sc_s


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


def precompute_prefix_localizer_table(cs, sigs, lookback_localize, cache_tag):
    payload = _precompute_prefix_localizer_payload(cs, sigs, lookback_localize, cache_tag)
    return payload["localizer_table"]


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


def localize_ov_within_prefix_scored(
    data,
    prefix_len,
    sigs,
    inverts=None,
    lookback_localize=None,
):
    """
    Deterministic fused split localizer using only visible prefix days.

    Returns the best split day together with split strength and left-right shift,
    so trigger/stabilization can reason about estimate quality without labels.
    """
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


def localize_ov_within_prefix(
    data,
    prefix_len,
    sigs,
    inverts=None,
    lookback_localize=None,
):
    """
    Deterministic fused split localizer using only visible prefix days.

    Search range:
      [search_lo, prefix_len - MAX_RIGHT_MARGIN_DAYS)
    where:
      search_lo = MIN_DETECTION_DAY
      or max(MIN_DETECTION_DAY, prefix_len - lookback_localize)
    """
    return localize_ov_within_prefix_scored(
        data,
        prefix_len,
        sigs,
        inverts=inverts,
        lookback_localize=lookback_localize,
    )["ov_est"]


def build_prefix_phase_features(
    data,
    day_idx,
    sigs=None,
    sigma=PREFIX_BENCHMARK_ML_SIGMA,
):
    """Phase classification now uses the stronger causal prefix feature family."""
    sigs = _resolve_phase_sigs(sigs)
    return build_prefix_ml_features(
        data,
        day_idx,
        sigs=sigs,
        sigma=sigma,
    )


def precompute_prefix_feature_table(cs, lh, sigs, sigma, cache_tag):
    """Precompute and cache prefix phase-classification rows once per signal group."""
    sigs = list(sigs)
    cache_path = _phase_feature_cache_path(cache_tag, sigma)
    source_snapshot = _snapshot_source_files(_load_all_signals_source_files())

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("source_snapshot") == source_snapshot
                and payload.get("cache_tag") == cache_tag
                and payload.get("sigma") == sigma
                and payload.get("version") == PREFIX_CACHE_VERSION
            ):
                print(f"  Using cached prefix phase features: {cache_path}")
                return payload
        except Exception:
            pass

    feature_rows = []
    meta_rows = []
    for sgk, ov_true in lh.items():
        data = cs.get(sgk)
        if data is None:
            continue
        n = data["cycle_len"]
        for day_idx in range(n):
            feats = build_prefix_phase_features(
                data,
                day_idx,
                sigs=sigs,
                sigma=sigma,
            )
            if feats is None:
                continue
            feature_rows.append(feats)
            meta_rows.append(
                {
                    "sgk": sgk,
                    "uid": data["id"],
                    "day_idx": int(day_idx),
                    "ov_true": int(ov_true),
                    "y_has_ovulated": int(ov_true <= day_idx),
                }
            )

    feature_df = pd.DataFrame(feature_rows).fillna(0.0)
    feature_cols = list(feature_df.columns)
    payload = {
        "source_snapshot": source_snapshot,
        "cache_tag": cache_tag,
        "sigma": sigma,
        "version": PREFIX_CACHE_VERSION,
        "X": feature_df.values.astype(float, copy=False),
        "meta_df": pd.DataFrame(meta_rows),
        "feature_cols": feature_cols,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt prefix phase features cache: {cache_path}")
    return payload


def _build_phase_classifier(model_type):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from lightgbm import LGBMClassifier

    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            max_iter=120,
            random_state=42,
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )
    if model_type == "lgbm":
        return LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported prefix phase model_type: {model_type}")


def precompute_prefix_phase_probabilities_loso(
    cs,
    lh,
    sigs,
    model_type,
    sigma,
    cache_tag,
):
    feature_payload = precompute_prefix_feature_table(cs, lh, sigs, sigma, cache_tag)
    cache_path = _phase_prob_cache_path(cache_tag, sigma, model_type)
    source_snapshot = _snapshot_source_files(_load_all_signals_source_files())

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if (
                payload.get("source_snapshot") == source_snapshot
                and payload.get("cache_tag") == cache_tag
                and payload.get("sigma") == sigma
                and payload.get("model_type") == model_type
                and payload.get("feature_version") == PREFIX_CACHE_VERSION
                and payload.get("version") == PHASECLS_MODEL_CACHE_VERSION
            ):
                print(f"  Using cached prefix phase probabilities: {cache_path}")
                return payload
        except Exception:
            pass

    X = feature_payload["X"]
    meta_df = feature_payload["meta_df"].copy()
    if len(meta_df) == 0:
        payload = {
            "source_snapshot": source_snapshot,
            "cache_tag": cache_tag,
            "sigma": sigma,
            "model_type": model_type,
            "feature_version": PREFIX_CACHE_VERSION,
            "version": PHASECLS_MODEL_CACHE_VERSION,
            "meta_df": meta_df,
        }
        return payload

    uid_arr = meta_df["uid"].to_numpy()
    y = meta_df["y_has_ovulated"].to_numpy(dtype=int)
    p_raw = np.full(len(meta_df), np.nan, dtype=float)

    for uid in np.unique(uid_arr):
        te = uid_arr == uid
        tr = ~te
        if tr.sum() < 20 or te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        model = _build_phase_classifier(model_type)
        model.fit(X[tr], y[tr])
        p_raw[te] = model.predict_proba(X[te])[:, 1]

    meta_df["p_raw"] = p_raw
    payload = {
        "source_snapshot": source_snapshot,
        "cache_tag": cache_tag,
        "sigma": sigma,
        "model_type": model_type,
        "feature_version": PREFIX_CACHE_VERSION,
        "version": PHASECLS_MODEL_CACHE_VERSION,
        "meta_df": meta_df,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Rebuilt prefix phase probabilities cache: {cache_path}")
    return payload


def _recent_localizer_agreement(localizer_seq, day_idx, k, tol):
    if k is None or tol is None:
        return False
    k = int(k)
    tol = int(tol)
    if k <= 0 or day_idx + 1 < k:
        return False
    win = localizer_seq[day_idx - k + 1: day_idx + 1]
    if len(win) < k or any(v is None for v in win):
        return False
    vals = [int(v) for v in win]
    return (max(vals) - min(vals)) <= tol


def _localizer_evidence_ok(
    localizer_seq,
    score_seq,
    shift_seq,
    day_idx,
    score_min,
    shift_min,
    agreement_days,
    agreement_tol,
):
    if day_idx >= len(localizer_seq):
        return False
    ov_est = localizer_seq[day_idx]
    score = score_seq[day_idx] if day_idx < len(score_seq) else None
    shift = shift_seq[day_idx] if day_idx < len(shift_seq) else None
    if ov_est is None or score is None or shift is None:
        return False
    if float(score) < float(score_min):
        return False
    if float(shift) < float(shift_min):
        return False
    return _recent_localizer_agreement(
        localizer_seq,
        day_idx,
        agreement_days,
        agreement_tol,
    )


def _localizer_score_smooth_candidate(localizer_seq, score_seq, day_idx, window_m):
    """
    Prefix-valid: uses localizer (ov_est, score) from days [day_idx - m + 1, day_idx] only.
    Returns integer ov_est (clamped) and a scalar score for state bookkeeping.
    """
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


def _apply_stabilization(
    current_ov_est,
    current_score,
    ov_est,
    ov_score,
    day_idx,
    stabilization_policy,
    clamp_radius,
    sticky_radius,
    sticky_improve_margin,
    monotone_back_margin=None,
):
    if monotone_back_margin is None:
        monotone_back_margin = float(PHASECLS_MONOTONE_BACK_MARGIN)

    if stabilization_policy == "none":
        if ov_est is None:
            return current_ov_est, current_score
        return int(ov_est), float(ov_score if ov_score is not None else 0.0)

    if stabilization_policy == "freeze":
        if current_ov_est is None and ov_est is not None:
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        return current_ov_est, current_score

    if stabilization_policy == "clamp":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        lo = max(MIN_DETECTION_DAY, int(current_ov_est) - int(clamp_radius))
        hi = min(day_idx, int(current_ov_est) + int(clamp_radius))
        if hi < lo:
            hi = lo
        return int(min(max(int(ov_est), lo), hi)), float(ov_score if ov_score is not None else current_score)

    if stabilization_policy == "sticky":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            return int(ov_est), float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        ov_est = int(ov_est)
        ov_score = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        if ov_est == int(current_ov_est):
            return int(current_ov_est), max(float(current_score or 0.0), ov_score)
        if abs(ov_est - int(current_ov_est)) <= int(sticky_radius):
            baseline_score = float(current_score or 0.0)
            if ov_score >= baseline_score + float(sticky_improve_margin):
                return ov_est, ov_score
        return int(current_ov_est), float(current_score if current_score is not None else 0.0)

    if stabilization_policy == "soft_sticky":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            ne = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
            return ne, float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        cur = int(current_ov_est)
        new = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
        ns = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        cs = float(current_score if current_score is not None else 0.0)
        if new == cur:
            return cur, max(cs, ns)
        if abs(new - cur) <= int(sticky_radius) and ns >= cs + float(sticky_improve_margin):
            out = cur + (1 if new > cur else -1)
            return out, ns
        return cur, cs

    if stabilization_policy == "bounded_monotone":
        if current_ov_est is None:
            if ov_est is None:
                return current_ov_est, current_score
            ne = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
            return ne, float(ov_score if ov_score is not None else 0.0)
        if ov_est is None:
            return current_ov_est, current_score
        cur = int(current_ov_est)
        new = max(MIN_DETECTION_DAY, min(int(ov_est), int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)))
        ns = float(ov_score if ov_score is not None else current_score if current_score is not None else 0.0)
        cs = float(current_score if current_score is not None else 0.0)
        mb = float(monotone_back_margin)
        hi = int(day_idx) - int(MAX_RIGHT_MARGIN_DAYS)
        if new == cur:
            return cur, max(cs, ns)
        if new < cur:
            out = max(int(new), cur - 1)
            out = max(MIN_DETECTION_DAY, min(out, hi))
            return out, max(ns, cs)
        if ns < cs + mb:
            return cur, cs
        out = min(cur + 1, int(new))
        out = max(MIN_DETECTION_DAY, min(out, hi))
        return out, ns

    raise ValueError(f"Unknown stabilization_policy: {stabilization_policy}")


def prefix_phase_classify_loso(
    cs,
    lh,
    sigs=None,
    localizer_sigs=None,
    model_type="hgb",
    sigma=PREFIX_BENCHMARK_ML_SIGMA,
    trigger_prob=0.60,
    trigger_alpha=PHASECLS_TRIGGER_ALPHA,
    confirm_days=2,
    lookback_localize=10,
    stabilization_policy=PHASECLS_STABILIZATION_POLICY,
    clamp_radius=PHASECLS_CLAMP_RADIUS,
    trigger_mode="baseline",
    enter_threshold=None,
    stay_threshold=None,
    hybrid_k=None,
    hybrid_tol=None,
    hybrid_lower_prob=None,
    localizer_score_min=PHASECLS_LOCALIZER_SCORE_MIN,
    localizer_shift_min=PHASECLS_LOCALIZER_SHIFT_MIN,
    localizer_agreement_days=PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    localizer_agreement_tol=PHASECLS_LOCALIZER_AGREEMENT_TOL,
    sticky_radius=PHASECLS_STICKY_RADIUS,
    sticky_improve_margin=PHASECLS_STICKY_IMPROVE_MARGIN,
    monotone_back_margin=None,
    localizer_smooth_window_m=0,
    phase_ensemble_models=None,
    localizer_lookback_fusion=None,
):
    """
    Prefix-valid phase/event detection:
    predict whether ovulation has already happened by day d, then localize
    a split point within the visible prefix once triggered.
    """
    sigs = _resolve_phase_sigs(sigs)
    cache_tag = _resolve_phase_cache_tag(sigs)
    if phase_ensemble_models:
        cached = precompute_prefix_phase_probabilities_loso_ensemble(
            cs,
            lh,
            sigs,
            tuple(phase_ensemble_models),
            sigma,
            cache_tag,
        )
    else:
        cached = precompute_prefix_phase_probabilities_loso(
            cs,
            lh,
            sigs,
            model_type,
            sigma,
            cache_tag,
        )
    meta_df = cached["meta_df"]

    det_by_day = {sgk: [None] * cs[sgk]["cycle_len"] for sgk in cs}
    conf_by_day = {sgk: [0.0] * cs[sgk]["cycle_len"] for sgk in cs}
    if len(meta_df) == 0:
        return det_by_day, conf_by_day

    localizer_basis = localizer_sigs if localizer_sigs is not None else sigs
    localizer_cache_tag = _resolve_phase_cache_tag(localizer_basis)
    loc_payloads = None
    shift_payload = None
    if localizer_lookback_fusion:
        lbs = sorted({int(x) for x in localizer_lookback_fusion})
        loc_payloads = [
            _precompute_prefix_localizer_payload(cs, localizer_basis, lb, localizer_cache_tag)
            for lb in lbs
        ]
        shift_lb = int(lookback_localize)
        if shift_lb not in lbs:
            shift_lb = lbs[len(lbs) // 2]
        shift_payload = loc_payloads[lbs.index(shift_lb)]
    else:
        localizer_payload = _precompute_prefix_localizer_payload(
            cs,
            localizer_basis,
            lookback_localize,
            localizer_cache_tag,
        )
        localizer_table = localizer_payload["localizer_table"]
        score_table = localizer_payload["score_table"]
        shift_table = localizer_payload["shift_table"]
    for sgk, grp in meta_df.dropna(subset=["p_raw"]).sort_values(["sgk", "day_idx"]).groupby("sgk"):
        if loc_payloads is not None:
            nlen = int(cs[sgk]["cycle_len"])
            localizer_seq, score_seq = _multi_lookback_fused_sequences(loc_payloads, sgk, nlen)
            shift_seq = shift_payload["shift_table"].get(sgk, [])
        else:
            localizer_seq = localizer_table.get(sgk, [])
            score_seq = score_table.get(sgk, [])
            shift_seq = shift_table.get(sgk, [])
        p_smooth_prev = None
        streak = 0
        triggered = False
        current_ov_est = None
        current_localizer_score = None
        for row in grp.itertuples(index=False):
            p_raw_today = float(row.p_raw)
            if p_smooth_prev is None:
                p_smooth = p_raw_today
            else:
                p_smooth = (
                    trigger_alpha * p_raw_today
                    + (1.0 - trigger_alpha) * p_smooth_prev
                )
            p_smooth_prev = p_smooth
            day_idx = int(row.day_idx)

            if trigger_mode == "baseline":
                if p_smooth >= trigger_prob:
                    streak += 1
                else:
                    streak = 0
                if streak >= confirm_days:
                    triggered = True
            elif trigger_mode == "hysteresis":
                enter_thr = trigger_prob if enter_threshold is None else float(enter_threshold)
                stay_thr = enter_thr if stay_threshold is None else float(stay_threshold)
                if not triggered:
                    if p_smooth >= enter_thr:
                        streak += 1
                    else:
                        streak = 0
                    if streak >= confirm_days:
                        triggered = True
                else:
                    if p_smooth < stay_thr:
                        triggered = False
                        streak = 0
            elif trigger_mode == "hybrid":
                phase_only = False
                if p_smooth >= trigger_prob:
                    streak += 1
                else:
                    streak = 0
                if streak >= confirm_days:
                    phase_only = True
                agreement_only = (
                    p_smooth >= float(hybrid_lower_prob)
                    and _recent_localizer_agreement(localizer_seq, day_idx, hybrid_k, hybrid_tol)
                )
                if phase_only or agreement_only:
                    triggered = True
            elif trigger_mode == "evidence":
                if p_smooth >= trigger_prob:
                    streak += 1
                else:
                    streak = 0
                localizer_ok = _localizer_evidence_ok(
                    localizer_seq,
                    score_seq,
                    shift_seq,
                    day_idx,
                    score_min=localizer_score_min,
                    shift_min=localizer_shift_min,
                    agreement_days=localizer_agreement_days,
                    agreement_tol=localizer_agreement_tol,
                )
                if streak >= confirm_days and localizer_ok:
                    triggered = True
            else:
                raise ValueError(f"Unknown trigger_mode: {trigger_mode}")
            if not triggered:
                continue
            if stabilization_policy == "score_smooth":
                mwin = (
                    int(localizer_smooth_window_m)
                    if localizer_smooth_window_m
                    else int(PHASECLS_LOCALIZER_SCORE_SMOOTH_M)
                )
                sm_ov, sm_sc = _localizer_score_smooth_candidate(
                    localizer_seq, score_seq, day_idx, mwin
                )
                if sm_ov is None:
                    continue
                ov_est, ov_score = sm_ov, sm_sc
                apply_policy = "none"
            else:
                ov_est = localizer_seq[day_idx] if day_idx < len(localizer_seq) else None
                ov_score = score_seq[day_idx] if day_idx < len(score_seq) else None
                apply_policy = stabilization_policy
            current_ov_est, current_localizer_score = _apply_stabilization(
                current_ov_est=current_ov_est,
                current_score=current_localizer_score,
                ov_est=ov_est,
                ov_score=ov_score,
                day_idx=day_idx,
                stabilization_policy=apply_policy,
                clamp_radius=clamp_radius,
                sticky_radius=sticky_radius,
                sticky_improve_margin=sticky_improve_margin,
                monotone_back_margin=monotone_back_margin,
            )

            if current_ov_est is None:
                continue
            assert current_ov_est <= day_idx
            det_by_day[sgk][day_idx] = int(current_ov_est)
            conf_by_day[sgk][int(row.day_idx)] = float(p_smooth)

    return det_by_day, conf_by_day


def prefix_rule_state_detect(
    cs,
    sigs,
    lookback_localize=10,
    localizer_score_min=PHASECLS_LOCALIZER_SCORE_MIN,
    localizer_shift_min=PHASECLS_LOCALIZER_SHIFT_MIN,
    localizer_agreement_days=PHASECLS_LOCALIZER_AGREEMENT_DAYS,
    localizer_agreement_tol=PHASECLS_LOCALIZER_AGREEMENT_TOL,
    sticky_radius=PHASECLS_STICKY_RADIUS,
    sticky_improve_margin=PHASECLS_STICKY_IMPROVE_MARGIN,
):
    """
    Prefix-valid rule/unsupervised detector:
    trigger only when the deterministic localizer itself is strong and stable.
    """
    cache_tag = _resolve_phase_cache_tag(sigs)
    payload = _precompute_prefix_localizer_payload(
        cs,
        sigs,
        lookback_localize,
        cache_tag,
    )
    localizer_table = payload["localizer_table"]
    score_table = payload["score_table"]
    shift_table = payload["shift_table"]
    det_by_day = {sgk: [None] * cs[sgk]["cycle_len"] for sgk in cs}
    conf_by_day = {sgk: [0.0] * cs[sgk]["cycle_len"] for sgk in cs}

    for sgk, data in cs.items():
        current_ov_est = None
        current_score = None
        localizer_seq = localizer_table.get(sgk, [])
        score_seq = score_table.get(sgk, [])
        shift_seq = shift_table.get(sgk, [])
        for day_idx in range(int(data["cycle_len"])):
            if not _localizer_evidence_ok(
                localizer_seq,
                score_seq,
                shift_seq,
                day_idx,
                score_min=localizer_score_min,
                shift_min=localizer_shift_min,
                agreement_days=localizer_agreement_days,
                agreement_tol=localizer_agreement_tol,
            ):
                continue
            ov_est = localizer_seq[day_idx] if day_idx < len(localizer_seq) else None
            ov_score = score_seq[day_idx] if day_idx < len(score_seq) else None
            current_ov_est, current_score = _apply_stabilization(
                current_ov_est=current_ov_est,
                current_score=current_score,
                ov_est=ov_est,
                ov_score=ov_score,
                day_idx=day_idx,
                stabilization_policy="sticky",
                clamp_radius=PHASECLS_CLAMP_RADIUS,
                sticky_radius=sticky_radius,
                sticky_improve_margin=sticky_improve_margin,
                monotone_back_margin=PHASECLS_MONOTONE_BACK_MARGIN,
            )
            if current_ov_est is None:
                continue
            assert current_ov_est <= day_idx
            det_by_day[sgk][day_idx] = int(current_ov_est)
            conf = float(max(0.0, min(1.0, (float(current_score or 0.0) - float(localizer_score_min)) / 2.0)))
            conf_by_day[sgk][day_idx] = conf
    return det_by_day, conf_by_day


def build_prefix_ml_features(
    data,
    day_idx,
    sigs=None,
    sigma=PREFIX_BENCHMARK_ML_SIGMA,
):
    """Build causal prefix-only features using only visible prefix days 1..d."""
    if sigs is None:
        sigs = PREFIX_ML_SIGNAL_GROUPS[-1][1]

    prefix_len = day_idx + 1
    if prefix_len < MIN_CYCLE_LEN_FOR_DETECTION:
        return None

    feats = {
        "prefix_day": float(prefix_len),
        "cycle_frac": float(prefix_len / max(data["hist_cycle_len"], 20.0)),
        "hist_clen": float(data["hist_cycle_len"]),
        "hist_cstd": float(data.get("hist_cycle_std", DEFAULT_HISTORY_CYCLE_STD)),
    }

    valid_signal_count = 0
    norm_series = []
    split_days = []

    for sk in sigs:
        raw = data.get(sk)
        if raw is None:
            continue
        prefix_raw = np.asarray(raw[:prefix_len], dtype=float)
        if np.isnan(prefix_raw).all():
            continue

        t = _clean(prefix_raw, sigma=sigma)
        s = pd.Series(t)
        name = sk
        rm3 = s.rolling(3, min_periods=1).mean()
        rm5 = s.rolling(5, min_periods=1).mean()
        rm7 = s.rolling(7, min_periods=1).mean()
        obs_ratio = float(np.mean(~np.isnan(prefix_raw)))
        diffs = np.diff(t)
        recent_slope = 0.0
        if prefix_len >= 3:
            xs = np.arange(min(5, prefix_len), dtype=float)
            ys = t[-len(xs):]
            recent_slope = float(np.polyfit(xs, ys, 1)[0])

        best_split_day = -1
        best_split_stat = 0.0
        best_split_shift = 0.0
        for sp in range(MIN_DETECTION_DAY, prefix_len - MAX_RIGHT_MARGIN_DAYS):
            left = t[:sp]
            right = t[sp:prefix_len]
            if len(left) < 2 or len(right) < 2:
                continue
            try:
                stat, _ = ttest_ind(right, left, alternative="greater")
            except Exception:
                continue
            if np.isnan(stat):
                continue
            if stat > best_split_stat:
                best_split_day = sp
                best_split_stat = float(stat)
                best_split_shift = float(np.mean(right) - np.mean(left))

        xover_idx = -1
        xover_gap = 0.0
        if len(rm3) >= 2:
            xdiff = np.asarray(rm3 - rm7, dtype=float)
            xover_gap = float(xdiff[-1])
            xover = np.where(np.diff(np.sign(xdiff)) != 0)[0]
            if len(xover) > 0:
                xover_idx = int(xover[-1])

        grad = np.gradient(t) if prefix_len >= 2 else np.array([0.0])
        q1, q2, q3, q4 = [float(np.mean(chunk)) for chunk in np.array_split(t, 4)]
        recent_window = min(5, prefix_len)
        recent_mean = float(np.mean(t[-recent_window:]))
        earlier_mean = (
            float(np.mean(t[:-recent_window]))
            if prefix_len > recent_window
            else float(np.mean(t))
        )
        mgd_day = int(np.argmax(grad)) if len(grad) > 0 else 0
        mgd_val = float(np.max(grad)) if len(grad) > 0 else 0.0
        nadir_day = int(np.argmin(t))
        feats[f"{name}_last"] = float(t[-1])
        feats[f"{name}_mean"] = float(np.mean(t))
        feats[f"{name}_std"] = float(np.std(t))
        feats[f"{name}_range"] = float(np.ptp(t))
        feats[f"{name}_skew"] = float(pd.Series(t).skew()) if prefix_len >= 3 else 0.0
        feats[f"{name}_kurt"] = float(pd.Series(t).kurtosis()) if prefix_len >= 4 else 0.0
        feats[f"{name}_obs_ratio"] = obs_ratio
        feats[f"{name}_rm3"] = float(rm3.iloc[-1]) if not np.isnan(rm3.iloc[-1]) else 0.0
        feats[f"{name}_rm5"] = float(rm5.iloc[-1]) if not np.isnan(rm5.iloc[-1]) else 0.0
        feats[f"{name}_rm7"] = float(rm7.iloc[-1]) if not np.isnan(rm7.iloc[-1]) else 0.0
        feats[f"{name}_rm3_rm7_gap"] = feats[f"{name}_rm3"] - feats[f"{name}_rm7"]
        feats[f"{name}_d1"] = float(t[-1] - t[-2]) if prefix_len >= 2 else 0.0
        feats[f"{name}_d3"] = float(t[-1] - t[-4]) if prefix_len >= 4 else 0.0
        feats[f"{name}_recent_slope"] = recent_slope
        feats[f"{name}_nadir_day"] = float(nadir_day)
        feats[f"{name}_grad_last"] = float(grad[-1])
        feats[f"{name}_grad_mean_abs"] = float(np.mean(np.abs(grad)))
        feats[f"{name}_mgd_day"] = float(mgd_day)
        feats[f"{name}_mgd_val"] = mgd_val
        feats[f"{name}_best_split_day"] = float(best_split_day)
        feats[f"{name}_best_split_stat"] = best_split_stat
        feats[f"{name}_best_split_shift"] = best_split_shift
        feats[f"{name}_xover_day"] = float(xover_idx)
        feats[f"{name}_xover_gap"] = xover_gap
        feats[f"{name}_half_diff"] = float(np.mean(t[prefix_len // 2 :]) - np.mean(t[: prefix_len // 2]))
        feats[f"{name}_q1"] = q1
        feats[f"{name}_q2"] = q2
        feats[f"{name}_q3"] = q3
        feats[f"{name}_q4"] = q4
        feats[f"{name}_last_minus_prefix_min"] = float(t[-1] - np.min(t))
        feats[f"{name}_last_minus_prefix_mean"] = float(t[-1] - np.mean(t))
        feats[f"{name}_recent_mean_minus_earlier_mean"] = recent_mean - earlier_mean
        for lag in FEATURE_AUTOCORR_LAGS:
            ac = s.autocorr(lag=lag)
            feats[f"{name}_ac{lag}"] = float(ac) if np.isfinite(ac) else 0.0

        std_t = np.std(t)
        if std_t > 1e-8:
            norm_series.append((t - np.mean(t)) / std_t)
        else:
            norm_series.append(np.zeros_like(t))
        if best_split_day >= 0:
            split_days.append(float(best_split_day))
        valid_signal_count += 1

    if valid_signal_count == 0:
        return None

    feats["valid_signal_count"] = float(valid_signal_count)
    if split_days:
        split_arr = np.asarray(split_days, dtype=float)
        feats["split_day_mean"] = float(np.mean(split_arr))
        feats["split_day_std"] = float(np.std(split_arr))
        feats["split_day_range"] = float(np.ptp(split_arr))
    else:
        feats["split_day_mean"] = 0.0
        feats["split_day_std"] = 0.0
        feats["split_day_range"] = 0.0

    if len(norm_series) >= 2:
        corrs = []
        last_gaps = []
        for i in range(len(norm_series)):
            for j in range(i + 1, len(norm_series)):
                c = np.corrcoef(norm_series[i], norm_series[j])[0, 1]
                corrs.append(float(c) if np.isfinite(c) else 0.0)
                last_gaps.append(float(abs(norm_series[i][-1] - norm_series[j][-1])))
        feats["pair_corr_mean"] = float(np.mean(corrs))
        feats["pair_corr_min"] = float(np.min(corrs))
        feats["pair_last_gap_mean"] = float(np.mean(last_gaps))
    else:
        feats["pair_corr_mean"] = 0.0
        feats["pair_corr_min"] = 0.0
        feats["pair_last_gap_mean"] = 0.0

    return feats


def prefix_ml_detect_loso(
    cs,
    lh,
    model_type="gbdt",
    sigs=None,
    sigma=PREFIX_BENCHMARK_ML_SIGMA,
):
    """Prefix-valid LOSO ML ovulation detector using causal day-by-day samples."""
    if sigs is None:
        sigs = PREFIX_ML_SIGNAL_GROUPS[-1][1]

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor

    labeled = [s for s in cs if s in lh]
    uniq_uids = np.unique([cs[s]["id"] for s in labeled])
    det_by_day = {sgk: [None] * cs[sgk]["cycle_len"] for sgk in cs}
    conf_by_day = {sgk: [0.0] * cs[sgk]["cycle_len"] for sgk in cs}
    day_rows = []

    for sgk in labeled:
        data = cs[sgk]
        ov_true = float(lh[sgk])
        n = data["cycle_len"]
        for day_idx in range(n):
            feats = build_prefix_ml_features(
                data,
                day_idx,
                sigs=sigs,
                sigma=sigma,
            )
            if feats is None:
                continue
            day_rows.append((sgk, data["id"], day_idx, feats, ov_true))

    for uid in uniq_uids:
        train_rows, train_targets, test_rows = [], [], []
        for sgk, row_uid, day_idx, feats, ov_true in day_rows:
            if row_uid == uid:
                test_rows.append((sgk, day_idx, feats))
            else:
                train_rows.append(feats)
                train_targets.append(ov_true)

        if len(train_rows) < 20 or len(test_rows) == 0:
            continue

        X_tr = pd.DataFrame(train_rows).fillna(0.0)
        feature_cols = list(X_tr.columns)
        y_tr = np.asarray(train_targets, dtype=float)
        X_te = pd.DataFrame([row for _, _, row in test_rows]).reindex(
            columns=feature_cols,
            fill_value=0.0,
        ).fillna(0.0)

        if model_type == "ridge":
            scaler = StandardScaler()
            X_tr_fit = scaler.fit_transform(X_tr.values)
            X_te_fit = scaler.transform(X_te.values)
            model = Ridge(alpha=1.0)
        elif model_type == "rf":
            X_tr_fit = X_tr.values
            X_te_fit = X_te.values
            model = RandomForestRegressor(
                n_estimators=160,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=1,
            )
        elif model_type == "gbdt":
            X_tr_fit = X_tr.values
            X_te_fit = X_te.values
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        elif model_type == "lgbm":
            X_tr_fit = X_tr
            X_te_fit = X_te
            model = LGBMRegressor(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=5,
                min_child_samples=10,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported prefix ML model_type: {model_type}")

        model.fit(X_tr_fit, y_tr)
        preds = model.predict(X_te_fit)
        for (sgk, day_idx, _), pred in zip(test_rows, preds):
            pred_day = int(round(pred))
            if pred_day <= day_idx and pred_day >= 0:
                det_by_day[sgk][day_idx] = pred_day
                conf_by_day[sgk][day_idx] = 0.5

    return det_by_day, conf_by_day


def gbdt_prefix_detect_loso(
    cs,
    lh,
    sigs=None,
    sigma=PREFIX_BENCHMARK_ML_SIGMA,
):
    """Backward-compatible prefix GBDT wrapper."""
    return prefix_ml_detect_loso(
        cs,
        lh,
        model_type="gbdt",
        sigs=sigs,
        sigma=sigma,
    )


def extract_features_v2(data, sigma=FEATURE_SIGMA):
    """Comprehensive multi-signal feature extraction. No cycle_len leakage."""
    feats = {}
    hcl = data["hist_cycle_len"]
    feats["hist_clen"] = hcl
    feats["hist_cstd"] = data.get("hist_cycle_std", DEFAULT_HISTORY_CYCLE_STD)
    exp_ov = hcl * EXPECTED_OVULATION_FRACTION
    n = data["cycle_len"]

    def _sig_feats(raw, prefix, sig=sigma, invert=False):
        if raw is None or np.isnan(raw).all():
            return {}
        t = _clean(raw, sigma=sig)
        if invert:
            t = -t
        f = {}
        f[f"{prefix}_mean"] = np.mean(t)
        f[f"{prefix}_std"] = np.std(t)
        f[f"{prefix}_range"] = np.ptp(t)
        f[f"{prefix}_skew"] = float(pd.Series(t).skew())
        f[f"{prefix}_kurt"] = float(pd.Series(t).kurtosis())
        f[f"{prefix}_nadir"] = int(np.argmin(t))
        f[f"{prefix}_nadir_dev"] = int(np.argmin(t)) - exp_ov
        grad = np.gradient(gaussian_filter1d(t, sigma=FEATURE_SIGMA))
        f[f"{prefix}_mgd"] = int(np.argmax(grad))
        f[f"{prefix}_mgd_dev"] = int(np.argmax(grad)) - exp_ov
        f[f"{prefix}_mgv"] = float(np.max(grad))
        for d in FEATURE_TTEST_DAYS:
            if MIN_DETECTION_DAY <= d < n - MAX_RIGHT_MARGIN_DAYS:
                try:
                    stat, _ = ttest_ind(t[d:], t[:d], alternative="greater")
                    f[f"{prefix}_tt{d}"] = float(stat) if not np.isnan(stat) else 0
                except:
                    f[f"{prefix}_tt{d}"] = 0
        best_sc, best_sp = -np.inf, n // 2
        for sp in range(MIN_DETECTION_DAY, n - MAX_RIGHT_MARGIN_DAYS):
            try:
                stat, _ = ttest_ind(t[sp:], t[:sp], alternative="greater")
                if not np.isnan(stat) and stat > best_sc:
                    best_sc = stat
                    best_sp = sp
            except:
                continue
        f[f"{prefix}_bs"] = best_sp
        f[f"{prefix}_bs_dev"] = best_sp - exp_ov
        f[f"{prefix}_bst"] = max(best_sc, 0)
        pre_m = np.mean(t[:best_sp])
        post_m = np.mean(t[best_sp:])
        f[f"{prefix}_shift"] = post_m - pre_m
        h1 = np.mean(t[:n // 2])
        h2 = np.mean(t[n // 2:])
        f[f"{prefix}_half_diff"] = h2 - h1
        q1 = np.mean(t[:n // 4])
        q2 = np.mean(t[n // 4:n // 2])
        q3 = np.mean(t[n // 2:3 * n // 4])
        q4 = np.mean(t[3 * n // 4:])
        f[f"{prefix}_q1"] = q1
        f[f"{prefix}_q2"] = q2
        f[f"{prefix}_q3"] = q3
        f[f"{prefix}_q4"] = q4
        ts = pd.Series(t)
        for lag in FEATURE_AUTOCORR_LAGS:
            ac = ts.autocorr(lag=lag)
            f[f"{prefix}_ac{lag}"] = float(ac) if not np.isnan(ac) else 0
        rm3 = ts.rolling(3, min_periods=1).mean()
        rm7 = ts.rolling(7, min_periods=1).mean()
        cross = np.where(np.diff(np.sign(rm3 - rm7)))[0]
        if len(cross) > 0:
            dists_c = np.abs(cross - exp_ov)
            f[f"{prefix}_xover"] = int(cross[np.argmin(dists_c)])
        else:
            f[f"{prefix}_xover"] = n // 2
        return f

    feats.update(_sig_feats(data.get("nightly_temperature"), "nt"))
    feats.update(_sig_feats(data.get("noct_temp"), "noct"))
    feats.update(_sig_feats(data.get("noct_hr_mean"), "nhr"))
    feats.update(_sig_feats(data.get("rhr"), "rhr"))
    feats.update(_sig_feats(data.get("rmssd_mean"), "rmssd", invert=True))
    feats.update(_sig_feats(data.get("hf_mean"), "hf", invert=True))
    feats.update(_sig_feats(data.get("lf_hf_ratio"), "lfhf"))

    # Cross-signal features
    for (s1, p1, inv1), (s2, p2, inv2) in [
        (("nightly_temperature", "nt", False), ("rmssd_mean", "rmssd", True)),
        (("nightly_temperature", "nt", False), ("noct_hr_mean", "nhr", False)),
        (("rmssd_mean", "rmssd", True), ("noct_hr_mean", "nhr", False)),
    ]:
        r1 = data.get(s1)
        r2 = data.get(s2)
        if r1 is not None and r2 is not None and not np.isnan(r1).all() and not np.isnan(r2).all():
            t1 = _clean(r1, sigma=sigma)
            t2 = _clean(r2, sigma=sigma)
            if inv1:
                t1 = -t1
            if inv2:
                t2 = -t2
            corr = np.corrcoef(t1, t2)[0, 1]
            feats[f"xcorr_{p1}_{p2}"] = float(corr) if not np.isnan(corr) else 0
            bs1 = feats.get(f"{p1}_bs", n // 2)
            bs2 = feats.get(f"{p2}_bs", n // 2)
            feats[f"bs_diff_{p1}_{p2}"] = abs(bs1 - bs2)
            feats[f"bs_mean_{p1}_{p2}"] = (bs1 + bs2) / 2

    for k in feats:
        if isinstance(feats[k], float) and (np.isnan(feats[k]) or np.isinf(feats[k])):
            feats[k] = 0.0
    return feats


def ml_detect_loso(cs, lh, model_type="ridge"):
    """LOSO ML detection with comprehensive features."""
    labeled = [s for s in cs if s in lh]
    all_f, all_t, all_id, all_s = [], [], [], []
    for sgk in labeled:
        feats = extract_features_v2(cs[sgk])
        if not feats or len(feats) < 5:
            continue
        all_f.append(feats)
        all_t.append(lh[sgk])
        all_id.append(cs[sgk]["id"])
        all_s.append(sgk)
    if len(all_f) < 10:
        return {}, {}

    df = pd.DataFrame(all_f).fillna(0)
    X = df.values
    y = np.array(all_t, dtype=float)
    ids = np.array(all_id)
    uniq = np.unique(ids)

    from sklearn.preprocessing import StandardScaler

    det, confs = {}, {}
    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        y_tr = y[tr]

        if model_type == "ridge":
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=300, max_depth=6,
                                      min_samples_leaf=3, random_state=42)
        elif model_type == "gbdt":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                          learning_rate=0.03, subsample=0.8,
                                          random_state=42)
        elif model_type == "elastic":
            from sklearn.linear_model import ElasticNet
            m = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)
        elif model_type == "svr":
            from sklearn.svm import SVR
            m = SVR(kernel="rbf", C=10.0, epsilon=0.5)
        elif model_type == "bayridge":
            from sklearn.linear_model import BayesianRidge
            m = BayesianRidge()
        elif model_type == "xgb":
            try:
                from xgboost import XGBRegressor
                m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0,
                                 random_state=42, verbosity=0)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                              learning_rate=0.03, random_state=42)
        elif model_type == "lgbm":
            try:
                from lightgbm import LGBMRegressor
                m = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, verbose=-1)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                              learning_rate=0.03, random_state=42)
        elif model_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            m = KNeighborsRegressor(n_neighbors=min(7, tr.sum()),
                                    weights="distance")
        elif model_type == "huber":
            from sklearn.linear_model import HuberRegressor
            m = HuberRegressor(max_iter=500)
        else:
            raise ValueError(model_type)

        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        test_sgks = [all_s[i] for i in np.where(te)[0]]
        for sgk, pred in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, pred))))
            confs[sgk] = 0.5
    return det, confs


def ml_phase_classify_loso(cs, lh):
    """
    Phase classification approach (Yu et al. 2022 inspired):
    Classify each day as follicular(0) vs luteal(1) using ML, then find boundary.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    labeled = [s for s in cs if s in lh]
    sigs_to_use = ["nightly_temperature", "noct_temp", "rhr",
                   "rmssd_mean", "lf_hf_ratio", "noct_hr_mean"]

    all_X, all_y, all_uid, all_sgk, all_dayidx = [], [], [], [], []
    for sgk in labeled:
        data = cs[sgk]
        ov = lh[sgk]
        n = data["cycle_len"]
        for i in range(n):
            row = []
            for sk in sigs_to_use:
                raw = data.get(sk)
                if raw is not None and not np.isnan(raw).all():
                    t = _clean(raw, sigma=FEATURE_SIGMA)
                    row.append(t[i])
                else:
                    row.append(0)
            row.append(i)
            row.append(data["hist_cycle_len"])
            all_X.append(row)
            all_y.append(1 if i >= ov else 0)
            all_uid.append(data["id"])
            all_sgk.append(sgk)
            all_dayidx.append(i)

    X = np.array(all_X)
    y = np.array(all_y)
    uids = np.array(all_uid)
    sgks = np.array(all_sgk)
    dayidxs = np.array(all_dayidx)
    uniq_uids = np.unique(uids)

    det, conf = {}, {}
    for uid in uniq_uids:
        te = uids == uid
        tr = ~te
        if tr.sum() < 20:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=42)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, 1]

        te_sgks = sgks[te]
        te_days = dayidxs[te]
        for sgk_val in np.unique(te_sgks):
            mask = te_sgks == sgk_val
            days = te_days[mask]
            probs = proba[mask]
            order = np.argsort(days)
            days = days[order]
            probs = probs[order]
            smoothed = gaussian_filter1d(probs, sigma=FEATURE_SIGMA)
            boundary = None
            for i in range(len(smoothed) - 1):
                if (
                    smoothed[i] < PHASE_CLASSIFIER_BOUNDARY_THRESHOLD
                    and smoothed[i + 1] >= PHASE_CLASSIFIER_BOUNDARY_THRESHOLD
                ):
                    boundary = int(days[i + 1])
                    break
            if boundary is None:
                boundary = int(
                    round(cs[sgk_val]["hist_cycle_len"] * EXPECTED_OVULATION_FRACTION)
                )
            det[sgk_val] = boundary
            conf[sgk_val] = float(np.max(probs) - np.min(probs))
    return det, conf


# =====================================================================
# C. 1D-CNN on multi-signal daily time series
# =====================================================================

def cnn_detect_loso(cs, lh, sigs=None, inverts=None, max_len=CNN_MAX_LEN):
    """1D-CNN regression on multi-signal daily series."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  [SKIP] PyTorch not available for CNN")
        return {}, {}

    if sigs is None:
        sigs = ["nightly_temperature", "noct_temp", "rhr",
                "rmssd_mean", "lf_hf_ratio", "noct_hr_mean"]
    if inverts is None:
        inverts = [False, False, False, True, False, False]

    labeled = [s for s in cs if s in lh]
    all_X, all_y, all_uid, all_sgk = [], [], [], []
    for sgk in labeled:
        data = cs[sgk]
        n = data["cycle_len"]
        channels = []
        valid = True
        for sk, inv in zip(sigs, inverts):
            raw = data.get(sk)
            if raw is None or np.isnan(raw).all():
                valid = False
                break
            t = _clean(raw, sigma=FEATURE_SIGMA)
            if inv:
                t = -t
            std = np.std(t)
            if std > 1e-8:
                t = (t - np.mean(t)) / std
            padded = np.zeros(max_len)
            padded[:min(n, max_len)] = t[:max_len]
            channels.append(padded)
        if not valid:
            continue
        all_X.append(np.stack(channels))
        all_y.append(lh[sgk])
        all_uid.append(data["id"])
        all_sgk.append(sgk)

    if len(all_X) < 15:
        return {}, {}

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    uids = np.array(all_uid)
    uniq = np.unique(uids)

    class CNN1D(nn.Module):
        def __init__(self, in_ch, seq_len):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.fc1 = nn.Linear(64 * 8, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.drop(self.relu(self.fc1(x)))
            return self.fc2(x).squeeze(-1)

    det, confs = {}, {}
    for uid in uniq:
        te = uids == uid
        tr = ~te
        if tr.sum() < 10:
            continue
        X_tr = torch.FloatTensor(X[tr])
        y_tr = torch.FloatTensor(y[tr])
        X_te = torch.FloatTensor(X[te])

        model = CNN1D(len(sigs), max_len)
        opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()

        model.train()
        for epoch in range(150):
            opt.zero_grad()
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te).numpy()
        test_sgks = [all_sgk[i] for i in np.where(te)[0]]
        for sgk, p in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, p))))
            confs[sgk] = 0.5
    return det, confs


# =====================================================================
# D. STACKING META-LEARNER
# =====================================================================

def stacking_detect(cs, lh, base_results):
    """Stacking: Ridge on base detector outputs + hist_clen as meta-features."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    labeled = [s for s in cs if s in lh]
    base_names = list(base_results.keys())
    rows, targets, uids, sgk_list = [], [], [], []
    for sgk in labeled:
        feats = [
            base_results[name][0].get(
                sgk, cs[sgk]["hist_cycle_len"] * EXPECTED_OVULATION_FRACTION
            )
                 for name in base_names]
        feats.append(cs[sgk]["hist_cycle_len"])
        feats.append(cs[sgk]["hist_cycle_len"] * EXPECTED_OVULATION_FRACTION)
        rows.append(feats)
        targets.append(lh[sgk])
        uids.append(cs[sgk]["id"])
        sgk_list.append(sgk)

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)
    ids = np.array(uids)
    uniq = np.unique(ids)

    det, confs = {}, {}
    for uid in uniq:
        te = ids == uid
        tr = ~te
        if tr.sum() < 5:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        m = Ridge(alpha=1.0)
        m.fit(X_tr, y[tr])
        preds = m.predict(X_te)
        test_sgks = [sgk_list[i] for i in np.where(te)[0]]
        for sgk, p in zip(test_sgks, preds):
            clen = cs[sgk]["cycle_len"]
            det[sgk] = int(round(max(5, min(clen - 3, p))))
            confs[sgk] = 0.6
    return det, confs


# =====================================================================
# E. WEIGHTED ENSEMBLE
# =====================================================================

def weighted_ensemble(results_list, cs, lh, top_n=5):
    """Weighted average of top-N detectors by ±2d accuracy."""
    labeled = set(s for s in cs if s in lh)
    scored = []
    for name, (d, c) in results_list:
        errs = [abs(d[s] - lh[s]) for s in d if s in labeled]
        if errs:
            acc2 = np.mean(np.array(errs) <= 2)
            scored.append((acc2, name, d, c))
    scored.sort(reverse=True)
    top = scored[:top_n]
    if not top:
        return {}, {}

    det, confs = {}, {}
    all_sgks = set()
    for _, _, d, _ in top:
        all_sgks.update(d.keys())
    for sgk in all_sgks:
        vals, ws = [], []
        for acc, _, d, c in top:
            if sgk in d:
                vals.append(d[sgk])
                ws.append(max(acc, 0.01))
        if vals:
            det[sgk] = int(round(np.average(vals, weights=ws)))
            confs[sgk] = np.mean(ws)
    return det, confs

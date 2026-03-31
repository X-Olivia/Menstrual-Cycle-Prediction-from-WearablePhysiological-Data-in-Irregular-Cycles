"""
EXPERIMENTAL: equal-weight LOSO phase-probability ensemble across classifier families.

Lazily imports single-model precompute from ``detectors_ml`` to avoid import cycles.
"""
from __future__ import annotations

import pickle

import numpy as np

from data import PROCESSED, _load_all_signals_source_files, _snapshot_source_files
from protocol import PREFIX_CACHE_VERSION, PHASECLS_MODEL_CACHE_VERSION


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
    from detectors_ml import precompute_prefix_phase_probabilities_loso

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

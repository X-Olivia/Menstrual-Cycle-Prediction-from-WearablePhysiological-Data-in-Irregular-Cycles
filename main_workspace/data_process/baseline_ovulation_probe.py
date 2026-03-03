"""
Ovulation signal learnability probe: use Logistic Regression on same-day features x_t to predict whether the day has ovulation probability.
Labels: y=1 for any day with ovulation_prob_fused >= prob_positive_threshold (multiple positives per cycle).
Evaluation: cycle-level accuracy — predicted ovulation day (argmax of model score in cycle) counts correct if it falls on any day that has ovulation probability in the cycle.
Run: from main_workspace execute  python -m model.baseline_ovulation_probe
"""
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import CYCLE_CSV, FULL_CSV, SLEEP_CSV, FEATURE_COLS, RANDOM_SEED, TEST_SUBJECT_RATIO
from .dataset import load_and_merge, add_ovulation_labels
from .split import split_fixed_test

# Days with ovulation_prob_fused >= this count as "has ovulation probability" (label and acceptable set)
PROB_POSITIVE_THRESHOLD = 0.05
# Clip z-score features to this range (data may contain invalid z-scores; see docs/Probe_Data_Issues_Analysis.md)
FEATURE_CLIP = 5.0

# Stage-one ablation: feature group definitions (all subsets of FEATURE_COLS)
# v2 groups add cycle_pos (P1) and biphasic shift (P5) ablations
ABLATION_GROUPS = {
    "temp_only": ["nightly_temperature_z", "wt_mean_z", "wt_std_z", "wt_min_z", "wt_max_z"],
    "hr_only": ["hr_mean_z", "hr_std_z", "hr_min_z", "hr_max_z", "resting_hr_z"],
    "hrv_only": ["rmssd_mean_z", "lf_mean_z", "hf_mean_z", "lf_hf_ratio_z"],
    "temp_hr": ["nightly_temperature_z", "wt_mean_z", "wt_std_z", "wt_min_z", "wt_max_z",
                "hr_mean_z", "hr_std_z", "hr_min_z", "hr_max_z", "resting_hr_z"],
    # v2 additions
    "cycle_pos_only": ["day_in_cycle", "day_in_cycle_frac"],
    "biphasic_only":  ["wt_shift_7v3", "temp_shift_7v3"],
    "temp_biphasic":  ["nightly_temperature_z", "wt_mean_z", "wt_std_z", "wt_min_z", "wt_max_z",
                       "wt_shift_7v3", "temp_shift_7v3"],
    "all": FEATURE_COLS,
}


def build_probe_table(
    cycle_path=CYCLE_CSV,
    full_path=SLEEP_CSV,
    min_cycle_confidence=0.6,
    prob_positive_threshold=PROB_POSITIVE_THRESHOLD,
    quiet=False,
):
    """Build per-day sample table. Each row: id, cycle_key, X, y, ovulation_prob.
    y=1 for any day with ovulation_prob_fused >= prob_positive_threshold (multiple positives per cycle).
    ovulation_prob is the raw probability for cycle-level evaluation (acceptable day = prob >= threshold).
    Keep only cycles with max(ovulation_prob_fused) >= min_cycle_confidence."""
    df = load_and_merge(cycle_path, full_path)
    df = add_ovulation_labels(df)

    rows = []
    n_cycles_dropped = 0
    for (uid, key), g in df.groupby(["id", "small_group_key"], sort=True):
        g = g.sort_values("day_in_study")
        X = g[FEATURE_COLS].values.astype(np.float32)
        X = np.clip(np.nan_to_num(X, nan=0.0), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
        prob = np.nan_to_num(g["ovulation_prob_fused"].values, nan=0.0).astype(np.float32)
        max_prob = float(np.max(prob))
        if max_prob < min_cycle_confidence:
            n_cycles_dropped += 1
            continue
        # Multiple positives: any day with ovulation probability >= threshold
        y_binary = (prob >= prob_positive_threshold).astype(np.float32)
        for i in range(len(g)):
            rows.append({
                "id": uid,
                "cycle_key": (uid, key),
                "X": X[i],
                "y": y_binary[i],
                "ovulation_prob": prob[i],
            })
    if n_cycles_dropped > 0 and not quiet:
        print(f"[probe] Filtered {n_cycles_dropped} low-confidence cycles (max_prob < {min_cycle_confidence})")
    return rows


def _cycle_level_accuracy(p_test, test_cycle_keys, test_ovulation_probs, prob_positive_threshold):
    """Cycle-level accuracy: for each cycle, predicted day = argmax of p_test; correct if that day has ovulation_prob >= threshold.
    Also returns random baseline = mean(acceptable_days/cycle_length) per cycle."""
    cycle_to_test_indices = defaultdict(list)
    for j, key in enumerate(test_cycle_keys):
        cycle_to_test_indices[key].append(j)
    correct, total = 0, 0
    random_baseline_sum = 0.0
    for key, indices in cycle_to_test_indices.items():
        if not indices:
            continue
        total += 1
        n_acceptable = sum(1 for i in indices if test_ovulation_probs[i] >= prob_positive_threshold)
        random_baseline_sum += (n_acceptable / len(indices)) if indices else 0
        pred_idx_in_cycle = indices[int(np.argmax(p_test[indices]))]
        acceptable = [i for i in indices if test_ovulation_probs[i] >= prob_positive_threshold]
        if pred_idx_in_cycle in acceptable:
            correct += 1
    acc = (correct / total) if total else float("nan")
    random_baseline = (random_baseline_sum / total) if total else float("nan")
    return acc, correct, total, random_baseline


def run_probe(
    cycle_path=CYCLE_CSV,
    full_path=SLEEP_CSV,
    seed=RANDOM_SEED,
    test_ratio=TEST_SUBJECT_RATIO,
    min_cycle_confidence=0.6,
    prob_positive_threshold=PROB_POSITIVE_THRESHOLD,
    feature_cols=None,
    verbose=True,
):
    """Split train/test by subject; fit LR on train (y=1 for days with ovulation_prob >= threshold).
    Main metric: cycle-level accuracy — predicted ovulation day in cycle counts correct if it lies in the set of days with ovulation probability."""
    rows = build_probe_table(
        cycle_path, full_path, min_cycle_confidence,
        prob_positive_threshold=prob_positive_threshold,
        quiet=not verbose,
    )
    if not rows:
        print("[probe] No data")
        return None
    ids = np.array([r["id"] for r in rows])
    cycle_keys = [r["cycle_key"] for r in rows]
    X_all = np.stack([r["X"] for r in rows])
    y_all = np.array([r["y"] for r in rows])
    ovulation_probs = np.array([r["ovulation_prob"] for r in rows])
    if feature_cols is not None:
        col_idx = [FEATURE_COLS.index(c) for c in feature_cols]
        X_all = X_all[:, col_idx]
    subject_ids = np.unique(ids)
    test_subjects, trainval_subjects = split_fixed_test(subject_ids, test_ratio, seed)
    trainval_subjects = list(trainval_subjects)
    test_subjects = set(test_subjects)

    trainval_mask = np.isin(ids, trainval_subjects)
    test_mask = np.isin(ids, list(test_subjects))
    X_train, y_train = X_all[trainval_mask], y_all[trainval_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    test_cycle_keys = [cycle_keys[i] for i in np.where(test_mask)[0]]
    test_ovulation_probs = ovulation_probs[test_mask]

    n_pos_train, n_pos_test = int(y_train.sum()), int(y_test.sum())
    if n_pos_test == 0 and verbose:
        print("[probe] Test set has no positive class (no day with ovulation_prob >= threshold)")

    # Sanity check (only when full features and verbose)
    if verbose and feature_cols is None:
        feat_mean_abs = np.mean(np.abs(X_train))
        print(f"[probe] Feature mean |x| (after clip ±{FEATURE_CLIP}): {feat_mean_abs:.4f}  (z-scores expect ~0.5–1; huge values suggest upstream z-score bugs)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=seed, class_weight=None)
    clf.fit(X_train_s, y_train)
    p_test = clf.predict_proba(X_test_s)[:, 1]

    # Main metric: cycle-level accuracy (predicted day in cycle is in acceptable set)
    cycle_acc, cycle_correct, cycle_total, random_baseline = _cycle_level_accuracy(
        p_test, test_cycle_keys, test_ovulation_probs, prob_positive_threshold
    )

    # Secondary: per-day AUC when there are positives
    if n_pos_test > 0:
        auc = roc_auc_score(y_test, p_test)
        pr_auc = average_precision_score(y_test, p_test)
        baseline_pr = float(y_test.mean())
        pr_ratio = pr_auc / baseline_pr if baseline_pr > 0 else float("nan")
    else:
        auc = pr_auc = baseline_pr = pr_ratio = float("nan")

    if not np.isnan(auc):
        if auc < 0.55:
            auc_conclusion, conclusion = "no signal", "Signal not separable; improve features or accept ceiling"
        elif auc < 0.65:
            auc_conclusion, conclusion = "weak signal", "Weak signal; optimization possible but limited"
        elif auc < 0.75:
            auc_conclusion, conclusion = "learnable", "Learnable; safe to optimize loss and structure"
        else:
            auc_conclusion, conclusion = "strong signal", "Strong signal; focus on model and loss optimization"
        pr_meaning = "weak" if (np.isnan(pr_ratio) or pr_ratio < 1.5) else ("present" if pr_ratio < 3 else "strong")
    else:
        auc_conclusion = pr_meaning = conclusion = "—"

    if verbose:
        dim = X_train.shape[1]
        print("[Ovulation probe] Labels: y=1 if ovulation_prob>={}; evaluation: cycle-level accuracy (predicted day in acceptable set)".format(prob_positive_threshold))
        print("  Train: n={}, positives={};  Test: n={}, positives={}".format(len(y_train), n_pos_train, len(y_test), n_pos_test))
        print("  Cycle-level accuracy = {}/{} = {:.4f}  (random baseline = {:.4f})".format(cycle_correct, cycle_total, cycle_acc, random_baseline))
        if not np.isnan(auc):
            print("  Per-day AUC = {:.4f} ({})  PR-AUC = {:.4f}  PR/baseline = {:.2f} ({})".format(auc, auc_conclusion, pr_auc, pr_ratio, pr_meaning))
        print("  Conclusion: {}".format(conclusion))
    return {
        "cycle_accuracy": cycle_acc,
        "cycle_correct": cycle_correct,
        "cycle_total": cycle_total,
        "random_baseline": random_baseline,
        "auc": auc,
        "pr_auc": pr_auc,
        "baseline_pr": baseline_pr,
        "pr_ratio": pr_ratio,
        "auc_conclusion": auc_conclusion,
        "pr_meaning": pr_meaning,
        "conclusion": conclusion,
    }


def run_probe_ablation(
    cycle_path=CYCLE_CSV,
    full_path=SLEEP_CSV,
    seed=RANDOM_SEED,
    test_ratio=TEST_SUBJECT_RATIO,
    min_cycle_confidence=0.6,
    prob_positive_threshold=PROB_POSITIVE_THRESHOLD,
    groups=None,
):
    """Stage one: feature ablation. Run probe for each feature group; report cycle-level accuracy and per-day AUC."""
    groups = groups or ABLATION_GROUPS
    print("=" * 60)
    print("Stage one: feature ablation (ovulation probe)")
    print("  Label: y=1 if ovulation_prob>={};  Eval: cycle acc = pred day in acceptable days".format(prob_positive_threshold))
    print("=" * 60)
    rows = build_probe_table(
        cycle_path, full_path, min_cycle_confidence,
        prob_positive_threshold=prob_positive_threshold,
    )
    if not rows:
        print("[probe] No data")
        return {}
    n_samples = len(rows)
    n_pos = int(sum(r["y"] for r in rows))
    print("  Samples: {} days, positives (days with ovulation prob>={}) {} (min_cycle_confidence={})".format(
        n_samples, prob_positive_threshold, n_pos, min_cycle_confidence))
    print("  (CycleAcc = pred day in acceptable set; random baseline = mean(acceptable_days/cycle_len) per cycle)\n")

    results = []
    by_name = {}
    for name, cols in groups.items():
        out = run_probe(
            cycle_path=cycle_path,
            full_path=full_path,
            seed=seed,
            test_ratio=test_ratio,
            min_cycle_confidence=min_cycle_confidence,
            prob_positive_threshold=prob_positive_threshold,
            feature_cols=cols,
            verbose=False,
        )
        by_name[name] = out or {}
        if out is None:
            results.append((name, len(cols), float("nan"), float("nan"), 0, 0, float("nan"), "—"))
        else:
            results.append((
                name,
                len(cols),
                out["cycle_accuracy"],
                out["auc"],
                out["cycle_correct"],
                out["cycle_total"],
                out["random_baseline"],
                out["auc_conclusion"],
            ))

    # Table: show correct/total and random baseline so single-digit denominators are visible
    print("Feature group       dim   CycleAcc(n/N)    RandBase  AUC     signal")
    print("-" * 72)
    for name, dim, cycle_acc, auc, correct, total, rand_base, sig in results:
        acc_str = "{:.3f}({}/{})".format(cycle_acc, int(correct), int(total)) if not np.isnan(cycle_acc) else "—"
        rb_str  = "{:.3f}".format(rand_base) if not np.isnan(rand_base) else "—"
        auc_str = "{:.3f}".format(auc) if not np.isnan(auc) else "—"
        print("  {:<18}  {:>2}   {:<17}  {}      {}   {}".format(name, dim, acc_str, rb_str, auc_str, sig))
    print("-" * 72)
    best = max((r for r in results if not np.isnan(r[2])), key=lambda x: x[2], default=None)
    if best:
        print("  Best CycleAcc: [{}] = {:.3f} ({}/{}); AUC = {:.3f}".format(
            best[0], best[2], int(best[4]), int(best[5]), best[3]))
    print()
    print("  Note: CycleAcc = fraction of test cycles where top-scored day is in acceptable set.")
    print("        RandBase = expected CycleAcc by chance (mean acceptable_days/cycle_len).")
    print("        biphasic AUC<0.5 indicates inverted signal (temp rise LAGS ovulation by ~3 days).")
    return by_name


def run_probe_loso(
    cycle_path=CYCLE_CSV,
    full_path=SLEEP_CSV,
    min_cycle_confidence=0.6,
    prob_positive_threshold=PROB_POSITIVE_THRESHOLD,
    groups=None,
):
    """Leave-One-Subject-Out cross-validation probe.
    With ~25 subjects, LOSO gives 25 folds instead of 1 fixed split, making CycleAcc estimates
    far more reliable. Each subject is held out as test once; scores are averaged across folds.
    Recommended over the fixed split for small N diagnostics.
    """
    groups = groups or ABLATION_GROUPS
    rows = build_probe_table(cycle_path, full_path, min_cycle_confidence,
                             prob_positive_threshold=prob_positive_threshold)
    if not rows:
        print("[probe LOSO] No data")
        return {}

    ids      = np.array([r["id"] for r in rows])
    ck       = [r["cycle_key"] for r in rows]
    X_all    = np.stack([r["X"] for r in rows])
    y_all    = np.array([r["y"] for r in rows])
    ov_probs = np.array([r["ovulation_prob"] for r in rows])
    subjects = np.unique(ids)

    print("=" * 60)
    print("Stage one (LOSO): feature ablation — {} subjects, {} days".format(len(subjects), len(rows)))
    print("  Label: y=1 if ovulation_prob>={};  Eval: mean CycleAcc over {} folds".format(
        prob_positive_threshold, len(subjects)))
    print("=" * 60)

    group_results = {}
    for name, cols in groups.items():
        col_idx = [FEATURE_COLS.index(c) for c in cols if c in FEATURE_COLS]
        X_g = X_all[:, col_idx]
        fold_accs, fold_aucs, fold_rands = [], [], []
        for test_subj in subjects:
            tr_mask = ids != test_subj
            te_mask = ids == test_subj
            if te_mask.sum() == 0:
                continue
            X_tr, y_tr = X_g[tr_mask], y_all[tr_mask]
            X_te, y_te = X_g[te_mask], y_all[te_mask]
            if y_tr.sum() == 0 or y_te.sum() == 0:
                continue
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight=None)
            clf.fit(X_tr_s, y_tr)
            p_te = clf.predict_proba(X_te_s)[:, 1]
            acc, _, _, rand = _cycle_level_accuracy(
                p_te,
                [ck[i] for i in np.where(te_mask)[0]],
                ov_probs[te_mask],
                prob_positive_threshold,
            )
            fold_accs.append(acc)
            fold_rands.append(rand)
            try:
                fold_aucs.append(roc_auc_score(y_te, p_te))
            except Exception:
                pass
        mean_acc  = float(np.nanmean(fold_accs))  if fold_accs  else float("nan")
        mean_auc  = float(np.nanmean(fold_aucs))  if fold_aucs  else float("nan")
        mean_rand = float(np.nanmean(fold_rands)) if fold_rands else float("nan")
        group_results[name] = {"loso_cycle_acc": mean_acc, "loso_auc": mean_auc,
                                "loso_rand_base": mean_rand, "n_folds": len(fold_accs)}

    print("Feature group       dim   LOSO_CycleAcc  LOSO_RandBase  LOSO_AUC")
    print("-" * 72)
    for name, cols in groups.items():
        r = group_results.get(name, {})
        acc_s  = "{:.3f}".format(r["loso_cycle_acc"])  if not np.isnan(r.get("loso_cycle_acc",  float("nan"))) else "—"
        rand_s = "{:.3f}".format(r["loso_rand_base"]) if not np.isnan(r.get("loso_rand_base", float("nan"))) else "—"
        auc_s  = "{:.3f}".format(r["loso_auc"])       if not np.isnan(r.get("loso_auc",        float("nan"))) else "—"
        folds  = r.get("n_folds", 0)
        print("  {:<18}  {:>2}   {}           {}         {}  (folds={})".format(
            name, len(cols), acc_s, rand_s, auc_s, folds))
    print("-" * 72)
    print("  LOSO_CycleAcc: mean fraction of per-subject test cycles correctly predicted.")
    return group_results


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    if mode == "full":
        run_probe()           # single full-feature probe with verbose output
    elif mode == "loso":
        run_probe_loso()      # LOSO cross-validation (more reliable with small N)
    else:
        run_probe_ablation()  # default: fixed split ablation table

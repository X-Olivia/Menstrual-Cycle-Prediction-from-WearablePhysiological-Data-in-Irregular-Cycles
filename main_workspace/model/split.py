"""Subject-based split: fixed test + K-fold for train/val only."""
import numpy as np


def get_unique_subjects(ids):
    """Deduplicate to get subject list."""
    return np.unique(ids)


def split_fixed_test(subject_ids, test_ratio, seed):
    """Hold out fixed test subjects first; rest used for K-fold train/val."""
    rng = np.random.default_rng(seed)
    subjects = get_unique_subjects(subject_ids)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = set(subjects[:n_test])
    trainval_subjects = subjects[n_test:]
    return test_subjects, trainval_subjects


def kfold_trainval(trainval_subjects, n_folds, seed):
    """K-fold for train/val only; each fold returns (train_subjects, val_subjects)."""
    rng = np.random.default_rng(seed)
    subjects = np.array(trainval_subjects)
    rng.shuffle(subjects)
    n = len(subjects)
    fold_size = n // n_folds
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        val_subjects = set(subjects[start:end])
        train_subjects = set(subjects) - val_subjects
        yield train_subjects, val_subjects

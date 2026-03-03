"""End-to-end experiment for GRU sequence model."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .seq_dataset import build_cycle_data, CycleSequenceDataset, collate_cycles
from .seq_model import CycleGRU, huber_loss_masked
from .evaluate import compute_metrics, print_metrics
from .config import RANDOM_SEED, TEST_SUBJECT_RATIO


def subject_split_cycles(cycles, test_ratio=0.15, seed=42):
    """Split cycles by subject (same logic as LightGBM pipeline)."""
    subjects = sorted(set(c["id"] for c in cycles))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n_test = max(1, int(len(subjects) * test_ratio))
    test_subj = set(subjects[:n_test])
    remain = [s for s in subjects if s not in test_subj]

    rng2 = np.random.RandomState(seed + 1)
    rng2.shuffle(remain)
    n_val = max(1, int(len(remain) * 0.2))
    val_subj = set(remain[:n_val])
    train_subj = set(remain[n_val:])

    train = [c for c in cycles if c["id"] in train_subj]
    val = [c for c in cycles if c["id"] in val_subj]
    test = [c for c in cycles if c["id"] in test_subj]

    print(
        f"[split] train={len(train)} cycles ({len(train_subj)} subj), "
        f"val={len(val)} ({len(val_subj)}), test={len(test)} ({len(test_subj)})"
    )
    return train, val, test


def train_gru(
    train_cycles,
    val_cycles,
    seq_dim,
    static_dim,
    hidden_dim=32,
    dropout=0.4,
    lr=1e-3,
    weight_decay=1e-3,
    max_epochs=300,
    patience=20,
    batch_size=16,
    seed=42,
):
    """Train GRU model with early stopping on validation MAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CycleSequenceDataset(train_cycles, augment=True)
    val_ds = CycleSequenceDataset(val_cycles, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_cycles
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, len(val_ds)), shuffle=False, collate_fn=collate_cycles
    )

    model = CycleGRU(
        seq_dim=seq_dim, static_dim=static_dim,
        hidden_dim=hidden_dim, dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_mae = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        for seq, static, label, mask, lengths in train_loader:
            optimizer.zero_grad()
            pred = model(seq, static, lengths)
            loss = huber_loss_masked(pred, label, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            all_pred, all_true = [], []
            for seq, static, label, mask, lengths in val_loader:
                pred = model(seq, static, lengths)
                for i in range(len(lengths)):
                    L = lengths[i].item()
                    all_pred.extend(pred[i, :L].numpy())
                    all_true.extend(label[i, :L].numpy())
            val_mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 50 == 0 or wait == 0:
            print(
                f"  Epoch {epoch+1:3d}: loss={train_loss/n_batches:.4f}, "
                f"val_mae={val_mae:.3f}, best={best_val_mae:.3f}, wait={wait}"
            )

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"[train] Best val MAE: {best_val_mae:.3f}")
    return model


def evaluate_gru(model, cycles):
    """Evaluate GRU on a set of cycles, returning flat pred/true arrays."""
    ds = CycleSequenceDataset(cycles, augment=False)
    loader = DataLoader(
        ds, batch_size=max(1, len(ds)), shuffle=False, collate_fn=collate_cycles
    )

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for seq, static, label, mask, lengths in loader:
            pred = model(seq, static, lengths)
            for i in range(len(lengths)):
                L = lengths[i].item()
                all_pred.extend(pred[i, :L].numpy())
                all_true.extend(label[i, :L].numpy())

    return np.array(all_pred), np.array(all_true)


def run_seq_experiment(seed=RANDOM_SEED):
    """Run single GRU experiment."""
    print("=" * 60)
    print("  GRU Sequence Model Experiment")
    print("=" * 60)

    cycles, seq_feats, static_feats = build_cycle_data()
    train_c, val_c, test_c = subject_split_cycles(
        cycles, test_ratio=TEST_SUBJECT_RATIO, seed=seed
    )

    model = train_gru(
        train_c, val_c,
        seq_dim=len(seq_feats),
        static_dim=len(static_feats),
        seed=seed,
    )

    pred_val, true_val = evaluate_gru(model, val_c)
    pred_test, true_test = evaluate_gru(model, test_c)

    val_overall, val_hz = print_metrics(
        {"pred": pred_val, "true": true_val}, "Validation"
    )
    test_overall, test_hz = print_metrics(
        {"pred": pred_test, "true": true_test}, "Test"
    )

    return {
        "model": model,
        "val": val_overall,
        "test": test_overall,
    }


def run_multi_seed_seq(n_seeds=5):
    """Multi-seed evaluation for GRU model."""
    print("=" * 60)
    print(f"  GRU Multi-Seed Evaluation ({n_seeds} seeds)")
    print("=" * 60)

    cycles, seq_feats, static_feats = build_cycle_data()

    test_maes, val_maes, test_acc3s = [], [], []

    for seed in range(42, 42 + n_seeds):
        print(f"\n{'─' * 40} Seed {seed} {'─' * 40}")
        train_c, val_c, test_c = subject_split_cycles(
            cycles, test_ratio=TEST_SUBJECT_RATIO, seed=seed
        )

        model = train_gru(
            train_c, val_c,
            seq_dim=len(seq_feats),
            static_dim=len(static_feats),
            seed=seed,
        )

        pred_test, true_test = evaluate_gru(model, test_c)
        pred_val, true_val = evaluate_gru(model, val_c)

        test_m = compute_metrics(pred_test, true_test)
        val_m = compute_metrics(pred_val, true_val)

        test_maes.append(test_m["mae"])
        val_maes.append(val_m["mae"])
        test_acc3s.append(test_m["acc_3d"])

        print(
            f"  → test MAE={test_m['mae']:.3f}, ±3d={test_m['acc_3d']:.3f}, "
            f"val MAE={val_m['mae']:.3f}"
        )

    print(f"\n{'=' * 60}")
    print(f"  GRU AGGREGATE ({n_seeds} seeds)")
    print(f"{'=' * 60}")
    print(f"  Test MAE: {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f}")
    print(f"  Test ±3d: {np.mean(test_acc3s):.3f} ± {np.std(test_acc3s):.3f}")
    print(f"  Val MAE:  {np.mean(val_maes):.3f} ± {np.std(val_maes):.3f}")

    return {
        "test_mae_mean": np.mean(test_maes),
        "test_mae_std": np.std(test_maes),
        "test_acc3_mean": np.mean(test_acc3s),
        "val_mae_mean": np.mean(val_maes),
    }


if __name__ == "__main__":
    run_seq_experiment()

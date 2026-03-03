"""Two-stage training: Stage1 joint optimization, Stage2 main-task fine-tuning."""
import random
import torch
from torch.utils.data import DataLoader

from .config import (
    LAMBDA_OV,
    LAMBDA_OV_STAGE2,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    LR_STAGE1,
    LR_STAGE2_HEAD,
    LR_STAGE2_BACKBONE,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP,
)
from .dataset import CycleSequenceDataset, collate_cycle_sequences
from .net import CycleModel
from .losses import total_loss, masked_menses_loss, masked_ovulation_bce

# Augmentation: randomly crop the start of each sequence during training
# by up to AUG_MAX_SKIP days. Simulates observing a cycle from a later starting point.
# Disabled in evaluation (model sees full sequence).
AUG_MAX_SKIP = 3


def _augment_batch(batch, max_skip=AUG_MAX_SKIP):
    """Randomly trim up to max_skip days from the start of each sequence in the batch."""
    X_orig       = batch["X"]
    lengths_orig = batch["lengths"]
    B, T, F      = X_orig.shape

    # First pass: compute skips and new lengths
    skips, lens_out = [], []
    for i in range(B):
        T_i  = int(lengths_orig[i].item())
        skip = random.randint(0, min(max_skip, T_i - 1))
        skips.append(skip)
        lens_out.append(T_i - skip)

    # Allocate outputs sized to the NEW max length (not the original padded length)
    T_new_max = max(lens_out)
    X_out   = torch.zeros(B, T_new_max, F,  dtype=X_orig.dtype)
    lm_out  = torch.zeros(B, T_new_max,     dtype=batch["y_menses"].dtype)
    lov_out = torch.zeros(B, T_new_max,     dtype=batch["y_ovulation"].dtype)
    mm_out  = torch.zeros(B, T_new_max,     dtype=batch["mask_menses"].dtype)
    mov_out = torch.zeros(B, T_new_max,     dtype=batch["mask_ovulation"].dtype)

    for i in range(B):
        T_i   = int(lengths_orig[i].item())
        skip  = skips[i]
        T_new = lens_out[i]
        X_out[i,   :T_new] = X_orig[i,                    skip:T_i]
        lm_out[i,  :T_new] = batch["y_menses"][i,         skip:T_i]
        lov_out[i, :T_new] = batch["y_ovulation"][i,      skip:T_i]
        mm_out[i,  :T_new] = batch["mask_menses"][i,      skip:T_i]
        mov_out[i, :T_new] = batch["mask_ovulation"][i,   skip:T_i]

    return {
        "X": X_out,
        "y_menses": lm_out,
        "y_ovulation": lov_out,
        "mask_menses": mm_out,
        "mask_ovulation": mov_out,
        "lengths": torch.tensor(lens_out, dtype=torch.long),
    }


def train_epoch(model, loader, optimizer, device, lambda_ov, stage=1, augment=True):
    model.train()
    total_l, total_m, total_ov = 0.0, 0.0, 0.0
    n = 0
    for batch in loader:
        if augment:
            batch = _augment_batch(batch)
        X        = batch["X"].to(device)
        lengths  = batch["lengths"].to(device)
        y_m      = batch["y_menses"].to(device)
        y_m_log  = torch.log1p(y_m)
        y_ov     = batch["y_ovulation"].to(device)
        mask_m   = batch["mask_menses"].to(device)
        mask_ov  = batch["mask_ovulation"].to(device)

        optimizer.zero_grad()
        y_m_pred, p_ov_pred = model(X, lengths)

        if stage == 1:
            loss, L_m, L_ov = total_loss(
                y_m_pred, y_m_log, mask_m, p_ov_pred, y_ov, mask_ov, lambda_ov
            )
        else:
            # Stage 2: menses fine-tuning with a small ovulation regularisation so that the
            # backbone representation learned in Stage 1 (HRV window + temperature shift) doesn't degrade.
            L_m  = masked_menses_loss(y_m_pred, y_m_log, mask_m)
            L_ov = masked_ovulation_bce(p_ov_pred, y_ov, mask_ov)
            loss = L_m + LAMBDA_OV_STAGE2 * L_ov

        loss.backward()
        # Gradient clipping: essential for RNN stability with variable-length sequences
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_l  += loss.item()
        total_m  += L_m.item()
        total_ov += L_ov.item()
        n += 1
    return total_l / n, total_m / n, total_ov / n


def eval_epoch(model, loader, device, lambda_ov):
    model.eval()
    total_l, total_m, total_ov = 0.0, 0.0, 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            X        = batch["X"].to(device)
            lengths  = batch["lengths"].to(device)
            y_m      = batch["y_menses"].to(device)
            y_m_log  = torch.log1p(y_m)
            y_ov     = batch["y_ovulation"].to(device)
            mask_m   = batch["mask_menses"].to(device)
            mask_ov  = batch["mask_ovulation"].to(device)

            y_m_pred, p_ov_pred = model(X, lengths)
            loss, L_m, L_ov = total_loss(
                y_m_pred, y_m_log, mask_m, p_ov_pred, y_ov, mask_ov, lambda_ov
            )
            total_l  += loss.item()
            total_m  += L_m.item()
            total_ov += L_ov.item()
            n += 1
    return total_l / n, total_m / n, total_ov / n


def run_stage1(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_STAGE1)
    best_val  = float("inf")
    wait      = 0
    best_state = None
    for epoch in range(STAGE1_EPOCHS):
        train_epoch(model, train_loader, optimizer, device, LAMBDA_OV, stage=1, augment=True)
        val_l, val_m, val_ov = eval_epoch(model, val_loader, device, LAMBDA_OV)
        if val_l < best_val:
            best_val   = val_l
            wait       = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_stage2(model, train_loader, val_loader, device):
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.menses_head.parameters())
        + list(model.ovulation_head_fwd.parameters())
        + list(model.ovulation_head_full.parameters())
    )
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": LR_STAGE2_BACKBONE},
            {"params": head_params,     "lr": LR_STAGE2_HEAD},
        ]
    )
    best_val   = float("inf")
    wait       = 0
    best_state = None
    for epoch in range(STAGE2_EPOCHS):
        train_epoch(model, train_loader, optimizer, device, LAMBDA_OV, stage=2, augment=True)
        val_l, val_m, _ = eval_epoch(model, val_loader, device, LAMBDA_OV)
        # Monitor menses loss (main task) for early stopping in Stage 2
        if val_m < best_val:
            best_val   = val_m
            wait       = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

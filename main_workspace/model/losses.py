"""Masked losses for main task and auxiliary task."""
import torch
import torch.nn.functional as F

from .config import POS_WEIGHT_OV


def masked_menses_loss(pred, target, mask):
    """Main task: L1 only for steps where mask=True."""
    if mask.sum() == 0:
        return pred.new_zeros(1)
    return F.l1_loss(pred[mask], target[mask])


def masked_ovulation_bce(pred, target, mask, pos_weight=None):
    """Auxiliary task: BCE only for steps where mask=True (supports soft labels).
    pos_weight corrects for class imbalance (~1:14 positive:negative ratio).
    For soft labels (0 < target < 1), pos_weight is applied proportionally.
    """
    if mask.sum() == 0:
        return pred.new_zeros(1)
    p = pred[mask]
    t = target[mask].clamp(1e-6, 1 - 1e-6)
    if pos_weight is None:
        pos_weight = POS_WEIGHT_OV
    # Scale BCE: weight = pos_weight for positive class, 1 for negative
    # For soft labels, interpolate the weight so the loss scale remains consistent
    w = t * pos_weight + (1.0 - t) * 1.0
    bce = F.binary_cross_entropy(p, t, reduction="none")
    return (bce * w).mean()


def total_loss(y_m_pred, y_m_true, mask_m, p_ov_pred, y_ov_true, mask_ov, lambda_ov):
    """L = L_menses + lambda * L_ovulation."""
    L_m  = masked_menses_loss(y_m_pred, y_m_true, mask_m)
    L_ov = masked_ovulation_bce(p_ov_pred, y_ov_true, mask_ov)
    return L_m + lambda_ov * L_ov, L_m, L_ov

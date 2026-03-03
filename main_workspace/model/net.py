"""Sequence multi-task model — cascade architecture.

Physiological reasoning chain:
  wearable signals → ovulation timing → luteal phase (~14 days) → next menses

Architecture:
  1. Bidirectional backbone produces h = [h_fwd | h_bwd].
  2. OvulationHeadFwd  (causal):       h_fwd  → p_ov_fwd  — used in the cascade to menses head.
  3. OvulationHeadFull (bidirectional): h_full → p_ov_full — used for direct ovulation supervision.
  4. MensesHead (causal):               [h_fwd, p_ov_fwd] → days_until_next_menses.

Why cascade?
  days_until_menses ≈ ovulation_day + luteal_length (~14 days).
  Feeding p_ov_fwd into the menses head explicitly encodes this physiological chain and
  forces the causal ovulation head to produce estimates useful for menses regression,
  not just BCE-optimal outputs.

Why two ovulation heads?
  OvulationHeadFull uses bidirectional context (valid because labels are retrospective)
  and provides a stronger training signal to the backbone.
  OvulationHeadFwd is causal (no future leakage) and feeds the menses cascade.
  Both share the same backbone weights.

Why h_fwd for menses?
  Using h_bwd at time t would reveal the cycle's last day_in_cycle value,
  trivially encoding days_until_next_menses = cycle_len − day_in_cycle. (Data leakage.)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import INPUT_DIM, HIDDEN_SIZE, RNN_TYPE, BIDIRECTIONAL, DROPOUT


# ── Backbone ─────────────────────────────────────────────────────────────────

class CycleRNN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                 rnn_type=RNN_TYPE, bidirectional=BIDIRECTIONAL, dropout=DROPOUT):
        super().__init__()
        self.hidden_size    = hidden_size
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1
        rnn_cls  = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_size, batch_first=True,
                           bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)

    @property
    def fwd_dim(self):
        return self.hidden_size

    @property
    def full_dim(self):
        return self.hidden_size * self.num_directions

    def forward(self, X, lengths):
        packed     = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _     = pad_packed_sequence(packed_out, batch_first=True)
        return self.drop(out)   # (B, T, full_dim)


# ── Shared MLP builder ────────────────────────────────────────────────────────

def _mlp(input_dim, mid_factor=2, dropout=DROPOUT):
    mid = max(input_dim // mid_factor, 16)
    return nn.Sequential(
        nn.Linear(input_dim, mid),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )


# ── Ovulation heads ───────────────────────────────────────────────────────────

class OvulationHeadFwd(nn.Module):
    """Causal ovulation head.  Input: h_fwd (forward-only, no future leakage).
    Output: p_ov_fwd ∈ (0,1).  Fed into the menses cascade."""

    def __init__(self, fwd_dim):
        super().__init__()
        self.net = _mlp(fwd_dim)

    def forward(self, h_fwd):
        return torch.sigmoid(self.net(h_fwd).squeeze(-1))   # (B, T)


class OvulationHeadFull(nn.Module):
    """Bidirectional ovulation head.  Input: h_full (forward + backward context).
    Output: p_ov_full ∈ (0,1).  Used only for direct ovulation supervision (L_ov).
    Not fed into the menses cascade, so no future leakage into menses prediction."""

    def __init__(self, full_dim):
        super().__init__()
        self.net = _mlp(full_dim)

    def forward(self, h_full):
        return torch.sigmoid(self.net(h_full).squeeze(-1))  # (B, T)


# ── Menses head (cascade) ─────────────────────────────────────────────────────

class MensesHead(nn.Module):
    """Causal regression: input = [h_fwd, p_ov_fwd].
    Explicitly encodes the physiological chain: ovulation timing → luteal phase → menses.
    L_menses back-propagates through p_ov_fwd, forcing OvulationHeadFwd to produce
    estimates that are useful for menses regression (not just BCE-optimal).
    Input dim = fwd_dim + 1."""

    def __init__(self, fwd_dim):
        super().__init__()
        # fwd_dim + 1 because p_ov_fwd is concatenated
        self.net = _mlp(fwd_dim + 1)

    def forward(self, h_fwd, p_ov_fwd):
        # p_ov_fwd: (B, T) → (B, T, 1) for concatenation
        p = p_ov_fwd.unsqueeze(-1)
        x = torch.cat([h_fwd, p], dim=-1)   # (B, T, fwd_dim+1)
        return self.net(x).squeeze(-1)       # (B, T)


# ── Full model ────────────────────────────────────────────────────────────────

class CycleModel(nn.Module):
    """Cascade + bidirectional supervision.

    forward() returns (y_menses_pred, p_ov_full) where:
      y_menses_pred  is computed causally via [h_fwd, p_ov_fwd]
      p_ov_full      is the bidirectional ovulation probability used for L_ov supervision
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_size=HIDDEN_SIZE,
                 rnn_type=RNN_TYPE, bidirectional=BIDIRECTIONAL):
        super().__init__()
        self.backbone           = CycleRNN(input_dim, hidden_size, rnn_type, bidirectional)
        self.ovulation_head_fwd = OvulationHeadFwd(self.backbone.fwd_dim)
        self.ovulation_head_full = OvulationHeadFull(self.backbone.full_dim)
        self.menses_head        = MensesHead(self.backbone.fwd_dim)
        self.hidden_size        = hidden_size

    def forward(self, X, lengths):
        h      = self.backbone(X, lengths)         # (B, T, full_dim)
        h_fwd  = h[:, :, :self.hidden_size]        # (B, T, fwd_dim)  — causal
        h_full = h                                  # (B, T, full_dim) — bidirectional

        # Causal ovulation estimate (fed into menses cascade)
        p_ov_fwd  = self.ovulation_head_fwd(h_fwd)   # (B, T)

        # Bidirectional ovulation estimate (used only for L_ov supervision)
        p_ov_full = self.ovulation_head_full(h_full)  # (B, T)

        # Menses prediction via cascade: [h_fwd, p_ov_fwd] → days_until_menses
        y_m = self.menses_head(h_fwd, p_ov_fwd)      # (B, T)

        # Return p_ov_full for L_ov supervision; p_ov_fwd is used internally in cascade
        return y_m, p_ov_full

"""Lightweight GRU sequence model for menstrual cycle prediction."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CycleGRU(nn.Module):
    """Single-layer GRU with small hidden dim + linear head.
    Outputs a prediction at every timestep so entire cycles can be
    processed in one forward pass."""

    def __init__(self, seq_dim=16, static_dim=6, hidden_dim=32, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim + static_dim, 1)

    def forward(self, seq, static, lengths):
        """
        seq:     (B, T, D_seq)
        static:  (B, T, D_static)
        lengths: (B,)
        Returns: (B, T) predictions at every timestep.
        """
        packed = pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)

        gru_out = self.dropout(gru_out)
        combined = torch.cat(
            [gru_out, static[:, : gru_out.size(1)]], dim=-1
        )
        return self.head(combined).squeeze(-1)


def huber_loss_masked(pred, target, mask, delta=3.0):
    """Huber loss only on valid (non-padded) positions."""
    diff = (pred - target).abs()
    loss = torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta),
    )
    return (loss * mask.float()).sum() / mask.float().sum()

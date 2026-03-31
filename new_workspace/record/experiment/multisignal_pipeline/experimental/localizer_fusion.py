"""EXPERIMENTAL: fuse deterministic localizer outputs across multiple lookback windows."""
from __future__ import annotations

import numpy as np


def multi_lookback_fused_sequences(payloads, sgk, n):
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

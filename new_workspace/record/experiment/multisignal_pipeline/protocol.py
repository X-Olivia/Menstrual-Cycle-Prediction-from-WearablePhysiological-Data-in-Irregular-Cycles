"""
Unified experiment protocol: re-exports mainline constants, then experimental knobs.

Import order is intentional: ``config.protocol_main`` defines shipped benchmark defaults;
``config.protocol_experimental`` adds sweeps, optional ensembles, and extra stabilization
hyperparameters without shadowing mainline values (no overlapping names).
"""
from __future__ import annotations

from config.protocol_main import *  # noqa: F403
from config.protocol_experimental import *  # noqa: F403

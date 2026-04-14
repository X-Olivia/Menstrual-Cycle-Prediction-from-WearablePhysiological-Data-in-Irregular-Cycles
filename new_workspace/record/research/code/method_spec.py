"""Explicit method specifications for subgroup evaluation (detect × countdown prior)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

CountdownPriorMode = Literal["population", "history_acl"]


@dataclass(frozen=True)
class MethodSpec:
    """One row in the subgroup results table."""

    name: str
    det_by_day: dict[str, list[int | None]] | None
    conf_by_day: dict[str, list[float]] | None
    use_population_only_prior: bool = False
    baseline_mode: str = "dynamic"
    countdown_prior_mode: CountdownPriorMode = "population"


def method_spec_from_tuple(t: tuple[Any, ...]) -> MethodSpec:
    """
    Backward-compatible conversion from legacy 3/4/5-tuple method rows.

    New code should construct `MethodSpec` explicitly so `countdown_prior_mode` is not
    inferred from the display `name` (except for this tuple shim).
    """
    if len(t) == 3:
        name, det, conf = t
        use_pop, bm = False, "dynamic"
    elif len(t) == 4:
        name, det, conf, use_pop = t
        bm = "dynamic"
    elif len(t) == 5:
        name, det, conf, use_pop, bm = t
    else:
        raise ValueError(f"Expected tuple of length 3–5, got {len(t)}: {t!r}")
    cd: CountdownPriorMode = "history_acl" if str(name) == "HistoryPrior-Menses" else "population"
    return MethodSpec(
        name=str(name),
        det_by_day=det,  # type: ignore[arg-type]
        conf_by_day=conf,  # type: ignore[arg-type]
        use_population_only_prior=bool(use_pop),
        baseline_mode=str(bm),
        countdown_prior_mode=cd,
    )


def coerce_method_specs(methods: Sequence[MethodSpec | tuple[Any, ...]]) -> list[MethodSpec]:
    out: list[MethodSpec] = []
    for m in methods:
        if isinstance(m, MethodSpec):
            out.append(m)
        else:
            out.append(method_spec_from_tuple(m))
    return out

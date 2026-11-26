"""Crossed listener/script bootstrap uncertainty.

The declared procedure (issue F-06/F-07) resamples listeners and scripts with
replacement, with script resampling stratified by bucket, 5,000 resamples and a
recorded seed. This module implements that procedure generically; callers pass
the statistic (mean naturalness or a pooled ratio) and the records.

Percentile intervals use linear interpolation between order statistics; the
method and seed are recorded in every returned report.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..errors import InputError

Statistic = Callable[[list[dict[str, Any]]], float]


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def mean_statistic(records: list[dict[str, Any]], *, field: str = "score") -> float:
    values = [float(record[field]) for record in records]
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)


def ratio_statistic(numerator_field: str, denominator_field: str) -> Statistic:
    def statistic(records: list[dict[str, Any]]) -> float:
        denominator = sum(float(record[denominator_field]) for record in records)
        if denominator <= 0:
            raise ValueError("zero denominator")
        return sum(float(record[numerator_field]) for record in records) / denominator

    return statistic


def crossed_bootstrap(
    records: list[dict[str, Any]],
    statistic: Statistic,
    *,
    resamples: int = 5000,
    seed: int = 3407,
    listener_field: str = "listener_id",
    script_field: str = "utterance_id",
    stratum_field: str | None = "bucket",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Resample listeners and (stratified) scripts; return an interval estimate."""
    if not records:
        raise InputError("no records supplied to the bootstrap")
    if resamples < 100:
        raise InputError("resamples must be at least 100 for a stable percentile interval")
    if not 0 < confidence < 1:
        raise InputError("confidence must be between 0 and 1")

    listeners = sorted({str(record[listener_field]) for record in records})
    if len(listeners) < 2:
        raise InputError("crossed bootstrap needs at least two distinct listeners")

    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    strata: dict[str, list[str]] = defaultdict(list)
    for record in records:
        listener = str(record[listener_field])
        script = str(record[script_field])
        indexed[(listener, script)].append(record)
        stratum = str(record[stratum_field]) if stratum_field else "__all__"
        if script not in strata[stratum]:
            strata[stratum].append(script)
    for scripts in strata.values():
        scripts.sort()

    rng = random.Random(seed)
    estimates: list[float] = []
    empty_draws = 0
    for _ in range(resamples):
        sampled_listeners = rng.choices(listeners, k=len(listeners))
        pooled: list[dict[str, Any]] = []
        for stratum in sorted(strata):
            scripts = strata[stratum]
            for script in rng.choices(scripts, k=len(scripts)):
                for listener in sampled_listeners:
                    pooled.extend(indexed.get((listener, script), ()))
        if not pooled:
            empty_draws += 1
            continue
        try:
            estimates.append(float(statistic(pooled)))
        except (ValueError, ZeroDivisionError):
            empty_draws += 1

    if not estimates:
        raise InputError("every bootstrap resample was empty; check listener/script coverage")
    estimates.sort()
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(statistic(records)),
        "lower": _percentile(estimates, alpha),
        "upper": _percentile(estimates, 1 - alpha),
        "confidence": confidence,
        "resamples": resamples,
        "valid_resamples": len(estimates),
        "empty_resamples": empty_draws,
        "seed": seed,
        "listeners": listeners,
        "script_field": script_field,
        "listener_field": listener_field,
        "stratum_field": stratum_field,
        "strata": {stratum: len(scripts) for stratum, scripts in sorted(strata.items())},
        "method": (
            f"crossed bootstrap: listeners resampled with replacement and "
            f"{'scripts resampled within strata' if stratum_field else 'scripts resampled'}; "
            f"percentile {confidence:.0%} interval"
        ),
    }


def mean_with_interval(
    records: list[dict[str, Any]],
    *,
    field: str = "score",
    resamples: int = 5000,
    seed: int = 3407,
    listener_field: str = "listener_id",
    script_field: str = "utterance_id",
    stratum_field: str | None = "bucket",
) -> dict[str, Any]:
    return crossed_bootstrap(
        records,
        lambda pooled: mean_statistic(pooled, field=field),
        resamples=resamples,
        seed=seed,
        listener_field=listener_field,
        script_field=script_field,
        stratum_field=stratum_field,
    )


def ratio_with_interval(
    records: list[dict[str, Any]],
    *,
    numerator_field: str,
    denominator_field: str,
    resamples: int = 5000,
    seed: int = 3407,
    listener_field: str = "listener_id",
    script_field: str = "utterance_id",
    stratum_field: str | None = "bucket",
) -> dict[str, Any]:
    return crossed_bootstrap(
        records,
        ratio_statistic(numerator_field, denominator_field),
        resamples=resamples,
        seed=seed,
        listener_field=listener_field,
        script_field=script_field,
        stratum_field=stratum_field,
    )


__all__ = [
    "crossed_bootstrap",
    "mean_statistic",
    "mean_with_interval",
    "ratio_statistic",
    "ratio_with_interval",
]

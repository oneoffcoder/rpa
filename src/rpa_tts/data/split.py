"""Leakage-safe train/dev/test assignment.

The split unit is a group (passage, recording session or related prompt), never
an individual clip. Groups are ordered by a seeded hash, so the assignment is
reproducible and independent of file ordering, and quotas are applied per bucket
so the locked test can report bucket-level results.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ..errors import InputError
from ..hashing import sha256_json

SPLITS: tuple[str, ...] = ("train", "dev", "test")


@dataclass(frozen=True)
class GroupPlan:
    bucket: str
    groups: list[str]


def _group_order(seed: int, groups: list[str]) -> list[str]:
    return sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(),
    )


def _quota(count: int, test_fraction: float, dev_fraction: float) -> tuple[int, int]:
    """(dev, test) group counts for one bucket, leaving at least one train group."""
    if count <= 1:
        return 0, 0
    if count == 2:
        return (0, 1) if test_fraction > 0 else (1, 0)
    dev = max(1, math.floor(count * dev_fraction)) if dev_fraction > 0 else 0
    test = max(1, math.floor(count * test_fraction)) if test_fraction > 0 else 0
    while dev + test > count - 1:
        if dev >= test and dev > 0:
            dev -= 1
        elif test > 0:
            test -= 1
        else:  # pragma: no cover - guarded by the loop condition
            break
    return dev, test


def assign_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int = 3407,
    dev_fraction: float = 0.1,
    test_fraction: float = 0.2,
    group_field: str = "group_id",
    bucket_field: str = "bucket",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return rows with a new ``split`` plus a summary of the assignment."""
    if not rows:
        raise InputError("cannot split an empty corpus")
    if not 0 <= dev_fraction < 1 or not 0 <= test_fraction < 1:
        raise InputError("dev/test fractions must be in [0, 1)")
    if dev_fraction + test_fraction >= 1:
        raise InputError("dev + test fractions must leave room for training rows")

    seen: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_buckets: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        record_id = str(row.get("id", ""))
        if not record_id:
            raise InputError(f"row {index} has no id")
        if record_id in seen:
            raise InputError(f"duplicate row id {record_id!r}")
        seen.add(record_id)
        group = str(row.get(group_field, "") or "")
        bucket = str(row.get(bucket_field, "") or "")
        if not group:
            raise InputError(
                f"row {record_id!r} has no {group_field}; unrelated clips must not leak"
            )
        if not bucket:
            raise InputError(f"row {record_id!r} has no {bucket_field}")
        groups[group].append(row)
        group_buckets[group].add(bucket)

    mixed = {group: sorted(buckets) for group, buckets in group_buckets.items() if len(buckets) > 1}
    if mixed:
        detail = ", ".join(f"{group}:{buckets}" for group, buckets in sorted(mixed.items())[:5])
        raise InputError(f"a group cannot span buckets ({detail})")

    bucket_groups: dict[str, list[str]] = defaultdict(list)
    for group, bucket in ((g, next(iter(b))) for g, b in group_buckets.items()):
        bucket_groups[bucket].append(group)

    assignments: dict[str, str] = {}
    warnings: list[str] = []
    bucket_summary: dict[str, dict[str, int]] = {}
    for bucket in sorted(bucket_groups):
        ordered = _group_order(seed, bucket_groups[bucket])
        dev, test = _quota(len(ordered), test_fraction, dev_fraction)
        for position, group in enumerate(ordered):
            if position < test:
                assignments[group] = "test"
            elif position < test + dev:
                assignments[group] = "dev"
            else:
                assignments[group] = "train"
        counts = Counter(assignments[group] for group in ordered)
        bucket_summary[bucket] = {name: counts.get(name, 0) for name in SPLITS}
        for name in ("dev", "test"):
            if counts.get(name, 0) == 0:
                warnings.append(f"bucket {bucket!r} has no {name} groups ({len(ordered)} groups)")

    overall = Counter(assignments.values())
    for name in SPLITS:
        if overall.get(name, 0) == 0:
            warnings.append(f"no {name} groups exist; add material before using this split")

    assigned = [{**row, "split": assignments[str(row[group_field])]} for row in rows]
    assigned.sort(key=lambda row: str(row["id"]))
    summary = {
        "seed": seed,
        "dev_fraction": dev_fraction,
        "test_fraction": test_fraction,
        "group_field": group_field,
        "bucket_field": bucket_field,
        "groups": len(groups),
        "rows": len(rows),
        "group_counts": {name: overall.get(name, 0) for name in SPLITS},
        "row_counts": dict(sorted(Counter(row["split"] for row in assigned).items())),
        "buckets": bucket_summary,
        "warnings": warnings,
        "assignment_sha256": sha256_json(sorted(assignments.items())),
        "method": "seeded sha256 ordering per bucket; whole groups assigned to one split",
    }
    return assigned, summary

"""Append-only learning-curve records (F-06).

One row per judged learning run. Appending validates the fixed columns and the
decision vocabulary so the curve cannot silently drift or overwrite a previous
run. The evidence id links the row to the hashed evidence index.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from ..errors import InputError, ValidationError
from ..schemas import LEARNING_CURVE_COLUMNS

RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{3,127}$")
EVIDENCE_ID = re.compile(r"^ev_[0-9a-f]{16}$")
DECISIONS = ("baseline", "promote", "hold", "reject", "inconclusive")


def _number(value: Any, field: str, *, low: float, high: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number, got {value!r}") from None
    if number != number:
        raise ValidationError(f"{field} must not be NaN")
    if number < low or (high is not None and number > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValidationError(f"{field} must be in {bound}, got {number}")
    return number


def validate_row(row: dict[str, Any], *, existing_ids: set[str] | None = None) -> dict[str, str]:
    missing = [column for column in LEARNING_CURVE_COLUMNS if column not in row]
    if missing:
        raise ValidationError(f"learning-curve row is missing columns: {missing}")
    run_id = str(row["run_id"])
    if not RUN_ID.fullmatch(run_id):
        raise ValidationError(f"invalid run_id: {run_id!r}")
    if existing_ids and run_id in existing_ids:
        raise ValidationError(f"run_id already has a learning-curve row: {run_id}")
    stage = str(row["stage"])
    if not re.fullmatch(r"F-0[5-8]", stage):
        raise ValidationError(f"stage must be F-05..F-08, got {stage!r}")
    if not str(row["dataset_version"]).strip():
        raise ValidationError("dataset_version must not be empty")
    _number(row["accepted_training_hours"], "accepted_training_hours", low=0)
    updates = _number(row["optimizer_updates"], "optimizer_updates", low=0)
    if updates != int(updates):
        raise ValidationError("optimizer_updates must be an integer")
    _number(row["dev_syllable_error"], "dev_syllable_error", low=0, high=1)
    _number(row["dev_naturalness_mean"], "dev_naturalness_mean", low=1, high=5)
    decision = str(row["decision"])
    if decision not in DECISIONS:
        raise ValidationError(f"decision must be one of {DECISIONS}, got {decision!r}")
    evidence_id = str(row["evidence_id"])
    if not EVIDENCE_ID.fullmatch(evidence_id):
        raise ValidationError(f"invalid evidence_id: {evidence_id!r}")
    return {column: str(row[column]) for column in LEARNING_CURVE_COLUMNS}


def load_curve(path: str | Path) -> list[dict[str, str]]:
    curve = Path(path)
    if not curve.is_file():
        return []
    with curve.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LEARNING_CURVE_COLUMNS:
            raise ValidationError(
                f"learning-curve header must be {LEARNING_CURVE_COLUMNS}, got {reader.fieldnames}"
            )
        rows = [
            dict(row) for row in reader if any(value and value.strip() for value in row.values())
        ]
    ids: set[str] = set()
    for index, row in enumerate(rows, 2):
        try:
            validate_row(row, existing_ids=ids)
        except ValidationError as exc:
            raise ValidationError(f"learning-curve line {index}: {exc}") from None
        ids.add(str(row["run_id"]))
    return rows


def append_row(path: str | Path, row: dict[str, Any]) -> dict[str, Any]:
    curve = Path(path)
    existing = load_curve(curve)
    validated = validate_row(row, existing_ids={item["run_id"] for item in existing})
    curve.parent.mkdir(parents=True, exist_ok=True)
    write_header = not curve.exists() or curve.stat().st_size == 0
    with curve.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEARNING_CURVE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(validated)
    return {"row": validated, "rows": len(existing) + 1, "curve": curve.as_posix()}


def row_from_run_manifest(
    manifest: dict[str, Any],
    *,
    dataset_version: str,
    dev_syllable_error: float,
    dev_naturalness_mean: float,
    decision: str,
    evidence_id: str,
) -> dict[str, Any]:
    if manifest.get("status") not in {"completed", "failed", "aborted"}:
        raise InputError(
            "a learning-curve row requires a completed/failed/aborted run manifest, "
            f"got status {manifest.get('status')!r}"
        )
    return {
        "run_id": manifest["run_id"],
        "stage": manifest["stage"],
        "dataset_version": dataset_version,
        "accepted_training_hours": manifest["dataset"]["unique_audio_hours"],
        "optimizer_updates": manifest["training"]["optimizer_updates"],
        "dev_syllable_error": dev_syllable_error,
        "dev_naturalness_mean": dev_naturalness_mean,
        "decision": decision,
        "evidence_id": evidence_id,
    }


__all__ = [
    "DECISIONS",
    "append_row",
    "load_curve",
    "row_from_run_manifest",
    "validate_row",
]

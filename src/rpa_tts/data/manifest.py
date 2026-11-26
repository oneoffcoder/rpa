"""Accepted-utterance manifest loading, validation and summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import InputError, ValidationError
from ..hashing import is_sha256_hex, sha256_file
from ..schemas import SPLITS
from ..schemas import validate as validate_schema
from ..text.normalize import normalize_rpa

PENDING_RIGHTS = {"UNRESOLVED", "pending", "PENDING", "none", "NONE", ""}
BUCKET_ORDER = [
    "ordinary_sentences",
    "targeted_contrasts",
    "questions_numbers_names",
    "continuous_passages",
    "calibration",
]


@dataclass(frozen=True)
class Problem:
    level: str
    row_index: int
    record_id: str
    field: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "row_index": self.row_index,
            "record_id": self.record_id,
            "field": self.field,
            "message": self.message,
        }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest with line-numbered errors for malformed input."""
    manifest_path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        stream = manifest_path.open(encoding="utf-8")
    except FileNotFoundError:
        raise InputError(f"manifest does not exist: {manifest_path}") from None
    with stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{manifest_path}:{number}: invalid JSON: {exc.msg}") from None
            if not isinstance(row, dict):
                raise InputError(f"{manifest_path}:{number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise InputError(f"manifest contains no rows: {manifest_path}")
    return rows


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> str:
    """Write deterministic JSONL and return its SHA-256."""
    ordered = sorted(rows, key=lambda row: str(row.get("id", "")))
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for row in ordered
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return sha256_file(destination)


def _check_normalized(row: dict[str, Any], index: int) -> list[Problem]:
    """The normalized transcript must be provably derived from the raw text."""
    problems: list[Problem] = []
    record_id = str(row.get("id", ""))
    raw = row.get("transcript_raw")
    normalized = row.get("transcript_normalized")
    if not isinstance(raw, str) or not isinstance(normalized, str):
        return problems
    try:
        derived = normalize_rpa(raw)
    except (TypeError, ValueError) as exc:  # defensive: normalize only raises for non-str
        problems.append(Problem("error", index, record_id, "transcript_raw", str(exc)))
        return problems
    expansions = row.get("expansion_ids") or []
    if not expansions and derived.normalized != normalized:
        problems.append(
            Problem(
                "error",
                index,
                record_id,
                "transcript_normalized",
                "normalized transcript is not the normalization of transcript_raw",
            )
        )
    checked = normalize_rpa(normalized)
    for issue in checked.issues:
        problems.append(Problem("error", index, record_id, "transcript_normalized", issue.message))
    return problems


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    purpose: str = "training",
    require_native_approval: bool | None = None,
) -> list[Problem]:
    """Return structural and cross-record problems; never mutate the rows.

    ``purpose='training'`` (default) additionally requires resolved rights and
    native approval, because F-05 admitted material feeds training. Use
    ``purpose='audit'`` for candidate-only checks.
    """
    if purpose not in {"audit", "training", "evaluation"}:
        raise InputError(f"unknown purpose: {purpose!r}")
    if require_native_approval is None:
        require_native_approval = purpose in {"training", "evaluation"}
    problems: list[Problem] = []
    seen_ids: dict[str, int] = {}
    group_splits: dict[str, set[str]] = {}

    for index, row in enumerate(rows):
        record_id = str(row.get("id", "")) if isinstance(row, dict) else ""
        try:
            validate_schema(row, "utterance-manifest.schema.json")
        except ValidationError as exc:
            problems.append(Problem("error", index, record_id, "record", str(exc)))
            continue

        if record_id in seen_ids:
            problems.append(
                Problem(
                    "error",
                    index,
                    record_id,
                    "id",
                    f"duplicate utterance id (first at row {seen_ids[record_id]})",
                )
            )
        else:
            seen_ids[record_id] = index

        group_id = str(row["group_id"])
        group_splits.setdefault(group_id, set()).add(str(row["split"]))

        if not is_sha256_hex(row["audio_sha256"]):
            problems.append(
                Problem("error", index, record_id, "audio_sha256", "not a SHA-256 hex digest")
            )

        if purpose in {"training", "evaluation"}:
            if str(row["rights_id"]).strip() in PENDING_RIGHTS:
                problems.append(
                    Problem(
                        "error",
                        index,
                        record_id,
                        "rights_id",
                        "rights are unresolved; record an approved rights id before admission",
                    )
                )
            if str(row["consent_id"]).strip() in PENDING_RIGHTS:
                problems.append(
                    Problem(
                        "error", index, record_id, "consent_id", "speaker consent is unresolved"
                    )
                )
            if require_native_approval and not row["reviewer"].get("native_approved"):
                problems.append(
                    Problem(
                        "error",
                        index,
                        record_id,
                        "reviewer.native_approved",
                        (
                            "row is not native-approved; admitted material requires one "
                            "verified review"
                        ),
                    )
                )

        problems.extend(_check_normalized(row, index))

        if row["bucket"] == "continuous_passages" and not row["source"].get("parent_passage_id"):
            problems.append(
                Problem(
                    "warning",
                    index,
                    record_id,
                    "source.parent_passage_id",
                    "continuous passage row has no parent passage id",
                )
            )
        if not row["source"].get("dialect"):
            problems.append(
                Problem("warning", index, record_id, "source.dialect", "dialect is not recorded")
            )
        start, end = row["source"].get("start_seconds"), row["source"].get("end_seconds")
        if start is not None and end is not None and end <= start:
            problems.append(
                Problem(
                    "error", index, record_id, "source.end_seconds", "passage offsets must increase"
                )
            )

    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            problems.append(
                Problem(
                    "error",
                    -1,
                    "",
                    "group_id",
                    (
                        f"group {group_id!r} spans splits {sorted(splits)}; "
                        "related audio must stay in one split"
                    ),
                )
            )
    return problems


def require_valid_rows(
    rows: list[dict[str, Any]],
    *,
    purpose: str = "training",
    require_native_approval: bool | None = None,
) -> None:
    problems = validate_rows(rows, purpose=purpose, require_native_approval=require_native_approval)
    errors = [problem for problem in problems if problem.level == "error"]
    if errors:
        details = "; ".join(
            f"row {problem.row_index}"
            + (f" ({problem.record_id})" if problem.record_id else "")
            + f" {problem.field}: {problem.message}"
            for problem in errors[:10]
        )
        if len(errors) > 10:
            details += f"; and {len(errors) - 10} more"
        raise ValidationError(f"manifest failed validation: {details}")


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Unique accepted hours, rows and groups by split and bucket."""
    by_split = {name: {"rows": 0, "seconds": 0.0} for name in SPLITS}
    by_bucket: dict[str, dict[str, float]] = {
        bucket: {"rows": 0, "seconds": 0.0} for bucket in BUCKET_ORDER
    }
    speakers: set[str] = set()
    groups: set[str] = set()
    for row in rows:
        split = row["split"]
        by_split[split]["rows"] += 1
        by_split[split]["seconds"] += float(row["duration_seconds"])
        bucket = row["bucket"]
        by_bucket.setdefault(bucket, {"rows": 0, "seconds": 0.0})
        by_bucket[bucket]["rows"] += 1
        by_bucket[bucket]["seconds"] += float(row["duration_seconds"])
        speakers.add(str(row["source"]["speaker_id"]))
        groups.add(str(row["group_id"]))
    return {
        "rows": len(rows),
        "groups": len(groups),
        "speakers": sorted(speakers),
        "by_split": {
            split: {
                "rows": counts["rows"],
                "hours": round(counts["seconds"] / 3600, 6),
            }
            for split, counts in by_split.items()
        },
        "by_bucket": {
            bucket: {
                "rows": counts["rows"],
                "hours": round(counts["seconds"] / 3600, 6),
            }
            for bucket, counts in by_bucket.items()
        },
        "accepted_training_hours": round(by_split["train"]["seconds"] / 3600, 6),
        "accepted_test_hours": round(by_split["test"]["seconds"] / 3600, 6),
        "accepted_dev_hours": round(by_split["dev"]["seconds"] / 3600, 6),
    }

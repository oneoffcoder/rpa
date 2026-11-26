"""Append-only evidence index shared by every recovery gate.

Every entry states one falsifiable claim, its gate, its status and the files
(hash-checked) that support it. Corrections are appended, never deleted; the
entry with the latest ``created_at`` for a gate is its effective status.
"""

from __future__ import annotations

import datetime as _dt
import secrets
from pathlib import Path
from typing import Any

from .errors import InputError, ValidationError
from .hashing import sha256_file
from .paths import evidence_index_path, find_repo_root
from .schemas import validate as validate_schema

STATUSES = {
    "not_started",
    "in_progress",
    "passed",
    "failed",
    "inconclusive",
    "blocked",
    "not_required",
}
INDEX_SCHEMA_VERSION = 1


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def load_index(path: str | Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    index_path = Path(path) if path else evidence_index_path(repo_root)
    if not index_path.is_file():
        return {"schema_version": INDEX_SCHEMA_VERSION, "entries": []}
    import json

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"evidence index is not valid JSON: {index_path}: {exc}") from None
    validate_index(data)
    return data


def validate_index(index: dict[str, Any]) -> None:
    try:
        validate_schema(index, "evidence-index.schema.json")
    except ValidationError:
        raise
    ids = [entry["id"] for entry in index["entries"]]
    if len(set(ids)) != len(ids):
        raise ValidationError("evidence index contains duplicate ids")


def _relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def make_entry(
    *,
    gate: str,
    claim: str,
    status: str,
    paths: list[Path] | tuple[Path, ...] = (),
    run_id: str | None = None,
    notes: str = "",
    repo_root: Path | None = None,
    evidence_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise InputError(f"unknown evidence status: {status!r}")
    root = repo_root or find_repo_root()
    hashed: list[dict[str, str]] = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file():
            raise InputError(f"evidence file does not exist: {candidate}")
        hashed.append({"path": _relative(candidate, root), "sha256": sha256_file(candidate)})
    entry = {
        "id": evidence_id or f"ev_{secrets.token_hex(8)}",
        "gate": gate,
        "claim": claim,
        "status": status,
        "created_at": created_at or utc_now(),
        "run_id": run_id,
        "paths": hashed,
        "notes": notes,
    }
    validate_schema(entry, "evidence-entry.schema.json")
    return entry


def append_entry(
    entry: dict[str, Any],
    path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    index_path = Path(path) if path else evidence_index_path(repo_root)
    index = load_index(index_path, repo_root=repo_root)
    if any(existing["id"] == entry["id"] for existing in index["entries"]):
        raise InputError(f"evidence id already exists: {entry['id']}")
    index["entries"].append(entry)
    validate_index(index)
    import json

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def latest_entry(index: dict[str, Any], gate: str) -> dict[str, Any] | None:
    entries = [entry for entry in index["entries"] if entry["gate"] == gate]
    if not entries:
        return None
    entries.sort(key=lambda entry: entry["created_at"])
    return entries[-1]


def gate_status(index: dict[str, Any], gate: str) -> dict[str, Any]:
    entry = latest_entry(index, gate)
    if entry is None:
        return {"gate": gate, "status": "not_started", "evidence_id": None, "claim": None}
    return {
        "gate": gate,
        "status": entry["status"],
        "evidence_id": entry["id"],
        "claim": entry["claim"],
        "created_at": entry["created_at"],
    }


__all__ = [
    "append_entry",
    "gate_status",
    "latest_entry",
    "load_index",
    "make_entry",
    "utc_now",
    "validate_index",
]

"""Corpus acquisition audit: decode, hash and flag every candidate recording.

This is a technical audit, not native review. It publishes defect and rejection
counts, keeps every row (including failures) and never edits source audio. An
audit manifest is input to native QA and to :mod:`rpa_tts.data.split`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InputError
from ..hashing import sha256_file
from ..text.normalize import normalize_rpa
from .audio import AUDIO_SUFFIXES, inspect_audio, require_audio_tools, review_flags

AUDIT_SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
LABEL_FIELDS = ("group_id", "bucket", "rights_id", "consent_id", "source", "reviewer")


def load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional sidecar labels keyed by recording id or file stem."""
    if path is None:
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{number}: invalid JSON: {exc.msg}") from None
            if not isinstance(row, dict) or "id" not in row:
                raise InputError(f"{path}:{number}: label row must be an object with 'id'")
            key = str(row["id"]).strip().lower()
            if key in labels:
                raise InputError(f"{path}:{number}: duplicate label id {key!r}")
            if not isinstance(row.get("transcript_raw", ""), str):
                raise InputError(f"{path}:{number}: transcript_raw must be a string")
            labels[key] = row
    return labels


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise InputError(f"path must be inside the repository root: {path}") from None


def _row_id(path: Path) -> str:
    candidate = re.sub(r"[^a-z0-9_-]+", "_", path.stem.strip().lower()).strip("_")
    return candidate


def audit_recording(
    path: Path,
    repo_root: Path,
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    recording_id = _row_id(path)
    label = labels.get(recording_id) or labels.get(path.stem.strip().lower()) or {}
    row: dict[str, Any] = {
        "id": recording_id,
        "audio_path": _relative(path, repo_root),
        "status": "ok",
        "errors": [],
        "quality_flags": [],
        "label_source": "sidecar" if label else "missing",
    }
    if not SAFE_ID.fullmatch(recording_id):
        row["status"] = "error"
        row["errors"].append(f"unsafe recording id derived from filename: {recording_id!r}")
    for field in LABEL_FIELDS:
        if field in label:
            row[field] = label[field]
    raw_transcript = label.get("transcript_raw")
    if raw_transcript:
        row["transcript_raw"] = raw_transcript
        normalized = normalize_rpa(raw_transcript)
        row["transcript_normalized"] = normalized.normalized
        row["normalization_version"] = normalized.version
        if normalized.issues:
            row["transcript_issues"] = [
                {"code": issue.code, "token": issue.token, "position": issue.position}
                for issue in normalized.issues
            ]
            row["status"] = "transcript_invalid"
    else:
        row["status"] = "needs_transcript"
    try:
        measurement = inspect_audio(path)
        row.update(measurement)
        flags = review_flags(measurement)
        row["quality_flags"] = flags
        if measurement["peak"] == 0:
            row["status"] = "error"
            row["errors"].append("decoded audio contains only zero samples")
    except (OSError, InputError) as exc:
        row["status"] = "error"
        row["errors"].append(str(exc))
    if row["errors"]:
        row["status"] = "error"
    return row


def _duplicate_groups(rows: list[dict[str, Any]], field: str) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value:
            groups[str(value)].append(row["audio_path"])
    return sorted(sorted(paths) for paths in groups.values() if len(paths) > 1)


def render_report(report: dict[str, Any]) -> str:
    duration = report["duration"]
    lines = [
        "# Candidate recording audit",
        "",
        "Technical only; native listening and rights review remain outstanding.",
        "",
        f"Manifest SHA-256: `{report['manifest_sha256']}`",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Audio files | {report['files']} |",
        f"| Technically valid | {report['valid_files']} |",
        f"| Awaiting transcript | {report['counts_by_status'].get('needs_transcript', 0)} |",
        f"| With errors | {report['counts_by_status'].get('error', 0)} |",
        f"| Decoded seconds | {duration['total_seconds']:.3f} |",
        f"| Decoded hours | {duration['total_seconds'] / 3600:.4f} |",
        f"| Byte-identical duplicate groups | {len(report['duplicate_groups']['source'])} |",
        f"| PCM-identical duplicate groups | {len(report['duplicate_groups']['pcm'])} |",
        f"| Files with review flags | {report['flagged_files']} |",
        "",
        "## Interpretation",
        "",
        "Duration is decoded sample duration including recorded silence, not accepted speech. "
        "Duplicate indices link related or repeated recordings for leakage-safe splitting. "
        "No trimming, denoising, gain change or source rewrite occurs. Advertised source "
        "duration is not usable-speech evidence.",
        "",
        "### Review flags",
        "",
    ]
    for key, value in sorted(report["quality_flag_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    if not report["quality_flag_counts"]:
        lines.append("No heuristic review flags.")
    lines.extend(["", "Full rows are in `manifest.jsonl`.", ""])
    return "\n".join(lines)


def audit_corpus(
    audio_dir: Path,
    repo_root: Path,
    output_dir: Path,
    *,
    labels_path: Path | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    require_audio_tools()
    if workers < 1:
        raise InputError("workers must be at least 1")
    repo_root = Path(repo_root).resolve()
    audio_dir = Path(audio_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not audio_dir.is_dir():
        raise InputError(f"audio directory does not exist: {audio_dir}")
    _relative(audio_dir, repo_root)
    files = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise InputError(f"no supported audio files found under {audio_dir}")
    labels = load_labels(labels_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda path: audit_recording(path, repo_root, labels), files))
    ids = Counter(row["id"] for row in rows)
    duplicate_ids = sorted(recording_id for recording_id, count in ids.items() if count > 1)
    for row in rows:
        if row["id"] in duplicate_ids:
            row["status"] = "error"
            row["errors"].append(f"duplicate recording id: {row['id']}")
    statuses = Counter(row["status"] for row in rows)
    durations = [row["duration_seconds"] for row in rows if "duration_seconds" in row]
    duration_summary: dict[str, float] = {"total_seconds": float(math.fsum(durations))}
    if durations:
        for key, percentile in (
            ("min_seconds", 0),
            ("median_seconds", 50),
            ("p95_seconds", 95),
            ("max_seconds", 100),
        ):
            duration_summary[key] = float(np.percentile(durations, percentile))
    error_rows = [{"id": row["id"], "errors": row["errors"]} for row in rows if row["errors"]]
    manifest_text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audio_directory": _relative(audio_dir, repo_root),
        "labels_path": _relative(labels_path, repo_root) if labels_path else None,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "ok": not any(
            status in {"error", "needs_transcript", "transcript_invalid"} for status in statuses
        ),
        "files": len(rows),
        "valid_files": sum(row["status"] == "ok" for row in rows),
        "counts_by_status": dict(sorted(statuses.items())),
        "counts_by_bucket": dict(
            sorted(Counter(row.get("bucket", "unlabeled") for row in rows).items())
        ),
        "duration": duration_summary,
        "sample_rates": dict(sorted(Counter(str(row.get("sample_rate")) for row in rows).items())),
        "channels": dict(sorted(Counter(str(row.get("channels")) for row in rows).items())),
        "codecs": dict(sorted(Counter(str(row.get("codec")) for row in rows).items())),
        "duplicate_ids": duplicate_ids,
        "duplicate_groups": {
            "source": _duplicate_groups(rows, "source_sha256"),
            "pcm": _duplicate_groups(rows, "pcm_sha256"),
        },
        "quality_flag_counts": dict(
            sorted(Counter(flag for row in rows for flag in row["quality_flags"]).items())
        ),
        "flagged_files": sum(bool(row["quality_flags"]) for row in rows),
        "errors": error_rows,
        "interpretation": (
            "Technical audit only; no native listening, rights clearance or accepted-hours claim."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def hash_paths(paths: list[Path]) -> dict[str, str]:
    """Convenience for evidence records; files must exist."""
    result: dict[str, str] = {}
    for path in paths:
        if not Path(path).is_file():
            raise InputError(f"cannot hash missing file: {path}")
        result[str(path)] = sha256_file(Path(path))
    return result

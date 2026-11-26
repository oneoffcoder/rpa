"""Byte-level verification of manifest audio against source files (F-05).

`corpus freeze` trusts the hashes recorded in a manifest. This module re-checks
the actual files: existence, SHA-256, decoded duration, sample rate, channel
count and decodability. It is intentionally separate from acceptance: a file can
be technically intact and still be native-rejected.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..hashing import sha256_file
from .audio import inspect_audio, require_audio_tools


def _verify_one(
    row: dict[str, Any],
    audio_root: Path,
    duration_tolerance_seconds: float,
) -> dict[str, Any]:
    record_id = str(row.get("id", ""))
    audio_path = audio_root / str(row.get("audio_path", ""))
    result: dict[str, Any] = {
        "id": record_id,
        "audio_path": str(row.get("audio_path", "")),
        "problems": [],
        "measured": None,
    }
    if not audio_path.is_file():
        result["problems"].append("missing_file")
        return result
    expected_hash = str(row.get("audio_sha256", ""))
    actual_hash = sha256_file(audio_path)
    if actual_hash != expected_hash:
        result["problems"].append(
            f"sha256_mismatch(recorded={expected_hash[:12]}..., actual={actual_hash[:12]}...)"
        )
    try:
        measurement = inspect_audio(audio_path)
    except (OSError, InputError) as exc:
        result["problems"].append(f"unreadable: {exc}")
        return result
    result["measured"] = {
        "duration_seconds": measurement["duration_seconds"],
        "sample_rate": measurement["sample_rate"],
        "channels": measurement["channels"],
        "codec": measurement["codec"],
        "source_sha256": measurement["source_sha256"],
    }
    recorded_duration = float(row.get("duration_seconds", 0.0))
    if abs(measurement["duration_seconds"] - recorded_duration) > duration_tolerance_seconds:
        result["problems"].append(
            "duration_mismatch"
            f"(recorded={recorded_duration:.3f}s, measured={measurement['duration_seconds']:.3f}s)"
        )
    if measurement["sample_rate"] != int(row.get("sample_rate", -1)):
        result["problems"].append(
            "sample_rate_mismatch"
            f"(recorded={row.get('sample_rate')}, measured={measurement['sample_rate']})"
        )
    if measurement["channels"] != int(row.get("channels", -1)):
        result["problems"].append(
            f"channels_mismatch(recorded={row.get('channels')}, measured={measurement['channels']})"
        )
    return result


def verify_audio(
    rows: list[dict[str, Any]],
    audio_root: str | Path,
    *,
    workers: int = 4,
    duration_tolerance_seconds: float = 0.05,
) -> dict[str, Any]:
    """Verify every manifest row against its file; return a deterministic report."""
    require_audio_tools()
    if not rows:
        raise InputError("manifest contains no rows")
    if workers < 1:
        raise InputError("workers must be at least 1")
    if duration_tolerance_seconds < 0:
        raise InputError("duration_tolerance_seconds must be non-negative")
    root = Path(audio_root).resolve()
    if not root.is_dir():
        raise InputError(f"audio root does not exist: {root}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(lambda row: _verify_one(row, root, duration_tolerance_seconds), rows)
        )
    failures = [result for result in results if result["problems"]]
    problem_counts: dict[str, int] = {}
    for failure in failures:
        for problem in failure["problems"]:
            key = problem.split("(")[0].split(":")[0]
            problem_counts[key] = problem_counts.get(key, 0) + 1
    measured = [result["measured"] for result in results if result["measured"]]
    return {
        "schema_version": 1,
        "audio_root": str(root),
        "rows": len(rows),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "ok": not failures,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "problem_counts": dict(sorted(problem_counts.items())),
        "failures": sorted(failures, key=lambda result: result["id"]),
        "measured_duration_seconds": round(sum(item["duration_seconds"] for item in measured), 6),
        "measured_hours": round(sum(item["duration_seconds"] for item in measured) / 3600, 6),
        "sample_rates": sorted({str(item["sample_rate"]) for item in measured}),
        "codecs": dict(sorted(Counter(str(item["codec"]) for item in measured).items())),
        "note": (
            "Technical integrity only; passing does not imply native acceptance, rights clearance "
            "or acoustic quality."
        ),
    }


__all__ = ["verify_audio"]

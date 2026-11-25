"""Inspect original audio without trimming, denoising, or changing source files."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from .orthography import CONSONANTS, ONSETS, TONES, VOWELS, expected_syllables, parse_syllable


AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a"})
SCHEMA_VERSION = 1


def require_audio_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Missing " + ", ".join(missing) + "; on macOS run: brew install ffmpeg")


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(args, capture_output=True, check=False, timeout=120)
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{Path(args[0]).name} failed: {message[:1500]}")
    return completed


def probe_audio(path: Path) -> dict[str, Any]:
    """Read the first audio stream, preserving source metadata."""
    info = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json", str(Path(path).resolve()),
    ]).stdout)
    streams = info.get("streams", [])
    if not streams:
        raise ValueError("No audio stream found")
    stream = streams[0]
    sample_rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("Invalid sample rate or channel count")
    duration = info.get("format", {}).get("duration")
    return {
        "codec": stream.get("codec_name", "unknown"),
        "sample_rate": sample_rate,
        "channels": channels,
        "source_duration_seconds": float(duration) if duration is not None else None,
    }


def decode_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Decode to mono float32, explicitly resampling when a rate is supplied.

    Passing a rate avoids a second metadata probe. FFmpeg applies its ordinary
    decoder delay/padding handling; we do not trim quiet audio. The first audio
    stream is selected explicitly.
    """
    if sample_rate is None:
        sample_rate = probe_audio(path)["sample_rate"]
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    data = _run([
        "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(Path(path).resolve()),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]).stdout
    waveform = np.frombuffer(data, dtype="<f4").copy()
    if not waveform.size:
        raise ValueError("Decoded audio is empty")
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded audio contains non-finite samples")
    return waveform, sample_rate


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        raise ValueError("The audio directory must be inside the repository") from None


def inspect_recording(path: Path, repo_root: Path) -> dict[str, Any]:
    """Record both integrity errors and non-destructive quality warnings."""
    row: dict[str, Any] = {
        "id": path.stem.lower(),
        "text": path.stem.lower(),
        "audio_path": _relative(path, repo_root.resolve()),
        "status": "ok",
        "errors": [],
        "quality_flags": [],
    }
    try:
        syllable = parse_syllable(path.stem)
        row.update(consonant=syllable.consonant, vowel=syllable.vowel, tone=syllable.tone)
    except ValueError as exc:
        row["errors"].append(str(exc))
    try:
        row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = probe_audio(path)
        row.update(metadata)
        samples, rate = decode_audio(path, metadata["sample_rate"])
        absolute = np.abs(samples)
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        near_full_scale = float(np.mean(absolute >= 0.999))
        # An amplitude heuristic is not a VAD, voiced-duration measurement, or
        # reason to remove aspiration, breathiness, creak, or unvoiced onsets.
        threshold = max(1e-4, peak * 0.01)
        active_fraction = float(np.mean(absolute > threshold))
        row.update(
            decoded_channels=1,
            duration_seconds=len(samples) / rate,
            num_samples=len(samples),
            pcm_sha256=hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest(),
            peak=peak,
            rms=rms,
            near_full_scale_fraction=near_full_scale,
            activity_fraction_heuristic=active_fraction,
        )
        if peak == 0:
            row["errors"].append("Decoded audio contains only zero samples")
        elif rms < 0.003:
            row["quality_flags"].append("low_rms_review")
        if near_full_scale > 0:
            row["quality_flags"].append("near_full_scale_review")
        if active_fraction < 0.1:
            row["quality_flags"].append("low_activity_review")
        if row["duration_seconds"] < 0.15:
            row["quality_flags"].append("very_short_review")
        if metadata["channels"] != 1:
            row["quality_flags"].append("multichannel_downmix_review")
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        row["errors"].append(str(exc))
    if row["errors"]:
        row["status"] = "error"
    return row


def _duplicates(rows: list[dict[str, Any]], field: str) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row["audio_path"])
    return sorted(sorted(paths) for paths in groups.values() if len(paths) > 1)


def _provenance(repo_root: Path, audio_dir: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "repository": "https://github.com/oneoffcoder/rpa",
        "audio_directory": _relative(audio_dir, repo_root),
        "license_notice": "README.md: Creative Commons Attribution 4.0 International, One-Off Coder",
        "speaker_id": "rpa_recordings",
        "speaker_identity_verified": False,
        "dialect_verified": False,
    }
    for name, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("audio_tree_sha", ["git", "rev-parse", "HEAD:" + _relative(audio_dir, repo_root)]),
    ):
        try:
            value = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=10)
            if value.returncode == 0:
                provenance[name] = value.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        if version.returncode == 0:
            provenance["decoder"] = version.stdout.splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return provenance


def render_report(report: dict[str, Any]) -> str:
    """A concise, deterministic human-readable companion to the full JSON."""
    coverage = report["coverage"]
    durations = report["duration"]
    lines = [
        "# RPA recording audit",
        "",
        f"Source commit: `{report['provenance'].get('git_commit', 'unavailable')}`.",
        f"Audio tree: `{report['provenance'].get('audio_tree_sha', 'unavailable')}`.",
        f"Manifest SHA-256: `{report['manifest_sha256']}`.",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Audio files | {report['files']} |",
        f"| Integrity-valid files | {report['valid_files']} |",
        f"| Expected unique syllables | {coverage['expected']} |",
        f"| Present expected syllables | {coverage['present']} |",
        f"| Missing syllables | {len(coverage['missing'])} |",
        f"| Unknown labels | {len(coverage['unknown'])} |",
        f"| Duplicate labels | {len(coverage['duplicate_labels'])} |",
        f"| Decoded audio, seconds | {durations['total_seconds']:.3f} |",
        f"| Decoded audio, minutes | {durations['total_seconds'] / 60:.3f} |",
        f"| Median duration, seconds | {durations.get('median_seconds', 0):.3f} |",
        f"| Byte-identical groups | {len(report['duplicate_groups']['source'])} |",
        f"| PCM-identical groups | {len(report['duplicate_groups']['pcm'])} |",
        f"| Files with review flags | {report['flagged_files']} |",
        "",
        "## Inventory",
        "",
        "This repository enumerates (56 consonants + an empty onset) × 13 vowels × 7 tones = **5,187** unique strings. "
        "The unmarked vowel-only syllables are already in that product. This inventory is not a complete validator of every Hmong dialect or spelling convention.",
        "",
        "## Quality and interpretation",
        "",
        "Every candidate file is probed and decoded. Duration measures decoded samples including any recorded silence; it is not speech-only duration. "
        "MP3 container duration can be longer because of encoder framing and padding. Flags are review heuristics, not automatic exclusions or proof of bad pronunciation. "
        "No trimming, denoising, loudness normalization, or source-file rewriting occurs.",
        "",
        "The filenames provide syllable labels. Speaker identity, dialect, transcription accuracy, lexical validity, and tone correctness still need native-speaker review. "
        "Isolated syllables alone do not establish natural sentence prosody.",
        "",
        "### Review flags",
        "",
    ]
    for key, value in sorted(report["quality_flag_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    if not report["quality_flag_counts"]:
        lines.append("No heuristic review flags.")
    lines.extend(["", "Full filenames, hashes, source metadata, coverage gaps, and errors are in `manifest.jsonl` and `report.json`.", ""])
    return "\n".join(lines)


def audit_corpus(
    repo_root: Path,
    output_dir: Path,
    workers: int = 4,
    audio_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit the complete corpus and return the same object written as JSON.

    Missing, duplicate, or unknown labels and decode failures make ``ok`` false.
    Review flags are advisory. Exact audio duplicates are reported separately
    so dataset preparation can keep them in one split or reject ambiguity.
    """
    require_audio_tools()
    if workers < 1:
        raise ValueError("workers must be positive")
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    audio_dir = Path(audio_dir).resolve() if audio_dir else repo_root / "source/_static/mp3"
    if not audio_dir.is_dir():
        raise ValueError(f"Audio directory does not exist: {audio_dir}")
    _relative(audio_dir, repo_root)
    files = sorted(p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda p: inspect_recording(p, repo_root), files))
    expected = expected_syllables()
    label_counts = Counter(row["text"] for row in rows)
    missing = sorted(set(expected) - set(label_counts))
    unknown = sorted(set(label_counts) - set(expected))
    duplicate_labels = {text: count for text, count in sorted(label_counts.items()) if count > 1}
    row_errors = [
        {"audio_path": row["audio_path"], "errors": row["errors"]}
        for row in rows if row["status"] == "error"
    ]
    durations = [row["duration_seconds"] for row in rows if "duration_seconds" in row]
    duration_summary: dict[str, float] = {"total_seconds": math.fsum(durations)}
    if durations:
        for key, percentile in (("min_seconds", 0), ("p05_seconds", 5), ("median_seconds", 50), ("p95_seconds", 95), ("max_seconds", 100)):
            duration_summary[key] = float(np.percentile(durations, percentile))
    error_count = len(missing) + len(unknown) + len(duplicate_labels) + len(row_errors)
    manifest_text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    corpus_hash_input = "".join(row["audio_path"] + "\0" + row.get("source_sha256", "unreadable") + "\n" for row in rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": _provenance(repo_root, audio_dir),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_hash_input.encode("utf-8")).hexdigest(),
        "ok": error_count == 0,
        "error_count": error_count,
        "files": len(rows),
        "valid_files": sum(row["status"] == "ok" for row in rows),
        "source_bytes": sum(path.stat().st_size for path in files),
        "coverage": {
            "expected": len(expected),
            "present": len(set(expected) & set(label_counts)),
            "consonants": len(CONSONANTS),
            "onsets_including_empty": len(ONSETS),
            "vowels": len(VOWELS),
            "tones": len(TONES),
            "missing": missing,
            "unknown": unknown,
            "duplicate_labels": duplicate_labels,
        },
        "duration": duration_summary,
        "sample_rates": dict(sorted(Counter(str(row["sample_rate"]) for row in rows if "sample_rate" in row).items())),
        "channels": dict(sorted(Counter(str(row["channels"]) for row in rows if "channels" in row).items())),
        "codecs": dict(sorted(Counter(row["codec"] for row in rows if "codec" in row).items())),
        "duplicate_groups": {"source": _duplicates(rows, "source_sha256"), "pcm": _duplicates(rows, "pcm_sha256")},
        "quality_flag_counts": dict(sorted(Counter(flag for row in rows for flag in row["quality_flags"]).items())),
        "flagged_files": sum(bool(row["quality_flags"]) for row in rows),
        "errors": row_errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report

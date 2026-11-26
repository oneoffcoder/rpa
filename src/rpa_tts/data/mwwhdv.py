"""Reproducible MWWHDV verse-corpus preparation.

The authorized source is the Hmong Daw UBS 2000 narration (audio fileset
`MWWHDVN2DA`, text `MWWHDV`). This module only handles public metadata and local
audio cutting; it never bundles audio. Callers must have the recorded training
permission and keep audio/weights private.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..hashing import sha256_file

API_BASE = "https://live.bible.is/api"
AUDIO_FILESET = "MWWHDVN2DA"
TEXT_FILESET = "MWWHDV"
EDITION = "Hmong Daw UBS 2000"
SAMPLE_RATE = 22050

BOOK_CHAPTERS: dict[str, int] = {
    "MAT": 28,
    "MRK": 16,
    "LUK": 24,
    "JHN": 21,
    "ACT": 28,
    "ROM": 16,
    "1CO": 16,
    "2CO": 13,
    "GAL": 6,
    "EPH": 6,
    "PHP": 4,
    "COL": 4,
    "1TH": 5,
    "2TH": 3,
    "1TI": 6,
    "2TI": 4,
    "TIT": 3,
    "PHM": 1,
    "HEB": 13,
    "JAS": 5,
    "1PE": 5,
    "2PE": 3,
    "1JN": 5,
    "2JN": 1,
    "3JN": 1,
    "JUD": 1,
    "REV": 22,
}


def verse_spans(timestamps: list[dict[str, Any]], chapter_duration: float) -> list[dict[str, Any]]:
    """Return start/end spans from Bible Brain verse timestamps.

    End is the next available verse start (not blindly ``verse + 1``) or the
    chapter duration, so gaps do not create a whole-chapter span.
    """
    starts = sorted(
        {
            int(row["verse_start"]): float(row["timestamp"])
            for row in timestamps
            if float(row.get("timestamp", -1)) >= 0
        }.items()
    )
    if not starts:
        raise InputError("no usable verse timestamps")
    if chapter_duration <= 0:
        raise InputError("chapter_duration must be positive")
    values = [value for _, value in starts]
    spans = []
    for verse, start in starts:
        later = [value for value in values if value > start]
        end = later[0] if later else chapter_duration
        if end <= start:
            continue
        spans.append(
            {
                "verse": verse,
                "start": start,
                "end": end,
                "duration": round(end - start, 4),
            }
        )
    return spans


def cut_verse(
    source_audio: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Cut one verse span to mono PCM WAV and return its measured duration."""
    if shutil.which(ffmpeg) is None:
        raise InputError(f"ffmpeg not found: {ffmpeg}")
    source = Path(source_audio)
    if not source.is_file():
        raise InputError(f"chapter audio does not exist: {source}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(source),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not output.is_file():
        raise InputError(f"ffmpeg did not write {output}")
    with wave.open(str(output), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        channels = handle.getnchannels()
    if rate != sample_rate or channels != 1:
        raise InputError(f"unexpected cut audio format: {rate} Hz, {channels} channel(s)")
    return {
        "path": output.as_posix(),
        "sample_rate": rate,
        "duration_seconds": round(frames / rate, 4),
        "sha256": sha256_file(output),
    }


def build_chapter_manifest(
    source_audio: str | Path,
    text_rows: list[dict[str, Any]],
    timestamp_rows: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    book: str,
    chapter: int,
    chapter_duration: float,
    ffmpeg: str = "ffmpeg",
) -> list[dict[str, Any]]:
    """Cut every timestamped verse and pair it with the same-edition verse text."""
    if book not in BOOK_CHAPTERS:
        raise InputError(f"unknown NT book: {book}")
    text_by_verse = {int(row["verse_start"]): str(row["verse_text"]) for row in text_rows}
    verse_dir = Path(output_dir) / "verses"
    rows: list[dict[str, Any]] = []
    for span in verse_spans(timestamp_rows, chapter_duration):
        verse = span["verse"]
        if verse not in text_by_verse:
            continue
        text = text_by_verse[verse].strip()
        if not text:
            continue
        cut = cut_verse(
            source_audio,
            span["start"],
            span["end"],
            verse_dir / f"{book}_{chapter}_{verse}.wav",
            ffmpeg=ffmpeg,
        )
        rows.append(
            {
                "id": f"{book}-{chapter}-{verse}",
                "reference": f"{book} {chapter}:{verse}",
                "audio_path": cut["path"],
                "text": text,
                "duration_seconds": cut["duration_seconds"],
                "expected_seconds": span["duration"],
                "sample_rate": cut["sample_rate"],
                "start": span["start"],
                "end": span["end"],
                "edition": EDITION,
                "audio_sha256": cut["sha256"],
            }
        )
    return rows


def audit_audio_uniqueness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag rows that share a WAV hash, which cannot be distinct verse pairs.

    A duplicate hash means the same waveform was cut for different transcripts or
    intervals; those rows are ambiguous until combined-verse alignment is done.
    """
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        digest = str(row.get("audio_sha256", "") or "")
        if not digest:
            raise InputError(f"row {row.get('id')!r} has no audio_sha256")
        by_hash.setdefault(digest, []).append(row)
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        groups.append(
            {
                "sha256": digest,
                "ids": [str(member["id"]) for member in members],
                "references": [str(member.get("reference", "")) for member in members],
                "intervals": [[member.get("start"), member.get("end")] for member in members],
                "texts": [str(member.get("text", "")) for member in members],
            }
        )
    ambiguous = sorted({row_id for group in groups for row_id in group["ids"]})
    return {
        "rows": len(rows),
        "distinct_audio_hashes": len(by_hash),
        "duplicate_groups": groups,
        "ambiguous_ids": ambiguous,
        "ok": not groups,
        "note": (
            "Duplicate hashes are ambiguous until combined-verse alignment; "
            "exclude them from training/evaluation splits."
        ),
    }


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "path": destination.as_posix(),
        "rows": len(rows),
        "seconds": round(sum(row["duration_seconds"] for row in rows), 3),
        "sha256": sha256_file(destination),
    }


def fetch_json(url: str, *, opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    with opener(url, timeout=60) as response:
        return json.load(response)


def fetch_chapter_metadata(book: str, chapter: int) -> dict[str, Any]:
    return fetch_json(
        f"{API_BASE}/bibles/filesets/{AUDIO_FILESET}"
        f"?book_id={book}&chapter_id={chapter}&type=audio_drama"
    )["data"][0]


def fetch_chapter_text(book: str, chapter: int) -> list[dict[str, Any]]:
    return fetch_json(f"{API_BASE}/bibles/filesets/{TEXT_FILESET}/{book}/{chapter}")["data"]


def fetch_chapter_timestamps(book: str, chapter: int) -> list[dict[str, Any]]:
    return fetch_json(f"{API_BASE}/timestamps/{AUDIO_FILESET}/{book}/{chapter}")["data"]


def download_file(
    url: str, destination: str | Path, *, opener: Callable[..., Any] = urllib.request.urlopen
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return target


__all__ = [
    "API_BASE",
    "AUDIO_FILESET",
    "BOOK_CHAPTERS",
    "EDITION",
    "TEXT_FILESET",
    "audit_audio_uniqueness",
    "build_chapter_manifest",
    "cut_verse",
    "download_file",
    "fetch_chapter_metadata",
    "fetch_chapter_text",
    "fetch_chapter_timestamps",
    "fetch_json",
    "verse_spans",
    "write_manifest",
]

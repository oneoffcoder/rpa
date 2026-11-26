"""Non-destructive audio inspection through ffmpeg/ffprobe.

Adapted from the retired PR #1 archive (``src/rpa_tts/audit.py``,
commit 96698f0689a489b16868ee85046cac6604dfbb77) with the same design rule:
probe and decode without trimming, denoising, normalizing gain or rewriting the
source file. Flags are review prompts, not automatic exclusions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InputError
from ..hashing import sha256_bytes, sha256_file

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".m4a", ".mp3"})
DEFAULT_TIMEOUT_SECONDS = 180


def require_audio_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise InputError(
            "missing audio tool(s): "
            + ", ".join(missing)
            + "; install ffmpeg and ensure ffmpeg/ffprobe are on PATH"
        )


def _run(
    args: list[str], timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(args, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise InputError(f"{Path(args[0]).name} timed out after {timeout}s on {args[-1]}") from None
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise InputError(f"{Path(args[0]).name} failed: {message[:1200]}")
    return completed


def ffmpeg_version() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (
        completed.stdout.splitlines()[0] if completed.returncode == 0 and completed.stdout else None
    )


def probe_audio(path: Path) -> dict[str, Any]:
    """Read the first audio stream plus container duration."""
    info = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels:format=duration",
                "-of",
                "json",
                str(Path(path).resolve()),
            ]
        ).stdout
    )
    streams = info.get("streams", [])
    if not streams:
        raise InputError(f"no audio stream found in {path}")
    stream = streams[0]
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError):
        raise InputError(f"audio stream metadata is incomplete in {path}") from None
    if sample_rate <= 0 or channels <= 0:
        raise InputError(f"invalid sample rate or channel count in {path}")
    duration = info.get("format", {}).get("duration")
    return {
        "codec": str(stream.get("codec_name", "unknown")),
        "sample_rate": sample_rate,
        "channels": channels,
        "container_duration_seconds": float(duration) if duration is not None else None,
    }


def decode_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Decode to mono float32; resample only when a rate is explicitly supplied."""
    if sample_rate is None:
        sample_rate = int(probe_audio(path)["sample_rate"])
    if sample_rate <= 0:
        raise InputError("sample_rate must be positive")
    data = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(Path(path).resolve()),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
    ).stdout
    waveform = np.frombuffer(data, dtype="<f4").copy()
    if waveform.size == 0:
        raise InputError(f"decoded audio is empty: {path}")
    if not np.isfinite(waveform).all():
        raise InputError(f"decoded audio contains non-finite samples: {path}")
    return waveform, sample_rate


def waveform_stats(waveform: np.ndarray) -> dict[str, float]:
    """Cheap, non-clinical signal measurements used only for review flags."""
    absolute = np.abs(waveform)
    peak = float(absolute.max()) if absolute.size else 0.0
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))) if absolute.size else 0.0
    threshold = max(1e-4, peak * 0.01)
    return {
        "peak": peak,
        "rms": rms,
        "near_full_scale_fraction": float(np.mean(absolute >= 0.999)) if absolute.size else 0.0,
        "activity_fraction_heuristic": float(np.mean(absolute > threshold))
        if absolute.size
        else 0.0,
    }


def inspect_audio(path: Path) -> dict[str, Any]:
    """Hash, probe and decode one file; raises InputError for unusable audio."""
    source_hash = sha256_file(path)
    metadata = probe_audio(path)
    waveform, rate = decode_audio(path, metadata["sample_rate"])
    stats = waveform_stats(waveform)
    pcm_hash = sha256_bytes(waveform.astype("<f4").tobytes())
    return {
        **metadata,
        "source_sha256": source_hash,
        "pcm_sha256": pcm_hash,
        "decoded_sample_rate": rate,
        "num_samples": int(len(waveform)),
        "duration_seconds": len(waveform) / rate,
        "peak": stats["peak"],
        "rms": stats["rms"],
        "near_full_scale_fraction": stats["near_full_scale_fraction"],
        "activity_fraction_heuristic": stats["activity_fraction_heuristic"],
    }


def review_flags(measurement: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if measurement["duration_seconds"] < 0.3:
        flags.append("very_short_review")
    if measurement["rms"] < 0.003:
        flags.append("low_rms_review")
    if measurement["near_full_scale_fraction"] > 0:
        flags.append("near_full_scale_review")
    if measurement["activity_fraction_heuristic"] < 0.1:
        flags.append("low_activity_review")
    if measurement["channels"] != 1:
        flags.append("multichannel_downmix_review")
    if measurement["codec"] in {"mp3", "aac"}:
        flags.append("lossy_source_review")
    return flags

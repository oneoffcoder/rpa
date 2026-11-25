"""A transparent recording-concatenation baseline, not neural synthesis."""

from pathlib import Path
import re

import numpy as np

from .audit import decode_audio, require_audio_tools
from .orthography import parse_syllable
from .prepare import SAMPLE_RATE, write_wav


def tokenize(text: str) -> list[str]:
    if not text.strip():
        raise ValueError("text must contain at least one syllable")
    unsupported = re.sub(r"[A-Za-z]+|[.,;:!?]|\s+", "", text)
    if unsupported:
        raise ValueError(f"unsupported characters in baseline text: {unsupported!r}; spell out numbers and expand abbreviations")
    tokens = re.findall(r"[A-Za-z]+|[.,;:!?]", text)
    syllables = [token for token in tokens if token.isalpha()]
    if not syllables:
        raise ValueError("text must contain at least one syllable")
    unknown = []
    for token in syllables:
        try:
            parse_syllable(token)
        except ValueError:
            unknown.append(token)
    if unknown:
        raise ValueError("outside this repository's recorded inventory: " + ", ".join(unknown))
    return [token.lower() for token in tokens]


def synthesize_baseline(
    text: str, output: Path, *, repo_root: Path, gap_ms: int = 40,
    punctuation_ms: int = 180,
) -> dict:
    """Join existing recordings, making all inserted pauses explicit."""
    if not 0 <= gap_ms <= 5000 or not 0 <= punctuation_ms <= 5000:
        raise ValueError("pause lengths must be between 0 and 5000 milliseconds")
    tokens = tokenize(text)
    if output.exists():
        raise ValueError(f"output already exists: {output}; choose a new filename")
    require_audio_tools()
    audio_dir = repo_root.resolve() / "source" / "_static" / "mp3"
    pieces = []
    syllable_count = 0
    pending_gap = 0
    for token in tokens:
        if not token.isalpha():
            pending_gap = punctuation_ms
            continue
        if pieces and pending_gap:
            pieces.append(np.zeros(SAMPLE_RATE * pending_gap // 1000, dtype=np.float32))
        source = audio_dir / f"{token}.mp3"
        if not source.is_file():
            raise ValueError(f"missing recording: {source}")
        samples, _ = decode_audio(source, sample_rate=SAMPLE_RATE)
        pieces.append(samples)
        syllable_count += 1
        pending_gap = gap_ms
    waveform = np.concatenate(pieces)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output, waveform)
    return {
        "method": "recording_concatenation", "text": text,
        "syllables": syllable_count, "output": str(output),
        "sample_rate": SAMPLE_RATE, "duration_seconds": len(waveform) / SAMPLE_RATE,
        "gap_ms": gap_ms, "punctuation_ms": punctuation_ms,
        "limitation": "Uses the full recording inventory; not a neural model or a fair unseen-syllable generalization baseline.",
    }

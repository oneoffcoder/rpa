from __future__ import annotations

import json
import shutil
import struct
import wave
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.data.mwwhdv import (
    build_chapter_manifest,
    cut_verse,
    verse_spans,
    write_manifest,
)
from rpa_tts.errors import InputError

FFMPEG = shutil.which("ffmpeg")


def _wave(path: Path, seconds: float = 5.0, rate: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(1000 * __import__("math").sin(2 * 3.14159 * 220 * i / rate)))
            for i in range(int(rate * seconds))
        )
        handle.writeframes(frames)
    return path


def test_verse_spans_uses_next_available_start():
    timestamps = [
        {"verse_start": "1", "timestamp": 1.0},
        {"verse_start": "3", "timestamp": 3.0},
        {"verse_start": "0", "timestamp": 0.0},
    ]
    spans = verse_spans(timestamps, 4.0)
    assert spans == [
        {"verse": 0, "start": 0.0, "end": 1.0, "duration": 1.0},
        {"verse": 1, "start": 1.0, "end": 3.0, "duration": 2.0},
        {"verse": 3, "start": 3.0, "end": 4.0, "duration": 1.0},
    ]


def test_verse_spans_requires_timestamps():
    with pytest.raises(InputError, match="no usable"):
        verse_spans([], 4.0)


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_cut_verse_measures_output(tmp_path):
    source = _wave(tmp_path / "chapter.wav")
    cut = cut_verse(source, 1.0, 3.0, tmp_path / "verses" / "v.wav")
    assert abs(cut["duration_seconds"] - 2.0) < 0.01
    assert cut["sample_rate"] == 22050
    assert cut["sha256"]


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_build_chapter_manifest_pairs_text_and_duration(tmp_path):
    source = _wave(tmp_path / "chapter.wav")
    text = [
        {"verse_start": 1, "verse_text": "Kuv hlub koj."},
        {"verse_start": 2, "verse_text": "Koj nyob li cas?"},
    ]
    timestamps = [
        {"verse_start": 1, "timestamp": 1.0},
        {"verse_start": 2, "timestamp": 2.5},
    ]
    rows = build_chapter_manifest(
        source,
        text,
        timestamps,
        tmp_path / "out",
        book="MAT",
        chapter=1,
        chapter_duration=5.0,
    )
    assert [row["id"] for row in rows] == ["MAT-1-1", "MAT-1-2"]
    assert rows[0]["text"] == "Kuv hlub koj."
    assert rows[0]["expected_seconds"] == 1.5
    assert abs(rows[0]["duration_seconds"] - 1.5) < 0.01
    assert rows[1]["expected_seconds"] == 2.5
    assert all(Path(row["audio_path"]).is_file() for row in rows)

    summary = write_manifest(rows, tmp_path / "manifest.jsonl")
    assert summary["rows"] == 2
    assert summary["sha256"]
    loaded = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert loaded == rows


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
def test_cli_mwwhdv_chapter(tmp_path, capsys):
    source = _wave(tmp_path / "chapter.wav")
    text = tmp_path / "text.json"
    text.write_text(
        json.dumps([{"verse_start": 1, "verse_text": "Kuv hlub koj."}]),
        encoding="utf-8",
    )
    timestamps = tmp_path / "timestamps.json"
    timestamps.write_text(
        json.dumps([{"verse_start": 1, "timestamp": 1.0}]),
        encoding="utf-8",
    )
    code = main(
        [
            "corpus",
            "mwwhdv-chapter",
            "--audio",
            str(source),
            "--text-json",
            str(text),
            "--timestamps-json",
            str(timestamps),
            "--book",
            "MAT",
            "--chapter",
            "1",
            "--chapter-duration",
            "5.0",
            "--output-dir",
            str(tmp_path / "out"),
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert payload["private_training_data"] is True

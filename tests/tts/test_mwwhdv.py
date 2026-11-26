from __future__ import annotations

import json
import shutil
import struct
import wave
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.data.mwwhdv import (
    audit_audio_uniqueness,
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


def test_audit_audio_uniqueness_flags_same_waveform(tmp_path):
    rows = [
        {
            "id": "REV-22-20",
            "reference": "REV 22:20",
            "audio_sha256": "same",
            "start": 298.23,
            "end": 327.18,
            "text": "one",
        },
        {
            "id": "REV-22-21",
            "reference": "REV 22:21",
            "audio_sha256": "same",
            "start": 298.23,
            "end": 327.18,
            "text": "two",
        },
        {
            "id": "MRK-1-1",
            "reference": "MRK 1:1",
            "audio_sha256": "unique",
            "start": 3.75,
            "end": 9.01,
            "text": "three",
        },
    ]
    report = audit_audio_uniqueness(rows)
    assert report["ok"] is False
    assert report["ambiguous_ids"] == ["REV-22-20", "REV-22-21"]
    assert report["distinct_audio_hashes"] == 2
    assert report["duplicate_groups"][0]["sha256"] == "same"
    assert audit_audio_uniqueness([rows[2]])["ok"] is True


def test_cli_mwwhdv_audit_fails_on_duplicates(tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "a", "reference": "A 1:1", "audio_sha256": "x"},
                {"id": "b", "reference": "A 1:2", "audio_sha256": "x"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    code = main(
        [
            "corpus",
            "mwwhdv-audit",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ambiguous_ids"] == ["a", "b"]
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["ok"] is False


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

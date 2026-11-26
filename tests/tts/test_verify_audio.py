from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.data.verify import verify_audio
from rpa_tts.errors import InputError
from rpa_tts.hashing import sha256_file

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg and ffprobe are required for audio verification tests",
    ),
]


def _write_wav(
    path: Path, *, seconds: float = 0.5, rate: int = 48000, frequency: int = 220
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * frequency * index / rate)))
        for index in range(samples)
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(frames)
    return path


def _row_for(row_factory, audio: Path, audio_root: Path, **overrides):
    row = row_factory(1)
    row["audio_path"] = audio.relative_to(audio_root).as_posix()
    row["audio_sha256"] = sha256_file(audio)
    row["duration_seconds"] = 0.5
    row["sample_rate"] = 48000
    row["channels"] = 1
    row.update(overrides)
    return row


def test_matching_audio_passes(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    report = verify_audio([_row_for(row_factory, audio, root)], root)
    assert report["ok"] is True
    assert report["passed"] == 1
    assert report["failed"] == 0
    assert report["measured_hours"] > 0
    assert report["sample_rates"] == ["48000"]


def test_tampered_audio_is_detected(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    row = _row_for(row_factory, audio, root)
    _write_wav(audio, frequency=330)
    report = verify_audio([row], root)
    assert report["ok"] is False
    assert report["failed"] == 1
    assert "sha256_mismatch" in report["problem_counts"]


def test_missing_audio_is_detected(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    row = _row_for(row_factory, audio, root)
    audio.unlink()
    report = verify_audio([row], root)
    assert report["problem_counts"] == {"missing_file": 1}
    assert report["failures"][0]["problems"] == ["missing_file"]


def test_duration_mismatch_is_detected(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    row = _row_for(row_factory, audio, root, duration_seconds=0.6)
    report = verify_audio([row], root, duration_tolerance_seconds=0.05)
    assert "duration_mismatch" in report["problem_counts"]
    tolerant = verify_audio([row], root, duration_tolerance_seconds=0.2)
    assert tolerant["ok"] is True


def test_sample_rate_and_channel_mismatches_are_detected(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    row = _row_for(row_factory, audio, root)
    row["sample_rate"] = 44100
    report = verify_audio([row], root)
    assert "sample_rate_mismatch" in report["problem_counts"]

    row = _row_for(row_factory, audio, root)
    row["channels"] = 2
    report = verify_audio([row], root)
    assert "channels_mismatch" in report["problem_counts"]


def test_unreadable_audio_is_reported(tmp_path, row_factory):
    root = tmp_path / "audio-root"
    audio = root / "audio" / "u0001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"not audio")
    row = _row_for(row_factory, audio, root, audio_sha256=sha256_file(audio))
    report = verify_audio([row], root)
    assert any(problem.startswith("unreadable") for problem in report["failures"][0]["problems"])


def test_missing_audio_root_and_rows_are_rejected(tmp_path, row_factory):
    row = row_factory(1)
    row["audio_path"] = "audio/u0001.wav"
    with pytest.raises(InputError, match="audio root does not exist"):
        verify_audio([row], tmp_path / "nope")
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(InputError, match="no rows"):
        verify_audio([], root)


def test_cli_verify_audio_reports_and_exits(tmp_path, row_factory, write_manifest_file, capsys):
    root = tmp_path / "audio-root"
    audio = _write_wav(root / "audio" / "u0001.wav")
    row = _row_for(row_factory, audio, root)
    manifest = write_manifest_file([row])
    code = main(
        [
            "corpus",
            "verify-audio",
            "--manifest",
            str(manifest),
            "--audio-root",
            str(root),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert (tmp_path / "report.json").is_file()

    _write_wav(audio, frequency=440)
    code = main(["corpus", "verify-audio", "--manifest", str(manifest), "--audio-root", str(root)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False

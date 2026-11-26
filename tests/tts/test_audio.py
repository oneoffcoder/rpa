from __future__ import annotations

import json
import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from rpa_tts.data.audio import decode_audio, inspect_audio, probe_audio

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg and ffprobe are required for audio integration tests",
    ),
]


def _write_wav(
    path: Path, *, seconds: float = 0.5, rate: int = 48000, frequency: int = 220
) -> Path:
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


def test_probe_and_decode_report_real_measurements(tmp_path):
    audio = _write_wav(tmp_path / "sample.wav")
    metadata = probe_audio(audio)
    assert metadata["sample_rate"] == 48000
    assert metadata["channels"] == 1
    waveform, rate = decode_audio(audio)
    assert rate == 48000
    assert len(waveform) == 24000
    measurement = inspect_audio(audio)
    assert measurement["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert measurement["rms"] > 0
    assert len(measurement["source_sha256"]) == 64
    assert len(measurement["pcm_sha256"]) == 64


def test_audit_corpus_flags_missing_transcripts_and_accepts_labeled_rows(tmp_path, repo_root):
    from rpa_tts.data.audit import audit_corpus

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "u0001.wav")
    _write_wav(audio_dir / "u0002.wav", frequency=330)
    cold = audit_corpus(audio_dir, tmp_path, tmp_path / "out-cold")
    assert cold["ok"] is False
    assert cold["counts_by_status"]["needs_transcript"] == 2

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "id": "u0001",
                    "transcript_raw": "kuv hlub koj",
                    "group_id": "p1",
                    "bucket": "ordinary_sentences",
                    "rights_id": "rights-owned-01",
                    "consent_id": "consent-01",
                },
                {
                    "id": "u0002",
                    "transcript_raw": "nws yog ib tug neeg",
                    "group_id": "p2",
                    "bucket": "ordinary_sentences",
                    "rights_id": "rights-owned-01",
                    "consent_id": "consent-01",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    warm = audit_corpus(audio_dir, tmp_path, tmp_path / "out-warm", labels_path=labels, workers=2)
    assert warm["ok"] is True
    assert warm["valid_files"] == 2
    assert warm["duration"]["total_seconds"] == pytest.approx(1.0, abs=0.05)
    rows = [
        json.loads(line)
        for line in (tmp_path / "out-warm" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["transcript_normalized"] for row in rows)
    assert (tmp_path / "out-warm" / "report.md").is_file()


def test_audit_records_duplicate_audio_for_leakage_awareness(tmp_path):
    from rpa_tts.data.audit import audit_corpus

    audio_dir = tmp_path / "audio"
    nested = audio_dir / "nested"
    nested.mkdir(parents=True)
    original = _write_wav(audio_dir / "u0001.wav")
    shutil.copyfile(original, nested / "u0001.wav")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "id": "u0001",
                "transcript_raw": "kuv hlub koj",
                "group_id": "p1",
                "bucket": "ordinary_sentences",
                "rights_id": "r1",
                "consent_id": "c1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_corpus(audio_dir, tmp_path, tmp_path / "out", labels_path=labels)
    assert report["ok"] is False
    assert report["duplicate_ids"] == ["u0001"]
    assert report["duplicate_groups"]["source"]


def test_transcript_outside_the_contract_is_flagged(tmp_path):
    from rpa_tts.data.audit import audit_corpus

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "u0001.wav")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": "u0001", "transcript_raw": "hello 3"}) + "\n", encoding="utf-8"
    )
    report = audit_corpus(audio_dir, tmp_path, tmp_path / "out", labels_path=labels)
    assert report["ok"] is False
    assert report["counts_by_status"]["transcript_invalid"] == 1


def test_empty_audio_directory_is_an_input_error(tmp_path):
    from rpa_tts.data.audit import audit_corpus
    from rpa_tts.errors import InputError

    (tmp_path / "audio").mkdir()
    with pytest.raises(InputError, match="no supported audio files"):
        audit_corpus(tmp_path / "audio", tmp_path, tmp_path / "out")

"""Shared fixtures and manifest builders for the TTS tests.

Fixtures are synthetic and deliberately small; they exist to exercise contracts,
leakage rules and scoring arithmetic, not to represent an accepted corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpa_tts.text.normalize import normalize_rpa  # noqa: E402

SAMPLE_TEXTS = [
    "kuv hlub koj",
    "nws yog ib tug neeg",
    "koj puas nyob zoo",
    "peb yuav mus tsev",
    "nws hais lus zoo",
]


def build_row(index: int = 1, **overrides):
    """One schema-valid accepted row; override only what a test is about."""
    text = overrides.pop("transcript_raw", SAMPLE_TEXTS[(index - 1) % len(SAMPLE_TEXTS)])
    normalized = normalize_rpa(text)
    row = {
        "id": f"u{index:04d}",
        "audio_path": f"audio/u{index:04d}.wav",
        "audio_sha256": f"{index:064x}"[:64],
        "duration_seconds": 2.5,
        "sample_rate": 48000,
        "channels": 1,
        "group_id": f"g{index:04d}",
        "bucket": "ordinary_sentences",
        "transcript_raw": text,
        "transcript_normalized": normalized.normalized,
        "normalization_version": normalized.version,
        "expansion_ids": [],
        "split": "train",
        "rights_id": "rights-owned-01",
        "consent_id": "consent-01",
        "source": {
            "source_id": "owned-recording-2026",
            "speaker_id": "spk-01",
            "edition": None,
            "url": None,
            "retrieval_date": None,
            "dialect": "white-hmong",
            "session_id": f"s{index:03d}",
            "parent_passage_id": None,
            "start_seconds": None,
            "end_seconds": None,
        },
        "reviewer": {
            "native_approved": True,
            "reviewer_ids": ["rev-01"],
            "reviewed_at": "2026-11-26T00:00:00+00:00",
            "decisions": ["accepted"],
            "notes": None,
        },
        "quality_tags": ["clean"],
        "preprocessing_version": "pp-1",
    }
    row.update(overrides)
    if "audio_sha256" in row and len(str(row["audio_sha256"])) != 64:
        row["audio_sha256"] = str(row["audio_sha256"]).ljust(64, "0")[:64]
    return row


def build_rows(count: int, **overrides) -> list[dict]:
    return [build_row(index, **overrides) for index in range(1, count + 1)]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def row_factory():
    return build_row


@pytest.fixture
def rows_factory():
    return build_rows


@pytest.fixture
def write_manifest_file(tmp_path):
    def write(rows: list[dict], name: str = "manifest.jsonl") -> Path:
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    return write


@pytest.fixture
def valid_rows() -> list[dict]:
    return build_rows(6)


@pytest.fixture
def config_path() -> Path:
    return REPO_ROOT / "configs" / "tts" / "default.yaml"


@pytest.fixture
def contract_path() -> Path:
    return REPO_ROOT / "docs" / "tts" / "recovery" / "language-contract.yaml"

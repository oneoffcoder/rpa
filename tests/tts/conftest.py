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


PACK_CONDITIONS = ["candidate", "natural", "baseline"]


def build_pack_definition(**overrides):
    """A small three-condition pack definition for instrument tests."""

    def item(script_id, bucket, text, **extra):
        return {
            "script_id": script_id,
            "bucket": bucket,
            "text": text,
            "condition_audio": {
                condition: f"audio/{condition}/{script_id}.wav" for condition in PACK_CONDITIONS
            },
            **extra,
        }

    definition = {
        "schema_version": 1,
        "pack_id": "calibration-test-01",
        "design": "calibration",
        "seed": 3407,
        "conditions": list(PACK_CONDITIONS),
        "items": [
            item("s001", "ordinary_sentences", "kuv hlub koj"),
            item(
                "s002",
                "targeted_contrasts",
                "nws yog ib tug neeg",
                critical=True,
                contrast={"options": ["neeg", "nees"], "expected": "neeg"},
            ),
            item("s003", "ordinary_sentences", "koj puas nyob zoo"),
        ],
        "listeners": [
            {
                "listener_id": f"L-{index:02d}",
                "roles": ["transcription", "naturalness", "contrast"],
            }
            for index in range(1, 4)
        ],
    }
    definition.update(overrides)
    return definition


def write_pack_responses(pack_dir, responses_dir, *, empty_codes=()):
    """Write valid phase-A/B response sheets from a built pack manifest."""
    import csv
    import json

    manifest = json.loads((Path(pack_dir) / "pack-manifest.json").read_text(encoding="utf-8"))
    items = {item["script_id"]: item for item in manifest["items"]}
    empty = set(empty_codes)
    responses_dir = Path(responses_dir)
    responses_dir.mkdir(parents=True, exist_ok=True)
    for listener in manifest["listeners"]:
        listener_id = listener["listener_id"]
        assignments = [
            entry for entry in manifest["assignments"] if entry["listener_id"] == listener_id
        ]
        with (responses_dir / f"{listener_id}-phase-a.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["code", "audio", "transcription", "naturalness", "replay_count", "comments"]
            )
            for entry in assignments:
                text = "" if entry["code"] in empty else items[entry["script_id"]]["text"]
                writer.writerow(
                    [
                        entry["code"],
                        entry["blind_audio"],
                        text,
                        4 if entry["condition"] == "candidate" else 5,
                        0,
                        "",
                    ]
                )
        contrasts = [
            entry
            for entry in manifest["contrast_assignments"]
            if entry["listener_id"] == listener_id
        ]
        if contrasts:
            with (responses_dir / f"{listener_id}-phase-b.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["code", "option_a", "option_b", "chosen", "confidence", "comments"]
                )
                for entry in contrasts:
                    writer.writerow(
                        [
                            entry["code"],
                            entry["options"][0],
                            entry["options"][1],
                            entry["expected"],
                            "",
                            "",
                        ]
                    )


@pytest.fixture
def pack_definition():
    return build_pack_definition


@pytest.fixture
def write_responses():
    return write_pack_responses


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

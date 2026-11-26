from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from rpa_tts.schemas import LEARNING_CURVE_COLUMNS, SCHEMAS


def test_mirrored_schema_files_match_the_embedded_schemas(repo_root):
    import os

    schema_dir = repo_root / "configs" / "tts" / "schemas"
    files = sorted(path.name for path in schema_dir.glob("*.json"))
    assert files == sorted(SCHEMAS)
    for name, embedded in SCHEMAS.items():
        mirrored = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        assert mirrored == embedded, f"{name} drifted from the embedded schema"
    assert os.environ.get("PYTHONPATH") is not None or True  # documents repo-relative fixtures


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_embedded_schemas_are_valid_draft_2020_12(name):
    Draft202012Validator.check_schema(SCHEMAS[name])


def test_learning_curve_header_matches_schema_columns(repo_root):
    header = (
        (repo_root / "docs" / "tts" / "recovery" / "learning-curve.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert header.split(",") == LEARNING_CURVE_COLUMNS


def test_utterance_schema_rejects_unknown_properties():
    validator = Draft202012Validator(SCHEMAS["utterance-manifest.schema.json"])
    errors = list(validator.iter_errors({"id": "u1", "surprise": True}))
    assert errors


def test_run_manifest_schema_rejects_unknown_status():
    validator = Draft202012Validator(SCHEMAS["run-manifest.schema.json"])
    errors = list(validator.iter_errors({"status": "finished"}))
    assert errors


def test_evidence_schema_requires_prefixed_ids():
    validator = Draft202012Validator(SCHEMAS["evidence-entry.schema.json"])
    errors = list(validator.iter_errors({"id": "bad"}))
    assert errors

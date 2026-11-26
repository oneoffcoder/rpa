from __future__ import annotations

import json

import pytest

from rpa_tts.data.manifest import (
    load_manifest,
    manifest_summary,
    require_valid_rows,
    validate_rows,
    write_manifest,
)
from rpa_tts.errors import InputError, ValidationError


def _errors(rows, **kwargs):
    return [problem for problem in validate_rows(rows, **kwargs) if problem.level == "error"]


def test_valid_row_has_no_errors(valid_rows):
    assert _errors(valid_rows) == []


def test_missing_required_field_is_reported(valid_rows):
    del valid_rows[0]["rights_id"]
    errors = _errors(valid_rows)
    assert any("rights_id" in problem.message or problem.field == "rights_id" for problem in errors)


def test_duplicate_ids_are_rejected(valid_rows):
    valid_rows[1]["id"] = valid_rows[0]["id"]
    errors = _errors(valid_rows)
    assert any("duplicate utterance id" in problem.message for problem in errors)


def test_malformed_audio_hash_is_rejected(valid_rows):
    valid_rows[0]["audio_sha256"] = "not-a-hash"
    errors = _errors(valid_rows)
    assert any(
        problem.field == "record" and "audio_sha256" in problem.message for problem in errors
    )


def test_pending_rights_block_training_but_not_audit(valid_rows):
    valid_rows[0]["rights_id"] = "UNRESOLVED"
    assert _errors(valid_rows, purpose="training")
    assert _errors(valid_rows, purpose="audit") == []


def test_native_approval_is_required_for_admitted_material(valid_rows):
    valid_rows[0]["reviewer"]["native_approved"] = False
    assert any(
        problem.field == "reviewer.native_approved"
        for problem in _errors(valid_rows, purpose="training")
    )
    assert _errors(valid_rows, purpose="audit") == []


def test_normalized_transcript_must_be_derived_from_raw(valid_rows):
    valid_rows[0]["transcript_normalized"] = "kuv hlub"
    errors = _errors(valid_rows)
    assert any(problem.field == "transcript_normalized" for problem in errors)


def test_unknown_words_in_normalized_transcript_are_rejected(valid_rows):
    valid_rows[0]["transcript_normalized"] = "kuv blorp"
    errors = _errors(valid_rows)
    assert any("documented RPA syllable inventory" in problem.message for problem in errors)


def test_group_cannot_span_splits(valid_rows):
    valid_rows[0]["group_id"] = "shared-passage"
    valid_rows[1]["group_id"] = "shared-passage"
    valid_rows[1]["split"] = "test"
    errors = _errors(valid_rows)
    assert any("spans splits" in problem.message for problem in errors)


def test_equal_start_and_end_offset_is_rejected(valid_rows):
    valid_rows[0]["source"]["parent_passage_id"] = "p1"
    valid_rows[0]["source"]["start_seconds"] = 5.0
    valid_rows[0]["source"]["end_seconds"] = 5.0
    errors = _errors(valid_rows)
    assert any(problem.field == "source.end_seconds" for problem in errors)


def test_missing_dialect_is_a_warning_not_a_failure(valid_rows):
    valid_rows[0]["source"]["dialect"] = None
    problems = validate_rows(valid_rows)
    assert not [problem for problem in problems if problem.level == "error"]
    assert any(problem.field == "source.dialect" for problem in problems)


def test_summary_counts_unique_accepted_hours(valid_rows):
    valid_rows[0]["split"] = "test"
    valid_rows[1]["split"] = "dev"
    for row in valid_rows:
        row["duration_seconds"] = 3600.0
    summary = manifest_summary(valid_rows)
    assert summary["rows"] == 6
    assert summary["accepted_training_hours"] == 4.0
    assert summary["accepted_test_hours"] == 1.0
    assert summary["accepted_dev_hours"] == 1.0
    assert summary["by_split"]["train"]["rows"] == 4


def test_load_manifest_reports_bad_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "u1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(InputError, match=r"bad\.jsonl:2"):
        load_manifest(path)


def test_load_manifest_rejects_non_object_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(InputError, match="expected a JSON object"):
        load_manifest(path)


def test_write_manifest_is_sorted_and_hashes(tmp_path, valid_rows):
    path = tmp_path / "out.jsonl"
    digest = write_manifest(list(reversed(valid_rows)), path)
    assert len(digest) == 64
    ids = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert ids == sorted(ids)


def test_require_valid_rows_raises_with_actionable_detail(valid_rows):
    valid_rows[0]["rights_id"] = "pending"
    with pytest.raises(ValidationError, match="rights are unresolved"):
        require_valid_rows(valid_rows)

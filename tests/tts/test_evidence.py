from __future__ import annotations

import json

import pytest

from rpa_tts.errors import InputError, ValidationError
from rpa_tts.evidence import (
    append_entry,
    gate_status,
    latest_entry,
    load_index,
    make_entry,
    validate_index,
)


def test_make_entry_hashes_support_files(tmp_path):
    payload = tmp_path / "proof.json"
    payload.write_text('{"status": "passed"}', encoding="utf-8")
    entry = make_entry(
        gate="F-01",
        claim="proof exists",
        status="passed",
        paths=[payload],
        repo_root=tmp_path,
        created_at="2026-11-26T00:00:00+00:00",
    )
    assert entry["id"].startswith("ev_")
    assert entry["paths"][0]["path"] == "proof.json"
    assert len(entry["paths"][0]["sha256"]) == 64


def test_missing_support_file_is_rejected(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        make_entry(
            gate="F-01", claim="x", status="passed", paths=[tmp_path / "nope"], repo_root=tmp_path
        )


def test_unknown_status_is_rejected(tmp_path):
    with pytest.raises(InputError, match="unknown evidence status"):
        make_entry(gate="F-01", claim="x", status="good", repo_root=tmp_path)


def test_append_and_latest_status(tmp_path):
    index_path = tmp_path / "index.json"
    first = make_entry(
        gate="F-01",
        claim="first",
        status="in_progress",
        repo_root=tmp_path,
        evidence_id="ev_0000000000000001",
        created_at="2026-11-26T00:00:00+00:00",
    )
    second = make_entry(
        gate="F-01",
        claim="second",
        status="passed",
        repo_root=tmp_path,
        evidence_id="ev_0000000000000002",
        created_at="2026-11-26T01:00:00+00:00",
    )
    append_entry(first, index_path, repo_root=tmp_path)
    index = append_entry(second, index_path, repo_root=tmp_path)
    assert len(index["entries"]) == 2
    assert latest_entry(index, "F-01")["id"] == "ev_0000000000000002"
    assert gate_status(index, "F-01")["status"] == "passed"
    assert gate_status(index, "F-04")["status"] == "not_started"
    assert load_index(index_path, repo_root=tmp_path)["entries"] == index["entries"]


def test_duplicate_evidence_id_is_rejected(tmp_path):
    index_path = tmp_path / "index.json"
    entry = make_entry(
        gate="F-01",
        claim="x",
        status="passed",
        repo_root=tmp_path,
        evidence_id="ev_0000000000000003",
    )
    append_entry(entry, index_path, repo_root=tmp_path)
    with pytest.raises(InputError, match="already exists"):
        append_entry(entry, index_path, repo_root=tmp_path)


def test_invalid_index_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        validate_index({"schema_version": 1, "entries": [{"id": "nope"}]})
    with pytest.raises(ValidationError):
        validate_index({"schema_version": 2, "entries": []})


def test_duplicate_ids_inside_an_index_are_rejected(tmp_path):
    entry = make_entry(
        gate="F-01",
        claim="x",
        status="passed",
        repo_root=tmp_path,
        evidence_id="ev_0000000000000004",
    )
    with pytest.raises(ValidationError, match="duplicate ids"):
        validate_index({"schema_version": 1, "entries": [entry, entry]})


def test_malformed_index_json_is_an_input_error(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        load_index(path, repo_root=tmp_path)


def test_effective_status_uses_latest_created_at(tmp_path):
    index = {"schema_version": 1, "entries": []}
    entry = make_entry(
        gate="F-02",
        claim="later",
        status="blocked",
        repo_root=tmp_path,
        evidence_id="ev_0000000000000005",
        created_at="2026-11-26T02:00:00+00:00",
    )
    json.dump(index, (tmp_path / "index.json").open("w"), indent=2)
    append_entry(entry, tmp_path / "index.json", repo_root=tmp_path)
    assert (
        gate_status(load_index(tmp_path / "index.json", repo_root=tmp_path), "F-02")["status"]
        == "blocked"
    )

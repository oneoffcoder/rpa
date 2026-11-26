"""The recovery runbook is part of the deliverable; keep it consistent and valid."""

from __future__ import annotations

import csv
import json

from rpa_tts.evidence import load_index, validate_index
from rpa_tts.paths import find_repo_root, recovery_dir
from rpa_tts.schemas import validate as validate_schema

ROOT = find_repo_root()
RECOVERY = recovery_dir(ROOT)


def _jsonl(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_every_decision_record_validates_and_ids_are_unique():
    entries = _jsonl(RECOVERY / "decisions.jsonl")
    assert entries
    for entry in entries:
        validate_schema(entry, "decision.schema.json")
    ids = [entry["id"] for entry in entries]
    assert len(set(ids)) == len(ids)
    assert {"D-01", "D-02", "D-03", "D-04", "D-05"} <= set(ids)


def test_initial_scope_and_model_decisions_are_recorded():
    by_id = {entry["id"]: entry for entry in _jsonl(RECOVERY / "decisions.jsonl")}
    assert by_id["D-01"]["status"] == "settled"
    assert by_id["D-02"]["status"] == "open"
    assert "Magpie" in by_id["D-02"]["decision"]
    assert by_id["D-03"]["status"] == "open"


def test_stage_decisions_are_absent_or_valid():
    path = RECOVERY / "stage-decisions.jsonl"
    for entry in _jsonl(path):
        validate_schema(entry, "stage-decision.schema.json")


def test_evidence_index_is_valid_and_covers_the_implemented_gates():
    index = load_index(RECOVERY / "evidence-index.json", repo_root=ROOT)
    validate_index(index)
    gates = {entry["gate"] for entry in index["entries"]}
    assert {"F-01", "F-02", "F-03", "F-04", "F-09"} <= gates
    statuses = {entry["status"] for entry in index["entries"]}
    assert statuses <= {
        "not_started",
        "in_progress",
        "passed",
        "failed",
        "inconclusive",
        "blocked",
        "not_required",
    }


def test_evidence_paths_still_resolve_with_matching_hashes():
    from rpa_tts.hashing import sha256_file

    index = load_index(RECOVERY / "evidence-index.json", repo_root=ROOT)
    for entry in index["entries"]:
        for recorded in entry["paths"]:
            path = ROOT / recorded["path"]
            assert path.is_file(), f"{entry['id']} references missing {recorded['path']}"
            assert sha256_file(path) == recorded["sha256"], f"{entry['id']} hash drift"


def test_status_table_lists_every_stage_and_the_not_claimed_section():
    readme = (RECOVERY / "README.md").read_text(encoding="utf-8")
    for index in range(1, 10):
        assert f"**F-0{index}**" in readme
    assert "Not claimed" in readme


def test_rights_matrix_has_required_columns_and_sources():
    with (RECOVERY / "rights-matrix.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    expected = {
        "source_id",
        "item",
        "url",
        "access_status",
        "training_permitted",
        "weight_publish_permitted",
        "redistribution_permitted",
        "evidence",
        "next_action",
    }
    assert expected <= set(rows[0])
    source_ids = {row["source_id"] for row in rows}
    assert {"cmu-mwwhdv", "pakorn2112", "owned-recordings-2026"} <= source_ids
    quarantined = next(row for row in rows if row["source_id"] == "pakorn2112")
    assert quarantined["training_permitted"] == "NO"


def test_coverage_ledger_states_the_screening_rule():
    text = (RECOVERY / "coverage-ledger.md").read_text(encoding="utf-8")
    assert "five distinct training examples" in text
    assert "not" in text.lower()


def test_language_contract_is_a_draft_with_no_invented_approvals():
    import yaml

    contract = yaml.safe_load((RECOVERY / "language-contract.yaml").read_text(encoding="utf-8"))
    assert contract["status"] == "draft"
    assert contract["critical_contrasts"] == []
    assert contract["approved_expansions"] == {}
    assert contract["dialect"]["approved_by"] == []


def test_handoff_records_the_required_resumption_fields():
    text = (RECOVERY / "handoff.md").read_text(encoding="utf-8")
    for field in ("Branch", "Commit", "Environment", "Checkpoint", "Next smallest executable step"):
        assert field in text

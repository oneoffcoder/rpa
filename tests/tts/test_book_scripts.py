from __future__ import annotations

import json

import pytest

from rpa_tts.data.book_scripts import draft_scripts, extract_candidates
from rpa_tts.errors import InputError

RST = """\
Dialog
======

Pov, "`Nyob zoo, Oo. <generate.html?q=nyob+zoo+oo>`_"
Oo, "`Kuv nyob Washington, DC. <generate.html?q=kuv+nyob+washington+dc>`_"
Pov, "`Nyob zoo, Oo. <generate.html?q=nyob+zoo+oo>`_"

Proverbs
--------

* `Kuv hlub koj thiab nws. <generate.html?q=kuv+hlub+koj+thiab+nws>`_
* `oo <generate.html?q=oo>`_
"""


def test_extracts_linked_sentences_with_source_and_dedupes():
    candidates = extract_candidates(RST, min_words=3)
    assert [candidate["text"] for candidate in candidates] == [
        "Nyob zoo, Oo.",
        "Kuv nyob Washington, DC.",
        "Kuv hlub koj thiab nws.",
    ]
    assert [candidate["script_id"] for candidate in candidates] == ["cal-001", "cal-002", "cal-003"]
    assert candidates[0]["source"]["line"] == 4
    assert candidates[0]["source"]["section"] == "Dialog"
    assert candidates[2]["source"]["section"] == "Proverbs"


def test_valid_candidates_are_pending_native_approval():
    candidates = extract_candidates(RST, min_words=3)
    valid = candidates[0]
    assert valid["status"] == "pending_native_approval"
    assert valid["normalized_text"] == "nyob zoo, oo."
    assert valid["issues"] == []
    assert valid["bucket"] == "ordinary_sentences"
    assert valid["syllables"] == ["nyob", "zoo", "oo"]


def test_out_of_contract_candidates_are_retained_with_issues():
    candidates = extract_candidates(RST, min_words=3)
    blocked = candidates[1]
    assert blocked["status"] == "blocked_contract_review"
    assert blocked["normalized_text"] is None
    tokens = {issue["token"] for issue in blocked["issues"]}
    assert {"Washington", "DC"} <= tokens


def test_proverbs_are_suggested_for_targeted_contrasts():
    candidates = extract_candidates(RST, min_words=3)
    assert candidates[2]["bucket"] == "targeted_contrasts"


def test_draft_scripts_writes_jsonl_and_report(tmp_path):
    source = tmp_path / "speaking.rst"
    source.write_text(RST, encoding="utf-8")
    output = tmp_path / "draft.jsonl"
    report_path = tmp_path / "report.md"
    report = draft_scripts(source, output, report_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == report["candidates"] == 3
    assert report["status_counts"] == {"blocked_contract_review": 1, "pending_native_approval": 2}
    assert report["bucket_counts"] == {"ordinary_sentences": 2, "targeted_contrasts": 1}
    assert report["blocked_tokens"] == {"DC": 1, "Washington": 1}
    assert report["output_sha256"]
    markdown = report_path.read_text(encoding="utf-8")
    assert "native approval" in markdown.lower()
    assert "not the locked test" in markdown


def test_empty_source_is_an_input_error(tmp_path):
    source = tmp_path / "speaking.rst"
    source.write_text("no links here\n", encoding="utf-8")
    with pytest.raises(InputError, match="no linked sentence candidates"):
        draft_scripts(source, tmp_path / "out.jsonl", tmp_path / "report.md")


def test_repository_speaking_page_yields_a_draft_pool(repo_root):
    candidates = extract_candidates(
        (repo_root / "source" / "speaking.rst").read_text(encoding="utf-8")
    )
    assert len(candidates) > 50
    statuses = {candidate["status"] for candidate in candidates}
    assert statuses == {"pending_native_approval", "blocked_contract_review"}
    buckets = {candidate["bucket"] for candidate in candidates}
    assert buckets == {"ordinary_sentences", "targeted_contrasts"}
    for candidate in candidates:
        if candidate["status"] == "pending_native_approval":
            assert candidate["normalized_text"] == candidate["text"].lower().strip()

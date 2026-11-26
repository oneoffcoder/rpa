from __future__ import annotations

import csv
import json

import pytest

from rpa_tts.cli import main
from rpa_tts.data.book_scripts import (
    REVIEW_COLUMNS,
    draft_scripts,
    extract_candidates,
    import_review,
    load_draft_rows,
    write_review_sheet,
)
from rpa_tts.errors import InputError, ValidationError

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


def _draft_file(tmp_path, source=RST):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rst = tmp_path / "speaking.rst"
    rst.write_text(source, encoding="utf-8")
    draft = tmp_path / "draft.jsonl"
    draft_scripts(rst, draft, tmp_path / "draft.md")
    return draft


def test_review_sheet_has_blank_decisions_and_blocked_rows(tmp_path):
    draft = _draft_file(tmp_path)
    result = write_review_sheet(draft, tmp_path / "review.csv", tmp_path / "review.md")
    with (tmp_path / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        header = stream.seek(0) or list(csv.reader(stream))[0]
    assert header == REVIEW_COLUMNS
    assert result["rows"] == 3
    assert result["status_counts"] == {
        "blocked_contract_review": 1,
        "pending_native_approval": 2,
    }
    assert result["blocked_tokens"] == {"DC": 1, "Washington": 1}
    assert all(row["decision"] == "" and row["edited_text"] == "" for row in rows)
    blocked = next(row for row in rows if row["status"] == "blocked_contract_review")
    assert "unknown_word:Washington" in blocked["issues"]
    report = (tmp_path / "review.md").read_text(encoding="utf-8")
    assert "Reviewer instructions" in report
    assert "Washington" in report
    assert "not the locked 300-item test" in report
    assert len(result["csv_sha256"]) == 64


def test_review_sheet_refuses_to_overwrite_without_flag(tmp_path):
    draft = _draft_file(tmp_path)
    write_review_sheet(draft, tmp_path / "review.csv", tmp_path / "review.md")
    with pytest.raises(InputError, match="already exists"):
        write_review_sheet(draft, tmp_path / "review.csv", tmp_path / "review.md")
    result = write_review_sheet(
        draft, tmp_path / "review.csv", tmp_path / "review.md", overwrite=True
    )
    assert result["rows"] == 3


def test_duplicate_script_id_in_draft_is_rejected(tmp_path):
    row = {
        "script_id": "cal-001",
        "text": "kuv hlub koj",
        "status": "pending_native_approval",
        "source": {"page": "source/speaking.rst", "line": 1},
        "bucket": "ordinary_sentences",
        "syllable_count": 3,
        "issues": [],
    }
    path = tmp_path / "draft.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(InputError, match="duplicate script_id"):
        load_draft_rows(path)


def test_blocked_row_without_issues_is_rejected(tmp_path):
    row = {
        "script_id": "cal-001",
        "text": "hello world",
        "status": "blocked_contract_review",
        "source": {"page": "source/speaking.rst", "line": 1},
        "bucket": "ordinary_sentences",
        "syllable_count": 0,
        "issues": [],
    }
    path = tmp_path / "draft.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(InputError, match="must list its contract issues"):
        load_draft_rows(path)


def test_cli_review_sheet(tmp_path, capsys):
    draft = _draft_file(tmp_path)
    code = main(
        [
            "corpus",
            "review-sheet",
            "--draft",
            str(draft),
            "--csv",
            str(tmp_path / "review.csv"),
            "--report",
            str(tmp_path / "review.md"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 3
    assert payload["status_counts"]["blocked_contract_review"] == 1


def test_repository_draft_pool_review_sheet(repo_root, tmp_path):
    draft = repo_root / "docs" / "tts" / "recovery" / "calibration-scripts.draft.jsonl"
    rows = load_draft_rows(draft)
    assert len(rows) == 272
    result = write_review_sheet(draft, tmp_path / "review.csv", tmp_path / "review.md")
    assert result["rows"] == 272
    assert result["status_counts"] == {
        "blocked_contract_review": 2,
        "pending_native_approval": 270,
    }


def _review_file(tmp_path):
    draft = _draft_file(tmp_path)
    write_review_sheet(draft, tmp_path / "review.csv", tmp_path / "review.md")
    return draft, tmp_path / "review.csv"


def _edit_review(path, mutate):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fill(rows):
    for row in rows:
        if row["script_id"] == "cal-001":
            row["decision"] = "accept"
        elif row["script_id"] == "cal-002":
            row["decision"] = "edit"
            row["edited_text"] = "kuv hlub koj"
            row["notes"] = "loanword removed"
        elif row["script_id"] == "cal-003":
            row["decision"] = "reject"
            row["notes"] = "not usable"


def test_import_review_produces_approved_set(tmp_path):
    draft, review = _review_file(tmp_path)
    _edit_review(review, _fill)
    result = import_review(
        draft,
        review,
        tmp_path / "approved.jsonl",
        tmp_path / "approved.md",
        reviewer_id="N-01",
    )
    assert result["counts"] == {"accept": 1, "edit": 1, "reject": 1}
    assert result["approved"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "approved.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["script_id"]: row for row in rows}
    assert by_id["cal-001"]["text"] == "nyob zoo, oo."
    assert by_id["cal-001"]["decision"] == "accept"
    assert by_id["cal-002"]["text"] == "kuv hlub koj"
    assert by_id["cal-002"]["original_text"] == "Kuv nyob Washington, DC."
    assert by_id["cal-002"]["decision"] == "edit"
    assert "cal-003" not in by_id
    assert all(row["reviewer_id"] == "N-01" for row in rows)
    report = (tmp_path / "approved.md").read_text(encoding="utf-8")
    assert "N-01" in report
    assert "loanword removed" in report
    assert "does not record consent" in report


def test_import_review_requires_every_row_decided(tmp_path):
    draft, review = _review_file(tmp_path)
    _edit_review(review, lambda rows: _fill(rows[1:]))
    with pytest.raises(ValidationError, match="undecided"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_import_review_requires_edited_text_for_edit(tmp_path):
    draft, review = _review_file(tmp_path)

    def mutate(rows):
        _fill(rows)
        for row in rows:
            if row["script_id"] == "cal-002":
                row["edited_text"] = ""

    _edit_review(review, mutate)
    with pytest.raises(ValidationError, match="no edited_text"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_import_review_rejects_invalid_edited_text(tmp_path):
    draft, review = _review_file(tmp_path)

    def mutate(rows):
        _fill(rows)
        for row in rows:
            if row["script_id"] == "cal-002":
                row["edited_text"] = "hello 3"

    _edit_review(review, mutate)
    with pytest.raises(ValidationError, match="not valid"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_import_review_rejects_accepting_a_blocked_row(tmp_path):
    draft, review = _review_file(tmp_path)

    def mutate(rows):
        _fill(rows)
        for row in rows:
            if row["script_id"] == "cal-002":
                row["decision"] = "accept"
                row["edited_text"] = ""

    _edit_review(review, mutate)
    with pytest.raises(ValidationError, match="cannot be accepted unchanged"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_import_review_rejects_unknown_duplicate_or_mismatched_rows(tmp_path):
    draft, review = _review_file(tmp_path)

    def add_unknown(rows):
        _fill(rows)
        rows.append({**rows[0], "script_id": "cal-999"})

    _edit_review(review, add_unknown)
    with pytest.raises(ValidationError, match="unknown script_id"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )

    draft, review = _review_file(tmp_path / "second")
    _edit_review(review, lambda rows: (_fill(rows), rows.append(dict(rows[0]))))
    with pytest.raises(ValidationError, match="duplicate script_id"):
        import_review(
            draft,
            review,
            tmp_path / "second" / "out.jsonl",
            tmp_path / "second" / "out.md",
            reviewer_id="N-01",
        )

    draft, review = _review_file(tmp_path / "third")

    def change_text(rows):
        _fill(rows)
        rows[0]["text"] = "Kuv hlub koj."

    _edit_review(review, change_text)
    with pytest.raises(ValidationError, match="differs from the frozen draft"):
        import_review(
            draft,
            review,
            tmp_path / "third" / "out.jsonl",
            tmp_path / "third" / "out.md",
            reviewer_id="N-01",
        )


def test_import_review_requires_all_draft_rows(tmp_path):
    draft, review = _review_file(tmp_path)

    def drop(rows):
        _fill(rows)
        rows.pop()

    _edit_review(review, drop)
    with pytest.raises(ValidationError, match="missing draft rows"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_import_review_validates_reviewer_id_and_overwrite(tmp_path):
    draft, review = _review_file(tmp_path)
    _edit_review(review, _fill)
    with pytest.raises(InputError, match="reviewer_id"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="bad id!"
        )
    import_review(draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01")
    with pytest.raises(InputError, match="already exists"):
        import_review(
            draft, review, tmp_path / "out.jsonl", tmp_path / "out.md", reviewer_id="N-01"
        )


def test_cli_review_import(tmp_path, capsys):
    draft, review = _review_file(tmp_path)
    _edit_review(review, _fill)
    code = main(
        [
            "corpus",
            "review-import",
            "--draft",
            str(draft),
            "--review",
            str(review),
            "--output-jsonl",
            str(tmp_path / "approved.jsonl"),
            "--output-report",
            str(tmp_path / "approved.md"),
            "--reviewer-id",
            "N-01",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["approved"] == 2
    assert payload["counts"]["reject"] == 1

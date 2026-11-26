from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.errors import InputError, ValidationError
from rpa_tts.evaluation.gate import evaluate_release
from rpa_tts.evaluation.metrics import contrast_summary
from rpa_tts.evaluation.review import apply_adjudications, apply_paragraph_reviews


def _scores() -> dict:
    return {
        "pack_id": "review-test",
        "contrasts": [
            {
                "code": "C1",
                "utterance_id": "s1",
                "listener_id": "L-01",
                "expected": "neeg",
                "chosen": "nees",
                "critical": True,
                "adjudicated": False,
                "confirmed_by": [],
            },
            {
                "code": "C2",
                "utterance_id": "s2",
                "listener_id": "L-02",
                "expected": "neeg",
                "chosen": "nees",
                "critical": True,
                "adjudicated": False,
                "confirmed_by": [],
            },
            {
                "code": "C3",
                "utterance_id": "s3",
                "listener_id": "L-03",
                "expected": "neeg",
                "chosen": "neeg",
                "critical": True,
                "adjudicated": False,
                "confirmed_by": [],
            },
            {
                "code": "C4",
                "utterance_id": "s4",
                "listener_id": "L-01",
                "expected": "neeg",
                "chosen": "nees",
                "critical": False,
                "adjudicated": False,
                "confirmed_by": [],
            },
        ],
        "transcriptions": [],
        "naturalness": [],
    }


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _adjudications(tmp_path: Path, rows: list[list[str]], name: str = "adjudications.csv") -> Path:
    return _write_csv(tmp_path / name, ["code", "verdict", "confirmed_by", "notes"], rows)


def test_confirmed_adjudications_enable_systematic_detection(tmp_path):
    path = _adjudications(
        tmp_path, [["C1", "confirmed", "a1;a2", ""], ["C2", "confirmed", "a1;a2", ""]]
    )
    merged = apply_adjudications(_scores(), path)
    assert [record["adjudicated"] for record in merged["contrasts"][:2]] == [True, True]
    summary = contrast_summary(merged["contrasts"])
    assert len(summary["systematic_critical_errors"]) == 1
    assert summary["systematic_critical_errors"][0]["distinct_utterances"] == 2
    assert summary["blocks_release"] is True
    assert merged["adjudications"]["counts"] == {"confirmed": 2, "rejected": 0, "uncertain": 0}
    assert merged["adjudications"]["unadjudicated_wrong_critical"] == 0


def test_unreviewed_wrong_critical_error_is_reported(tmp_path):
    path = _adjudications(tmp_path, [["C1", "confirmed", "a1;a2", ""]])
    merged = apply_adjudications(_scores(), path)
    assert merged["adjudications"]["unadjudicated_wrong_critical"] == 1


def test_rejected_and_uncertain_keep_the_raw_choice(tmp_path):
    path = _adjudications(
        tmp_path,
        [["C1", "rejected", "", "not an error"], ["C2", "uncertain", "", "unclear"]],
    )
    merged = apply_adjudications(_scores(), path)
    first, second = merged["contrasts"][0], merged["contrasts"][1]
    assert first["chosen"] == "nees" and second["chosen"] == "nees"
    assert first["adjudicated"] is False and second["adjudicated"] is False
    assert first["adjudication"]["verdict"] == "rejected"
    assert second["adjudication"]["verdict"] == "uncertain"
    assert merged["adjudications"]["counts"] == {"confirmed": 0, "rejected": 1, "uncertain": 1}


@pytest.mark.parametrize(
    "rows, message",
    [
        ([["ZZ", "confirmed", "a1;a2", ""]], "unknown contrast code"),
        (
            [["C1", "confirmed", "a1;a2", ""], ["C1", "confirmed", "a1;a2", ""]],
            "duplicate adjudication code",
        ),
        ([["C1", "maybe", "a1;a2", ""]], "not confirmed/rejected/uncertain"),
        ([["C3", "confirmed", "a1;a2", ""]], "correct/no-response"),
        ([["C4", "confirmed", "a1;a2", ""]], "non-critical item"),
        ([["C1", "confirmed", "a1", ""]], "at least two distinct adjudicators"),
        ([["C1", "confirmed", "a1;bad id!", ""]], "unsafe adjudicator id"),
        ([["C1", "rejected", "a1", ""]], "must not list confirming adjudicators"),
    ],
)
def test_invalid_adjudications_are_rejected(tmp_path, rows, message):
    path = _adjudications(tmp_path, rows)
    with pytest.raises(ValidationError, match=message):
        apply_adjudications(_scores(), path)


def test_empty_adjudication_sheet_is_rejected(tmp_path):
    path = _adjudications(tmp_path, [])
    with pytest.raises(InputError, match="no rows"):
        apply_adjudications(_scores(), path)


def test_missing_adjudication_column_is_rejected(tmp_path):
    path = _write_csv(tmp_path / "bad.csv", ["code", "verdict"], [["C1", "confirmed"]])
    with pytest.raises(InputError, match="missing columns"):
        apply_adjudications(_scores(), path)


def _paragraph_sheet(tmp_path: Path, rows: list[list[str]], name: str = "paragraphs.csv") -> Path:
    return _write_csv(
        tmp_path / name,
        ["script_id", "missing_clauses", "repeated_clauses", "language_switch", "notes"],
        rows,
    )


def test_paragraph_reviews_merge_counts(tmp_path):
    path = _paragraph_sheet(
        tmp_path,
        [
            ["p1", "1", "0", "no", ""],
            ["p2", "0", "1", "yes", "switched to English"],
        ],
    )
    merged = apply_paragraph_reviews(_scores(), path)
    paragraphs = merged["paragraphs"]
    assert paragraphs["items"] == 2
    assert paragraphs["missing_clauses"] == 1
    assert paragraphs["repeated_clauses"] == 1
    assert paragraphs["language_switch_items"] == 1
    verdict = evaluate_release(
        thresholds={
            "pooled_syllable_error_max": 0.05,
            "bucket_syllable_error_max": 0.10,
            "contrast_accuracy_min": 0.95,
            "naturalness_mean_min": 4.0,
            "naturalness_lower_bound_min": 3.8,
            "naturalness_gap_max": 0.5,
        },
        syllable={"totals": {"error_rate": 0.01, "reference_syllables": 1000}, "by_bucket": {}},
        contrast={"accuracy": 1.0, "judgments": 10, "systematic_critical_errors": []},
        naturalness={"count": 10, "mean": 4.5, "lower": 4.2},
        natural={"mean": 4.7},
        paragraphs=merged["paragraphs"],
        catastrophic_outputs=0,
        panel_acceptance=True,
        required_buckets=[],
    )
    assert verdict["verdict"] == "rejected"
    assert "paragraph_clauses" in verdict["failed_rules"]


def test_paragraph_review_can_be_validated_against_a_pack(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack-manifest.json").write_text(
        json.dumps({"items": [{"script_id": "p1", "kind": "paragraph"}]}), encoding="utf-8"
    )
    path = _paragraph_sheet(tmp_path, [["p2", "0", "0", "no", ""]])
    with pytest.raises(ValidationError, match="not a paragraph item"):
        apply_paragraph_reviews(_scores(), path, pack_dir=pack)


def test_pack_without_paragraph_items_is_rejected(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pack-manifest.json").write_text(
        json.dumps({"items": [{"script_id": "s1", "kind": "sentence"}]}), encoding="utf-8"
    )
    path = _paragraph_sheet(tmp_path, [["p1", "0", "0", "no", ""]])
    with pytest.raises(InputError, match="no paragraph items"):
        apply_paragraph_reviews(_scores(), path, pack_dir=pack)


@pytest.mark.parametrize(
    "row, message",
    [
        (["p1", "-1", "0", "no", ""], "non-negative"),
        (["p1", "x", "0", "no", ""], "must be integers"),
        (["p1", "0", "0", "maybe", ""], "boolean-like"),
        (["p1", "0", "0", "no", ""], "duplicate paragraph id"),
    ],
)
def test_invalid_paragraph_reviews_are_rejected(tmp_path, row, message):
    rows = (
        [row, ["p1", "0", "0", "no", ""]]
        if row[0] == "p1" and message != "duplicate paragraph id"
        else [row]
    )
    if message == "duplicate paragraph id":
        rows = [["p1", "0", "0", "no", ""], ["p1", "1", "0", "no", ""]]
    path = _paragraph_sheet(tmp_path, rows)
    with pytest.raises(ValidationError, match=message):
        apply_paragraph_reviews(_scores(), path)


def test_cli_adjudicate_and_paragraphs(tmp_path, capsys):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_scores()), encoding="utf-8")
    adjudications = _adjudications(
        tmp_path, [["C1", "confirmed", "a1;a2", ""], ["C2", "confirmed", "a1;a2", ""]]
    )
    merged_path = tmp_path / "merged.json"
    code = main(
        [
            "evaluate",
            "adjudicate",
            "--scores",
            str(scores_path),
            "--adjudications",
            str(adjudications),
            "--output",
            str(merged_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["confirmed"] == 2

    paragraphs = _paragraph_sheet(tmp_path, [["p1", "0", "0", "no", ""]])
    final_path = tmp_path / "final.json"
    code = main(
        [
            "evaluate",
            "paragraphs",
            "--scores",
            str(merged_path),
            "--reviews",
            str(paragraphs),
            "--output",
            str(final_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"] == 1
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["adjudications"]["counts"]["confirmed"] == 2
    assert final["paragraphs"]["items"] == 1

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rpa_tts.errors import InputError, ValidationError
from rpa_tts.evaluation.metrics import contrast_summary, naturalness_summary, score_transcriptions
from rpa_tts.evaluation.pack import (
    PHASE_A_COLUMNS,
    PHASE_B_COLUMNS,
    build_pack,
    load_definition,
    load_responses,
    stage_audio,
)


def _definition_file(tmp_path: Path, definition: dict) -> Path:
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


def _build(tmp_path: Path, definition: dict, name: str = "pack"):
    loaded = load_definition(_definition_file(tmp_path, definition))
    result = build_pack(loaded, tmp_path / name)
    return loaded, result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_pack_assignments_are_balanced_and_blind(tmp_path, pack_definition):
    _, result = _build(tmp_path, pack_definition())
    manifest = result["manifest"]
    summary = result["summary"]
    assert len(manifest["assignments"]) == 9
    assert summary["warnings"] == []
    for counts in summary["per_listener"].values():
        assert counts == {"baseline": 1, "candidate": 1, "natural": 1}
    for counts in summary["per_script"].values():
        assert counts == {"baseline": 1, "candidate": 1, "natural": 1}
    codes = [entry["code"] for entry in manifest["assignments"]]
    assert len(set(codes)) == 9
    assert all(len(code) == 12 and code.isupper() for code in codes)


def test_listener_sheets_do_not_reveal_condition_or_expected(tmp_path, pack_definition):
    _, result = _build(tmp_path, pack_definition())
    pack = tmp_path / "pack"
    for sheet in sorted((pack / "listener-sheets").glob("L-*-phase-a.csv")):
        header = sheet.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == PHASE_A_COLUMNS
        assert "condition" not in header
        assert "source_audio" not in header
        rows = _read_csv(sheet)
        assert len(rows) == 3
        assert all(row["audio"].startswith("audio/") for row in rows)
    for sheet in sorted((pack / "listener-sheets").glob("L-*-phase-b.csv")):
        header = sheet.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == PHASE_B_COLUMNS
        assert "expected" not in header
        rows = _read_csv(sheet)
        assert len(rows) == 1
        assert {rows[0]["option_a"], rows[0]["option_b"]} == {"neeg", "nees"}
    keys = _read_csv(pack / "listener-sheets" / "L-01-key.csv")
    assert {row["condition"] for row in keys} == {"candidate", "natural", "baseline"}
    coordinator = _read_csv(pack / "coordinator" / "assignments.csv")
    assert len(coordinator) == 9


def test_pack_is_reproducible_for_a_seed_and_changes_with_another(tmp_path, pack_definition):
    _, first = _build(tmp_path, pack_definition(), "pack-a")
    _, second = _build(tmp_path, pack_definition(), "pack-b")
    assert first["manifest"]["assignments"] == second["manifest"]["assignments"]
    assert first["manifest"]["contrast_assignments"] == second["manifest"]["contrast_assignments"]
    _, other = _build(tmp_path, pack_definition(seed=99), "pack-c")
    assert [a["code"] for a in other["manifest"]["assignments"]] != [
        a["code"] for a in first["manifest"]["assignments"]
    ]


def test_pack_refuses_to_overwrite_a_nonempty_output(tmp_path, pack_definition):
    _, _ = _build(tmp_path, pack_definition())
    with pytest.raises(InputError, match="already exists"):
        _build(tmp_path, pack_definition())


def test_audio_staging_is_blind_and_hashes_sources(tmp_path, pack_definition):
    _, _ = _build(tmp_path, pack_definition())
    root = tmp_path / "source"
    for condition in ("candidate", "natural", "baseline"):
        directory = root / "audio" / condition
        directory.mkdir(parents=True)
        for script in ("s001", "s002", "s003"):
            (directory / f"{script}.wav").write_bytes(f"{condition}-{script}".encode())
    staged = stage_audio(tmp_path / "pack", root)
    manifest = staged["manifest"]
    assert manifest["audio_staged"] is True
    assert len(manifest["staged_audio"]) == 9
    assert len(manifest["staged_audio_sha256"]) == 64
    for record in manifest["staged_audio"]:
        target = tmp_path / "pack" / record["blind_audio"]
        assert target.is_file()
        assert target.read_bytes() in {
            b"candidate-s001",
            b"natural-s001",
            b"baseline-s001",
            b"candidate-s002",
            b"natural-s002",
            b"baseline-s002",
            b"candidate-s003",
            b"natural-s003",
            b"baseline-s003",
        }
    with pytest.raises(InputError, match="already exists"):
        stage_audio(tmp_path / "pack", root)


def test_ingest_produces_scores_for_metrics_and_gate(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    scores = load_responses(tmp_path / "pack", tmp_path / "responses")
    assert len(scores["transcriptions"]) == 9
    assert len(scores["naturalness"]) == 9
    assert len(scores["contrasts"]) == 3
    assert scores["catastrophic_outputs"] == 0
    candidate = naturalness_summary(scores["naturalness"], condition="candidate")
    assert candidate["count"] == 3
    assert candidate["mean"] == 4.0
    syllable = score_transcriptions(scores["transcriptions"])
    assert syllable["pooled_error_rate"] == 0.0
    contrast = contrast_summary(scores["contrasts"])
    assert contrast["accuracy"] == 1.0
    assert contrast["blocks_release"] is False


def test_empty_transcription_remains_a_catastrophic_failure(
    tmp_path, pack_definition, write_responses
):
    _, result = _build(tmp_path, pack_definition())
    code = result["manifest"]["assignments"][0]["code"]
    write_responses(tmp_path / "pack", tmp_path / "responses", empty_codes={code})
    scores = load_responses(tmp_path / "pack", tmp_path / "responses")
    assert scores["catastrophic_outputs"] == 1
    syllable = score_transcriptions(scores["transcriptions"])
    assert syllable["totals"]["empty_outputs"] == 1
    assert syllable["totals"]["deletions"] > 0
    assert syllable["pooled_error_rate"] > 0


def _mutate_sheet(path: Path, mutate) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0].keys())
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_missing_response_row_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(tmp_path / "responses" / "L-01-phase-a.csv", lambda rows: rows.pop())
    with pytest.raises(ValidationError, match="missing phase-A codes"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_duplicate_response_code_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(
        tmp_path / "responses" / "L-01-phase-a.csv", lambda rows: rows.append(dict(rows[0]))
    )
    with pytest.raises(ValidationError, match="duplicate code"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_out_of_range_naturalness_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(
        tmp_path / "responses" / "L-01-phase-a.csv",
        lambda rows: rows[0].update(naturalness="9"),
    )
    with pytest.raises(ValidationError, match="outside 1-5"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_replay_cap_is_enforced(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(
        tmp_path / "responses" / "L-01-phase-a.csv",
        lambda rows: rows[0].update(replay_count="3"),
    )
    with pytest.raises(ValidationError, match="exceeds cap"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_unknown_code_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(
        tmp_path / "responses" / "L-01-phase-a.csv",
        lambda rows: rows[0].update(code="DEADBEEF0000"),
    )
    with pytest.raises(ValidationError, match="unknown code"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_missing_phase_b_file_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    (tmp_path / "responses" / "L-01-phase-b.csv").unlink()
    with pytest.raises(InputError, match="phase-b"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_invalid_contrast_choice_is_rejected(tmp_path, pack_definition, write_responses):
    _, _ = _build(tmp_path, pack_definition())
    write_responses(tmp_path / "pack", tmp_path / "responses")
    _mutate_sheet(
        tmp_path / "responses" / "L-01-phase-b.csv",
        lambda rows: rows[0].update(chosen="blorp"),
    )
    with pytest.raises(ValidationError, match="not one of"):
        load_responses(tmp_path / "pack", tmp_path / "responses")


def test_qualification_pack_rejects_calibration_bucket(tmp_path, pack_definition):
    definition = pack_definition(design="qualification")
    definition["items"][0]["bucket"] = "calibration"
    with pytest.raises(ValidationError, match="calibration bucket"):
        load_definition(_definition_file(tmp_path, definition))


def test_qualification_pack_requires_a_critical_contrast(tmp_path, pack_definition):
    definition = pack_definition(design="qualification")
    definition["items"] = [item for item in definition["items"] if not item.get("critical")]
    with pytest.raises(ValidationError, match="critical contrast"):
        load_definition(_definition_file(tmp_path, definition))


def test_item_text_must_be_valid_rpa(tmp_path, pack_definition):
    definition = pack_definition()
    definition["items"][0]["text"] = "hello world"
    with pytest.raises(ValidationError, match="outside the RPA contract"):
        load_definition(_definition_file(tmp_path, definition))


def test_audio_must_cover_every_condition(tmp_path, pack_definition):
    definition = pack_definition()
    del definition["items"][0]["condition_audio"]["baseline"]
    with pytest.raises(ValidationError, match="must exactly match conditions"):
        load_definition(_definition_file(tmp_path, definition))


def test_repository_template_definition_is_valid(repo_root):
    loaded = load_definition(
        repo_root / "docs" / "tts" / "recovery" / "listening-pack.template.json"
    )
    assert loaded["design"] == "calibration"
    assert len(loaded["listeners"]) == 3
    assert sum(1 for item in loaded["items"] if item.get("critical")) == 1


def _qualification_definition(
    *,
    ordinary: int = 100,
    targeted: int = 80,
    questions: int = 60,
    continuous: int = 60,
    paragraphs: int = 20,
    listeners: int = 12,
    critical: int = 40,
    paragraph_duration: float = 45.0,
    conditions: list[str] | None = None,
    drop_contrast_at: int | None = None,
):
    conditions = conditions or ["candidate", "natural", "baseline"]
    items = []

    def add(prefix: str, count: int, bucket: str, text: str):
        for index in range(count):
            script_id = f"{prefix}{index:03d}"
            items.append(
                {
                    "script_id": script_id,
                    "bucket": bucket,
                    "text": text,
                    "condition_audio": {
                        condition: f"audio/{condition}/{script_id}.wav" for condition in conditions
                    },
                }
            )

    add("o", ordinary, "ordinary_sentences", "kuv hlub koj")
    for index in range(targeted):
        script_id = f"t{index:03d}"
        extra: dict = {
            "critical": index < critical,
            "contrast": {"options": ["neeg", "nees"], "expected": "neeg"},
        }
        if drop_contrast_at is not None and index == drop_contrast_at:
            extra.pop("contrast")
            extra["critical"] = False
        items.append(
            {
                "script_id": script_id,
                "bucket": "targeted_contrasts",
                "text": "nws yog ib tug neeg",
                "condition_audio": {
                    condition: f"audio/{condition}/{script_id}.wav" for condition in conditions
                },
                **extra,
            }
        )
    add("q", questions, "questions_numbers_names", "koj puas nyob zoo")
    add("c", continuous, "continuous_passages", "peb yuav mus tsev")
    for index in range(paragraphs):
        script_id = f"p{index:03d}"
        items.append(
            {
                "script_id": script_id,
                "bucket": "continuous_passages",
                "kind": "paragraph",
                "duration_seconds": paragraph_duration,
                "text": "kuv hlub koj",
                "condition_audio": {
                    condition: f"audio/{condition}/{script_id}.wav" for condition in conditions
                },
            }
        )
    return {
        "schema_version": 1,
        "pack_id": "qualification-test-01",
        "design": "qualification",
        "seed": 3407,
        "conditions": conditions,
        "items": items,
        "listeners": [
            {
                "listener_id": f"L-{index:02d}",
                "roles": ["transcription", "naturalness", "contrast"],
            }
            for index in range(1, listeners + 1)
        ],
    }


def test_complete_qualification_pack_builds(tmp_path):
    _, result = _build(tmp_path, _qualification_definition())
    manifest = result["manifest"]
    check = manifest["design_check"]
    assert check["applicable"] is True
    assert check["passed"] is True
    assert check["violations"] == []
    assert check["counts"]["sentences"] == 300
    assert check["counts"]["paragraphs"] == 20
    assert check["counts"]["listeners"] == 12
    assert check["counts"]["by_bucket"] == {
        "continuous_passages": 60,
        "ordinary_sentences": 100,
        "questions_numbers_names": 60,
        "targeted_contrasts": 80,
    }
    assert len(manifest["assignments"]) == 320 * 12
    assert result["summary"]["design_check"]["passed"] is True


def test_calibration_pack_is_exempt_from_qualification_composition(tmp_path, pack_definition):
    _, result = _build(tmp_path, pack_definition())
    assert result["manifest"]["design_check"]["applicable"] is False
    assert result["summary"]["design_check"]["passed"] is True


def test_qualification_requires_exact_sentence_total(tmp_path):
    with pytest.raises(ValidationError, match="sentences: 299"):
        _build(tmp_path, _qualification_definition(ordinary=99))


def test_qualification_requires_exact_bucket_counts(tmp_path):
    with pytest.raises(ValidationError, match="bucket targeted_contrasts: 79 != 80"):
        _build(tmp_path, _qualification_definition(targeted=79))


def test_qualification_requires_all_twenty_paragraphs(tmp_path):
    with pytest.raises(ValidationError, match="paragraphs: 19 != 20"):
        _build(tmp_path, _qualification_definition(paragraphs=19))


def test_qualification_requires_paragraph_duration_range(tmp_path):
    with pytest.raises(ValidationError, match="outside 30.0-90.0s"):
        _build(tmp_path, _qualification_definition(paragraph_duration=20.0))


def test_qualification_requires_twelve_listeners(tmp_path):
    with pytest.raises(ValidationError, match="listeners: 11 < 12"):
        _build(tmp_path, _qualification_definition(listeners=11))


def test_qualification_requires_candidate_condition(tmp_path):
    with pytest.raises(ValidationError, match="must include 'candidate'"):
        _build(
            tmp_path,
            _qualification_definition(conditions=["natural", "baseline", "other"]),
        )


def test_qualification_targeted_items_need_contrast(tmp_path):
    with pytest.raises(ValidationError, match="without contrast options"):
        _build(tmp_path, _qualification_definition(drop_contrast_at=5))

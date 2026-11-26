from __future__ import annotations

import json

import pytest

from rpa_tts.cli import main

BUCKETS = [
    "ordinary_sentences",
    "targeted_contrasts",
    "questions_numbers_names",
    "continuous_passages",
]


def _scores(*, panel=True, catastrophic=0):
    transcriptions = [
        {
            "utterance_id": f"u{index}",
            "listener_id": "l1",
            "bucket": bucket,
            "reference": "kuv hlub koj",
            "hypothesis": "kuv hlub koj",
        }
        for index, bucket in enumerate(BUCKETS)
    ]
    contrasts = [
        {
            "utterance_id": f"c{index}",
            "listener_id": f"l{index % 4}",
            "expected": "hlub",
            "chosen": "hlub",
        }
        for index in range(20)
    ]
    naturalness = [
        {
            "utterance_id": f"n{utterance}",
            "listener_id": f"l{listener}",
            "bucket": BUCKETS[utterance % len(BUCKETS)],
            "condition": "candidate",
            "score": 4 if (utterance + listener) % 2 else 5,
        }
        for utterance in range(6)
        for listener in range(4)
    ]
    natural = [
        {
            "utterance_id": row["utterance_id"],
            "listener_id": row["listener_id"],
            "bucket": row["bucket"],
            "condition": "natural",
            "score": 5,
        }
        for row in naturalness
    ]
    return {
        "transcriptions": transcriptions,
        "contrasts": contrasts,
        "naturalness": naturalness,
        "natural": natural,
        "paragraphs": {"items": 20, "missing_clauses": 0, "repeated_clauses": 0},
        "catastrophic_outputs": catastrophic,
        "panel_acceptance": panel,
    }


def test_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0


def test_text_normalize_reports_contract_state(capsys):
    assert main(["text", "normalize", "--text", "Kuv Hlub Koj"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["normalized"] == "kuv hlub koj"
    assert payload["valid"] is True

    assert main(["text", "normalize", "--text", "hello 3"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert {issue["code"] for issue in payload["issues"]} == {"unknown_word", "unapproved_digits"}


def test_text_normalize_requires_exactly_one_input(capsys):
    assert main(["text", "normalize"]) == 2
    assert "exactly one" in capsys.readouterr().err


def test_environment_check_writes_inventory(tmp_path, config_path, capsys):
    output = tmp_path / "environment.json"
    code = main(["environment", "check", "--output", str(output), "--config", str(config_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["environment"]["hostname"]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_text_validate_prints_problems(tmp_path, valid_rows, write_manifest_file, capsys):
    valid_rows[0]["rights_id"] = "UNRESOLVED"
    path = write_manifest_file(valid_rows)
    assert main(["text", "validate", "--manifest", str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"]
    assert payload["summary"]["rows"] == len(valid_rows)

    assert main(["text", "validate", "--manifest", str(path), "--purpose", "audit"]) == 0


def test_model_check_reports_not_ready(tmp_path, config_path, capsys):
    assert main(["model", "check", "--config", str(config_path), "--repo-root", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_lanes"] == []
    assert payload["lanes"]["magpie"]["ready"] is False


def test_synthesis_is_gated_before_any_audio(tmp_path, config_path, capsys):
    code = main(
        [
            "synthesize",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--text",
            "kuv hlub koj",
            "--output",
            str(tmp_path / "out.wav"),
            "--config",
            str(config_path),
        ]
    )
    assert code == 3
    assert "F-07" in capsys.readouterr().err
    assert not (tmp_path / "out.wav").exists()


def test_training_is_gated(tmp_path, config_path, write_manifest_file, valid_rows, capsys):
    path = write_manifest_file(valid_rows)
    code = main(
        [
            "train",
            "run",
            "--config",
            str(config_path),
            "--manifest",
            str(path),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert code == 3
    assert "F-04" in capsys.readouterr().err


def test_corpus_split_writes_manifest_and_report(
    tmp_path, write_manifest_file, rows_factory, capsys
):
    rows = rows_factory(12)
    path = write_manifest_file(rows)
    output = tmp_path / "split.jsonl"
    code = main(
        ["corpus", "split", "--manifest", str(path), "--output", str(output), "--seed", "42"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["split"]["groups"] == 12
    assert output.is_file()
    assert (tmp_path / "split.report.json").is_file()


def test_evaluate_metrics_and_gate_accept_a_passing_score_set(tmp_path, config_path, capsys):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_scores()), encoding="utf-8")
    metrics_output = tmp_path / "metrics.json"
    assert (
        main(
            [
                "evaluate",
                "metrics",
                "--scores",
                str(scores_path),
                "--output",
                str(metrics_output),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["syllable"]["pooled_error_rate"] == 0.0
    assert metrics["contrast"]["accuracy"] == 1.0
    assert metrics["naturalness"]["interval"]["lower"] >= 3.8

    gate_output = tmp_path / "gate.json"
    assert (
        main(
            [
                "evaluate",
                "gate",
                "--scores",
                str(scores_path),
                "--output",
                str(gate_output),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    gate = json.loads(capsys.readouterr().out)
    assert gate["verdict"]["verdict"] == "accepted"


def test_evaluate_gate_rejects_panel_refusal(tmp_path, config_path, capsys):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_scores(panel=False)), encoding="utf-8")
    code = main(
        [
            "evaluate",
            "gate",
            "--scores",
            str(scores_path),
            "--output",
            str(tmp_path / "gate.json"),
            "--config",
            str(config_path),
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"]["verdict"] == "rejected"
    assert "panel_rejected" in payload["verdict"]["failed_rules"]


def test_evidence_add_appends_one_entry(tmp_path, capsys):
    index = tmp_path / "index.json"
    code = main(
        [
            "evidence",
            "add",
            "--gate",
            "F-01",
            "--claim",
            "workspace created",
            "--status",
            "in_progress",
            "--index",
            str(index),
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"] == 1
    assert json.loads(index.read_text(encoding="utf-8"))["entries"][0]["gate"] == "F-01"


def test_train_plan_requires_an_explicit_stage_decision(
    tmp_path, config_path, write_manifest_file, valid_rows, capsys
):
    path = write_manifest_file(valid_rows)
    assert (
        main(
            [
                "train",
                "plan",
                "--config",
                str(config_path),
                "--manifest",
                str(path),
                "--output",
                str(tmp_path / "plan"),
                "--objective",
                "2-hour adaptation",
                "--baseline",
                "pilot",
                "--max-gpu-hours",
                "4",
                "--measurement-set",
                "40 fresh development utterances",
                "--inference-policy",
                "frozen sampling",
                "--raters",
                "6 eligible listeners",
                "--next-increment",
                "5 hours if improvement",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage_decision"]["max_gpu_hours"] == 4
    assert payload["blockers"]


def test_evaluate_pack_and_ingest_round_trip(
    tmp_path, pack_definition, write_responses, config_path, capsys
):
    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(pack_definition()), encoding="utf-8")
    pack_dir = tmp_path / "pack"
    assert (
        main(["evaluate", "pack", "--definition", str(definition_path), "--output", str(pack_dir)])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["assignments"] == 9
    assert payload["audio_staged"] is False

    write_responses(pack_dir, tmp_path / "responses")
    scores_path = tmp_path / "scores.json"
    assert (
        main(
            [
                "evaluate",
                "ingest",
                "--pack",
                str(pack_dir),
                "--responses",
                str(tmp_path / "responses"),
                "--output",
                str(scores_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcriptions"] == 9
    assert payload["catastrophic_outputs"] == 0

    assert (
        main(
            [
                "evaluate",
                "metrics",
                "--scores",
                str(scores_path),
                "--output",
                str(tmp_path / "metrics.json"),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["syllable"]["pooled_error_rate"] == 0.0


def test_run_manifest_validate_rejects_garbage(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")
    assert main(["run-manifest", "validate", "--path", str(path)]) == 2

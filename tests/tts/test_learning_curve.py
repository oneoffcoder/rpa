from __future__ import annotations

import csv
import json

import pytest

from rpa_tts.cli import main
from rpa_tts.config import load_config
from rpa_tts.errors import InputError, ValidationError
from rpa_tts.schemas import LEARNING_CURVE_COLUMNS
from rpa_tts.training.learning_curve import (
    append_row,
    load_curve,
    row_from_run_manifest,
    validate_row,
)
from rpa_tts.training.run_manifest import build_run_manifest, write_run_manifest

ENVIRONMENT = {
    "hostname": "testhost",
    "platform": {"system": "Windows", "release": "11"},
    "python": "3.12.16",
    "packages": {"torch": "2.9.0", "torch_cuda_runtime": "12.8"},
    "gpu": {
        "name": "NVIDIA GeForce RTX 5090",
        "memory_total_mb": 32607,
        "driver_version": "621.35",
    },
}


def _row(**overrides):
    row = {
        "run_id": "f06-20261126t090000z-abc123",
        "stage": "F-06",
        "dataset_version": "pilot-a",
        "accepted_training_hours": 1.5,
        "optimizer_updates": 1200,
        "dev_syllable_error": 0.08,
        "dev_naturalness_mean": 3.6,
        "decision": "hold",
        "evidence_id": "ev_0123456789abcdef",
    }
    row.update(overrides)
    return row


def test_append_writes_header_and_validates(tmp_path):
    curve = tmp_path / "learning-curve.csv"
    result = append_row(curve, _row())
    assert result["rows"] == 1
    header = curve.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == LEARNING_CURVE_COLUMNS
    rows = load_curve(curve)
    assert rows[0]["run_id"] == "f06-20261126t090000z-abc123"
    append_row(curve, _row(run_id="f06-20261126t090500z-def456"))
    assert len(load_curve(curve)) == 2


def test_duplicate_run_id_is_rejected(tmp_path):
    curve = tmp_path / "learning-curve.csv"
    append_row(curve, _row())
    with pytest.raises(ValidationError, match="already has a learning-curve row"):
        append_row(curve, _row())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"stage": "F-04"}, "stage must be F-05"),
        ({"decision": "maybe"}, "decision must be one of"),
        ({"evidence_id": "nope"}, "invalid evidence_id"),
        ({"run_id": "Bad Run!"}, "invalid run_id"),
        ({"dev_syllable_error": 1.2}, "dev_syllable_error"),
        ({"dev_naturalness_mean": 0.5}, "dev_naturalness_mean"),
        ({"accepted_training_hours": -1}, "accepted_training_hours"),
        ({"optimizer_updates": 1.5}, "must be an integer"),
        ({"dataset_version": ""}, "dataset_version"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, overrides, message):
    with pytest.raises(ValidationError, match=message):
        validate_row(_row(**overrides))


def test_missing_column_is_rejected():
    row = _row()
    del row["decision"]
    with pytest.raises(ValidationError, match="missing columns"):
        validate_row(row)


def test_header_mismatch_is_rejected(tmp_path):
    curve = tmp_path / "learning-curve.csv"
    curve.write_text("run_id,stage\nx,y\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="header must be"):
        load_curve(curve)


def test_row_from_planned_run_is_rejected():
    with pytest.raises(InputError, match="completed/failed/aborted"):
        row_from_run_manifest(
            {"status": "planned", "run_id": "x", "stage": "F-06"},
            dataset_version="pilot-a",
            dev_syllable_error=0.1,
            dev_naturalness_mean=3.0,
            decision="reject",
            evidence_id="ev_0123456789abcdef",
        )


def test_row_from_completed_manifest_uses_run_fields():
    manifest = {
        "status": "completed",
        "run_id": "f06-20261126t090000z-abc123",
        "stage": "F-06",
        "dataset": {"unique_audio_hours": 2.0},
        "training": {"optimizer_updates": 2500},
    }
    row = row_from_run_manifest(
        manifest,
        dataset_version="pilot-a",
        dev_syllable_error=0.07,
        dev_naturalness_mean=3.9,
        decision="promote",
        evidence_id="ev_0123456789abcdef",
    )
    assert row["accepted_training_hours"] == 2.0
    assert row["optimizer_updates"] == 2500
    assert row["decision"] == "promote"


def test_cli_train_record_appends_a_row(
    tmp_path, config_path, write_manifest_file, rows_factory, capsys
):
    rows = rows_factory(4)
    dataset_path = write_manifest_file(rows)
    config = load_config(config_path)
    manifest = build_run_manifest(
        config=config,
        run_id="f06-20261126t090000z-abc123",
        stage="F-06",
        command="rpa-tts train run",
        dataset_manifest=dataset_path,
        rows=rows,
        environment=ENVIRONMENT,
        training={"optimizer_updates": 800},
        status="completed",
    )
    run_path = write_run_manifest(manifest, tmp_path / "run-manifest.json")
    curve = tmp_path / "learning-curve.csv"
    code = main(
        [
            "train",
            "record",
            "--run-manifest",
            str(run_path),
            "--dataset-version",
            "pilot-a",
            "--dev-syllable-error",
            "0.06",
            "--dev-naturalness",
            "3.9",
            "--decision",
            "promote",
            "--evidence-id",
            "ev_0123456789abcdef",
            "--curve",
            str(curve),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    with curve.open(encoding="utf-8", newline="") as stream:
        stored = list(csv.DictReader(stream))
    assert stored[0]["run_id"] == "f06-20261126t090000z-abc123"
    assert stored[0]["optimizer_updates"] == "800"
    assert stored[0]["decision"] == "promote"

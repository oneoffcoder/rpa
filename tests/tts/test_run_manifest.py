from __future__ import annotations

import json

import pytest

from rpa_tts.config import load_config
from rpa_tts.errors import InputError
from rpa_tts.training.run_manifest import (
    build_run_manifest,
    code_state,
    hash_outputs,
    load_run_manifest,
    new_run_id,
    validate_output_hashes,
    write_run_manifest,
)

FAKE_ENVIRONMENT = {
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


def _manifest(config_path, tmp_path, write_manifest_file, rows_factory):
    rows = rows_factory(6)
    path = write_manifest_file(rows)
    return load_config(config_path), rows, path


def test_build_run_manifest_records_hashes_and_hours(
    config_path, tmp_path, write_manifest_file, rows_factory
):
    config, rows, path = _manifest(config_path, tmp_path, write_manifest_file, rows_factory)
    manifest = build_run_manifest(
        config=config,
        run_id=new_run_id("F-06", suffix="abc123"),
        stage="F-06",
        command="rpa-tts train run",
        dataset_manifest=path,
        rows=rows,
        environment=FAKE_ENVIRONMENT,
        training={"effective_batch_size": 4, "gradient_accumulation": 8, "max_steps": 100},
    )
    assert manifest["dataset"]["rows"] == 6
    assert manifest["dataset"]["unique_audio_hours"] == pytest.approx(6 * 2.5 / 3600, abs=1e-6)
    assert manifest["dataset"]["split_rows"] == {"train": 6, "dev": 0, "test": 0}
    assert manifest["training"]["effective_batch_size"] == 4
    assert manifest["training"]["gradient_accumulation"] == 8
    assert manifest["environment"]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert manifest["config"]["sha256"]
    assert len(manifest["dataset"]["manifest_sha256"]) == 64
    assert manifest["code"]["branch"]


def test_run_manifest_round_trip_and_overwrite_refusal(
    config_path, tmp_path, write_manifest_file, rows_factory
):
    config, rows, path = _manifest(config_path, tmp_path, write_manifest_file, rows_factory)
    manifest = build_run_manifest(
        config=config,
        run_id="f06-test-run-0001",
        stage="F-06",
        command="rpa-tts train run",
        dataset_manifest=path,
        rows=rows,
        environment=FAKE_ENVIRONMENT,
    )
    destination = tmp_path / "runs" / "run-manifest.json"
    write_run_manifest(manifest, destination)
    assert load_run_manifest(destination)["run_id"] == "f06-test-run-0001"
    with pytest.raises(InputError, match="already exists"):
        write_run_manifest(manifest, destination)
    write_run_manifest(manifest, destination, overwrite=True)


def test_output_hashes_are_checked(tmp_path, config_path, write_manifest_file, rows_factory):
    config, rows, dataset_path = _manifest(config_path, tmp_path, write_manifest_file, rows_factory)
    output = tmp_path / "checkpoint.bin"
    output.write_bytes(b"checkpoint")
    manifest = build_run_manifest(
        config=config,
        run_id="f06-output-check-0001",
        stage="F-06",
        command="rpa-tts train run",
        dataset_manifest=dataset_path,
        rows=rows,
        environment=FAKE_ENVIRONMENT,
        outputs=hash_outputs([(output, "checkpoint")], tmp_path),
    )
    assert validate_output_hashes(manifest, repo_root=tmp_path) == []
    output.write_bytes(b"tampered")
    problems = validate_output_hashes(manifest, repo_root=tmp_path)
    assert any("hash changed" in problem for problem in problems)
    output.unlink()
    assert any(
        "missing output" in problem
        for problem in validate_output_hashes(manifest, repo_root=tmp_path)
    )


def test_missing_output_file_is_rejected_during_hashing(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        hash_outputs([(tmp_path / "missing.bin", "checkpoint")], tmp_path)


def test_invalid_run_ids_are_rejected(tmp_path, config_path, write_manifest_file, rows_factory):
    config, rows, path = _manifest(config_path, tmp_path, write_manifest_file, rows_factory)
    with pytest.raises(InputError, match="invalid run id"):
        build_run_manifest(
            config=config,
            run_id="Bad Run!",
            stage="F-06",
            command="x",
            dataset_manifest=path,
            rows=rows,
            environment=FAKE_ENVIRONMENT,
        )
    with pytest.raises(InputError, match="schema-safe"):
        new_run_id("F-06", suffix="!")


def test_run_id_format_is_deterministic_with_a_fixed_time():
    import datetime as dt

    moment = dt.datetime(2026, 11, 26, 8, 0, 0, tzinfo=dt.UTC)
    assert new_run_id("F-06", suffix="abc", now=moment) == "f06-20261126t080000z-abc"


def test_code_state_reports_a_commit():
    state = code_state()
    assert state["commit"] and len(state["commit"]) >= 7
    assert isinstance(state["dirty"], bool)


def test_load_run_manifest_rejects_missing_file(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        load_run_manifest(tmp_path / "nope.json")


def test_load_run_manifest_rejects_invalid_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    with pytest.raises(Exception, match="validation failed"):
        load_run_manifest(path)

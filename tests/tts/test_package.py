from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.errors import InputError, ValidationError
from rpa_tts.hashing import sha256_file
from rpa_tts.inference.synthesize import inspect_package, load_model_manifest


def _write_package(tmp_path: Path, **overrides) -> Path:
    directory = tmp_path / "package"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.bin"
    checkpoint.write_bytes(b"accepted-weights")
    config = directory / "config.json"
    config.write_text(json.dumps({"sample_rate": 24000}), encoding="utf-8")
    benchmark_manifest = directory / "benchmark.jsonl"
    benchmark_manifest.write_text('{"item": "frozen"}\n', encoding="utf-8")
    benchmark_scores = directory / "scores.json"
    benchmark_scores.write_text('{"result": "accepted"}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "accepted",
        "voice_id": "voice-white-hmong-neutral-1",
        "model_version": "v0.1.0",
        "normalization_version": "rpa-norm-1",
        "text_contract": "rpa-chars-1",
        "model_sample_rate": 24000,
        "checkpoint_path": "checkpoint.bin",
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": "config.json",
        "config_sha256": sha256_file(config),
        "created_at": "2026-11-26T00:00:00+00:00",
        "run_id": None,
        "license": "MPL-2.0",
        "rights_id": "rights-owned-01",
        "benchmark": {
            "manifest_path": "benchmark.jsonl",
            "manifest_sha256": sha256_file(benchmark_manifest),
            "scores_path": "scores.json",
            "scores_sha256": sha256_file(benchmark_scores),
            "gate_verdict": "accepted",
        },
        "notes": "",
    }
    manifest.update(overrides)
    (directory / "model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def test_accepted_package_inspects_clean(tmp_path):
    directory = _write_package(tmp_path)
    report = inspect_package(directory)
    assert report["ok"] is True
    names = {check["name"] for check in report["checks"]}
    assert {
        "manifest",
        "status_accepted",
        "checkpoint",
        "config",
        "benchmark_verdict",
        "benchmark_manifest",
        "benchmark_scores",
        "rights_resolved",
        "normalization_contract",
    } == names
    assert all(check["status"] == "passed" for check in report["checks"])
    assert report["manifest"]["voice_id"] == "voice-white-hmong-neutral-1"


def test_load_model_manifest_returns_accepted_manifest(tmp_path):
    directory = _write_package(tmp_path)
    manifest = load_model_manifest(directory)
    assert manifest["status"] == "accepted"
    assert manifest["benchmark"]["gate_verdict"] == "accepted"


def test_missing_manifest_reports_failure_and_load_raises(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    report = inspect_package(directory)
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "manifest"
    assert report["manifest"] is None
    with pytest.raises(InputError, match="no model-manifest.json"):
        load_model_manifest(directory)


def test_missing_package_directory_is_an_input_error(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        inspect_package(tmp_path / "nope")


def test_candidate_status_is_not_usable(tmp_path):
    directory = _write_package(tmp_path, status="candidate")
    report = inspect_package(directory)
    assert report["ok"] is False
    assert any(
        check["name"] == "status_accepted" and check["status"] == "failed"
        for check in report["checks"]
    )
    with pytest.raises(InputError, match="status_accepted"):
        load_model_manifest(directory)


def test_tampered_checkpoint_is_detected(tmp_path):
    directory = _write_package(tmp_path)
    (directory / "checkpoint.bin").write_bytes(b"tampered")
    report = inspect_package(directory)
    assert report["ok"] is False
    assert any(
        check["name"] == "checkpoint" and check["status"] == "failed" for check in report["checks"]
    )


def test_missing_config_is_detected(tmp_path):
    directory = _write_package(tmp_path)
    (directory / "config.json").unlink()
    report = inspect_package(directory)
    assert any(
        check["name"] == "config" and check["status"] == "failed" for check in report["checks"]
    )


def test_pending_rights_are_not_packaged(tmp_path):
    directory = _write_package(tmp_path, rights_id="UNRESOLVED")
    report = inspect_package(directory)
    assert report["ok"] is False
    assert any(
        check["name"] == "rights_resolved" and check["status"] == "failed"
        for check in report["checks"]
    )


def test_rejected_benchmark_verdict_blocks_packaging(tmp_path):
    directory = _write_package(tmp_path)
    manifest_path = directory / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["gate_verdict"] = "rejected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = inspect_package(directory)
    assert report["ok"] is False
    assert any(
        check["name"] == "benchmark_verdict" and check["status"] == "failed"
        for check in report["checks"]
    )


def test_missing_benchmark_file_is_detected(tmp_path):
    directory = _write_package(tmp_path)
    (directory / "scores.json").unlink()
    report = inspect_package(directory)
    assert any(
        check["name"] == "benchmark_scores" and check["status"] == "failed"
        for check in report["checks"]
    )


def test_invalid_manifest_schema_is_rejected(tmp_path):
    directory = _write_package(tmp_path)
    manifest_path = directory / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["text_contract"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError, match="validation failed"):
        inspect_package(directory)


def test_cli_package_inspect_exit_codes(tmp_path, capsys):
    directory = _write_package(tmp_path)
    assert main(["package", "inspect", "--checkpoint", str(directory)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True

    (directory / "checkpoint.bin").write_bytes(b"changed")
    assert main(["package", "inspect", "--checkpoint", str(directory)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_synthesis_and_build_remain_gated(tmp_path, config_path, capsys):
    directory = _write_package(tmp_path)
    code = main(
        [
            "synthesize",
            "--checkpoint",
            str(directory),
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

    code = main(
        ["package", "build", "--checkpoint", str(directory), "--output", str(tmp_path / "dist")]
    )
    assert code == 3
    assert "F-08" in capsys.readouterr().err

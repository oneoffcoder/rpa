from __future__ import annotations

import pytest

from rpa_tts.config import load_config
from rpa_tts.errors import GateNotPassedError, InputError, RpaTtsError
from rpa_tts.evidence import append_entry, make_entry
from rpa_tts.training import backend


def test_magpie_lane_reports_missing_artifacts_without_crashing(config_path):
    config = load_config(config_path)
    report = backend.check_lane(config, "magpie")
    assert report["ready"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["codec_sample_rate"]["status"] == "passed"
    assert checks["nemo_commit_immutable"]["status"] == "passed"
    assert checks["model_artifact"]["status"] == "not_started"
    assert "resolve lane checks" in report["next_action"]


def test_vits_lane_is_not_ready_until_the_implementation_is_pinned(config_path):
    config = load_config(config_path)
    report = backend.check_lane(config, "vits")
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["original_commit"]["status"] == "passed"
    assert checks["implementation_pinned"]["status"] == "not_started"
    assert report["ready"] is False


def test_unknown_lane_is_an_input_error(config_path):
    config = load_config(config_path)
    with pytest.raises(InputError, match="unknown model lane"):
        backend.check_lane(config, "tacotron")


def test_training_is_blocked_until_f04_passes(config_path, tmp_path):
    config = load_config(config_path)
    with pytest.raises(GateNotPassedError, match="F-04"):
        backend.require_training_backend(config, repo_root=tmp_path)


def test_passed_gate_without_verified_artifacts_still_blocks(config_path, tmp_path):
    config = load_config(config_path)
    entry = make_entry(
        gate="F-04",
        claim="synthetic test evidence",
        status="passed",
        repo_root=tmp_path,
        evidence_id="ev_0123456789abcdef",
    )
    append_entry(
        entry, tmp_path / "docs" / "tts" / "recovery" / "evidence-index.json", repo_root=tmp_path
    )
    with pytest.raises(RpaTtsError, match="no lane has verified artifacts"):
        backend.require_training_backend(config, repo_root=tmp_path)


def test_training_command_must_be_configured(config_path):
    config = load_config(config_path)
    with pytest.raises(InputError, match="no configured training wrapper"):
        backend.training_command(config, "magpie")


def test_configured_training_command_is_returned(config_path):
    config = load_config(config_path)
    config.data["model_lanes"]["magpie"]["training"] = {"command": ["python", "train.py"]}
    assert backend.training_command(config, "magpie") == ["python", "train.py"]
    config.data["model_lanes"]["magpie"]["training"] = {"command": "python train.py"}
    with pytest.raises(InputError, match="list of strings"):
        backend.training_command(config, "magpie")


def test_artifact_hash_mismatch_is_reported(config_path, tmp_path):
    config = load_config(config_path)
    artifact = tmp_path / "model.nemo"
    artifact.write_bytes(b"model")
    config.data["model_lanes"]["magpie"]["artifacts"] = {
        "model_path": "model.nemo",
        "model_sha256": "0" * 64,
    }
    report = backend.check_lane(config, "magpie", repo_root=tmp_path)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["model_artifact"]["status"] == "failed"
    assert report["ready"] is False

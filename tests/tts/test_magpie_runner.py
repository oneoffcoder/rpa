from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from rpa_tts.cli import main
from rpa_tts.config import load_config
from rpa_tts.errors import InputError
from rpa_tts.training.magpie_runner import (
    build_finetune_command,
    diagnose_environment,
    effective_packaged_config,
    prepare_magpie_manifest,
    run_upstream,
)


def _tar_with_configs(path: Path, configs: list[dict]) -> Path:
    with tarfile.open(path, "w") as archive:
        for config in configs:
            data = yaml.safe_dump(config).encode("utf-8")
            info = tarfile.TarInfo("./model_config.yaml")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_effective_packaged_config_uses_last_member(tmp_path):
    artifact = _tar_with_configs(
        tmp_path / "model.nemo",
        [
            {"nemo_version": "2.7.0", "target": "a", "model_type": "decoder_ce"},
            {"nemo_version": "2.8.0rc0", "target": "b", "model_type": "decoder_ce"},
            {"nemo_version": "2.8.0rc0", "target": "c", "model_type": "decoder_ce"},
        ],
    )
    destination = tmp_path / "effective.yaml"
    resolved = effective_packaged_config(artifact, destination=destination)
    assert resolved["member_count"] == 3
    assert resolved["selected"]["target"] == "c"
    assert resolved["effective_sha256"]
    assert yaml.safe_load(destination.read_text(encoding="utf-8"))["target"] == "c"


def test_effective_config_requires_artifact(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        effective_packaged_config(tmp_path / "missing.nemo")


def test_run_upstream_captures_outcome(tmp_path):
    result = run_upstream([sys.executable, "-c", "print('hi')"], cwd=tmp_path)
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert "hi" in result["stdout_tail"]

    failed = run_upstream([sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path)
    assert failed["ok"] is False
    assert failed["returncode"] == 3


def test_prepare_magpie_manifest_uses_same_speaker_context(tmp_path, rows_factory):
    rows = rows_factory(4)
    rows[0]["duration_seconds"] = 5.0
    rows[1]["duration_seconds"] = 4.0
    rows[2]["duration_seconds"] = 2.0
    rows[3]["duration_seconds"] = 2.0
    for row in rows[:3]:
        row["source"]["speaker_id"] = "spk-01"
    rows[3]["source"]["speaker_id"] = "spk-02"
    result = prepare_magpie_manifest(rows, tmp_path / "train.json")
    prepared = [
        json.loads(line)
        for line in (tmp_path / "train.json").read_text(encoding="utf-8").splitlines()
    ]
    assert result["rows"] == 4
    assert result["speakers"] == ["spk-01", "spk-02"]
    by_id = {row["utterance_id"]: row for row in prepared}
    assert by_id["u0001"]["context_audio_filepath"] == rows[1]["audio_path"]
    assert by_id["u0002"]["context_audio_filepath"] == rows[0]["audio_path"]
    assert by_id["u0003"]["context_audio_filepath"] == rows[0]["audio_path"]
    # spk-02 has a single short row: no other same-speaker candidate, so self-context.
    assert by_id["u0004"]["context_audio_filepath"] == rows[3]["audio_path"]
    assert result["self_context_rows"] == 1
    assert result["cross_context_rows"] == 3
    assert by_id["u0001"]["text"] == rows[0]["transcript_normalized"]


def test_prepare_magpie_manifest_requires_rows(tmp_path):
    with pytest.raises(InputError, match="zero rows"):
        prepare_magpie_manifest([], tmp_path / "train.json")


def test_build_finetune_command_matches_pinned_upstream(tmp_path):
    init = tmp_path / "pretrained.ckpt"
    init.write_bytes(b"weights")
    train = tmp_path / "train.json"
    train.write_text("{}\n", encoding="utf-8")
    val = tmp_path / "val.json"
    val.write_text("{}\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    built = build_finetune_command(
        repo_root=tmp_path,
        init_checkpoint=init,
        exp_dir=tmp_path / "exp",
        train_manifest=train,
        val_manifest=val,
        audio_dir=audio,
        max_steps=1,
    )
    command = built["command"]
    assert command[1] == "examples/tts/magpietts.py"
    assert "--config-name=magpietts" in command
    assert any(part.startswith("+init_from_ptl_ckpt=") for part in command)
    assert "+model.text_tokenizers.hmong_chartokenizer._target_=AutoTokenizer" in command
    assert any("pretrained_model=google/byt5-small" in part for part in command)
    assert any("model.codecmodel_path=nvidia/nemo-nano-codec" in part for part in command)
    assert "trainer.precision=32" in command
    assert "trainer.devices=1" in command
    assert "+trainer.max_steps=1" in command
    assert "~model.optim.sched" in command
    assert built["gated_steps"] == 1

    with pytest.raises(InputError, match="init checkpoint does not exist"):
        build_finetune_command(
            repo_root=tmp_path,
            init_checkpoint=tmp_path / "missing.ckpt",
            exp_dir=tmp_path / "exp",
            train_manifest=train,
            val_manifest=val,
            audio_dir=audio,
        )


def test_cli_finetune_and_private_render_fail_with_the_defect(repo_root, tmp_path, capsys):
    config = repo_root / "configs" / "tts" / "f04-magpie.yaml"
    code = main(
        [
            "train",
            "finetune",
            "--config",
            str(config),
            "--init-checkpoint",
            str(tmp_path / "pretrained.ckpt"),
            "--exp-dir",
            str(tmp_path / "exp"),
            "--train-manifest",
            str(tmp_path / "train.json"),
            "--val-manifest",
            str(tmp_path / "val.json"),
            "--audio-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert code == 3
    assert "F-04" in capsys.readouterr().err

    output = tmp_path / "private.wav"
    code = main(
        [
            "train",
            "render-private",
            "--config",
            str(config),
            "--artifact",
            str(tmp_path / "model.nemo"),
            "--text",
            "kuv hlub koj",
            "--output",
            str(output),
            "--metadata",
            str(tmp_path / "private.json"),
        ]
    )
    assert code == 3
    assert "F-04" in capsys.readouterr().err
    assert not output.exists()


def test_diagnose_reports_first_defect_and_artifact_checks(repo_root, config_path):
    config = load_config(repo_root / "configs" / "tts" / "f04-magpie.yaml")
    report = diagnose_environment(config, repo_root=repo_root)
    checks = {check["name"]: check for check in report["checks"]}
    assert report["ok"] is False
    assert report["first_defect"]["code"]
    assert checks["tokenizer_pin"]["status"] == "passed"
    assert checks["calibration_data"]["status"] == "pending"
    if (
        repo_root / "data" / "artifacts" / "magpie-v2607" / "magpie_tts_multilingual_357m.nemo"
    ).is_file():
        assert checks["artifacts"]["status"] == "passed"
        assert report["packaged_config"]["member_count"] == 5
        assert checks["packaged_config"]["status"] == "passed"
    assert "Diagnosis only" in report["note"]


def test_cli_diagnose_and_train_gate_report_the_defect(repo_root, tmp_path, capsys):
    assert (
        main(
            [
                "model",
                "diagnose",
                "--config",
                str(repo_root / "configs" / "tts" / "f04-magpie.yaml"),
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["first_defect"]

    output = tmp_path / "gate"
    code = main(
        [
            "train",
            "gate",
            "--config",
            str(repo_root / "configs" / "tts" / "f04-magpie.yaml"),
            "--output",
            str(output),
        ]
    )
    assert code == 3
    assert "F-04" in capsys.readouterr().err
    diagnosis = json.loads((output / "f04-diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis["ok"] is False
    assert diagnosis["defects"]

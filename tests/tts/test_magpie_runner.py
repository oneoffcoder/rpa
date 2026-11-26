from __future__ import annotations

import io
import json
import math
import sys
import tarfile
from pathlib import Path

import pytest
import torch
import yaml

from rpa_tts.cli import main
from rpa_tts.config import load_config
from rpa_tts.errors import InputError, UnsupportedInputError
from rpa_tts.training.magpie_runner import (
    MagpieRunnerError,
    build_finetune_command,
    diagnose_environment,
    effective_packaged_config,
    parameter_fingerprint,
    prepare_magpie_manifest,
    render_private_audio,
    run_upstream,
    save_and_reload,
    verify_resume_checkpoint,
    write_rng_state,
)


class FakeModel:
    def __init__(self, weights=(1.0, 2.0, 3.0, 4.0), sample_rate: int = 24000):
        self.weights = torch.tensor(weights)
        self.sample_rate = sample_rate

    def state_dict(self):
        return {"layer.weight": self.weights}

    def save_to(self, path):
        torch.save(self.state_dict(), path)

    def do_tts(self, text, language, apply_TN, speaker_index):
        t = torch.linspace(0, 0.1, 2400)
        audio = 0.1 * torch.sin(2 * math.pi * 220 * t)
        return audio, int(audio.numel())


def _loader(path):
    model = FakeModel()
    model.weights = torch.load(path, weights_only=True)["layer.weight"]
    return model


def _tar_with_configs(path: Path, configs: list[dict]) -> Path:
    with tarfile.open(path, "w") as archive:
        for config in configs:
            data = yaml.safe_dump(config).encode("utf-8")
            info = tarfile.TarInfo("./model_config.yaml")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_fingerprint_detects_swapped_equal_sum_weights():
    first = FakeModel((1.0, 2.0, 3.0, 4.0))
    swapped = FakeModel((3.0, 4.0, 1.0, 2.0))
    assert float(first.weights.sum()) == float(swapped.weights.sum())
    assert float(first.weights.abs().sum()) == float(swapped.weights.abs().sum())
    assert parameter_fingerprint(first) != parameter_fingerprint(swapped)


def test_fingerprint_covers_dtype_shape_and_names():
    base = FakeModel((1.0, 2.0))
    as_double = FakeModel((1.0, 2.0))
    as_double.weights = as_double.weights.double()
    assert parameter_fingerprint(base) != parameter_fingerprint(as_double)


def test_weights_roundtrip_matches_and_mismatch_detected(tmp_path):
    model = FakeModel()
    report = save_and_reload(model, tmp_path / "weights.nemo", loader=_loader)
    assert report["match"] is True
    assert report["scope"] == "weights_and_config_only"
    assert report["resume_eligible"] is False
    assert report["sha256"]

    def wrong_loader(path):
        changed = FakeModel((9.0, 2.0, 3.0, 4.0))
        return changed

    mismatch = save_and_reload(model, tmp_path / "other.nemo", loader=wrong_loader)
    assert mismatch["match"] is False


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


def test_resume_checkpoint_requires_optimizer_scheduler_step_and_rng(tmp_path):
    checkpoint = tmp_path / "trainer.ckpt"
    torch.save(
        {
            "state_dict": {"w": torch.ones(2)},
            "optimizer_states": [{"state": {}}],
            "lr_schedulers": [{"last_epoch": 4}],
            "global_step": 4,
            "epoch": 0,
        },
        checkpoint,
    )
    rng = write_rng_state(tmp_path / "rng.pt")
    report = verify_resume_checkpoint(checkpoint, rng_state_path=rng["path"])
    assert report["passed"] is True
    assert report["global_step"] == 4
    assert report["optimizer_states"] == 1
    assert report["scheduler_states"] == 1

    with pytest.raises(MagpieRunnerError, match="RNG sidecar is missing"):
        verify_resume_checkpoint(checkpoint, rng_state_path=tmp_path / "missing.pt")


def test_weights_only_nemo_cannot_resume(tmp_path):
    artifact = tmp_path / "weights.nemo"
    artifact.write_bytes(b"weights")
    with pytest.raises(MagpieRunnerError, match="weights/config artifact"):
        verify_resume_checkpoint(artifact, rng_state_path=tmp_path / "rng.pt")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.pop("optimizer_states"), "missing required key"),
        (lambda data: data.update(optimizer_states=[]), "no optimizer state"),
        (lambda data: data.update(lr_schedulers=[]), "no scheduler state"),
        (lambda data: data.update(global_step=0), "global_step is zero"),
    ],
)
def test_invalid_resume_checkpoints_are_rejected(tmp_path, mutate, message):
    data = {
        "state_dict": {"w": torch.ones(2)},
        "optimizer_states": [{"state": {}}],
        "lr_schedulers": [{}],
        "global_step": 2,
        "epoch": 0,
    }
    mutate(data)
    checkpoint = tmp_path / "bad.ckpt"
    torch.save(data, checkpoint)
    rng = write_rng_state(tmp_path / "rng.pt")
    with pytest.raises(MagpieRunnerError, match=message):
        verify_resume_checkpoint(checkpoint, rng_state_path=rng["path"])


def test_private_render_writes_audio_and_marks_not_release(tmp_path):
    repo = tmp_path / "repo"
    (repo / "source").mkdir(parents=True)
    output = repo / "data" / "runs" / "r1" / "private.wav"
    report = render_private_audio(
        FakeModel(),
        "Kuv hlub koj",
        output,
        output.with_suffix(".json"),
        repo_root=repo,
        sample_rate=24000,
    )
    assert output.is_file()
    assert report["private_evaluation"] is True
    assert report["release_eligible"] is False
    assert report["text"] == "kuv hlub koj"
    assert report["audio_sha256"]
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["release_eligible"] is False


def test_private_render_refuses_book_source_path(tmp_path):
    repo = tmp_path / "repo"
    (repo / "source").mkdir(parents=True)
    with pytest.raises(InputError, match="may not be written under source"):
        render_private_audio(
            FakeModel(),
            "kuv hlub koj",
            repo / "source" / "lesson.wav",
            repo / "data" / "meta.json",
            repo_root=repo,
        )


def test_private_render_rejects_invalid_text(tmp_path):
    with pytest.raises(UnsupportedInputError):
        render_private_audio(
            FakeModel(),
            "hello 3",
            tmp_path / "data" / "x.wav",
            tmp_path / "data" / "x.json",
            repo_root=tmp_path,
        )


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

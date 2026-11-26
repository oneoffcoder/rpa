from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from rpa_tts.errors import InputError, UnsupportedInputError  # noqa: E402
from rpa_tts.training.magpie_runner import (  # noqa: E402
    MagpieRunnerError,
    evaluate_verse_loss,
    make_verse_batch,
    parameter_fingerprint,
    read_pcm16_wav,
    render_private_audio,
    save_and_reload,
    save_training_checkpoint,
    train_verse_steps,
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


def _pcm_wav(path: Path, seconds: float = 1.0, rate: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(2000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(rate * seconds))
        )
        handle.writeframes(frames)
    return path


class FakeTrainModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.tokenizer = self

    def encode(self, text, tokenizer_name=None):
        return [ord(char) for char in text[:8]]

    def parameters(self):
        return [self.weight]

    def state_dict(self):
        return {"weight": self.weight.detach()}

    def train(self):
        return self

    def eval(self):
        return self

    def training_step(self, batch, step):
        return self.weight * batch["audio"].abs().mean() + 1.0

    def process_batch(self, batch):
        return {"loss": self.training_step(batch, 0)}


def test_read_pcm16_wav_and_batch(tmp_path):
    path = _pcm_wav(tmp_path / "v.wav")
    samples, rate = read_pcm16_wav(path)
    assert rate == 22050
    assert samples.dtype.name == "float32"
    model = FakeTrainModel()
    batch = make_verse_batch(model, {"audio_path": str(path), "text": "kuv"}, device="cpu")
    assert batch["text"].shape == (1, 3)
    assert batch["audio"].shape == (1, 22050)
    assert batch["text_lens"].item() == 3


def test_evaluate_and_train_verse_steps(tmp_path):
    path = _pcm_wav(tmp_path / "v.wav")
    rows = [{"audio_path": str(path), "text": "kuv hlub koj"}]
    model = FakeTrainModel()
    base = evaluate_verse_loss(model, rows, device="cpu")
    assert base["rows"] == 1
    assert math.isfinite(base["mean"])
    report = train_verse_steps(
        model,
        rows,
        steps=3,
        learning_rate=1e-3,
        device="cpu",
        eval_rows=rows,
        eval_every=2,
        log_every=2,
    )
    assert report["steps"] == 3
    assert report["losses_finite"] is True
    assert len(report["history"]) == 2  # one eval and one train log
    assert report["optimizer"].state_dict()
    assert report["scheduler"].state_dict()


def test_save_training_checkpoint_is_resumable(tmp_path):
    path = _pcm_wav(tmp_path / "v.wav")
    rows = [{"audio_path": str(path), "text": "kuv"}]
    model = FakeTrainModel()
    report = train_verse_steps(model, rows, steps=2, device="cpu")
    saved = save_training_checkpoint(
        model,
        report["optimizer"],
        report["scheduler"],
        tmp_path / "checkpoint.pt",
        step=2,
        rng_state_path=tmp_path / "rng.pt",
    )
    assert saved["resume_verification"]["passed"] is True
    assert saved["resume_verification"]["global_step"] == 2
    assert saved["sha256"]


def test_read_pcm16_rejects_stereo(tmp_path):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\x00\x00\x00\x00" * 100)
    with pytest.raises(InputError, match="mono 16-bit"):
        read_pcm16_wav(path)


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

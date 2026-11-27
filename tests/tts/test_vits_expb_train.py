"""Regression tests for starting a new VITS fine-tune from recovered weights."""

from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

try:
    from TTS.tts.configs.vits_config import VitsConfig
except ModuleNotFoundError as error:
    if error.name != "TTS":
        raise
    raise unittest.SkipTest("pinned Coqui runtime is optional") from error

from rpa_tts.text.coqui_characters import (  # noqa: E402
    DEFAULT_VOCABULARY,
    RpaCharacters,
    tokenizer_contract,
)
from scripts import vits_expb_train as expb  # noqa: E402


def configuration(tokens=DEFAULT_VOCABULARY):
    config = VitsConfig()
    config.characters = RpaCharacters(tokens).to_config()
    config.model_args.num_chars = len(tokens)
    return config


def reordered_tokens():
    tokens = list(DEFAULT_VOCABULARY)
    tokens[4], tokens[5] = tokens[5], tokens[4]
    return tuple(tokens)


class ExperimentBFineTuneTest(unittest.TestCase):
    def test_training_config_preserves_recovered_order_and_sets_both_rates(self):
        config = expb.training_config(Path("run"), configuration(reordered_tokens()), 100, 16, 1e-4)
        self.assertEqual(tokenizer_contract(config.to_dict())["tokens"], reordered_tokens())
        self.assertEqual((config.epochs, config.batch_size), (100, 16))
        self.assertEqual((config.lr_disc, config.lr_gen), (1e-4, 1e-4))

    def test_cleaned_dataset_projects_transcripts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cleaned"
            source.mkdir()
            rows = []
            for i, split in enumerate(("train", "dev")):
                with wave.open(str(source / f"{split}.wav"), "wb") as audio:
                    audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
                    audio.writeframes(bytes([i, 0]) * 200)
                rows.append(
                    {
                        "id": split,
                        "split": split,
                        "text": "Kuv Yog Hmoob.",
                        "audio_path": f"{split}.wav",
                    }
                )
            (source / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            run = root / "run"
            with patch.object(expb, "CLEANED_SOURCE", source):
                summary = expb.prepare_cleaned_dataset(run)
            self.assertEqual(summary["projected_transcripts"], {"train": 1, "dev": 1})
            dev = run / "dataset/metadata_dev.csv"
            self.assertEqual(dev.read_text(encoding="utf-8"), "dev|kuv yog hmoob.|kuv yog hmoob.\n")
            self.assertEqual(summary["metadata_sha256"]["dev"], expb.sha256(dev))

    def test_mismatched_order_fails_before_checkpoint_loading(self):
        with patch.object(expb.torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "differs from the verified recovery bundle"):
                expb.initialize_model(
                    Path("baseline.pth"), configuration(reordered_tokens()), configuration()
                )
            load.assert_not_called()

    def test_fresh_trainer_rejects_inherited_epoch_step_and_optimizer_state(self):
        config = configuration()
        optimizers = [
            SimpleNamespace(state={}, param_groups=[{"lr": lr}])
            for lr in (config.lr_disc, config.lr_gen)
        ]
        trainer = SimpleNamespace(epochs_done=0, total_steps_done=0, optimizer=optimizers)
        expb.assert_fresh_trainer(trainer, config)
        for field, old_value in (("epochs_done", 88), ("total_steps_done", 35461)):
            setattr(trainer, field, old_value)
            with self.assertRaisesRegex(RuntimeError, "epoch index 0 and step 0"):
                expb.assert_fresh_trainer(trainer, config)
            setattr(trainer, field, 0)
        optimizers[0].state["old"] = object()
        with self.assertRaisesRegex(RuntimeError, "fresh optimizers"):
            expb.assert_fresh_trainer(trainer, config)

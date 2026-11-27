"""Regression checks using the pinned Coqui runtime when installed."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    from TTS.config import load_config
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
except ModuleNotFoundError as error:
    if error.name != "TTS":
        raise
    raise unittest.SkipTest("pinned Coqui runtime is optional") from error

from rpa_tts.text.coqui_characters import (  # noqa: E402
    DEFAULT_VOCABULARY,
    RpaCharacters,
    ordered_vocabulary,
    verify_checkpoint_tokenizer,
    vocabulary_sha256,
)


def configuration(tokens=DEFAULT_VOCABULARY):
    config = VitsConfig()
    config.characters = RpaCharacters(tokens).to_config()
    config.model_args.num_chars = len(tokens)
    return config


class CharacterMapTest(unittest.TestCase):
    def test_round_trip_in_fresh_processes_with_distinct_hash_seeds(self):
        from tempfile import TemporaryDirectory

        differing = list(DEFAULT_VOCABULARY)
        differing[4], differing[5] = differing[5], differing[4]
        config = configuration(tuple(differing))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config.save_json(str(path))
            command = """
import json,sys
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer
tokenizer, config = TTSTokenizer.init_from_config(load_config(sys.argv[1]))
text = "kuv yog hmoob."
ids = tokenizer.text_to_ids(text)
print(json.dumps([config.characters.vocab_dict, ids]))
"""
            results = [
                json.loads(
                    subprocess.check_output(
                        [sys.executable, "-c", command, str(path)],
                        env={**os.environ, "PYTHONHASHSEED": seed},
                    ).splitlines()[-1]
                )
                for seed in ("1", "2", "12345")
            ]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            self.assertEqual(results[0][0], differing)
            tokenizer, _ = TTSTokenizer.init_from_config(load_config(str(path)))
            self.assertEqual(results[0][1], tokenizer.text_to_ids("kuv yog hmoob."))

    def test_missing_malformed_and_reserved_tokens_rejected(self):
        tokens = list(DEFAULT_VOCABULARY)
        for bad in (
            None,
            {},
            "abc",
            tokens[:-1],
            tokens[:-1] + [tokens[0]],
            [tokens[1], tokens[0], *tokens[2:]],
            tokens[:-1] + [[]],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ordered_vocabulary(bad)

    def test_checkpoint_compatibility_checks_exact_order_roles_flags_and_rows(self):
        import torch

        config = configuration()
        state = {
            "config": config.to_dict(),
            "model": {"text_encoder.emb.weight": torch.zeros(len(DEFAULT_VOCABULARY), 3)},
        }
        verify_checkpoint_tokenizer(state, config)
        for change in ("swap", "blank_flag", "bos_eos_flag", "missing", "roles", "width", "short"):
            bad = copy.deepcopy(state)
            if change == "swap":
                tokens = bad["config"]["characters"]["vocab_dict"]
                tokens[4], tokens[5] = tokens[5], tokens[4]
                self.assertNotEqual(
                    vocabulary_sha256(tuple(tokens)), vocabulary_sha256(DEFAULT_VOCABULARY)
                )
            elif change == "blank_flag":
                bad["config"]["add_blank"] = not config.add_blank
            elif change == "bos_eos_flag":
                bad["config"]["enable_eos_bos_chars"] = not config.enable_eos_bos_chars
            elif change == "missing":
                bad["config"]["characters"]["vocab_dict"] = None
            elif change == "roles":
                bad["config"]["characters"]["pad"] = "<bos>"
            elif change == "width":
                bad["config"]["model_args"]["num_chars"] = len(DEFAULT_VOCABULARY) - 1
            else:
                bad["model"]["text_encoder.emb.weight"] = torch.zeros(1, 3)
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_checkpoint_tokenizer(bad, config)

    def test_recipe_resume_rejects_different_map_on_either_trainer_checkpoint(self):
        from tempfile import TemporaryDirectory

        import torch

        scripts = Path(__file__).resolve().parents[2] / "scripts"
        sys.path.insert(0, str(scripts))
        try:
            from vits_coqui_setup import settings, validate_resume_tokenizers
        finally:
            sys.path.remove(str(scripts))
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            config = settings(run, epochs=1, batch_size=2)
            config.save_json(str(run / "config.json"))
            loaded = load_config(str(run / "config.json"))
            state = {
                "config": loaded.to_dict(),
                "model": {"text_encoder.emb.weight": torch.zeros(len(DEFAULT_VOCABULARY), 3)},
            }
            files = [run / "checkpoint_20.pth", run / "best_model_10.pth"]
            for file in files:
                torch.save(state, file)
            validate_resume_tokenizers(run, loaded)
            bad = copy.deepcopy(state)
            tokens = bad["config"]["characters"]["vocab_dict"]
            tokens[4], tokens[5] = tokens[5], tokens[4]
            for file in files:
                with self.subTest(file=file):
                    torch.save(bad, file)
                    with self.assertRaisesRegex(ValueError, file.name):
                        validate_resume_tokenizers(run, loaded)
                    torch.save(state, file)

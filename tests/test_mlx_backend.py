"""CPU tests for the boundary around the optional Apple Silicon backend.

These tests exercise data safety and orchestration with an injected backend.
They are not evidence that MLX, a checkpoint, or Hmong speech works on a Mac.
"""

import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from rpa_tts import mlx_backend as backend


class MLXBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "train.jsonl"
        self.out = self.root / "canary"

    def manifest_with(self, count=3, split="train", rate=24000, channels=1):
        rows = []
        audio_dir = self.root / "audio"
        audio_dir.mkdir(exist_ok=True)
        for i in range(count):
            audio = audio_dir / f"syllable-{i}.wav"
            samples = np.tile(np.array([-32768, 0, 16384, 32767], dtype="<i2"), 1800 * channels)
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframes(samples.tobytes())
            rows.append({"id": f"syllable-{i}", "text": f"kab{i}", "split": split,
                         "audio_path": f"audio/{audio.name}", "duration_seconds": 7200 / rate})
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return rows

    def fake_runtime(self):
        wrapper = SimpleNamespace(sample_rate=24000, load_adapter=MagicMock())
        wrapper.full_model = SimpleNamespace(
            config=SimpleNamespace(tts_model_type="base"),
            speech_tokenizer=SimpleNamespace(has_encoder=True),
            eval=MagicMock(),
            generate=MagicMock(return_value=iter([
                SimpleNamespace(audio=np.full(2400, 0.1, dtype=np.float32), sample_rate=24000)
            ])),
        )
        fast = SimpleNamespace(
            from_pretrained=MagicMock(return_value=(wrapper, object())),
            get_peft_model=MagicMock(side_effect=lambda model, **kwargs: model),
            for_inference=MagicMock(side_effect=lambda model: model),
        )
        captured = {}

        class Trainer:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def train(self):
                batch = captured["train_dataset"][:1]
                captured["decoded_batch"] = batch
                adapters = Path(captured["args"].output_dir) / "adapters"
                adapters.mkdir()
                (adapters / "adapters.safetensors").write_bytes(b"test-double-only")
                (adapters / "adapter_config.json").write_text(json.dumps({
                    "lora_parameters": {"rank": 16, "alpha": 16, "dropout": 0.0,
                                        "keys": ["q_proj", "v_proj"]}
                }))
                return SimpleNamespace(metrics={"train_loss": 1.23, "train_runtime": 0})

        runtime = SimpleNamespace(
            np=np,
            mx=SimpleNamespace(random=SimpleNamespace(seed=MagicMock()),
                               metal=SimpleNamespace(is_available=lambda: True),
                               get_peak_memory=lambda: 123456),
            FastTTSModel=fast,
            TTSDataCollator=MagicMock(),
            TTSSFTConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TTSSFTTrainer=Trainer,
            snapshot_download=MagicMock(return_value=str(
                self.root / "models--mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16" /
                "snapshots" / ("a" * 40)
            )),
        )
        return runtime, captured, wrapper

    def test_short_training_audio_and_relative_paths_are_preserved(self):
        self.manifest_with()
        rows = backend.read_training_manifest(self.manifest)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["duration_seconds"], 0.3)
        self.assertTrue(Path(rows[0]["audio_path"]).is_absolute())
        data = backend._AudioDataset(rows, np)[0]
        self.assertEqual(data["audio"]["sampling_rate"], 24000)
        self.assertEqual(data["audio"]["array"].dtype, np.float32)
        np.testing.assert_array_equal(data["audio"]["array"][:4],
                                      np.array([-1, 0, 0.5, 32767 / 32768], dtype=np.float32))

    def test_held_out_or_unlabelled_rows_are_rejected(self):
        for split in ("valid", "test", None):
            with self.subTest(split=split):
                self.manifest_with(split=split)
                with self.assertRaisesRegex(ValueError, "only split='train'"):
                    backend.read_training_manifest(self.manifest)

    def test_resampling_is_never_implied_by_a_rate_label(self):
        self.manifest_with(rate=44100)
        with self.assertRaisesRegex(ValueError, "mono PCM16 WAV at 24000"):
            backend.read_training_manifest(self.manifest)

    def test_prepared_audio_hash_is_checked(self):
        rows = self.manifest_with(count=1)
        rows[0]["audio_sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "Audio SHA-256"):
            backend.read_training_manifest(self.manifest)

    def test_duplicates_stale_duration_and_truncated_audio_fail(self):
        rows = self.manifest_with()
        self.manifest.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            backend.read_training_manifest(self.manifest)
        rows[0]["duration_seconds"] = 5
        self.manifest.write_text(json.dumps(rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "Duration metadata"):
            backend.read_training_manifest(self.manifest)
        self.manifest_with(count=1)
        records = backend.read_training_manifest(self.manifest)
        path = Path(records[0]["audio_path"])
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            backend._AudioDataset(records, np)[0]

    def test_linux_preflight_does_not_import_or_download_models(self):
        self.manifest_with()
        with patch.object(backend, "_is_apple_silicon", return_value=False), \
             patch.object(backend, "_load_runtime", side_effect=AssertionError("must stay lazy")):
            result = backend.preflight(self.manifest)
        self.assertFalse(result["ready"])
        self.assertFalse(result["model_downloaded"])
        self.assertEqual(result["manifest"]["samples"], 3)
        with patch.object(backend, "_is_apple_silicon", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "native arm64 Python"):
                backend._load_runtime()

    def test_training_records_real_artifacts_and_decodes_only_training(self):
        self.manifest_with()
        runtime, captured, _ = self.fake_runtime()
        with patch.object(backend, "_load_runtime", return_value=runtime):
            result = backend.train(self.manifest, self.out, steps=2, max_samples=2)
        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["training_samples"], 2)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["model_revision"], "a" * 40)
        self.assertEqual(result["peak_memory_bytes"], 123456)
        self.assertGreater(result["training_runtime_seconds"], 0)
        self.assertEqual(captured["decoded_batch"][0]["audio"]["sampling_rate"], 24000)
        self.assertFalse(hasattr(captured["args"], "eval_dataset"))
        self.assertFalse(hasattr(captured["args"], "warmup_steps"))
        self.assertTrue((self.out / "run.json").exists())
        self.assertEqual(len(result["adapter_sha256"]), 2)

    def test_all_samples_option_and_output_overwrite_guard(self):
        self.manifest_with()
        runtime, _, _ = self.fake_runtime()
        with patch.object(backend, "_load_runtime", return_value=runtime):
            result = backend.train(self.manifest, self.out, steps=1, max_samples=0)
            self.assertEqual(result["training_samples"], 3)
            with self.assertRaises(FileExistsError):
                backend.train(self.manifest, self.out)

    def test_failure_cannot_be_reported_as_a_trained_checkpoint(self):
        self.manifest_with()
        runtime, _, wrapper = self.fake_runtime()
        wrapper.full_model.speech_tokenizer.has_encoder = False
        with patch.object(backend, "_load_runtime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "no working speech tokenizer encoder"):
                backend.train(self.manifest, self.out)
        run = json.loads((self.out / "run.json").read_text())
        self.assertEqual(run["status"], "failed")
        self.assertNotIn("adapter_sha256", run)

    def test_inference_restores_lora_before_full_qwen_generation(self):
        self.manifest_with()
        runtime, _, wrapper = self.fake_runtime()
        with patch.object(backend, "_load_runtime", return_value=runtime):
            backend.train(self.manifest, self.out, steps=1)
            metadata = backend.synthesize(self.out / "adapters", "kuv", self.root / "kuv.wav")
        runtime.snapshot_download.assert_called_with(repo_id=backend.DEFAULT_MODEL, revision="a" * 40)
        runtime.FastTTSModel.get_peft_model.assert_called_with(
            wrapper, r=16, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]
        )
        wrapper.load_adapter.assert_called_once_with(str((self.out / "adapters").resolve()))
        wrapper.full_model.generate.assert_called_once_with(
            text="kuv", lang_code="auto", max_tokens=128, stream=False
        )
        self.assertEqual(metadata["duration_seconds"], 0.1)
        self.assertTrue((self.root / "kuv.wav.json").exists())
        with wave.open(str(self.root / "kuv.wav")) as audio:
            self.assertEqual(audio.getframerate(), 24000)
            self.assertEqual(audio.getsampwidth(), 2)

    def test_adapter_tampering_is_rejected_before_model_download(self):
        self.manifest_with()
        runtime, _, _ = self.fake_runtime()
        with patch.object(backend, "_load_runtime", return_value=runtime):
            backend.train(self.manifest, self.out, steps=1)
        (self.out / "adapters" / "adapters.safetensors").write_bytes(b"changed")
        with patch.object(backend, "_load_runtime", side_effect=AssertionError("must stay lazy")):
            with self.assertRaisesRegex(ValueError, "does not match"):
                backend.synthesize(self.out / "adapters", "kuv", self.root / "out.wav")


if __name__ == "__main__":
    unittest.main()

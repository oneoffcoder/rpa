"""Regression checks for the two-stage target shift and causal depth conditioning."""

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("mlx_audio"), "optional Apple Silicon backend")
class JointTargetTests(unittest.TestCase):
    def test_padding_prompt_and_eos_are_excluded_from_residual_targets(self):
        import mlx.core as mx
        from rpa_tts import joint_mlx as joint
        codes = mx.arange(32).reshape(16, 2)
        wrapper = SimpleNamespace(
            codec_adapter=SimpleNamespace(encode_all_codebooks=lambda x, sr: codes),
            full_model=SimpleNamespace(config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=99))))
        collated = {"labels": mx.array([[-100, -100, -100, 0, 1, 99, -100, -100]]),
                    "inputs_embeds": mx.arange(8)[None, :, None]}
        args = (wrapper, lambda samples: collated, {"audio": {"array": np.zeros(3), "sampling_rate": 24000}})
        inputs, frames, targets = joint.training_example(*args)
        self.assertEqual(inputs.flatten().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(frames.shape, (2, 16))
        self.assertEqual(targets.tolist(), [0, 1, 99])
        collated["labels"] = mx.array([[-100, -100, -100, 0, -100, 1, 99, -100]])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            joint.training_example(*args)

    def test_parallel_depth_heads_match_real_incremental_heads_without_future_codes(self):
        import mlx.core as mx
        from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerConfig
        from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerForConditionalGeneration
        from rpa_tts import joint_mlx as joint
        mx.random.seed(36)
        tiny = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_code_groups=16)
        config = Qwen3TTSTalkerConfig(**tiny, text_hidden_size=16, text_vocab_size=32,
                                      rope_scaling={"mrope_section": [2, 1, 1]}, code_predictor_config=tiny)
        talker = Qwen3TTSTalkerForConditionalGeneration(config)
        prefix = mx.random.normal((1, 3, 16))
        codes = mx.random.randint(0, 32, (2, 16))
        audio = talker.get_input_embeddings()(codes[:, :1])
        for j, embed in enumerate(talker.code_predictor.codec_embedding):
            audio = audio + embed(codes[:, j + 1:j + 2])
        inputs = mx.concatenate([prefix, audio[:, 0][None]], axis=1)
        pad = mx.zeros((1, 1, 16))
        wrapper = SimpleNamespace(model=talker, full_model=SimpleNamespace(talker=talker,
                                  _prepare_generation_inputs=lambda text, language: (prefix, pad, pad)))
        example = (inputs, codes, None)
        for relative_error in joint.parity(wrapper, example, "po").values():
            self.assertLess(relative_error, 1e-5)
        # A future depth target may change later heads only, never the head predicting it.
        altered = mx.array(codes)
        altered[0, 4] = (codes[0, 4] + 1) % 32
        _, before = joint.logits(talker, inputs, codes)
        _, after = joint.logits(talker, inputs, altered)
        np.testing.assert_array_equal(np.asarray(before[0, :4]), np.asarray(after[0, :4]))
        self.assertFalse(np.array_equal(np.asarray(before[0, 4:]), np.asarray(after[0, 4:])))

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from rpa_tts.baseline import synthesize_baseline, tokenize


class BaselineTests(unittest.TestCase):
    def test_case_and_punctuation_keep_tone_letters(self):
        self.assertEqual(tokenize("Kuv yog Hmoob."), ["kuv", "yog", "hmoob", "."])

    def test_unknown_text_is_rejected_without_dropping_tokens(self):
        for text in ("", "...", "hello", "kuv 12", "aé", "ka_d"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                tokenize(text)

    def test_pause_duration_and_sample_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "source" / "_static" / "mp3"
            audio.mkdir(parents=True)
            (audio / "kuv.mp3").touch()
            (audio / "yog.mp3").touch()
            with patch("rpa_tts.baseline.require_audio_tools"), patch(
                "rpa_tts.baseline.decode_audio", return_value=(np.ones(2400) * 0.1, 24000)
            ):
                result = synthesize_baseline("Kuv, yog.", root / "result.wav", repo_root=root)
            self.assertAlmostEqual(result["duration_seconds"], 0.38)
            with wave.open(str(root / "result.wav")) as handle:
                self.assertEqual(handle.getframerate(), 24000)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()

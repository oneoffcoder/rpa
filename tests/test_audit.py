import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import wave

import numpy as np

from rpa_tts.audit import audit_corpus, decode_audio, inspect_recording


def write_wav(path, samples, sample_rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "source/_static/mp3"
        self.audio.mkdir(parents=True)

    def test_decode_duration_and_resampling_are_explicit(self):
        path = self.audio / "caj.wav"
        samples = np.sin(np.arange(4410) * 2 * np.pi * 220 / 22050) * 10000
        write_wav(path, samples)
        row = inspect_recording(path, self.root)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["sample_rate"], 22050)
        self.assertEqual(row["num_samples"], 4410)
        self.assertAlmostEqual(row["duration_seconds"], 0.2)
        self.assertEqual(row["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        resampled, rate = decode_audio(path, 24000)
        self.assertEqual(rate, 24000)
        self.assertEqual(len(resampled), 4800)
        self.assertEqual(resampled.dtype, np.float32)

    def test_audit_exposes_gaps_corruption_silence_and_duplicate_content(self):
        samples = np.sin(np.arange(4410) * 2 * np.pi * 220 / 22050) * 10000
        write_wav(self.audio / "caj.wav", samples)
        shutil.copyfile(self.audio / "caj.wav", self.audio / "cas.wav")
        write_wav(self.audio / "cam.wav", np.zeros(4410))
        write_wav(self.audio / "unknown.wav", samples)
        (self.audio / "ca.mp3").write_bytes(b"not audio")
        output = self.root / "audit"
        report = audit_corpus(self.root, output, workers=2)
        rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
        by_label = {row["text"]: row for row in rows}
        self.assertFalse(report["ok"])
        self.assertEqual(report["files"], 5)
        self.assertEqual(report["valid_files"], 2)
        self.assertEqual(report["coverage"]["present"], 4)
        self.assertIn("cab", report["coverage"]["missing"])
        self.assertEqual(report["coverage"]["unknown"], ["unknown"])
        self.assertEqual(by_label["cam"]["status"], "error")
        self.assertIn("zero samples", by_label["cam"]["errors"][0])
        self.assertEqual(by_label["ca"]["status"], "error")
        self.assertEqual(len(report["duplicate_groups"]["pcm"]), 1)
        self.assertEqual(len(report["duplicate_groups"]["pcm"][0]), 3)
        self.assertEqual(json.loads((output / "report.json").read_text()), report)
        self.assertIn("5,187", (output / "report.md").read_text())

    def test_review_flags_do_not_exclude_audio(self):
        path = self.audio / "cag.wav"
        write_wav(path, [32767, -32768] * 2205)
        row = inspect_recording(path, self.root)
        self.assertEqual(row["status"], "ok")
        self.assertIn("near_full_scale_review", row["quality_flags"])

    def test_partial_mp3_corruption_is_not_accepted_as_shorter_audio(self):
        # FFmpeg normally returns exit 0 after some damaged MP3 packets while
        # emitting a shorter waveform. -xerror must make that an integrity error.
        original = Path(__file__).resolve().parents[1] / "source/_static/mp3/caj.mp3"
        corrupted = bytearray(original.read_bytes())
        corrupted[1000:1200] = b"\0" * 200
        path = self.audio / "caj.mp3"
        path.write_bytes(corrupted)
        row = inspect_recording(path, self.root)
        self.assertEqual(row["status"], "error")
        self.assertTrue(any("ffmpeg failed" in message for message in row["errors"]))

    def test_duplicate_labels_are_reported(self):
        write_wav(self.audio / "caj.wav", [1000, -1000] * 2205)
        write_wav(self.audio / "alternate/caj.wav", [2000, -2000] * 2205)
        report = audit_corpus(self.root, self.root / "audit")
        self.assertEqual(report["coverage"]["duplicate_labels"], {"caj": 2})
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()

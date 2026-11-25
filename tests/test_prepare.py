import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rpa_tts.orthography import expected_syllables
from rpa_tts.prepare import assign_splits, load_jsonl, prepare_corpus, write_wav


def records():
    return [
        {"id": s.text, "text": s.text, "consonant": s.consonant, "vowel": s.vowel,
         "tone": s.tone, "source_sha256": "source-" + s.text, "pcm_sha256": "pcm-" + s.text,
         "audio_path": f"audio/{s.text}.wav", "status": "ok"}
        for s in expected_syllables().values() if s.consonant in {"c", "d"} and s.vowel in {"a", "e"}
    ]


class SplitTests(unittest.TestCase):
    def test_deterministic_order_independent_disjoint_and_component_covered(self):
        rows = records()
        result, summary = assign_splits(rows)
        reverse, reverse_summary = assign_splits(list(reversed(rows)))
        self.assertEqual(result, reverse)
        self.assertEqual(summary, reverse_summary)
        self.assertEqual({r["split"] for r in result}, {"train", "valid", "test"})
        self.assertEqual(len(result), len(rows))
        train = [r for r in result if r["split"] == "train"]
        for key in ("consonant", "vowel", "tone"):
            self.assertEqual({r[key] for r in rows}, {r[key] for r in train})

    def test_identical_audio_and_repeated_text_are_kept_together(self):
        rows = records()
        repeated = dict(rows[0], id="repeat", source_sha256="distinct-file")
        rows.append(repeated)
        # Also link a differently labelled example by identical decoded audio.
        rows[1]["pcm_sha256"] = rows[0]["pcm_sha256"]
        result, _ = assign_splits(rows)
        assignments = {r["id"]: r["split"] for r in result}
        self.assertEqual(assignments["repeat"], assignments[rows[0]["id"]])
        self.assertEqual(assignments[rows[1]["id"]], assignments[rows[0]["id"]])

    def test_cv_split_holds_out_entire_tone_families(self):
        rows = [
            {"id": s.text, "text": s.text, "consonant": s.consonant,
             "vowel": s.vowel, "tone": s.tone, "source_sha256": s.text,
             "pcm_sha256": s.text}
            for s in expected_syllables().values()
            if s.consonant in {"c", "d", "f"} and s.vowel in {"a", "e", "i"}
        ]
        result, _ = assign_splits(rows, split_unit="cv")
        families = {}
        for row in result:
            key = row["consonant"], row["vowel"]
            families.setdefault(key, set()).add(row["split"])
        self.assertTrue(all(len(labels) == 1 for labels in families.values()))

    def test_invalid_fractions_and_single_group_rejected(self):
        for valid, test in [(0, 0.1), (0.9, 0.2), (float("nan"), 0.1)]:
            with self.assertRaises(ValueError):
                assign_splits(records(), valid_fraction=valid, test_fraction=test)
        with self.assertRaises(ValueError):
            assign_splits(records()[:1])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "audio").mkdir()
        self.rows = records()
        for i, row in enumerate(self.rows):
            waveform = np.sin(np.arange(2400) * (0.05 + i * 0.001)).astype(np.float32) * 0.1
            path = self.root / row["audio_path"]
            write_wav(path, waveform)
            row["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = self.root / "manifest.jsonl"
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def test_conversion_writes_portable_manifests_and_preserves_short_audio(self):
        waveform = np.ones(2400, dtype=np.float32) * 0.1
        with patch("rpa_tts.prepare.decode_audio", return_value=(waveform, 24000)):
            summary = prepare_corpus(self.manifest, self.root, self.root / "dataset", workers=2)
        self.assertEqual(sum(summary["split"]["recording_counts"].values()), len(self.rows))
        for split in ("train", "valid", "test"):
            for row in load_jsonl(self.root / "dataset" / f"{split}.jsonl"):
                self.assertEqual(row["split"], split)
                self.assertEqual(row["duration_seconds"], 0.1)
                self.assertTrue((self.root / "dataset" / row["audio_path"]).is_file())

    def test_changed_source_cannot_reuse_old_audit(self):
        (self.root / self.rows[0]["audio_path"]).write_bytes(b"changed")
        with patch("rpa_tts.prepare.decode_audio", return_value=(np.ones(20), 24000)):
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                prepare_corpus(self.manifest, self.root, self.root / "dataset")
        self.assertFalse((self.root / "dataset").exists())

    def test_mislabeled_components_and_external_paths_are_rejected(self):
        self.rows[0]["tone"] = "wrong"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            prepare_corpus(self.manifest, self.root, self.root / "dataset")
        self.rows[0]["tone"] = "j"
        self.rows[0]["audio_path"] = "../outside.wav"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "outside repository"):
            prepare_corpus(self.manifest, self.root, self.root / "dataset")

    def test_existing_output_is_not_replaced(self):
        destination = self.root / "dataset"
        destination.mkdir()
        (destination / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_corpus(self.manifest, self.root, destination)
        self.assertEqual((destination / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

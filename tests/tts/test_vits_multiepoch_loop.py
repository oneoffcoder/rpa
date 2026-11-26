"""GPU integration check for the real VITS entry point and a small paired corpus."""
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("RPA_VITS_INTEGRATION") == "1",
                     "requires the VITS GPU environment and the paired corpus")
class VitsLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        sys.path.insert(0, str(scripts))
        cls.addClassCleanup(sys.path.remove, str(scripts))
        cls.loop = importlib.import_module("vits_multiepoch_train")

    def test_legacy_cursor(self):
        m = self.loop
        for position in (0, 9):
            epoch, cursor, order = m.resume_cursor(
                dict(epoch=1, position=position, order=m.epoch_order(9, 0), seed=m.SEED), 9)
            self.assertEqual((epoch, cursor, order), (0, 9, m.epoch_order(9, 0)))
        with self.assertRaises(ValueError):
            m.resume_cursor(dict(epoch=1, position=4, order=list(range(9)), seed=m.SEED), 9)

    def test_real_resume_and_epoch_boundary(self):
        m = self.loop
        train, dev, splits = m.load_rows()
        train = [row for row in train if 2 < row["duration_seconds"] < 8][:3]
        dev = [row for row in dev if 2 < row["duration_seconds"] < 8][:3]
        self.assertEqual((len(train), len(dev)), (3, 3))
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "load_rows", return_value=(train, dev, splits)):
            root = Path(temp)
            source = root / "checkpoint.pt"
            options = ["--root", temp, "--batch-size", "2", "--validation-samples", "2",
                       "--checkpoint-every", "2"]
            def status():
                return json.loads((root / "status.json").read_text(encoding="utf-8"))

            def start(*args):
                self.assertEqual(m.main([*options, *args]), 0)

            start("--max-updates", "1")
            self.assertEqual((status()["step"], status()["position"], status()["stop_reason"]),
                             (1, 2, "update_limit"))
            with patch.object(m, "save", wraps=m.save) as saved:
                start("--resume", str(source), "--epochs", "1")
                # A periodic save, epoch boundary, and terminal condition coincide.
                self.assertEqual(saved.call_count, 1)
            first = status()
            self.assertEqual((first["step"], first["epoch"], first["position"]), (2, 1, 0))
            self.assertEqual(first["boundary"]["validation"]["rows"], 2)
            self.assertEqual(first["best_epoch"], 1)
            self.assertEqual(first["stop_reason"], "epoch_limit")
            ck = m.torch.load(source, map_location="cpu", weights_only=False)
            self.assertEqual(ck["order"], m.epoch_order(3, 1))
            for name in ("disc_opt", "gen_opt"):
                states = ck[name]["state"].values()
                self.assertTrue(states)
                self.assertEqual({state["step"].item() for state in states}, {2})
            del ck
            immutable = m.digest(root / "checkpoint-epoch1.pt")
            start("--resume", str(source), "--epochs", "2")
            second = status()
            self.assertEqual((second["step"], second["rows_seen"], second["unique_seen"]), (4, 6, 3))
            self.assertEqual(immutable, m.digest(root / "checkpoint-epoch1.pt"))
            self.assertTrue((root / second["best_checkpoint"]).is_file())
            self.assertTrue((root / "boundaries" / "epoch2" / "spaced.wav").is_file())
            with self.assertRaises(FileNotFoundError):
                start("--resume", str(root / "missing.pt"))


if __name__ == "__main__":
    unittest.main()

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from python.orpheus_listen import connection, state
from python.orpheus_segments import PROTOCOL, RATE, assemble, chunk_prompts, complete_groups


class SegmentTest(unittest.TestCase):
    def test_spelling_is_preserved(self):
        data = json.loads(PROTOCOL.read_text())
        for item in data["prompts"]:
            chunks = chunk_prompts(item)
            self.assertEqual(item["text"].rstrip(".").split(),
                             " ".join(c.rstrip(".") for c in chunks).split())
        for chunks in (["po", "po"], ["poj", "po"], ["po", "", "poj"]):
            with self.assertRaises(ValueError):
                chunk_prompts(dict(id="contrast", text="po poj", chunks=chunks))

    def test_keeps_every_sample_and_only_tops_up_quiet(self):
        left = np.array([500, 0, 0], dtype=np.int16)
        right = np.array([0, -600], dtype=np.int16)
        wave, joins = assemble([left,right])
        self.assertEqual(joins[0]["added_samples"], int(.12*RATE)-3)
        np.testing.assert_array_equal(wave[:3],left)
        np.testing.assert_array_equal(wave[-2:],right)
        left = np.pad(left,(0,RATE))
        wave, joins = assemble([left,right])
        self.assertEqual(joins[0]["added_samples"],0)
        np.testing.assert_array_equal(wave,np.concatenate([left,right]))
        np.testing.assert_array_equal(assemble([right])[0],right)

    def test_corrupt_interior_keeps_every_complete_frame(self):
        a = 128266 + np.arange(7)*4096 + 3
        codes, bad = complete_groups(np.concatenate([a,a[1:4],a,a]))
        self.assertEqual(bad,[7,8,9])
        np.testing.assert_array_equal(codes,np.full((3,7),3))
        with self.assertRaises(ValueError):
            complete_groups(a[:4])

    def test_reveal_after_all_first_responses(self):
        key=[dict(id=i,text="secret text",arm="whole") for i in ("a","b")]
        with tempfile.TemporaryDirectory() as tmp, connection(Path(tmp)) as db:
            def insert(item,phase,data):
                db.execute("INSERT INTO responses VALUES (?,?,?,?,?)",("l",item,phase,"date",json.dumps(data)))
            insert("a","first",dict(transcription="heard"))
            self.assertNotIn("text",state(db,key,"l"))
            with self.assertRaises(sqlite3.IntegrityError):
                insert("a","first",dict(transcription="rewrite"))
            insert("b","first",dict(transcription="uncertain"))
            phase2=state(db,key,"l")
            self.assertEqual(phase2["first"],"heard")
            self.assertNotIn("arm",phase2)
            for item in key:
                insert(item["id"],"rating",dict(tones="uncertain"))
            self.assertEqual(state(db,key,"l")["phase"],"done")

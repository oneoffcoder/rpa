from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.errors import InputError
from rpa_tts.evaluation.ab_player import render_ab_player, write_ab_player


def _key(tmp_path: Path) -> Path:
    audio = tmp_path / "audio"
    audio.mkdir()
    for name in ("item01_A.wav", "item01_B.wav"):
        (audio / name).write_bytes(b"fake")
    key = tmp_path / "answer-key.json"
    key.write_text(
        json.dumps(
            [
                {
                    "item": "item01",
                    "reference": "MRK 1:1",
                    "text": "Kuv hlub koj.",
                    "conditions": {
                        "A": {
                            "condition": "trained",
                            "audio_path": str(audio / "item01_A.wav"),
                            "seconds": 1.2,
                        },
                        "B": {
                            "condition": "base",
                            "audio_path": str(audio / "item01_B.wav"),
                            "seconds": 1.4,
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return key


def test_player_is_blinded_and_keeps_key_separate(tmp_path):
    key = _key(tmp_path)
    html = render_ab_player(json.loads(key.read_text(encoding="utf-8")))
    assert "Slot A" in html
    assert "Slot B" in html
    assert "Kuv hlub koj." in html
    assert "item01_A.wav" in html
    assert "item01_B.wav" in html
    assert "condition" not in html.replace("condition mapping", "")
    assert "trained" not in html
    assert "base" not in html


def test_write_player_checks_audio(tmp_path):
    key = _key(tmp_path)
    result = write_ab_player(key, tmp_path / "player.html")
    assert result["items"] == 1
    assert (tmp_path / "player.html").is_file()
    (tmp_path / "audio" / "item01_B.wav").unlink()
    with pytest.raises(InputError, match="audio file is missing"):
        write_ab_player(key, tmp_path / "other.html")


def test_cli_ab_player(tmp_path, capsys):
    key = _key(tmp_path)
    code = main(
        ["evaluate", "ab-player", "--answer-key", str(key), "--output", str(tmp_path / "p.html")]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"] == 1
    assert (tmp_path / "p.html").is_file()

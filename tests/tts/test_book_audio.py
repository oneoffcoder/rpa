from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.data.book_audio import (
    audio_reference_report,
    extract_playback_links,
    render_reference_report,
    write_reference_report,
)
from rpa_tts.errors import InputError

EXPECTED = {"kuv", "hlub", "koj"}


def _pages(tmp_path: Path, body: str) -> Path:
    pages = tmp_path / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "speaking.rst").write_text(body, encoding="utf-8")
    return pages


def _audio(tmp_path: Path, stems=EXPECTED) -> Path:
    audio = tmp_path / "mp3"
    audio.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (audio / f"{stem}.mp3").write_bytes(b"fake")
    return audio


def test_extracts_links_with_tokens(tmp_path):
    pages = _pages(
        tmp_path,
        "`a <generate.html?q=kuv+hlub+koj>`_ and `b <generate.html?q=kuv+hlub+koj>`_\n",
    )
    links = extract_playback_links(pages)
    assert len(links) == 2
    assert links[0]["tokens"] == ["kuv", "hlub", "koj"]
    assert links[0]["page"] == "speaking.rst"
    assert links[0]["line"] == 1


def test_clean_references_pass(tmp_path):
    pages = _pages(tmp_path, "`a <generate.html?q=kuv+hlub+koj>`_\n")
    report = audio_reference_report(pages, _audio(tmp_path), expected_inventory=EXPECTED)
    assert report["ok"] is True
    assert report["links"] == 1
    assert report["playable_tokens"] == 3
    assert report["dropped_tokens"] == {}
    assert report["inventory"]["missing_count"] == 0


def test_non_inventory_tokens_are_warnings_not_failures(tmp_path):
    pages = _pages(
        tmp_path,
        "`a <generate.html?q=kuv+washington+dc>`_\n",
    )
    report = audio_reference_report(pages, _audio(tmp_path), expected_inventory=EXPECTED)
    assert report["ok"] is True
    assert report["dropped_tokens"] == {"dc": 1, "washington": 1}
    assert report["warnings"]
    markdown = render_reference_report(report)
    assert "washington" in markdown
    assert "does not change that behavior" in markdown


def test_link_with_no_playable_tokens_fails(tmp_path):
    pages = _pages(tmp_path, "`a <generate.html?q=washington+dc>`_\n")
    report = audio_reference_report(pages, _audio(tmp_path), expected_inventory=EXPECTED)
    assert report["ok"] is False
    assert len(report["zero_playable_links"]) == 1
    assert any("no playable syllable" in error for error in report["errors"])


def test_missing_and_unexpected_audio_files_fail(tmp_path):
    pages = _pages(tmp_path, "`a <generate.html?q=kuv+hlub+koj>`_\n")
    audio = _audio(tmp_path, stems={"kuv", "hlub"})
    report = audio_reference_report(pages, audio, expected_inventory=EXPECTED)
    assert report["ok"] is False
    assert report["inventory"]["missing"] == ["koj"]
    assert report["missing_audio"] == {"koj": 1}

    audio = _audio(tmp_path / "second", stems=EXPECTED | {"extra"})
    pages = _pages(tmp_path / "second", "`a <generate.html?q=kuv>`_\n")
    report = audio_reference_report(pages, audio, expected_inventory=EXPECTED)
    assert report["inventory"]["unexpected"] == ["extra"]
    assert report["ok"] is False


def test_write_reference_report_outputs(tmp_path):
    pages = _pages(tmp_path, "`a <generate.html?q=kuv>`_\n")
    audio = _audio(tmp_path)
    result = write_reference_report(
        pages,
        audio,
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        expected_inventory=EXPECTED,
    )
    assert result["ok"] is True
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["ok"] is True
    assert "**OK**" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_missing_directories_are_input_errors(tmp_path):
    with pytest.raises(InputError, match="page directory"):
        extract_playback_links(tmp_path / "nope")
    pages = _pages(tmp_path, "`a <generate.html?q=kuv>`_\n")
    with pytest.raises(InputError, match="audio directory"):
        audio_reference_report(pages, tmp_path / "nope", expected_inventory=EXPECTED)


def test_cli_book_audio_on_the_repository(repo_root, tmp_path, capsys):
    code = main(
        [
            "corpus",
            "book-audio",
            "--page-dir",
            str(repo_root / "source"),
            "--audio-dir",
            str(repo_root / "source" / "_static" / "mp3"),
            "--output-json",
            str(tmp_path / "report.json"),
            "--output-md",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["inventory"]["expected"] == 5187
    assert payload["inventory"]["audio_files"] == 5187
    assert payload["links"] == 281
    assert payload["zero_playable_links"] == []
    assert payload["dropped_tokens"] == {"dc": 1, "washington": 1}

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpa_tts.cli import main
from rpa_tts.data.book_html import (
    html_reference_report,
    render_html_reference_report,
    write_html_reference_report,
)
from rpa_tts.errors import InputError

EXPECTED = {"kuv", "hlub"}

HTML = """\
<html><head>
<link href="_static/css/theme.css?v=abc">
<script src="_static/js/custom.js"></script>
</head><body>
<a href="_static/mp3/kuv.mp3">kuv</a>
<a href="generate.html?q=kuv+hlub">listen</a>
<a href="https://example.com/x">external</a>
</body></html>
"""


def _build(tmp_path: Path, body: str = HTML, *, stems=EXPECTED, sitemap: bool = True) -> Path:
    build = tmp_path / "build"
    (build / "_static" / "css").mkdir(parents=True, exist_ok=True)
    (build / "_static" / "js").mkdir(parents=True, exist_ok=True)
    (build / "_static" / "mp3").mkdir(parents=True, exist_ok=True)
    (build / "index.html").write_text(body, encoding="utf-8")
    (build / "_static" / "css" / "theme.css").write_text("body{}", encoding="utf-8")
    (build / "_static" / "js" / "custom.js").write_text("//", encoding="utf-8")
    for stem in stems:
        (build / "_static" / "mp3" / f"{stem}.mp3").write_bytes(b"fake")
    if sitemap:
        (build / "sitemap.xml").write_text("<urlset/>", encoding="utf-8")
    return build


def test_clean_built_references_pass(tmp_path):
    report = html_reference_report(_build(tmp_path), expected_inventory=EXPECTED)
    assert report["ok"] is True
    assert report["pages"] == 1
    assert report["checked_local_references"] == 3
    assert report["generate_links"] == 1
    assert report["playable_tokens"] == 2
    assert report["inventory"]["built_audio_files"] == 2
    assert report["sitemap_present"] is True
    assert report["errors"] == []


def test_missing_local_reference_fails(tmp_path):
    body = HTML + '<a href="_static/js/missing.js">missing</a>\n'
    report = html_reference_report(_build(tmp_path, body), expected_inventory=EXPECTED)
    assert report["ok"] is False
    assert report["missing_reference_count"] == 1
    assert report["missing_references"][0]["reference"] == "_static/js/missing.js"
    assert any("do not resolve" in error for error in report["errors"])


def test_reference_escaping_the_build_fails(tmp_path):
    body = HTML + '<a href="../secret.txt">secret</a>\n'
    report = html_reference_report(_build(tmp_path, body), expected_inventory=EXPECTED)
    assert report["ok"] is False
    assert len(report["outside_build"]) == 1
    assert any("escape the build" in error for error in report["errors"])


def test_zero_playable_generate_link_fails(tmp_path):
    body = HTML + '<a href="generate.html?q=washington+dc">bad</a>\n'
    report = html_reference_report(_build(tmp_path, body), expected_inventory=EXPECTED)
    assert report["ok"] is False
    assert len(report["zero_playable_links"]) == 1
    assert report["dropped_tokens"] == {"dc": 1, "washington": 1}


def test_missing_built_inventory_fails(tmp_path):
    report = html_reference_report(
        _build(tmp_path, stems={"kuv"}), expected_inventory={"kuv", "koj"}
    )
    assert report["ok"] is False
    assert report["inventory"]["missing"] == ["koj"]
    assert report["missing_token_audio"] == {}


def test_missing_sitemap_is_reported_but_not_fatal(tmp_path):
    report = html_reference_report(_build(tmp_path, sitemap=False), expected_inventory=EXPECTED)
    assert report["sitemap_present"] is False
    assert report["ok"] is True


def test_write_html_report_outputs(tmp_path):
    build = _build(tmp_path)
    result = write_html_reference_report(
        build,
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        expected_inventory=EXPECTED,
    )
    assert result["ok"] is True
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["ok"] is True
    assert "**OK**" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_missing_build_directory_is_an_input_error(tmp_path):
    with pytest.raises(InputError, match="build directory"):
        html_reference_report(tmp_path / "nope", expected_inventory=EXPECTED)


def test_render_failed_report_lists_errors(tmp_path):
    body = HTML + '<a href="_static/js/missing.js">missing</a>\n'
    report = html_reference_report(_build(tmp_path, body), expected_inventory=EXPECTED)
    markdown = render_html_reference_report(report)
    assert "**FAILED**" in markdown
    assert "missing.js" in markdown


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "build" / "html" / "index.html").is_file(),
    reason="built Sphinx HTML is not present",
)
def test_repository_built_site(repo_root, tmp_path, capsys):
    code = main(
        [
            "corpus",
            "book-html",
            "--build-dir",
            str(repo_root / "build" / "html"),
            "--output-json",
            str(tmp_path / "report.json"),
            "--output-md",
            str(tmp_path / "report.md"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["pages"] >= 12
    assert payload["generate_links"] == 281
    assert payload["inventory"]["expected"] == 5187
    assert payload["inventory"]["built_audio_files"] == 5187
    assert payload["missing_reference_count"] == 0

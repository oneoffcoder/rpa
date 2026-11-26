"""Reference integrity of the built Sphinx HTML (verification layer 1).

The source-level playback check verifies `generate.html?q=...` links against the
syllable inventory. This check verifies the *built* site: every local `href`/`src`
resolves to an emitted file, query strings are ignored, generated assets
(theme/JS/blockdiag images) exist, the built MP3 inventory matches, and no
generate link has zero playable tokens.

It does not execute JavaScript or prove audible browser playback; it catches
missing/renamed assets and broken build-time references.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..text.orthography import expected_syllables

REF_PATTERN = re.compile(r'(?:href|src)="([^"]+)"')
GENERATE_PREFIX = "generate.html?q="
AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".ogg", ".flac", ".m4a"})


def _is_external(reference: str) -> bool:
    lowered = reference.lower()
    return (
        lowered.startswith(("#", "javascript:", "mailto:", "data:", "tel:", "//"))
        or lowered.startswith("http://")
        or lowered.startswith("https://")
    )


def html_reference_report(
    build_dir: str | Path,
    *,
    expected_inventory: set[str] | None = None,
) -> dict[str, Any]:
    build = Path(build_dir)
    if not build.is_dir() or not (build / "index.html").is_file():
        raise InputError(f"build directory with index.html not found: {build}")
    expected = expected_inventory if expected_inventory is not None else set(expected_syllables())
    pages = sorted(build.rglob("*.html"))
    missing: list[dict[str, Any]] = []
    outside_build: list[dict[str, Any]] = []
    checked_refs = 0
    refs_by_kind: dict[str, int] = {}
    generate_links: list[dict[str, Any]] = []
    tokens_total = 0
    playable_total = 0
    dropped_tokens: dict[str, int] = {}
    missing_token_audio: dict[str, int] = {}
    zero_playable: list[dict[str, Any]] = []

    for page in pages:
        try:
            text = page.read_text(encoding="utf-8")
        except OSError:
            continue
        relative_page = page.relative_to(build).as_posix()
        for match in REF_PATTERN.finditer(text):
            raw = html.unescape(match.group(1)).strip()
            if not raw or _is_external(raw):
                continue
            if raw.startswith(GENERATE_PREFIX):
                query = urllib.parse.unquote(raw[len(GENERATE_PREFIX) :])
                tokens = [token.lower() for token in query.split("+") if token]
                tokens_total += len(tokens)
                playable = [token for token in tokens if token in expected]
                playable_total += len(playable)
                for token in tokens:
                    if token not in expected:
                        dropped_tokens[token] = dropped_tokens.get(token, 0) + 1
                for token in playable:
                    if not (build / "_static" / "mp3" / f"{token}.mp3").is_file():
                        missing_token_audio[token] = missing_token_audio.get(token, 0) + 1
                if not playable:
                    zero_playable.append({"page": relative_page, "query": query})
                generate_links.append({"page": relative_page, "query": query})
                continue
            path_part = raw.split("#", 1)[0].split("?", 1)[0]
            if not path_part:
                continue
            checked_refs += 1
            target = (page.parent / urllib.parse.unquote(path_part)).resolve()
            try:
                target.relative_to(build.resolve())
            except ValueError:
                outside_build.append({"page": relative_page, "reference": raw})
                continue
            suffix = target.suffix.lower()
            kind = (
                "audio"
                if suffix in AUDIO_SUFFIXES
                else "style"
                if suffix == ".css"
                else "script"
                if suffix == ".js"
                else "image"
                if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp"}
                else "page"
                if suffix == ".html"
                else "other"
            )
            refs_by_kind[kind] = refs_by_kind.get(kind, 0) + 1
            if not target.is_file():
                missing.append({"page": relative_page, "reference": raw, "kind": kind})

    built_audio = build / "_static" / "mp3"
    built_stems = (
        {
            path.stem.lower()
            for path in built_audio.iterdir()
            if path.suffix.lower() in AUDIO_SUFFIXES
        }
        if built_audio.is_dir()
        else set()
    )
    missing_inventory = sorted(expected - built_stems)
    unexpected_audio = sorted(built_stems - expected)
    sitemap = (build / "sitemap.xml").is_file()

    errors: list[str] = []
    if missing:
        errors.append(f"{len(missing)} local references do not resolve")
    if outside_build:
        errors.append(f"{len(outside_build)} references escape the build directory")
    if missing_inventory:
        errors.append(f"{len(missing_inventory)} expected syllables have no built MP3")
    if unexpected_audio:
        errors.append(f"{len(unexpected_audio)} built MP3s are outside the documented inventory")
    if missing_token_audio:
        errors.append(f"{len(missing_token_audio)} playable tokens have no built MP3")
    if zero_playable:
        errors.append(f"{len(zero_playable)} generate links have no playable tokens")
    return {
        "schema_version": 1,
        "build_dir": build.as_posix(),
        "pages": len(pages),
        "checked_local_references": checked_refs,
        "references_by_kind": dict(sorted(refs_by_kind.items())),
        "missing_references": missing[:100],
        "missing_reference_count": len(missing),
        "outside_build": outside_build[:20],
        "generate_links": len(generate_links),
        "generate_tokens": tokens_total,
        "playable_tokens": playable_total,
        "dropped_tokens": dict(sorted(dropped_tokens.items())),
        "missing_token_audio": dict(sorted(missing_token_audio.items())),
        "zero_playable_links": zero_playable[:50],
        "inventory": {
            "expected": len(expected),
            "built_audio_files": len(built_stems),
            "missing_count": len(missing_inventory),
            "missing": missing_inventory[:50],
            "unexpected_count": len(unexpected_audio),
            "unexpected": unexpected_audio[:50],
        },
        "sitemap_present": sitemap,
        "ok": not errors,
        "errors": errors,
        "interpretation": (
            "Static built-HTML integrity only. It does not execute JavaScript, verify "
            "audible playback, pronunciation or native acceptance."
        ),
    }


def render_html_reference_report(report: dict[str, Any]) -> str:
    lines = [
        "# Built HTML reference check",
        "",
        "Verifies that the built Sphinx site's local `href`/`src` targets exist,",
        "that the built syllable MP3 inventory matches, and that no",
        "`generate.html?q=...` link has zero playable tokens.",
        "",
        f"Build: `{report['build_dir']}`.",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| HTML pages | {report['pages']} |",
        f"| Checked local references | {report['checked_local_references']} |",
        f"| Missing references | {report['missing_reference_count']} |",
        f"| References escaping the build | {len(report['outside_build'])} |",
        f"| Generate links | {report['generate_links']} |",
        f"| Generate tokens | {report['generate_tokens']} |",
        f"| Playable tokens | {report['playable_tokens']} |",
        f"| Built audio files | {report['inventory']['built_audio_files']} |",
        f"| Missing expected audio | {report['inventory']['missing_count']} |",
        f"| Unexpected audio | {report['inventory']['unexpected_count']} |",
        f"| Zero-playable links | {len(report['zero_playable_links'])} |",
        f"| Sitemap present | {report['sitemap_present']} |",
        "",
        "## Result",
        "",
        "**OK**" if report["ok"] else "**FAILED**",
        "",
    ]
    if report["errors"]:
        lines.extend(["### Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    if report["missing_references"]:
        lines.extend(["### Missing references (first 100)", ""])
        for item in report["missing_references"]:
            lines.append(f"- `{item['page']}` → `{item['reference']}` ({item['kind']})")
        lines.append("")
    if report["dropped_tokens"]:
        lines.extend(
            [
                "### Non-inventory tokens (pre-existing player behavior)",
                "",
                ", ".join(
                    f"`{token}` ({count})"
                    for token, count in sorted(report["dropped_tokens"].items())
                ),
                "",
            ]
        )
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def write_html_reference_report(
    build_dir: str | Path,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    expected_inventory: set[str] | None = None,
) -> dict[str, Any]:
    report = html_reference_report(build_dir, expected_inventory=expected_inventory)
    if output_json:
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report["output_json"] = destination.as_posix()
    if output_md:
        destination = Path(output_md)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_html_reference_report(report), encoding="utf-8")
        report["output_md"] = destination.as_posix()
    return report


__all__ = [
    "html_reference_report",
    "render_html_reference_report",
    "write_html_reference_report",
]

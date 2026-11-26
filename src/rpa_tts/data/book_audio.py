"""Verification of the book's existing syllable playback references.

The book's `generate.html?q=...` links are intercepted by `custom.js`, which
splits the query on `+`, drops any token not in its syllable list, and plays
`_static/mp3/<token>.mp3`. This check verifies that inventory and references still
line up, without modifying the book or its audio:

* the MP3 file set equals the documented 5,187-syllable inventory;
* every playable query token has an MP3;
* no link is left with zero playable tokens.

Tokens outside the syllable inventory (for example proper names such as
`washington`, `dc`) are reported as warnings: the existing player silently drops
them, and that pre-existing behavior is disclosed rather than "fixed" by this
research branch.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..text.orthography import expected_syllables

LINK_PATTERN = re.compile(r"generate\.html\?q=([A-Za-z0-9+%._-]+)")
AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".ogg", ".flac", ".m4a"})


def extract_playback_links(page_dir: str | Path) -> list[dict[str, Any]]:
    pages = Path(page_dir)
    if not pages.is_dir():
        raise InputError(f"page directory does not exist: {pages}")
    links: list[dict[str, Any]] = []
    for page in sorted(pages.glob("*.rst")):
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for match in LINK_PATTERN.finditer(line):
                raw_query = match.group(1)
                query = urllib.parse.unquote(raw_query)
                tokens = [token.lower() for token in query.split("+") if token]
                links.append(
                    {
                        "page": page.name,
                        "line": number,
                        "query": query,
                        "tokens": tokens,
                    }
                )
    return links


def audio_reference_report(
    page_dir: str | Path,
    audio_dir: str | Path,
    *,
    expected_inventory: set[str] | None = None,
) -> dict[str, Any]:
    pages = Path(page_dir)
    audio = Path(audio_dir)
    if not audio.is_dir():
        raise InputError(f"audio directory does not exist: {audio}")
    expected = expected_inventory if expected_inventory is not None else set(expected_syllables())
    files = sorted(
        path for path in audio.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    stems = {path.stem.lower() for path in files}
    links = extract_playback_links(pages)
    missing_inventory = sorted(expected - stems)
    unexpected_files = sorted(stems - expected)

    total_tokens = 0
    playable_tokens = 0
    dropped_tokens: dict[str, int] = {}
    missing_audio: dict[str, int] = {}
    zero_playable: list[dict[str, Any]] = []
    links_with_dropped: list[dict[str, Any]] = []
    for link in links:
        tokens = link["tokens"]
        total_tokens += len(tokens)
        playable = [token for token in tokens if token in expected]
        dropped = [token for token in tokens if token not in expected]
        playable_tokens += len(playable)
        for token in dropped:
            dropped_tokens[token] = dropped_tokens.get(token, 0) + 1
        missing = [token for token in playable if token not in stems]
        for token in missing:
            missing_audio[token] = missing_audio.get(token, 0) + 1
        if not playable:
            zero_playable.append(
                {
                    "page": link["page"],
                    "line": link["line"],
                    "query": link["query"],
                }
            )
        if dropped:
            links_with_dropped.append(
                {
                    "page": link["page"],
                    "line": link["line"],
                    "query": link["query"],
                    "dropped": dropped,
                }
            )

    errors: list[str] = []
    if missing_inventory:
        errors.append(f"{len(missing_inventory)} expected syllables have no audio file")
    if unexpected_files:
        errors.append(f"{len(unexpected_files)} audio files are outside the documented inventory")
    if missing_audio:
        errors.append(f"{len(missing_audio)} playable query tokens have no audio file")
    if zero_playable:
        errors.append(f"{len(zero_playable)} links have no playable syllable token")
    return {
        "schema_version": 1,
        "page_dir": pages.as_posix(),
        "audio_dir": audio.as_posix(),
        "links": len(links),
        "total_query_tokens": total_tokens,
        "playable_tokens": playable_tokens,
        "inventory": {
            "expected": len(expected),
            "audio_files": len(files),
            "missing": missing_inventory[:50],
            "missing_count": len(missing_inventory),
            "unexpected": unexpected_files[:50],
            "unexpected_count": len(unexpected_files),
        },
        "dropped_tokens": dict(sorted(dropped_tokens.items())),
        "missing_audio": dict(sorted(missing_audio.items())),
        "zero_playable_links": zero_playable,
        "links_with_dropped_tokens": links_with_dropped,
        "ok": not errors,
        "errors": errors,
        "warnings": (
            []
            if not dropped_tokens
            else [
                "Non-inventory tokens are silently dropped by the existing player "
                f"({', '.join(sorted(dropped_tokens))}); disclosed as pre-existing book behavior."
            ]
        ),
        "interpretation": (
            "Static reference integrity only. It does not prove browser playback, pronunciation, "
            "native acceptance or that stitching sounds natural."
        ),
    }


def render_reference_report(report: dict[str, Any]) -> str:
    lines = [
        "# Book syllable-playback reference check",
        "",
        "Verifies the existing `generate.html?q=...` links against the documented",
        "5,187-syllable inventory and `_static/mp3` files. It does not modify the book.",
        "",
        f"Pages: `{report['page_dir']}`; audio: `{report['audio_dir']}`.",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Playback links | {report['links']} |",
        f"| Query tokens | {report['total_query_tokens']} |",
        f"| Playable tokens | {report['playable_tokens']} |",
        f"| Expected syllables | {report['inventory']['expected']} |",
        f"| Audio files | {report['inventory']['audio_files']} |",
        f"| Missing audio for expected syllables | {report['inventory']['missing_count']} |",
        f"| Unrecognized audio files | {report['inventory']['unexpected_count']} |",
        f"| Playable tokens with no audio | {len(report['missing_audio'])} |",
        f"| Links with zero playable tokens | {len(report['zero_playable_links'])} |",
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
    lines.extend(["## Non-inventory tokens (disclosed as pre-existing)", ""])
    if report["dropped_tokens"]:
        for token, count in sorted(report["dropped_tokens"].items()):
            lines.append(f"- `{token}` ({count})")
        lines.extend(
            [
                "",
                "The existing player filters these out, so their audio is not heard. This "
                "branch does not change that behavior; a later authorized content fix may "
                "replace the text or add an approved recording.",
                "",
            ]
        )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def write_reference_report(
    page_dir: str | Path,
    audio_dir: str | Path,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    expected_inventory: set[str] | None = None,
) -> dict[str, Any]:
    report = audio_reference_report(page_dir, audio_dir, expected_inventory=expected_inventory)
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
        destination.write_text(render_reference_report(report), encoding="utf-8")
        report["output_md"] = destination.as_posix()
    return report


__all__ = [
    "audio_reference_report",
    "extract_playback_links",
    "render_reference_report",
    "write_reference_report",
]

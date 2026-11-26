"""Draft calibration/development scripts from the book's native-authored examples.

The book (CC BY 4.0) contains linked Hmong sentences in its dialog and proverb
sections. They are useful candidate scripts, but they are **not approved** and
they are not fresh final-test material: the native lead must select, edit and
approve scripts before recording, and exposed scripts cannot be reused in the
locked test.

Extraction is deterministic and lossless: each candidate keeps its source page
and line, raw text, normalization form and any contract issues. Blocked
candidates are retained with their issues so native reviewers can decide on
proper names/loanwords instead of the tool silently dropping them.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..errors import InputError
from ..hashing import sha256_file
from ..text.normalize import normalize_rpa

SCHEMA_VERSION = 1
REVIEW_COLUMNS = [
    "script_id",
    "source_page",
    "source_line",
    "section",
    "bucket",
    "status",
    "syllable_count",
    "text",
    "issues",
    "decision",
    "edited_text",
    "notes",
]
DRAFT_STATUSES = {"pending_native_approval", "blocked_contract_review"}
# Matches RST links such as: `Nyob zoo, Oo. <generate.html?q=nyob+zoo+oo>`_
_LINK = re.compile(r"`([^`<>]+?)\s*<generate\.html\?q=[^>]*>`_")


def _section_for_line(lines: list[str], line_number: int) -> str:
    """Return the nearest preceding RST section heading, if any."""
    for index in range(min(line_number, len(lines)) - 1, -1, -1):
        candidate = lines[index].strip()
        if not candidate:
            continue
        if index + 1 < len(lines) and re.fullmatch(r"[-=~^]{3,}", lines[index + 1].strip()):
            return candidate
    return ""


def extract_candidates(
    rst_text: str, *, min_words: int = 3, source_page: str = "source/speaking.rst"
) -> list[dict[str, Any]]:
    """Extract linked Hmong sentence candidates with source line and section."""
    if min_words < 1:
        raise ValueError("min_words must be at least 1")
    lines = rst_text.splitlines()
    candidates: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines, 1):
        for match in _LINK.finditer(line):
            text = match.group(1).strip().strip('"').strip()
            if not text:
                continue
            words = [word for word in re.findall(r"[A-Za-z]+", text)]
            if len(words) < min_words:
                continue
            normalized = normalize_rpa(text)
            if normalized.issues:
                status = "blocked_contract_review"
            else:
                status = "pending_native_approval"
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "text": text,
                    "source": {
                        "page": source_page,
                        "line": line_index,
                        "section": _section_for_line(lines, line_index),
                    },
                    "normalized_text": normalized.normalized if not normalized.issues else None,
                    "syllables": list(normalized.syllables),
                    "syllable_count": len(normalized.syllables),
                    "word_count": len(words),
                    "bucket": "targeted_contrasts"
                    if _section_for_line(lines, line_index) == "Proverbs"
                    else "ordinary_sentences",
                    "kind": "sentence",
                    "status": status,
                    "issues": [
                        {
                            "code": issue.code,
                            "token": issue.token,
                            "position": issue.position,
                            "message": issue.message,
                        }
                        for issue in normalized.issues
                    ],
                    "normalization_version": normalized.version,
                    "notes": "Draft from book example; native approval and recording required.",
                }
            )
    # Deduplicate by raw text, keeping the first occurrence.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["text"] in seen:
            continue
        seen.add(candidate["text"])
        unique.append(candidate)
    unique.sort(key=lambda candidate: candidate["source"]["line"])
    for index, candidate in enumerate(unique, 1):
        candidate["script_id"] = f"cal-{index:03d}"
    return unique


def draft_scripts(
    rst_path: str | Path,
    output_jsonl: str | Path,
    output_report: str | Path,
    *,
    min_words: int = 3,
) -> dict[str, Any]:
    source = Path(rst_path)
    if not source.is_file():
        raise InputError(f"RST source does not exist: {source}")
    rst_text = source.read_text(encoding="utf-8")
    candidates = extract_candidates(rst_text, min_words=min_words, source_page=source.as_posix())
    if not candidates:
        raise InputError("no linked sentence candidates were found")
    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in candidates),
        encoding="utf-8",
    )
    status_counts = Counter(row["status"] for row in candidates)
    bucket_counts = Counter(row["bucket"] for row in candidates)
    blocked_tokens = Counter(
        issue["token"] for row in candidates for issue in row["issues"] if issue["token"]
    )
    syllables = [row["syllable_count"] for row in candidates if row["syllable_count"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": source.as_posix(),
        "source_sha256": sha256_file(source),
        "output": output.as_posix(),
        "output_sha256": sha256_file(output),
        "candidates": len(candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "syllable_total": sum(syllables),
        "median_syllables": sorted(syllables)[len(syllables) // 2] if syllables else 0,
        "estimated_minutes_at_4s": round(len(candidates) * 4 / 60, 2),
        "blocked_tokens": dict(sorted(blocked_tokens.items())),
        "note": (
            "Draft only. Native review selects/edits and approves scripts before recording; "
            "these public book sentences are calibration/development material, not the locked test."
        ),
    }
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report_markdown(report), encoding="utf-8")
    report["report"] = report_path.as_posix()
    return report


def _render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Draft calibration/development scripts (from the book)",
        "",
        f"Source: `{report['source']}` (SHA-256 `{report['source_sha256']}`).",
        f"Candidate JSONL: `{report['output']}` (SHA-256 `{report['output_sha256']}`).",
        "",
        "**Draft only — native approval and recording are outstanding. These public",
        "book sentences are calibration/development material, not the locked test.**",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Candidates | {report['candidates']} |",
        f"| Pending native approval | "
        f"{report['status_counts'].get('pending_native_approval', 0)} |",
        f"| Blocked contract review | "
        f"{report['status_counts'].get('blocked_contract_review', 0)} |",
        f"| Total syllables | {report['syllable_total']} |",
        f"| Median syllables per candidate | {report['median_syllables']} |",
        f"| Estimated recording time at 4 s/script | {report['estimated_minutes_at_4s']} min |",
        "",
        "## Buckets",
        "",
    ]
    for bucket, count in sorted(report["bucket_counts"].items()):
        lines.append(f"- `{bucket}`: {count}")
    lines.extend(["", "## Blocked tokens (native decision required)", ""])
    if report["blocked_tokens"]:
        for token, count in sorted(report["blocked_tokens"].items()):
            lines.append(f"- `{token}` ({count})")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "1. Native lead selects roughly 100–150 calibration clips (10–15 minutes),",
            "   edits/approves the text and decides every blocked proper name/loanword",
            "   (approved spelling or an explicit product decision).",
            "2. Record fresh independent development and final-test scripts; do not",
            "   reuse these exposed book sentences in the locked 300-item suite.",
            "3. Feed approved items into a listening-pack definition when stimuli exist.",
            "",
        ]
    )
    return "\n".join(lines)


def load_draft_rows(path: str | Path) -> list[dict[str, Any]]:
    draft = Path(path)
    if not draft.is_file():
        raise InputError(f"draft script JSONL does not exist: {draft}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(draft.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{draft}:{number}: invalid JSON: {exc.msg}") from None
        if not isinstance(row, dict):
            raise InputError(f"{draft}:{number}: expected a JSON object")
        for field in (
            "script_id",
            "text",
            "status",
            "source",
            "bucket",
            "syllable_count",
            "issues",
        ):
            if field not in row:
                raise InputError(f"{draft}:{number}: missing {field!r}")
        script_id = str(row["script_id"])
        if script_id in seen:
            raise InputError(f"{draft}:{number}: duplicate script_id {script_id!r}")
        seen.add(script_id)
        if row["status"] not in DRAFT_STATUSES:
            raise InputError(f"{draft}:{number}: unknown draft status {row['status']!r}")
        source = row["source"]
        if not isinstance(source, dict) or "page" not in source or "line" not in source:
            raise InputError(f"{draft}:{number}: source must contain page and line")
        issues = row["issues"]
        if not isinstance(issues, list):
            raise InputError(f"{draft}:{number}: issues must be a list")
        if row["status"] == "blocked_contract_review" and not issues:
            raise InputError(f"{draft}:{number}: blocked row must list its contract issues")
        rows.append(row)
    if not rows:
        raise InputError(f"draft script JSONL contains no rows: {draft}")
    rows.sort(
        key=lambda row: (
            str(row["source"]["page"]),
            int(row["source"]["line"]),
            str(row["script_id"]),
        )
    )
    return rows


def _issue_text(issues: list[dict[str, Any]]) -> str:
    parts = []
    for issue in issues:
        token = issue.get("token") or ""
        parts.append(f"{issue.get('code', 'issue')}:{token}@{issue.get('position', '?')}")
    return "; ".join(parts)


def write_review_sheet(
    draft_path: str | Path,
    csv_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a reviewer CSV (decisions blank) and an instructions report."""
    draft = Path(draft_path)
    rows = load_draft_rows(draft)
    csv_destination = Path(csv_path)
    report_destination = Path(report_path)
    if not overwrite:
        for destination in (csv_destination, report_destination):
            if destination.exists():
                raise InputError(
                    f"{destination} already exists; a filled review sheet must not be overwritten "
                    "(use --overwrite only for a deliberate regeneration)"
                )
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "script_id": row["script_id"],
                    "source_page": row["source"]["page"],
                    "source_line": row["source"]["line"],
                    "section": row["source"].get("section", ""),
                    "bucket": row["bucket"],
                    "status": row["status"],
                    "syllable_count": row["syllable_count"],
                    "text": row["text"],
                    "issues": _issue_text(row["issues"]),
                    "decision": "",
                    "edited_text": "",
                    "notes": "",
                }
            )
    status_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    blocked_tokens: dict[str, int] = {}
    blocked_rows = []
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
        for issue in row["issues"]:
            token = issue.get("token") or ""
            if token:
                blocked_tokens[token] = blocked_tokens.get(token, 0) + 1
        if row["status"] == "blocked_contract_review":
            blocked_rows.append(row)
    lines = [
        "# Native review worksheet — draft calibration/development scripts",
        "",
        f"Draft JSONL: `{draft.as_posix()}`.",
        f"Review CSV: `{csv_destination.as_posix()}`.",
        f"Rows: {len(rows)}; pending approval: {status_counts.get('pending_native_approval', 0)}; "
        f"blocked: {status_counts.get('blocked_contract_review', 0)}.",
        "",
        "**Draft only. No row is approved or recorded. These public book sentences are",
        "calibration/development material, not the locked 300-item test.**",
        "",
        "## Reviewer instructions",
        "",
        "1. Set `decision` on every row to `accept`, `edit` or `reject`.",
        "2. For `edit`, put the approved RPA text in `edited_text`; it must pass",
        "   `rpa-norm-1` with no digits, foreign text or unapproved punctuation.",
        "3. Blocked rows need a spelling/product decision (approved spelling or",
        "   explicit rejection); record it in `notes` and, if editing, `edited_text`.",
        "4. Record reviewer identity/consent separately under stable private IDs; do",
        "   not put names or contact details in this worksheet.",
        "5. Selected scripts still need native approval and a consented speaker",
        "   recording before any pack or training use.",
        "",
        "## Counts by bucket",
        "",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"- `{bucket}`: {count}")
    lines.extend(["", "## Blocked rows (decision required)", ""])
    if blocked_rows:
        lines.append("| Script | Line | Text | Issues |")
        lines.append("| --- | ---: | --- | --- |")
        for row in blocked_rows:
            text = str(row["text"]).replace("|", "\\|")
            lines.append(
                f"| {row['script_id']} | {row['source']['line']} | {text} | "
                f"{_issue_text(row['issues'])} |"
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Blocked tokens",
            "",
            ", ".join(f"`{token}` ({count})" for token, count in sorted(blocked_tokens.items()))
            if blocked_tokens
            else "None.",
            "",
        ]
    )
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text("\n".join(lines), encoding="utf-8")
    return {
        "csv": csv_destination.as_posix(),
        "report": report_destination.as_posix(),
        "rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "blocked_tokens": dict(sorted(blocked_tokens.items())),
        "csv_sha256": sha256_file(csv_destination),
        "report_sha256": sha256_file(report_destination),
    }

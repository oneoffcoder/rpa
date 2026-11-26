"""Post-scoring native review ingestion.

Blind transcription/naturalness/contrast responses are captured first. Two
human review steps follow and must be merged without altering the raw responses:

* **adjudication** of critical-contrast errors, with at least two distinct native
  adjudicators for a confirmed error (issue F-07);
* **paragraph review** of missing/repeated clauses and unintended language
  switches.

Both steps are separate from the listener session so the raw response record
stays auditable. Confirmed adjudications are what the gate uses to detect an
unresolved systematic critical-contrast error.
"""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

from ..errors import InputError, ValidationError

ADJUDICATION_COLUMNS = ["code", "verdict", "confirmed_by", "notes"]
PARAGRAPH_COLUMNS = [
    "script_id",
    "missing_clauses",
    "repeated_clauses",
    "language_switch",
    "notes",
]
ADJUDICATION_VERDICTS = {"confirmed", "rejected", "uncertain"}
ADJUDICATOR_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")


def _read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputError(f"review sheet does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError(f"review sheet has no header: {path}")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise InputError(f"{path} is missing columns: {missing}")
        rows = []
        for row in reader:
            if any((value or "").strip() for value in row.values()):
                rows.append({key: (value or "").strip() for key, value in row.items()})
        return rows


def _parse_adjudicators(value: str) -> list[str]:
    identifiers = [part.strip() for part in re.split(r"[;|,]", value) if part.strip()]
    unique: list[str] = []
    for identifier in identifiers:
        if identifier not in unique:
            unique.append(identifier)
    return unique


def apply_adjudications(scores: dict[str, Any], adjudications_path: str | Path) -> dict[str, Any]:
    """Merge confirmed/rejected/uncertain adjudications into contrast records."""
    path = Path(adjudications_path)
    rows = _read_csv(path, ADJUDICATION_COLUMNS)
    if not rows:
        raise InputError("adjudication sheet has no rows")
    result = copy.deepcopy(scores)
    contrasts = result.get("contrasts") or []
    by_code = {str(record.get("code")): record for record in contrasts if record.get("code")}
    if not by_code:
        raise InputError("scores payload has no contrast records to adjudicate")

    problems: list[str] = []
    seen: set[str] = set()
    counts = {"confirmed": 0, "rejected": 0, "uncertain": 0}
    unadjudicated_wrong_critical = 0
    for index, row in enumerate(rows, 2):
        code = row["code"]
        verdict = row["verdict"].lower()
        if code in seen:
            problems.append(f"line {index}: duplicate adjudication code {code!r}")
            continue
        seen.add(code)
        if code not in by_code:
            problems.append(f"line {index}: unknown contrast code {code!r}")
            continue
        if verdict not in ADJUDICATION_VERDICTS:
            problems.append(
                f"line {index}: verdict {verdict!r} is not confirmed/rejected/uncertain"
            )
            continue
        record = by_code[code]
        chosen = str(record.get("chosen") or "").strip().lower()
        expected = str(record.get("expected") or "").strip().lower()
        wrong = bool(chosen) and chosen != expected
        adjudicators = _parse_adjudicators(row["confirmed_by"])
        if verdict == "confirmed":
            if not wrong:
                problems.append(
                    f"line {index}: cannot confirm an error for a correct/no-response choice"
                )
                continue
            if not record.get("critical"):
                problems.append(f"line {index}: cannot confirm an error on a non-critical item")
                continue
            if len(adjudicators) < 2:
                problems.append(
                    f"line {index}: a confirmed error needs at least two distinct adjudicators"
                )
                continue
            for identifier in adjudicators:
                if not ADJUDICATOR_ID.fullmatch(identifier):
                    problems.append(f"line {index}: unsafe adjudicator id {identifier!r}")
                    break
            else:
                record["adjudicated"] = True
                record["confirmed_by"] = adjudicators
        else:
            if adjudicators and verdict == "rejected":
                problems.append(
                    f"line {index}: a rejected error must not list confirming adjudicators"
                )
                continue
        record["adjudication"] = {
            "verdict": verdict,
            "confirmed_by": adjudicators if verdict == "confirmed" else [],
            "notes": row["notes"],
            "sheet": path.name,
        }
        counts[verdict] += 1

    if problems:
        raise ValidationError("adjudication failed validation: " + "; ".join(problems[:12]))

    for record in contrasts:
        chosen = str(record.get("chosen") or "").strip().lower()
        expected = str(record.get("expected") or "").strip().lower()
        if (
            record.get("critical")
            and chosen
            and chosen != expected
            and not record.get("adjudicated")
            and code_of(record) not in seen
        ):
            unadjudicated_wrong_critical += 1

    result["adjudications"] = {
        "sheet": path.name,
        "rows": len(rows),
        "counts": counts,
        "unadjudicated_wrong_critical": unadjudicated_wrong_critical,
        "note": (
            "Confirmed errors require two distinct adjudicators and remain part of the raw "
            "response record; rejected and uncertain decisions do not erase the original choice."
        ),
    }
    return result


def code_of(record: dict[str, Any]) -> str:
    return str(record.get("code") or "")


def _parse_bool(value: str, *, field: str, line: int) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n", ""}:
        return False
    raise ValidationError(f"line {line}: {field} must be a boolean-like value, got {value!r}")


def apply_paragraph_reviews(
    scores: dict[str, Any], reviews_path: str | Path, *, pack_dir: str | Path | None = None
) -> dict[str, Any]:
    """Merge paragraph-level human review counts into the scores payload."""
    path = Path(reviews_path)
    rows = _read_csv(path, PARAGRAPH_COLUMNS)
    if not rows:
        raise InputError("paragraph review sheet has no rows")
    result = copy.deepcopy(scores)
    paragraph_ids: set[str] | None = None
    if pack_dir is not None:
        manifest_path = Path(pack_dir) / "pack-manifest.json"
        if not manifest_path.is_file():
            raise InputError(f"pack manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paragraph_ids = {
            str(item["script_id"]) for item in manifest["items"] if item.get("kind") == "paragraph"
        }
        if not paragraph_ids:
            raise InputError("pack has no paragraph items to review")

    problems: list[str] = []
    seen: set[str] = set()
    total_missing = 0
    total_repeated = 0
    language_switches = 0
    items: list[dict[str, Any]] = []
    for line, row in enumerate(rows, 2):
        script_id = row["script_id"]
        if script_id in seen:
            problems.append(f"line {line}: duplicate paragraph id {script_id!r}")
            continue
        seen.add(script_id)
        if paragraph_ids is not None and script_id not in paragraph_ids:
            problems.append(f"line {line}: {script_id!r} is not a paragraph item in the pack")
            continue
        try:
            missing = int(row["missing_clauses"])
            repeated = int(row["repeated_clauses"])
        except ValueError:
            problems.append(f"line {line}: clause counts must be integers")
            continue
        if missing < 0 or repeated < 0:
            problems.append(f"line {line}: clause counts must be non-negative")
            continue
        switch = _parse_bool(row["language_switch"], field="language_switch", line=line)
        total_missing += missing
        total_repeated += repeated
        language_switches += int(switch)
        items.append(
            {
                "script_id": script_id,
                "missing_clauses": missing,
                "repeated_clauses": repeated,
                "language_switch": switch,
                "notes": row["notes"],
            }
        )
    if problems:
        raise ValidationError("paragraph review failed validation: " + "; ".join(problems[:12]))

    result["paragraphs"] = {
        "sheet": path.name,
        "items": len(items),
        "missing_clauses": total_missing,
        "repeated_clauses": total_repeated,
        "language_switch_items": language_switches,
        "per_item": items,
        "note": "Human paragraph review; counts are not inferred by the transcription tool.",
    }
    return result

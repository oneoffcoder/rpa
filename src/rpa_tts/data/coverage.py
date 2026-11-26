"""Coverage ledger for declared attested cells and critical contrasts.

Counts are distinct utterance examples per split plus token occurrences and
distinct spellings. The project screening target (default five examples per
prioritized attested cell) is a screening count, not a completeness guarantee;
every missing or underfilled cell is reported.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..data.contract import ContrastCell
from ..text.normalize import normalize_rpa
from ..text.orthography import parse_syllable

SPLITS = ("train", "dev", "test")


def _syllables(row: dict[str, Any]) -> list[str]:
    return list(normalize_rpa(str(row["transcript_normalized"])).syllables)


def coverage_report(
    rows: list[dict[str, Any]],
    cells: list[ContrastCell] | tuple[ContrastCell, ...],
    *,
    min_examples: int = 5,
) -> dict[str, Any]:
    if min_examples < 1:
        raise ValueError("min_examples must be at least 1")
    per_cell: dict[str, dict[str, Any]] = {
        cell.id: {
            "id": cell.id,
            "type": cell.type,
            "consonant": cell.consonant,
            "vowel": cell.vowel,
            "tone": cell.tone,
            "attested": cell.attested,
            "note": cell.note,
            "splits": {split: {"examples": 0, "occurrences": 0} for split in SPLITS},
            "forms": set(),
        }
        for cell in cells
    }
    onset_tone: Counter[str] = Counter()
    vowel_tone: Counter[str] = Counter()

    for row in rows:
        split = str(row["split"])
        seen_in_row: set[str] = set()
        for text in _syllables(row):
            syllable = parse_syllable(text)
            onset_tone[f"{syllable.consonant}_{syllable.tone}"] += 1
            vowel_tone[f"{syllable.vowel}_{syllable.tone}"] += 1
            for cell in cells:
                if not cell.matches(syllable.consonant, syllable.vowel, syllable.tone):
                    continue
                entry = per_cell[cell.id]
                entry["splits"][split]["occurrences"] += 1
                entry["forms"].add(text)
                if cell.id not in seen_in_row:
                    entry["splits"][split]["examples"] += 1
                    seen_in_row.add(cell.id)

    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        entry = per_cell[cell.id]
        train_examples = entry["splits"]["train"]["examples"]
        if not cell.attested:
            status = "unattested_excluded"
        elif train_examples == 0:
            status = "missing"
        elif train_examples < min_examples:
            status = "underfilled"
        else:
            status = "ok"
        cell_rows.append(
            {
                **{
                    key: entry[key]
                    for key in ("id", "type", "consonant", "vowel", "tone", "attested", "note")
                },
                "status": status,
                "splits": entry["splits"],
                "distinct_forms": sorted(entry["forms"]),
                "screening_target_met": cell.attested
                and train_examples >= min_examples
                and entry["splits"]["test"]["examples"] > 0,
            }
        )

    return {
        "min_examples_per_attested_cell": min_examples,
        "declared_cells": len(cells),
        "attested_cells": sum(cell.attested for cell in cells),
        "cells": cell_rows,
        "missing": [row["id"] for row in cell_rows if row["status"] == "missing"],
        "underfilled": [row["id"] for row in cell_rows if row["status"] == "underfilled"],
        "screening_passed": bool([row for row in cell_rows if row["attested"]])
        and all(row["screening_target_met"] for row in cell_rows if row["attested"]),
        "observed_onset_tone": dict(sorted(onset_tone.items())),
        "observed_vowel_tone": dict(sorted(vowel_tone.items())),
        "interpretation": (
            "Distinct-utterance screening counts. Passing this screen does not establish language "
            "completeness, acoustic balance or model quality."
        ),
    }


def render_coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Coverage ledger",
        "",
        f"Screening target: at least {report['min_examples_per_attested_cell']} distinct training "
        "examples per attested cell, plus a held-out context.",
        "",
        f"Declared cells: {report['declared_cells']} ({report['attested_cells']} attested).",
        "",
        "| Cell | Type | Components | Train | Dev | Test | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["cells"]:
        tone = "" if row["tone"] == "_" else row["tone"]
        components = f"{row['consonant'] or ''}{row['vowel']}{tone}"
        lines.append(
            f"| {row['id']} | {row['type']} | `{components}` | "
            f"{row['splits']['train']['examples']} | {row['splits']['dev']['examples']} | "
            f"{row['splits']['test']['examples']} | {row['status']} |"
        )
    if not report["cells"]:
        lines.append("| — | — | — | 0 | 0 | 0 | no cells declared (contract is a draft) |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Unattested cells are excluded explicitly rather than counted as coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def coverage_from_contract(
    rows: list[dict[str, Any]],
    contract: Any,
    *,
    min_examples: int = 5,
) -> dict[str, Any]:
    return coverage_report(rows, contract.critical_contrasts, min_examples=min_examples)

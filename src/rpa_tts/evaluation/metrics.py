"""Human-scoring metrics: pooled syllable errors, naturalness and contrast accuracy.

Rules from the issue (F-07):

* Syllable error = pooled substitutions + deletions + insertions over the pooled
  reference syllable count across eligible independent transcriptions.
* Empty outputs incur deletions for every reference syllable and remain counted.
* Contrast accuracy always reports its exact judgment denominator and confusion
  matrix; ties are counted and reported, never dropped.
* Naturalness is scored separately from voice preference on a frozen 1-5 scale.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from ..errors import InputError
from ..text.normalize import normalize_rpa

_WORD = re.compile(r"[A-Za-z]+")
TIE_TOKENS = {"", "uncertain", "tie", "unsure", "none", "no_response"}


def listener_syllables(text: str) -> list[str]:
    """Lenient tokenizer for human transcriptions.

    Unknown spellings stay in the sequence so they can count as errors; they
    are not silently dropped and they never raise.
    """
    return [match.lower() for match in _WORD.findall(text)]


def reference_syllables(text: str) -> list[str]:
    """Strict reference tokenizer: the reference must be valid RPA text."""
    normalized = normalize_rpa(text, strict=True)
    return list(normalized.syllables)


def syllable_ops(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """Return (errors, substitutions, deletions, insertions) deterministically.

    Ties between edit paths are broken by the lexicographic order of the count
    tuple, so repeated scoring of the same pair always reports the same split.
    """
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    # Each cell is (errors, substitutions, deletions, insertions).
    table: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0)] * columns for _ in range(rows)]
    for i in range(1, rows):
        previous = table[i - 1][0]
        table[i][0] = (previous[0] + 1, 0, previous[2] + 1, 0)
    for j in range(1, columns):
        previous = table[0][j - 1]
        table[0][j] = (previous[0] + 1, 0, 0, previous[3] + 1)
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidate = table[i - 1][j - 1]
            substitute = (candidate[0] + 1, candidate[1] + 1, candidate[2], candidate[3])
            deleted = table[i - 1][j]
            deletion = (deleted[0] + 1, deleted[1], deleted[2] + 1, deleted[3])
            inserted = table[i][j - 1]
            insertion = (inserted[0] + 1, inserted[1], inserted[2], inserted[3] + 1)
            table[i][j] = min(substitute, deletion, insertion)
    errors, substitutions, deletions, insertions = table[-1][-1]
    return errors, substitutions, deletions, insertions


def _as_syllables(value: Any, *, reference: bool, field: str) -> list[str]:
    if isinstance(value, list) and all(isinstance(token, str) for token in value):
        return [token.strip().lower() for token in value if token.strip()]
    if isinstance(value, str):
        return reference_syllables(value) if reference else listener_syllables(value)
    raise InputError(f"{field} must be a string or list of syllable strings")


def score_transcriptions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score blind transcription records; failed outputs stay in denominators."""
    if not records:
        raise InputError("no transcription records supplied")
    totals = {
        "utterances": 0,
        "judgments": 0,
        "reference_syllables": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "errors": 0,
        "empty_outputs": 0,
    }
    by_bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "judgments": 0,
            "reference_syllables": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "errors": 0,
            "empty_outputs": 0,
        }
    )
    by_listener: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "judgments": 0,
            "reference_syllables": 0,
            "errors": 0,
            "empty_outputs": 0,
        }
    )
    seen_utterances: set[str] = set()

    for index, record in enumerate(records):
        for field in ("utterance_id", "listener_id", "bucket", "reference", "hypothesis"):
            if field not in record:
                raise InputError(f"transcription record {index} is missing {field!r}")
        reference = _as_syllables(record["reference"], reference=True, field="reference")
        if not reference:
            raise InputError(f"transcription record {index} has no reference syllables")
        hypothesis = _as_syllables(record["hypothesis"], reference=False, field="hypothesis")
        errors, substitutions, deletions, insertions = syllable_ops(reference, hypothesis)
        empty = not hypothesis
        bucket = str(record["bucket"])
        listener = str(record["listener_id"])
        seen_utterances.add(str(record["utterance_id"]))

        totals["judgments"] += 1
        totals["reference_syllables"] += len(reference)
        totals["substitutions"] += substitutions
        totals["deletions"] += deletions
        totals["insertions"] += insertions
        totals["errors"] += errors
        totals["empty_outputs"] += int(empty)

        bucket_block = by_bucket[bucket]
        bucket_block["judgments"] += 1
        bucket_block["reference_syllables"] += len(reference)
        bucket_block["substitutions"] += substitutions
        bucket_block["deletions"] += deletions
        bucket_block["insertions"] += insertions
        bucket_block["errors"] += errors
        bucket_block["empty_outputs"] += int(empty)

        listener_block = by_listener[listener]
        listener_block["judgments"] += 1
        listener_block["reference_syllables"] += len(reference)
        listener_block["errors"] += errors
        listener_block["empty_outputs"] += int(empty)

    totals["utterances"] = len(seen_utterances)

    def rate(block: dict[str, int]) -> float:
        denominator = block.get("reference_syllables", 0)
        return round(block.get("errors", 0) / denominator, 6) if denominator else 0.0

    totals["error_rate"] = rate(totals)
    bucket_report = {
        bucket: {**values, "error_rate": rate(values)}
        for bucket, values in sorted(by_bucket.items())
    }
    listener_report = {
        listener: {**values, "error_rate": rate(values)}
        for listener, values in sorted(by_listener.items())
    }
    return {
        "totals": totals,
        "by_bucket": bucket_report,
        "by_listener": listener_report,
        "pooled_error_rate": totals["error_rate"],
        "interpretation": (
            "Pooled human syllable errors across eligible independent transcriptions; failed and "
            "empty outputs remain in the denominators."
        ),
    }


def naturalness_summary(
    records: list[dict[str, Any]], *, condition: str = "candidate"
) -> dict[str, Any]:
    """Summarize the frozen 1-5 naturalness scale for one condition."""
    scores: list[int] = []
    listeners: set[str] = set()
    utterances: set[str] = set()
    distribution: Counter[int] = Counter()
    for index, record in enumerate(records):
        if record.get("condition") != condition:
            continue
        score = record.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise InputError(f"naturalness record {index} must have an integer score from 1 to 5")
        scores.append(score)
        distribution[score] += 1
        listeners.add(str(record.get("listener_id", "")))
        utterances.add(str(record.get("utterance_id", "")))
    if not scores:
        return {
            "condition": condition,
            "count": 0,
            "mean": None,
            "listeners": 0,
            "utterances": 0,
            "distribution": {},
        }
    return {
        "condition": condition,
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 6),
        "listeners": len(listeners),
        "utterances": len(utterances),
        "distribution": {str(score): count for score, count in sorted(distribution.items())},
    }


def contrast_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Forced-choice contrast identification with explicit denominators."""
    if not records:
        raise InputError("no contrast records supplied")
    judgments = len(records)
    correct = 0
    ties = 0
    confusion: Counter[tuple[str, str]] = Counter()
    critical_errors: dict[tuple[str, str], set[str]] = defaultdict(set)
    confirmations: dict[tuple[str, str], set[str]] = defaultdict(set)

    for index, record in enumerate(records):
        for field in ("utterance_id", "expected", "chosen"):
            if field not in record:
                raise InputError(f"contrast record {index} is missing {field!r}")
        expected = str(record["expected"]).strip().lower()
        raw_chosen = record["chosen"]
        chosen = "" if raw_chosen is None else str(raw_chosen).strip().lower()
        if chosen in TIE_TOKENS:
            ties += 1
        confusion[(expected, chosen or "no_response")] += 1
        if chosen == expected:
            correct += 1
            continue
        if record.get("critical") and record.get("adjudicated"):
            key = (expected, chosen or "no_response")
            critical_errors[key].add(str(record["utterance_id"]))
            for adjudicator in record.get("confirmed_by") or []:
                confirmations[key].add(str(adjudicator))

    systematic = [
        {
            "expected": key[0],
            "chosen": key[1],
            "utterances": sorted(utterances),
            "distinct_utterances": len(utterances),
            "adjudicators": sorted(confirmations.get(key, set())),
        }
        for key, utterances in sorted(critical_errors.items())
        if len(utterances) >= 2 and len(confirmations.get(key, set())) >= 2
    ]
    return {
        "judgments": judgments,
        "correct": correct,
        "accuracy": round(correct / judgments, 6),
        "ties": ties,
        "denominator": judgments,
        "confusion": {
            f"{expected}->{chosen}": count
            for (expected, chosen), count in sorted(confusion.items())
        },
        "systematic_critical_errors": systematic,
        "blocks_release": bool(systematic),
    }


__all__ = [
    "contrast_summary",
    "listener_syllables",
    "naturalness_summary",
    "reference_syllables",
    "score_transcriptions",
    "syllable_ops",
]

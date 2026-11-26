"""Lossless RPA normalization and validation.

The contract (docs/tts/recovery/language-contract.yaml) requires raw text,
surface text and the model-facing normalized text to be retained together with
the normalization version. Nothing is discarded silently:

* Unicode is NFC-normalized.
* Typographic quotes/dashes/ellipsis are mapped to ASCII; the mapping is recorded.
* Runs of whitespace collapse to one space; leading/trailing whitespace is removed.
* The model-facing form is lowercased; the surface form preserves case.
* Digits, foreign characters, unsupported punctuation and unrecorded syllables
  are reported as issues. They are never "fixed" by an English cleaner.

Callers decide whether issues are fatal (``normalize_rpa(..., strict=True)``).
The approved number/date/currency expansion is applied separately, from the
language contract, by :mod:`rpa_tts.text.numbers`.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .. import NORMALIZATION_VERSION
from ..errors import UnsupportedInputError
from .orthography import parse_syllable

PUNCTUATION: frozenset[str] = frozenset(".,!?;:")

# Typographic variants that carry no RPA lexical information. Every substitution
# is recorded in ``transformations`` so a reviewer can see what changed.
CHARACTER_MAP: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
}

ISSUE_EMPTY_TEXT = "empty_text"
ISSUE_UNKNOWN_WORD = "unknown_word"
ISSUE_UNAPPROVED_DIGITS = "unapproved_digits"
ISSUE_UNSUPPORTED_CHARACTER = "unsupported_character"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    position: int
    token: str = ""


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    surface: str
    normalized: str
    tokens: tuple[str, ...]
    syllables: tuple[str, ...]
    issues: tuple[Issue, ...]
    transformations: tuple[str, ...]
    version: str = NORMALIZATION_VERSION

    @property
    def is_valid(self) -> bool:
        return not self.issues and bool(self.tokens)

    def require_valid(self) -> NormalizedText:
        if any(issue.code == ISSUE_EMPTY_TEXT for issue in self.issues):
            raise UnsupportedInputError(f"empty RPA text (normalization {self.version})")
        if self.issues:
            details = "; ".join(
                f"{issue.code} at {issue.position}: {issue.message}" for issue in self.issues
            )
            raise UnsupportedInputError(
                "text is outside the approved RPA contract "
                f"(normalization {self.version}): {details}"
            )
        return self


def _to_nfc(text: str) -> tuple[str, bool]:
    normalized = unicodedata.normalize("NFC", text)
    return normalized, normalized != text


def _map_characters(text: str) -> tuple[str, list[str]]:
    transformations: list[str] = []
    output: list[str] = []
    for character in text:
        replacement = CHARACTER_MAP.get(character)
        if replacement is None:
            output.append(character)
        else:
            output.append(replacement)
            transformations.append(f"mapped U+{ord(character):04X} to {replacement!r}")
    return "".join(output), transformations


def _collapse_whitespace(text: str) -> tuple[str, bool]:
    collapsed = " ".join(text.split())
    return collapsed, collapsed != text


def _classify(surface: str) -> tuple[tuple[str, ...], tuple[str, ...], list[Issue]]:
    """Split the surface form into words and punctuation, validating each word."""
    tokens: list[str] = []
    syllables: list[str] = []
    issues: list[Issue] = []
    word: list[str] = []
    word_start = 0

    def flush() -> None:
        nonlocal word, word_start
        if not word:
            return
        raw_word = "".join(word)
        candidate = raw_word.lower()
        position = word_start
        word = []
        try:
            syllable = parse_syllable(candidate)
        except ValueError:
            issues.append(
                Issue(
                    code=ISSUE_UNKNOWN_WORD,
                    message=f"{raw_word!r} is not in the documented RPA syllable inventory",
                    position=position,
                    token=raw_word,
                )
            )
            return
        tokens.append(candidate)
        syllables.append(syllable.text)

    for position, character in enumerate(surface):
        if character.isalpha():
            if not word:
                word_start = position
            word.append(character)
            continue
        flush()
        if character.isspace():
            continue
        if character in PUNCTUATION:
            tokens.append(character)
            continue
        if character.isdigit():
            issues.append(
                Issue(
                    code=ISSUE_UNAPPROVED_DIGITS,
                    message=(
                        f"digit {character!r} needs an approved number/date/currency expansion; "
                        "none is adopted yet"
                    ),
                    position=position,
                    token=character,
                )
            )
            continue
        issues.append(
            Issue(
                code=ISSUE_UNSUPPORTED_CHARACTER,
                message=f"character U+{ord(character):04X} {character!r} is not approved RPA text",
                position=position,
                token=character,
            )
        )
    flush()
    return tuple(tokens), tuple(syllables), issues


def normalize_rpa(text: str, *, strict: bool = False) -> NormalizedText:
    """Return the raw/surface/model forms plus issues for one text span.

    ``strict=True`` raises :class:`UnsupportedInputError` instead of returning
    issues, and also rejects empty text.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    transformations: list[str] = []

    nfc, changed = _to_nfc(text)
    if changed:
        transformations.append("applied Unicode NFC")

    mapped, mapping_notes = _map_characters(nfc)
    transformations.extend(mapping_notes)

    surface, collapsed = _collapse_whitespace(mapped)
    if collapsed:
        transformations.append("collapsed whitespace runs")

    lowered = surface.lower()
    if lowered != surface:
        transformations.append("lowercased model-facing form")

    tokens, syllables, issues = _classify(surface)
    if not surface:
        issues.insert(0, Issue(code=ISSUE_EMPTY_TEXT, message="text is empty", position=0))

    result = NormalizedText(
        raw=text,
        surface=surface,
        normalized=lowered,
        tokens=tokens,
        syllables=syllables,
        issues=tuple(issues),
        transformations=tuple(transformations),
    )
    if strict:
        result.require_valid()
    return result

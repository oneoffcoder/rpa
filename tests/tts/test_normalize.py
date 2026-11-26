from __future__ import annotations

import pytest

from rpa_tts.errors import UnsupportedInputError
from rpa_tts.text.normalize import (
    ISSUE_EMPTY_TEXT,
    ISSUE_UNAPPROVED_DIGITS,
    ISSUE_UNKNOWN_WORD,
    ISSUE_UNSUPPORTED_CHARACTER,
    normalize_rpa,
)


def test_valid_sentence_preserves_every_tone_and_word():
    normalized = normalize_rpa("kuv hlub koj")
    assert normalized.normalized == "kuv hlub koj"
    assert normalized.syllables == ("kuv", "hlub", "koj")
    assert normalized.is_valid


def test_case_is_folded_only_in_the_model_form_and_recorded():
    normalized = normalize_rpa("Kuv Hlub Koj")
    assert normalized.surface == "Kuv Hlub Koj"
    assert normalized.normalized == "kuv hlub koj"
    assert any("lowercased" in note for note in normalized.transformations)
    assert normalized.raw == "Kuv Hlub Koj"


def test_whitespace_collapse_keeps_words_and_punctuation():
    normalized = normalize_rpa("  kuv   hlub\nkoj.  ")
    assert normalized.surface == "kuv hlub koj."
    assert normalized.tokens == ("kuv", "hlub", "koj", ".")


def test_typographic_quotes_are_mapped_and_the_change_is_recorded():
    normalized = normalize_rpa("\u201ckuv\u201d")
    assert normalized.surface.startswith('"kuv"')
    assert any("mapped U+201C" in note for note in normalized.transformations)
    assert any(issue.code == ISSUE_UNSUPPORTED_CHARACTER for issue in normalized.issues)


def test_nfc_normalization_is_applied():
    decomposed = "ku\u0301"
    normalized = normalize_rpa(decomposed)
    assert normalized.surface == "k\u00fa"
    assert any("NFC" in note for note in normalized.transformations)
    assert any(issue.code == ISSUE_UNKNOWN_WORD for issue in normalized.issues)


def test_digits_are_rejected_not_guessed():
    normalized = normalize_rpa("kuv muaj 3 tug")
    assert any(issue.code == ISSUE_UNAPPROVED_DIGITS for issue in normalized.issues)
    assert not normalized.is_valid


def test_unknown_foreign_words_are_reported_with_position():
    normalized = normalize_rpa("kuv 3 hello")
    codes = [issue.code for issue in normalized.issues]
    assert ISSUE_UNAPPROVED_DIGITS in codes
    assert ISSUE_UNKNOWN_WORD in codes
    unknown = next(issue for issue in normalized.issues if issue.code == ISSUE_UNKNOWN_WORD)
    assert unknown.token == "hello"
    assert unknown.position == 6


def test_empty_text_is_reported_and_strict_mode_raises():
    normalized = normalize_rpa("   ")
    assert any(issue.code == ISSUE_EMPTY_TEXT for issue in normalized.issues)
    with pytest.raises(UnsupportedInputError, match="empty RPA text"):
        normalize_rpa("   ", strict=True)


def test_strict_mode_lists_every_issue_in_the_error():
    with pytest.raises(UnsupportedInputError) as caught:
        normalize_rpa("hello 3", strict=True)
    message = str(caught.value)
    assert ISSUE_UNKNOWN_WORD in message
    assert ISSUE_UNAPPROVED_DIGITS in message


def test_non_string_input_is_a_type_error():
    with pytest.raises(TypeError):
        normalize_rpa(None)  # type: ignore[arg-type]


def test_unmarked_tone_and_multigraph_survive_normalization():
    normalized = normalize_rpa("Nplhaj ku")
    assert normalized.syllables == ("nplhaj", "ku")
    assert normalized.normalized == "nplhaj ku"

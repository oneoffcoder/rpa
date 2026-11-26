from __future__ import annotations

import pytest

from rpa_tts.text.orthography import (
    CONSONANTS,
    MARKED_TONES,
    ONSETS,
    TONES,
    VOWELS,
    expected_syllables,
    inventory_size,
    is_syllable,
    parse_syllable,
)


def test_inventory_matches_documented_product():
    assert inventory_size() == len(ONSETS) * len(VOWELS) * len(TONES) == 5187
    assert len(expected_syllables()) == 5187
    assert len(CONSONANTS) == 56
    assert len(TONES) == 7


def test_unmarked_tone_is_an_explicit_inventory_entry():
    syllable = parse_syllable("ku")
    assert (syllable.consonant, syllable.vowel, syllable.tone) == ("k", "u", "_")
    assert syllable.tone_letter == ""


def test_vowel_only_syllable_has_empty_onset():
    syllable = parse_syllable("ib")
    assert syllable.consonant == ""
    assert (syllable.vowel, syllable.tone) == ("i", "b")


def test_longest_onset_multigraph_is_recognized():
    syllable = parse_syllable("nplhaj")
    assert syllable.consonant == "nplh"
    assert syllable.vowel == "a"
    assert syllable.tone == "j"


def test_parse_accepts_case_and_surrounding_whitespace_but_not_punctuation():
    assert parse_syllable("  KUV\n").text == "kuv"
    with pytest.raises(ValueError, match="outside this repository"):
        parse_syllable("kuv!")


@pytest.mark.parametrize("bad", ["hello", "xyz", "kuv kuv", "", "3", "hmong"])
def test_unrecorded_spellings_are_rejected(bad):
    assert is_syllable(bad) is False
    with pytest.raises(ValueError):
        parse_syllable(bad)


def test_every_marked_tone_letter_is_distinguishable():
    seen = set()
    for tone in MARKED_TONES:
        syllable = parse_syllable(f"ka{tone}")
        assert syllable.tone == tone
        seen.add(syllable.tone_letter)
    assert seen == set(MARKED_TONES)

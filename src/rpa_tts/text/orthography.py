"""Finite RPA (Romanized Popular Alphabet) syllable inventory used by this repository.

Selectively adapted from the retired PR #1 archive
(commit 96698f0689a489b16868ee85046cac6604dfbb77, ``src/rpa_tts/orthography.py``),
which encoded the same inventory. The archive's MLX experiment code is not
reused.

This is the inventory recorded by the book's syllable recordings: 56 onsets plus
the empty onset, 13 vowels and 7 tone symbols (including the unmarked tone),
giving 5,187 unique strings. It is a project inventory, not a complete validator
of every Hmong dialect, spelling convention, loanword or proper name. Unknown
strings are rejected; they are never silently respelled or phonemized.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

CONSONANTS: tuple[str, ...] = (
    "c",
    "d",
    "f",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "x",
    "y",
    "z",
    "ch",
    "dh",
    "hl",
    "hm",
    "hn",
    "kh",
    "ml",
    "nc",
    "nk",
    "np",
    "nq",
    "nr",
    "nt",
    "ny",
    "ph",
    "pl",
    "qh",
    "rh",
    "th",
    "ts",
    "tx",
    "xy",
    "hml",
    "hny",
    "nch",
    "nkh",
    "nph",
    "npl",
    "nqh",
    "nrh",
    "nth",
    "nts",
    "ntx",
    "plh",
    "tsh",
    "txh",
    "nplh",
    "ntsh",
    "ntxh",
)

ONSETS: tuple[str, ...] = ("", *CONSONANTS)

VOWELS: tuple[str, ...] = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "w",
    "ai",
    "au",
    "aw",
    "ee",
    "ia",
    "oo",
    "ua",
)

MARKED_TONES: tuple[str, ...] = ("j", "s", "v", "m", "g", "b")
UNMARKED_TONE = "_"
TONES: tuple[str, ...] = (*MARKED_TONES, UNMARKED_TONE)

TONE_LETTERS: tuple[str, ...] = MARKED_TONES


@dataclass(frozen=True)
class Syllable:
    """One inventory entry; ``tone`` is ``_`` when no tone letter is written."""

    text: str
    consonant: str
    vowel: str
    tone: str

    @property
    def tone_letter(self) -> str:
        return "" if self.tone == UNMARKED_TONE else self.tone


def _spell(consonant: str, vowel: str, tone: str) -> str:
    return consonant + vowel + ("" if tone == UNMARKED_TONE else tone)


def expected_syllables() -> dict[str, Syllable]:
    """Return a fresh mapping over every unique recorded CVT combination."""
    return {
        _spell(consonant, vowel, tone): Syllable(
            _spell(consonant, vowel, tone), consonant, vowel, tone
        )
        for consonant, vowel, tone in product(ONSETS, VOWELS, TONES)
    }


_SYLLABLES: dict[str, Syllable] = expected_syllables()


def inventory_size() -> int:
    return len(_SYLLABLES)


def parse_syllable(text: str) -> Syllable:
    """Parse one written syllable; never substitute an unrecorded spelling.

    Case differences and surrounding whitespace are accepted. Punctuation,
    digits, other dialects' spellings and unrecorded tone letters are rejected
    rather than guessed, so callers can report the exact token to a native
    reviewer.
    """
    if not isinstance(text, str):
        raise TypeError(f"syllable must be a string, got {type(text).__name__}")
    key = text.strip().lower()
    try:
        return _SYLLABLES[key]
    except KeyError:
        raise ValueError(
            f"Syllable {text!r} is outside this repository's documented CVT inventory"
        ) from None


def is_syllable(text: str) -> bool:
    try:
        parse_syllable(text)
    except ValueError:
        return False
    return True


def component_index() -> dict[str, dict[str, tuple[str, ...]]]:
    """Inventory component lists, useful for coverage-ledger configuration."""
    return {"onsets": list(ONSETS), "vowels": list(VOWELS), "tones": list(TONES)}

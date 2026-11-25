"""The finite inventory documented in this repository, not all Hmong spelling.

The empty onset and the unmarked tone are real inventory entries. Vowels alone
are already included in their product; counting them again yields the old 5,200
figure instead of 5,187 unique strings.
"""

from dataclasses import dataclass
from itertools import product


CONSONANTS = (
    "c", "d", "f", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t",
    "v", "x", "y", "z", "ch", "dh", "hl", "hm", "hn", "kh", "ml", "nc",
    "nk", "np", "nq", "nr", "nt", "ny", "ph", "pl", "qh", "rh", "th",
    "ts", "tx", "xy", "hml", "hny", "nch", "nkh", "nph", "npl", "nqh",
    "nrh", "nth", "nts", "ntx", "plh", "tsh", "txh", "nplh", "ntsh", "ntxh",
)
ONSETS = ("", *CONSONANTS)
VOWELS = ("a", "e", "i", "o", "u", "w", "ai", "au", "aw", "ee", "ia", "oo", "ua")
TONES = ("j", "s", "v", "m", "g", "b", "_")


@dataclass(frozen=True)
class Syllable:
    text: str
    consonant: str
    vowel: str
    tone: str


def expected_syllables() -> dict[str, Syllable]:
    """Return a fresh mapping covering every unique recorded CVT combination."""
    return {
        c + v + ("" if t == "_" else t): Syllable(
            c + v + ("" if t == "_" else t), c, v, t
        )
        for c, v, t in product(ONSETS, VOWELS, TONES)
    }


_SYLLABLES = expected_syllables()


def parse_syllable(text: str) -> Syllable:
    """Parse one spelling; never substitute an unrecognized token silently.

    This accepts case differences and surrounding whitespace, but deliberately
    does not strip punctuation or infer pronunciations for other dialects,
    loanwords, spellings, or tone markers absent from the source inventory.
    """
    normalized = text.strip().lower()
    try:
        return _SYLLABLES[normalized]
    except KeyError:
        raise ValueError(f"Syllable is outside this repository's CVT inventory: {text!r}") from None

"""Character-level RPA text contract for the single-speaker VITS fallback.

Why characters rather than phonemes:

* RPA tone letters are lexical; the repository has no native-approved Hmong G2P
  or phoneme inventory, and the issue forbids silently using an English
  phonemizer.
* A character sequence preserves every tone letter, multigraph onset and vowel
  exactly, and is trivially reversible for auditing.
* A phoneme/tone representation is a later controlled intervention only if
  observed errors justify it (issue F-02).

The contract is versioned (`rpa-chars-1`) and validated against the documented
5,187-syllable inventory. Callers pass text already normalized by
:mod:`rpa_tts.text.normalize`; :func:`encode_rpa` is the convenience wrapper that
normalizes strictly first, so digits/foreign text cannot slip into training.
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Iterable

from ..errors import InputError, UnsupportedInputError
from .normalize import PUNCTUATION, normalize_rpa
from .orthography import expected_syllables

TEXT_CONTRACT_VERSION = "rpa-chars-1"

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

CHARS: tuple[str, ...] = (" ",) + tuple(sorted(PUNCTUATION)) + tuple(string.ascii_lowercase)
TOKENS: tuple[str, ...] = SPECIAL_TOKENS + CHARS
TOKEN_TO_ID: dict[str, int] = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN: dict[int, str] = {index: token for token, index in TOKEN_TO_ID.items()}
CHAR_TO_ID: dict[str, int] = {character: TOKEN_TO_ID[character] for character in CHARS}
SPECIAL_IDS = frozenset(TOKEN_TO_ID[token] for token in SPECIAL_TOKENS)


def vocab_size() -> int:
    return len(TOKENS)


def encode(text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Encode already-normalized RPA text; unknown characters are rejected."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    if not text:
        raise InputError("cannot encode empty text")
    ids: list[int] = []
    if add_bos:
        ids.append(BOS_ID)
    for position, character in enumerate(text):
        token_id = CHAR_TO_ID.get(character)
        if token_id is None:
            name = unicodedata.name(character, "UNKNOWN")
            raise UnsupportedInputError(
                f"character {character!r} (U+{ord(character):04X} {name}) at position {position} "
                f"is outside the {TEXT_CONTRACT_VERSION} character set; pass normalized RPA text"
            )
        ids.append(token_id)
    if add_eos:
        ids.append(EOS_ID)
    return ids


def encode_rpa(text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Strictly normalize RPA text, then encode it."""
    normalized = normalize_rpa(text, strict=True)
    return encode(normalized.normalized, add_bos=add_bos, add_eos=add_eos)


def decode(ids: Iterable[int], *, strip_specials: bool = True) -> str:
    """Decode token ids back to text, rejecting unknown or non-integer ids."""
    characters: list[str] = []
    for index, token_id in enumerate(ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise InputError(f"token id at position {index} must be an integer, got {token_id!r}")
        token = ID_TO_TOKEN.get(token_id)
        if token is None:
            raise InputError(f"token id {token_id} at position {index} is outside the vocabulary")
        if token in SPECIAL_TOKENS:
            if not strip_specials:
                characters.append(token)
            continue
        characters.append(token)
    return "".join(characters)


def validate_inventory() -> dict[str, object]:
    """Prove every documented syllable and character round-trips losslessly."""
    syllables = expected_syllables()
    missing_chars = sorted(set("".join(syllables)) - set(CHARS))
    failures: list[dict[str, object]] = []
    for spelling in syllables:
        ids = encode(spelling)
        decoded = decode(ids)
        if decoded != spelling:
            failures.append({"spelling": spelling, "decoded": decoded, "ids": ids})
    return {
        "text_contract_version": TEXT_CONTRACT_VERSION,
        "vocab_size": vocab_size(),
        "special_tokens": list(SPECIAL_TOKENS),
        "characters": len(CHARS),
        "inventory_syllables": len(syllables),
        "missing_characters": missing_chars,
        "round_trip_failures": failures[:20],
        "failure_count": len(failures),
        "passed": not missing_chars and not failures,
    }


__all__ = [
    "BOS_ID",
    "CHARS",
    "CHAR_TO_ID",
    "EOS_ID",
    "PAD_ID",
    "SPECIAL_TOKENS",
    "TEXT_CONTRACT_VERSION",
    "TOKENS",
    "TOKEN_TO_ID",
    "UNK_ID",
    "decode",
    "encode",
    "encode_rpa",
    "validate_inventory",
    "vocab_size",
]

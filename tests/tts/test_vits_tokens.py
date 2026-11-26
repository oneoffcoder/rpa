from __future__ import annotations

import pytest

from rpa_tts.errors import InputError, UnsupportedInputError
from rpa_tts.text.orthography import expected_syllables
from rpa_tts.text.vits_tokens import (
    BOS_ID,
    CHARS,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    TEXT_CONTRACT_VERSION,
    TOKENS,
    UNK_ID,
    decode,
    encode,
    encode_rpa,
    validate_inventory,
    vocab_size,
)


def test_contract_is_versioned_and_stable():
    assert TEXT_CONTRACT_VERSION == "rpa-chars-1"
    assert SPECIAL_TOKENS == ("<pad>", "<bos>", "<eos>", "<unk>")
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)
    assert len(TOKENS) == vocab_size() == 37
    assert TOKENS[4] == " "


def test_every_documented_syllable_round_trips():
    report = validate_inventory()
    assert report["passed"] is True
    assert report["missing_characters"] == []
    assert report["failure_count"] == 0
    assert report["inventory_syllables"] == len(expected_syllables()) == 5187
    assert report["vocab_size"] == 37


def test_sentence_with_punctuation_round_trips():
    text = "kuv hlub koj. nws yog ib tug neeg!"
    ids = encode(text)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert decode(ids) == text
    assert decode(ids, strip_specials=False).startswith("<bos>")
    assert decode(ids, strip_specials=False).endswith("<eos>")


def test_encode_rpa_normalizes_case_and_whitespace():
    ids = encode_rpa("  Kuv   hlub koj. ")
    assert decode(ids) == "kuv hlub koj."


@pytest.mark.parametrize("text", ["kuv 3", "kuv \u00e9", 'kuv "hlub"', "kuv blorp"])
def test_out_of_contract_text_is_rejected(text):
    with pytest.raises((InputError, UnsupportedInputError)):
        encode_rpa(text)


def test_encode_rejects_digits_in_already_normalized_text():
    with pytest.raises(UnsupportedInputError, match="outside the rpa-chars-1 character set"):
        encode("kuv 3")


def test_empty_text_is_rejected():
    with pytest.raises(InputError, match="empty"):
        encode("")


def test_decode_rejects_unknown_and_non_integer_ids():
    with pytest.raises(InputError, match="outside the vocabulary"):
        decode([BOS_ID, 999])
    with pytest.raises(InputError, match="must be an integer"):
        decode([BOS_ID, "x"])  # type: ignore[list-item]
    with pytest.raises(InputError, match="must be an integer"):
        decode([True])  # type: ignore[list-item]


def test_all_inventory_characters_are_in_the_vocabulary():
    inventory_chars = set("".join(expected_syllables()))
    assert inventory_chars <= set(CHARS)
    assert inventory_chars == set("abcdefghijklmnopqrstuvwxyz")


def test_special_tokens_are_not_emitted_by_normalization():
    ids = encode_rpa("kuv hlub koj")
    assert not any(token in {"<pad>", "<unk>"} for token in decode(ids, strip_specials=False))

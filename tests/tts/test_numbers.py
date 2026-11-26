from __future__ import annotations

import pytest

from rpa_tts.errors import UnsupportedInputError
from rpa_tts.text.numbers import NumberPolicy, default_policy


def test_default_policy_has_no_approved_expansion():
    policy = default_policy()
    assert policy.approved == {}
    assert policy.unknown_tokens("kuv muaj 3 tug") == ["3"]


def test_unapproved_numbers_raise_in_strict_mode():
    with pytest.raises(UnsupportedInputError, match="no approved number"):
        default_policy().expand("kuv muaj $5", strict=True)


def test_approved_expansion_is_applied_whole():
    policy = NumberPolicy(approved={"3": "peb"})
    expanded, applied = policy.expand("kuv muaj 3 tug")
    assert expanded == "kuv muaj peb tug"
    assert applied == ("3",)


def test_expansion_never_replaces_part_of_a_longer_number():
    policy = NumberPolicy(approved={"3": "peb"})
    with pytest.raises(UnsupportedInputError, match=r"\$?13"):
        policy.expand("kuv muaj 13 tug", strict=True)


def test_non_strict_mode_returns_unapproved_tokens_unchanged():
    policy = NumberPolicy(approved={"3": "peb"})
    expanded, applied = policy.expand("3 thiab 13", strict=False)
    assert expanded == "peb thiab 13"
    assert applied == ("3",)


def test_text_without_numbers_is_unchanged():
    expanded, applied = default_policy().expand("kuv hlub koj")
    assert expanded == "kuv hlub koj"
    assert applied == ()

"""Approved number/date/currency expansion.

No expansion table is adopted yet: choosing Hmong number words, date order and
currency phrasing is a native language decision recorded in
``docs/tts/recovery/language-contract.yaml`` (``approved_expansions``). Until
that table is approved, digits are rejected with an explicit error rather than
being read as English numbers or dropped.

The mechanics are implemented and tested with synthetic tables so an approved
mapping can be enabled without new code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..errors import UnsupportedInputError

# Digit runs with common date/currency decorations. The whole match must be an
# approved key: partial replacement would silently change the meaning.
_NUMERIC_PATTERN = re.compile(r"[$€£¥]?\d[\d,./:\-]*\d?%?")


@dataclass(frozen=True)
class NumberPolicy:
    approved: dict[str, str] = field(default_factory=dict)
    contract_version: str = "draft-0"

    def unknown_tokens(self, text: str) -> list[str]:
        return [match for match in _NUMERIC_PATTERN.findall(text) if match not in self.approved]

    def expand(self, text: str, *, strict: bool = True) -> tuple[str, tuple[str, ...]]:
        """Replace approved numeric tokens; return the text and applied tokens.

        Overlapping or partial replacements are impossible because the pattern
        is replaced wholesale. Unapproved tokens raise when ``strict``.
        """
        applied: list[str] = []
        unknown: list[str] = []

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            replacement = self.approved.get(token)
            if replacement is None:
                unknown.append(token)
                return token
            applied.append(token)
            return replacement

        expanded = _NUMERIC_PATTERN.sub(replace, text)
        if unknown and strict:
            raise UnsupportedInputError(
                "no approved number/date/currency expansion for: "
                + ", ".join(sorted(set(unknown)))
                + " (add the exact written form to language-contract.yaml "
                "`approved_expansions` after native approval)"
            )
        return expanded, tuple(applied)


def default_policy() -> NumberPolicy:
    return NumberPolicy(approved={}, contract_version="draft-0")

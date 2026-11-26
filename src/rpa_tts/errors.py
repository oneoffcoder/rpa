"""Typed failures so the CLI can report a precise exit status and next action."""

from __future__ import annotations


class RpaTtsError(Exception):
    """Base class for expected, user-actionable failures."""


class InputError(RpaTtsError, ValueError):
    """Malformed or unusable input (bad path, manifest, config, argument)."""


class UnsupportedInputError(InputError):
    """Text outside the approved contract; never silently cleaned or phonemized.

    The message lists every issue so a caller can preserve the raw text and
    route it to an approved correction path.
    """


class ValidationError(RpaTtsError):
    """A schema or cross-record invariant failed."""


class GateNotPassedError(RpaTtsError):
    """A downstream operation was requested before its recovery gate passed.

    Gated commands exist so interfaces are stable and discoverable, but they
    must refuse to run until recorded evidence shows the dependency passed.
    """

    def __init__(self, gate: str, message: str, next_action: str = "") -> None:
        self.gate = gate
        self.next_action = next_action
        text = f"{gate} has not passed: {message}"
        if next_action:
            text += f" Next action: {next_action}"
        super().__init__(text)

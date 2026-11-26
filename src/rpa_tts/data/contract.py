"""Language contract: the approved speech target and its coverage screen.

The contract is a draft until Jee and a native White Hmong lead approve it
(issue F-02). Code never substitutes an English phonemizer or a guessed
expansion; missing approval is reported as a gate state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..errors import InputError, ValidationError
from ..text.numbers import NumberPolicy
from ..text.orthography import ONSETS, TONES, VOWELS

CONTRACT_VERSION_PATTERN = "draft-"
CELL_TYPES = {"onset_tone", "vowel_tone"}


@dataclass(frozen=True)
class ContrastCell:
    id: str
    type: str
    consonant: str | None
    vowel: str
    tone: str
    attested: bool
    note: str = ""

    def matches(self, consonant: str, vowel: str, tone: str) -> bool:
        if self.vowel != vowel or self.tone != tone:
            return False
        if self.type == "vowel_tone":
            return True
        return (self.consonant or "") == consonant


@dataclass(frozen=True)
class LanguageContract:
    path: Path
    status: str
    dialect_name: str
    normalization_version: str
    number_policy: NumberPolicy
    critical_contrasts: tuple[ContrastCell, ...]
    buckets: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def approved_for_training(self) -> bool:
        return self.status == "approved" and bool(self.critical_contrasts)

    @property
    def attested_cells(self) -> tuple[ContrastCell, ...]:
        return tuple(cell for cell in self.critical_contrasts if cell.attested)

    @classmethod
    def load(cls, path: str | Path) -> LanguageContract:
        contract_path = Path(path).expanduser().resolve()
        try:
            data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise InputError(f"language contract does not exist: {contract_path}") from None
        except yaml.YAMLError as exc:
            raise InputError(f"language contract is not valid YAML: {exc}") from None
        return cls.from_mapping(data, contract_path)

    @classmethod
    def from_mapping(cls, data: Any, path: Path) -> LanguageContract:
        if not isinstance(data, dict):
            raise InputError(f"language contract root must be a mapping: {path}")
        for key in (
            "schema_version",
            "status",
            "dialect",
            "normalization",
            "critical_contrasts",
            "buckets",
        ):
            if key not in data:
                raise InputError(f"language contract is missing {key!r}: {path}")
        if data["schema_version"] != 1:
            raise InputError("language contract schema_version must be 1")
        status = str(data["status"])
        if status not in {"draft", "approved", "rejected"}:
            raise InputError(
                f"language contract status must be draft/approved/rejected, got {status!r}"
            )
        dialect = data["dialect"]
        if not isinstance(dialect, dict) or not dialect.get("name"):
            raise InputError("language contract dialect.name is required")
        normalization = data["normalization"]
        if not isinstance(normalization, dict) or not normalization.get("version"):
            raise InputError("language contract normalization.version is required")
        if not str(normalization["version"]).startswith("rpa-norm-"):
            raise InputError("normalization.version must look like rpa-norm-N")
        expansions = data.get("approved_expansions") or {}
        if not isinstance(expansions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in expansions.items()
        ):
            raise InputError("approved_expansions must map written forms to approved expansions")

        cells: list[ContrastCell] = []
        seen: set[str] = set()
        for index, raw in enumerate(data["critical_contrasts"]):
            if not isinstance(raw, dict):
                raise InputError(f"critical_contrasts[{index}] must be a mapping")
            cell_id = str(raw.get("id", ""))
            if not cell_id or cell_id in seen:
                raise InputError(f"critical_contrasts[{index}] needs a unique id")
            seen.add(cell_id)
            cell_type = str(raw.get("type", "onset_tone"))
            if cell_type not in CELL_TYPES:
                raise InputError(
                    f"critical_contrasts[{index}].type must be onset_tone or vowel_tone"
                )
            vowel = str(raw.get("vowel", ""))
            tone = str(raw.get("tone", ""))
            consonant = raw.get("consonant")
            if vowel not in VOWELS:
                raise InputError(
                    f"critical_contrasts[{index}].vowel is not an RPA vowel: {vowel!r}"
                )
            if tone not in TONES:
                raise InputError(f"critical_contrasts[{index}].tone is not an RPA tone: {tone!r}")
            if cell_type == "onset_tone":
                if consonant not in ONSETS:
                    raise InputError(
                        f"critical_contrasts[{index}].consonant is not an RPA onset: {consonant!r}"
                    )
            elif consonant not in (None, ""):
                raise InputError(
                    f"critical_contrasts[{index}] is vowel_tone and must not set consonant"
                )
            cells.append(
                ContrastCell(
                    id=cell_id,
                    type=cell_type,
                    consonant=None if cell_type == "vowel_tone" else str(consonant),
                    vowel=vowel,
                    tone=tone,
                    attested=bool(raw.get("attested", True)),
                    note=str(raw.get("note", "")),
                )
            )
        buckets = data["buckets"]
        if not isinstance(buckets, list) or not all(isinstance(bucket, str) for bucket in buckets):
            raise ValidationError("buckets must be a list of strings")
        return cls(
            path=path,
            status=status,
            dialect_name=str(dialect["name"]),
            normalization_version=str(normalization["version"]),
            number_policy=NumberPolicy(approved=dict(expansions)),
            critical_contrasts=tuple(cells),
            buckets=tuple(buckets),
            raw=data,
        )

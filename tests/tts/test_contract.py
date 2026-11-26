from __future__ import annotations

import pytest
import yaml

from rpa_tts.data.contract import LanguageContract
from rpa_tts.errors import InputError

BASE = {
    "schema_version": 1,
    "status": "draft",
    "dialect": {"name": "White Hmong / Hmong Daw / Hmoob Dawb"},
    "normalization": {"version": "rpa-norm-1"},
    "approved_expansions": {},
    "critical_contrasts": [
        {"id": "C-01", "type": "onset_tone", "consonant": "hl", "vowel": "u", "tone": "b"},
        {"id": "C-02", "type": "vowel_tone", "vowel": "ai", "tone": "s", "attested": False},
    ],
    "buckets": [
        "ordinary_sentences",
        "targeted_contrasts",
        "questions_numbers_names",
        "continuous_passages",
        "calibration",
    ],
}


def _write(tmp_path, data):
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_a_draft_contract_with_cells(tmp_path):
    contract = LanguageContract.load(_write(tmp_path, BASE))
    assert contract.dialect_name.startswith("White Hmong")
    assert contract.approved_for_training is False
    assert [cell.id for cell in contract.critical_contrasts] == ["C-01", "C-02"]
    assert [cell.id for cell in contract.attested_cells] == ["C-01"]
    assert contract.number_policy.approved == {}


def test_contract_cells_match_components(tmp_path):
    contract = LanguageContract.load(_write(tmp_path, BASE))
    onset_cell, vowel_cell = contract.critical_contrasts
    assert onset_cell.matches("hl", "u", "b")
    assert not onset_cell.matches("h", "u", "b")
    assert vowel_cell.matches("h", "ai", "s")  # any onset
    assert not vowel_cell.matches("h", "ai", "j")


def test_approved_expansions_become_the_number_policy(tmp_path):
    data = dict(BASE)
    data["approved_expansions"] = {"3": "peb"}
    contract = LanguageContract.load(_write(tmp_path, data))
    assert contract.number_policy.approved == {"3": "peb"}


def test_approved_status_requires_declared_cells(tmp_path):
    data = dict(BASE)
    data["status"] = "approved"
    data["critical_contrasts"] = []
    contract = LanguageContract.load(_write(tmp_path, data))
    assert contract.approved_for_training is False


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data.update(status="maybe"), "status"),
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data["critical_contrasts"][0].update(vowel="xx"), "vowel"),
        (lambda data: data["critical_contrasts"][0].update(tone="z"), "tone"),
        (lambda data: data["critical_contrasts"][0].update(consonant="hlh"), "onset"),
        (
            lambda data: data["critical_contrasts"][1].update(consonant="h"),
            "must not set consonant",
        ),
        (
            lambda data: data["critical_contrasts"].append(dict(data["critical_contrasts"][0])),
            "unique id",
        ),
        (lambda data: data["normalization"].update(version="english"), "rpa-norm"),
        (lambda data: data["approved_expansions"].update({"3": 3}), "approved_expansions"),
    ],
)
def test_invalid_contracts_are_rejected(tmp_path, mutate, message):
    data = {
        **BASE,
        "critical_contrasts": [dict(cell) for cell in BASE["critical_contrasts"]],
        "normalization": dict(BASE["normalization"]),
        "approved_expansions": {},
    }
    mutate(data)
    with pytest.raises(InputError, match=message):
        LanguageContract.load(_write(tmp_path, data))


def test_missing_contract_file_is_an_input_error(tmp_path):
    with pytest.raises(InputError, match="does not exist"):
        LanguageContract.load(tmp_path / "missing.yaml")


def test_repository_contract_is_readable(contract_path):
    contract = LanguageContract.load(contract_path)
    assert contract.status in {"draft", "approved", "rejected"}
    assert contract.normalization_version == "rpa-norm-1"

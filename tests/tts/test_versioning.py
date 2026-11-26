from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rpa_tts.cli import main
from rpa_tts.data.manifest import load_manifest, write_manifest
from rpa_tts.data.versioning import (
    freeze_dataset,
    load_dataset_config,
    verify_dataset_config,
)
from rpa_tts.errors import InputError, ValidationError
from rpa_tts.text.normalize import normalize_rpa


def _dataset(rows_factory):
    rows = rows_factory(7)
    for index, row in enumerate(rows):
        row["id"] = f"u{index:04d}"
        row["group_id"] = f"g{index:04d}"
        row["bucket"] = "ordinary_sentences"
        row["split"] = "train" if index < 5 else ("dev" if index == 5 else "test")
        row["transcript_raw"] = "kuv hlub koj"
        row["transcript_normalized"] = normalize_rpa("kuv hlub koj").normalized
    return rows


def _write_rows(rows, path: Path) -> Path:
    write_manifest(rows, path)
    return path


def test_freeze_writes_immutable_manifest_and_config(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    contract = approved_contract()
    result = freeze_dataset(
        manifest,
        "pilot-a",
        contract_path=contract,
        manifests_dir=tmp_path / "manifests",
        configs_dir=tmp_path / "configs",
        repo_root=tmp_path,
        notes="pilot freeze",
    )
    assert result["manifest_sha256"]
    assert result["coverage"]["screening_passed"] is True
    assert result["summary"]["by_split"]["train"]["rows"] == 5
    config_path = tmp_path / "configs" / "pilot-a.yaml"
    config = load_dataset_config(config_path)
    assert config["dataset_version"] == "pilot-a"
    assert config["split_rows"] == {"train": 5, "dev": 1, "test": 1}
    assert config["contract"]["screening_passed"] is True
    assert config["rights_ids"] == ["rights-owned-01"]
    assert config["speakers"] == ["spk-01"]
    assert config["notes"] == "pilot freeze"
    verify = verify_dataset_config(config_path, repo_root=tmp_path)
    assert verify["ok"] is True
    assert verify["rows"] == 7


def test_freeze_refuses_to_rewrite_a_version(tmp_path, rows_factory, approved_contract):
    manifest = _write_rows(_dataset(rows_factory), tmp_path / "accepted.jsonl")
    contract = approved_contract()
    freeze_dataset(
        manifest,
        "pilot-a",
        contract_path=contract,
        manifests_dir=tmp_path / "manifests",
        configs_dir=tmp_path / "configs",
        repo_root=tmp_path,
    )
    with pytest.raises(InputError, match="already exists"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=contract,
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_freeze_requires_an_approved_contract(tmp_path, rows_factory, approved_contract):
    manifest = _write_rows(_dataset(rows_factory), tmp_path / "accepted.jsonl")
    draft = approved_contract(status="draft", name="draft.yaml")
    with pytest.raises(ValidationError, match="not approved for training"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=draft,
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_freeze_requires_all_splits(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    for row in rows:
        row["split"] = "train"
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    with pytest.raises(ValidationError, match="non-empty train/dev/test"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=approved_contract(),
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_freeze_requires_coverage_screen(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    for row in rows:
        row["transcript_raw"] = "nws yog ib tug neeg"
        row["transcript_normalized"] = normalize_rpa(row["transcript_raw"]).normalized
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    with pytest.raises(ValidationError, match="coverage screen"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=approved_contract(),
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_freeze_rejects_unresolved_rights(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    rows[0]["rights_id"] = "UNRESOLVED"
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    with pytest.raises(ValidationError, match="rights are unresolved"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=approved_contract(),
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_freeze_rejects_group_leakage(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    rows[6]["group_id"] = rows[0]["group_id"]
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    with pytest.raises(ValidationError, match="spans splits"):
        freeze_dataset(
            manifest,
            "pilot-a",
            contract_path=approved_contract(),
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_invalid_version_name_is_rejected(tmp_path, rows_factory, approved_contract):
    manifest = _write_rows(_dataset(rows_factory), tmp_path / "accepted.jsonl")
    with pytest.raises(InputError, match="dataset version"):
        freeze_dataset(
            manifest,
            "Bad Version!",
            contract_path=approved_contract(),
            manifests_dir=tmp_path / "manifests",
            configs_dir=tmp_path / "configs",
            repo_root=tmp_path,
        )


def test_verify_detects_manifest_tampering(tmp_path, rows_factory, approved_contract):
    manifest = _write_rows(_dataset(rows_factory), tmp_path / "accepted.jsonl")
    result = freeze_dataset(
        manifest,
        "pilot-a",
        contract_path=approved_contract(),
        manifests_dir=tmp_path / "manifests",
        configs_dir=tmp_path / "configs",
        repo_root=tmp_path,
    )
    frozen = tmp_path / result["manifest_path"]
    rows = load_manifest(frozen)
    rows[0]["notes"] = "changed"
    write_manifest(rows, frozen)
    with pytest.raises(ValidationError, match="hash changed"):
        verify_dataset_config(tmp_path / "configs" / "pilot-a.yaml", repo_root=tmp_path)


def test_load_dataset_config_requires_schema(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(Exception, match="validation failed"):
        load_dataset_config(path)
    with pytest.raises(InputError, match="does not exist"):
        load_dataset_config(tmp_path / "missing.yaml")


def test_bucket_summary_only_counts_present_buckets(tmp_path, rows_factory, approved_contract):
    rows = _dataset(rows_factory)
    rows[0]["bucket"] = "targeted_contrasts"
    rows[0]["transcript_raw"] = "kuv hlub koj"
    rows[0]["transcript_normalized"] = normalize_rpa(rows[0]["transcript_raw"]).normalized
    manifest = _write_rows(rows, tmp_path / "accepted.jsonl")
    result = freeze_dataset(
        manifest,
        "pilot-a",
        contract_path=approved_contract(),
        manifests_dir=tmp_path / "manifests",
        configs_dir=tmp_path / "configs",
        repo_root=tmp_path,
    )
    config = load_dataset_config(tmp_path / "configs" / "pilot-a.yaml")
    assert set(config["buckets"]) == {"ordinary_sentences", "targeted_contrasts"}
    assert config["buckets"]["targeted_contrasts"] == 1
    assert json.loads(json.dumps(result["config"]))["dataset_version"] == "pilot-a"


def test_cli_freeze_and_verify_dataset(tmp_path, rows_factory, approved_contract, capsys):
    manifest = _write_rows(_dataset(rows_factory), tmp_path / "accepted.jsonl")
    contract = approved_contract()
    code = main(
        [
            "corpus",
            "freeze",
            "--manifest",
            str(manifest),
            "--version",
            "pilot-a",
            "--contract",
            str(contract),
            "--manifests-dir",
            str(tmp_path / "manifests"),
            "--configs-dir",
            str(tmp_path / "configs"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_sha256"]
    config_path = tmp_path / "configs" / "pilot-a.yaml"
    assert config_path.is_file()

    code = main(["corpus", "verify-dataset", "--config", str(config_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dataset_version"] == "pilot-a"

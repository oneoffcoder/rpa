"""Freeze an accepted dataset version (F-05) and record its configuration.

Freezing is deliberately strict: it requires native-approved rows with resolved
rights, all three splits, no group leakage, and a passing coverage screen against
the approved language contract. A frozen version is immutable; a change creates a
new version rather than rewriting history.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Any

from ..data.contract import LanguageContract
from ..data.coverage import coverage_from_contract
from ..data.manifest import (
    load_manifest,
    manifest_summary,
    require_valid_rows,
    write_manifest,
)
from ..errors import InputError, ValidationError
from ..hashing import sha256_file
from ..paths import find_repo_root
from ..schemas import validate as validate_schema

VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
DATASET_CONFIG_SCHEMA_VERSION = 1


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def freeze_dataset(
    manifest_path: str | Path,
    version: str,
    *,
    contract_path: str | Path,
    manifests_dir: str | Path | None = None,
    configs_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    min_examples: int = 5,
    notes: str = "",
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else find_repo_root()
    source = Path(manifest_path)
    if not VERSION_PATTERN.fullmatch(version):
        raise InputError(
            f"dataset version must match ^[a-z0-9][a-z0-9._-]{{2,63}}$ (got {version!r})"
        )
    rows = load_manifest(source)
    require_valid_rows(rows, purpose="training")
    summary = manifest_summary(rows)
    missing_splits = [
        split for split in ("train", "dev", "test") if summary["by_split"][split]["rows"] == 0
    ]
    if missing_splits:
        raise ValidationError(
            f"dataset freeze needs non-empty train/dev/test splits; empty: {missing_splits}"
        )

    contract = LanguageContract.load(contract_path)
    if not contract.approved_for_training:
        raise ValidationError(
            "language contract is not approved for training; F-02 native approval and attested "
            "cells are required before freezing a dataset version"
        )
    coverage = coverage_from_contract(rows, contract, min_examples=min_examples)
    if not coverage["screening_passed"]:
        raise ValidationError(
            "coverage screen is not passed; declared attested cells are missing or underfilled "
            f"(missing={coverage['missing']}, underfilled={coverage['underfilled']})"
        )

    manifests_root = Path(manifests_dir) if manifests_dir else root / "data" / "manifests"
    configs_root = Path(configs_dir) if configs_dir else root / "configs" / "tts" / "data"
    version_dir = manifests_root / version
    manifest_out = version_dir / "accepted.jsonl"
    config_out = configs_root / f"{version}.yaml"
    if version_dir.exists() or config_out.exists():
        raise InputError(
            f"dataset version {version!r} already exists; freeze a new version instead of rewriting"
        )

    manifest_sha = write_manifest(rows, manifest_out)
    preprocessing_versions = sorted({str(row["preprocessing_version"]) for row in rows})
    normalization_versions = sorted({str(row["normalization_version"]) for row in rows})
    rights_ids = sorted({str(row["rights_id"]) for row in rows})
    speakers = sorted({str(row["source"]["speaker_id"]) for row in rows})
    buckets = {
        bucket: counts["rows"] for bucket, counts in summary["by_bucket"].items() if counts["rows"]
    }
    config = {
        "schema_version": DATASET_CONFIG_SCHEMA_VERSION,
        "dataset_version": version,
        "created_at": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
        "manifest_path": _relative(manifest_out, root),
        "manifest_sha256": manifest_sha,
        "normalization_version": normalization_versions[0]
        if len(normalization_versions) == 1
        else "mixed",
        "preprocessing_versions": preprocessing_versions,
        "split_rows": {
            split: summary["by_split"][split]["rows"] for split in ("train", "dev", "test")
        },
        "split_hours": {
            split: summary["by_split"][split]["hours"] for split in ("train", "dev", "test")
        },
        "buckets": buckets,
        "rights_ids": rights_ids,
        "speakers": speakers,
        "contract": {
            "path": _relative(Path(contract_path), root),
            "sha256": sha256_file(Path(contract_path)),
            "status": contract.status,
            "declared_cells": coverage["declared_cells"],
            "screening_passed": coverage["screening_passed"],
            "missing_cells": coverage["missing"],
            "underfilled_cells": coverage["underfilled"],
        },
        "notes": notes,
    }
    validate_schema(config, "dataset-version.schema.json")
    config_out.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    try:
        config_out.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    except BaseException:
        manifest_out.unlink(missing_ok=True)
        if version_dir.exists() and not any(version_dir.iterdir()):
            version_dir.rmdir()
        raise
    return {
        "config": config,
        "config_path": _relative(config_out, root),
        "manifest_path": _relative(manifest_out, root),
        "manifest_sha256": manifest_sha,
        "summary": summary,
        "coverage": {
            "declared_cells": coverage["declared_cells"],
            "missing": coverage["missing"],
            "underfilled": coverage["underfilled"],
            "screening_passed": coverage["screening_passed"],
        },
    }


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise InputError(f"dataset config does not exist: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_schema(data, "dataset-version.schema.json")
    return data


def verify_dataset_config(
    path: str | Path, *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Re-hash the frozen manifest and report resolvability."""
    root = Path(repo_root) if repo_root else find_repo_root()
    config = load_dataset_config(path)
    manifest = root / config["manifest_path"]
    if not manifest.is_file():
        raise InputError(f"frozen manifest does not exist: {manifest}")
    actual = sha256_file(manifest)
    manifest_sha = actual
    if actual != config["manifest_sha256"]:
        raise ValidationError(
            f"frozen manifest hash changed: recorded {config['manifest_sha256']}, actual {actual}"
        )
    rows = load_manifest(manifest)
    normalization = sorted({str(row["normalization_version"]) for row in rows})
    return {
        "dataset_version": config["dataset_version"],
        "manifest_path": config["manifest_path"],
        "manifest_sha256": manifest_sha,
        "rows": len(rows),
        "normalization_versions": normalization,
        "splits": config["split_rows"],
        "hours": config["split_hours"],
        "ok": True,
    }


def dataset_json(config: dict[str, Any]) -> str:
    import json

    return json.dumps(config, indent=2, sort_keys=True)


__all__ = [
    "freeze_dataset",
    "load_dataset_config",
    "verify_dataset_config",
]

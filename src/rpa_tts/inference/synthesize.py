"""Local synthesis interface and package inspection.

Synthesis remains gated on F-07. Package structure can be inspected today so a
reviewer can see whether an accepted package is complete before any audio is
produced; inspection is not an acceptance judgment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import InputError, RpaTtsError, ValidationError
from ..hashing import sha256_file
from ..schemas import validate as validate_schema
from ..text.normalize import normalize_rpa
from ..training.gates import require_gate_passed

MODEL_MANIFEST = "model-manifest.json"
PENDING_RIGHTS = {"UNRESOLVED", "pending", "PENDING", "none", "NONE", ""}


def _load_manifest_json(directory: Path) -> dict[str, Any]:
    manifest_path = directory / MODEL_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{manifest_path} is not valid JSON: {exc}") from None
    if not isinstance(manifest, dict):
        raise ValidationError(f"{manifest_path} root must be an object")
    validate_schema(manifest, "model-manifest.schema.json")
    return manifest


def _check_file(directory: Path, relative: str, expected_sha: str, label: str) -> dict[str, str]:
    path = directory / relative
    if not path.is_file():
        return {"name": label, "status": "failed", "detail": f"missing file: {relative}"}
    actual = sha256_file(path)
    if actual != expected_sha:
        return {
            "name": label,
            "status": "failed",
            "detail": f"{relative} sha256 does not match the manifest",
        }
    return {"name": label, "status": "passed", "detail": f"{relative} hash matches"}


def inspect_package(package_dir: str | Path) -> dict[str, Any]:
    """Return a check-by-check package report without producing audio."""
    directory = Path(package_dir)
    if not directory.is_dir():
        raise InputError(f"package directory does not exist: {directory}")
    manifest_path = directory / MODEL_MANIFEST
    if not manifest_path.is_file():
        return {
            "ok": False,
            "package": str(directory.resolve()),
            "checks": [
                {
                    "name": "manifest",
                    "status": "failed",
                    "detail": f"missing {MODEL_MANIFEST}",
                }
            ],
            "manifest": None,
        }
    manifest = _load_manifest_json(directory)
    checks: list[dict[str, str]] = [
        {"name": "manifest", "status": "passed", "detail": f"{MODEL_MANIFEST} is schema-valid"}
    ]
    checks.append(
        {
            "name": "status_accepted",
            "status": "passed" if manifest["status"] == "accepted" else "failed",
            "detail": f"status={manifest['status']!r}; only an accepted package is usable",
        }
    )
    checks.append(
        _check_file(
            directory,
            str(manifest["checkpoint_path"]),
            str(manifest["checkpoint_sha256"]),
            "checkpoint",
        )
    )
    checks.append(
        _check_file(
            directory,
            str(manifest["config_path"]),
            str(manifest["config_sha256"]),
            "config",
        )
    )
    verdict = manifest["benchmark"]["gate_verdict"]
    checks.append(
        {
            "name": "benchmark_verdict",
            "status": "passed" if verdict == "accepted" else "failed",
            "detail": f"qualification verdict={verdict!r}; packaging requires an accepted verdict",
        }
    )
    checks.append(
        _check_file(
            directory,
            str(manifest["benchmark"]["manifest_path"]),
            str(manifest["benchmark"]["manifest_sha256"]),
            "benchmark_manifest",
        )
    )
    checks.append(
        _check_file(
            directory,
            str(manifest["benchmark"]["scores_path"]),
            str(manifest["benchmark"]["scores_sha256"]),
            "benchmark_scores",
        )
    )
    rights = str(manifest["rights_id"]).strip()
    checks.append(
        {
            "name": "rights_resolved",
            "status": "failed" if rights in PENDING_RIGHTS else "passed",
            "detail": (
                "rights/consent are unresolved"
                if rights in PENDING_RIGHTS
                else f"rights id={rights!r}"
            ),
        }
    )
    checks.append(
        {
            "name": "normalization_contract",
            "status": "passed",
            "detail": f"normalization={manifest['normalization_version']} "
            f"text_contract={manifest['text_contract']}",
        }
    )
    ok = all(check["status"] == "passed" for check in checks)
    return {
        "ok": ok,
        "package": str(directory.resolve()),
        "checks": checks,
        "manifest": manifest,
        "note": "Inspection is structural and hash-based; it is not a new native judgment.",
    }


def load_model_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load a usable accepted package or raise with all failing checks."""
    directory = Path(checkpoint_dir)
    if not (directory / MODEL_MANIFEST).is_file():
        raise InputError(
            f"no {MODEL_MANIFEST} in {directory}; an accepted package must record its checkpoint, "
            "config, text contract and benchmark hashes and be reviewable"
        )
    report = inspect_package(directory)
    if not report["ok"]:
        failures = "; ".join(
            f"{check['name']}: {check['detail']}"
            for check in report["checks"]
            if check["status"] != "passed"
        )
        raise InputError(f"model package is not usable: {failures}")
    return report["manifest"]


def synthesize(
    text: str,
    *,
    checkpoint_dir: str | Path,
    output_path: str | Path,
    config: Config,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Synthesize one UTF-8 RPA string after validating the accepted package.

    The F-07 gate is enforced before anything is written. The actual acoustic
    call belongs to the qualified lane's packaged wrapper; that wrapper does not
    exist yet, so this function raises rather than emitting unqualified audio.
    """
    normalized = normalize_rpa(text, strict=True)
    require_gate_passed("F-07", operation="synthesis", repo_root=repo_root)
    manifest = load_model_manifest(checkpoint_dir)
    if manifest["normalization_version"] != normalized.version:
        raise InputError(
            f"text normalization {normalized.version} does not match the packaged model "
            f"{manifest['normalization_version']}"
        )
    raise RpaTtsError(
        "F-07 passed and the package is valid, but the lane inference wrapper is not implemented "
        "yet. Implement it from the qualified recipe, record its command in "
        "docs/tts/recovery/handoff.md, and then synthesize."
    )


def package(
    checkpoint_dir: str | Path, output_dir: str | Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build a distributable package from an accepted checkpoint (gated)."""
    require_gate_passed("F-08", operation="package build", repo_root=repo_root)
    raise RpaTtsError(
        "F-08 requires a passed native qualification and distribution rights; packaging code is "
        "intentionally not enabled before then."
    )


__all__ = [
    "MODEL_MANIFEST",
    "inspect_package",
    "load_model_manifest",
    "package",
    "synthesize",
]

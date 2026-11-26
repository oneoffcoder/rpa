"""Local synthesis interface. Gated: no unqualified model may synthesize lessons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import InputError, RpaTtsError
from ..hashing import sha256_file
from ..text.normalize import normalize_rpa
from ..training.gates import require_gate_passed

MODEL_MANIFEST = "model-manifest.json"


def load_model_manifest(checkpoint_dir: str | Path) -> dict[str, Any]:
    directory = Path(checkpoint_dir)
    manifest_path = directory / MODEL_MANIFEST
    if not manifest_path.is_file():
        raise InputError(
            f"no {MODEL_MANIFEST} in {directory}; an accepted package must record its checkpoint, "
            "config and normalization hashes and be reviewable"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{manifest_path} is not valid JSON: {exc}") from None
    required = {
        "status",
        "voice_id",
        "model_version",
        "normalization_version",
        "model_sample_rate",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_sha256",
    }
    missing = required - set(manifest)
    if missing:
        raise InputError(f"{manifest_path} is missing keys: {sorted(missing)}")
    if manifest["status"] != "accepted":
        raise InputError(
            f"model package status is {manifest['status']!r}; "
            "only an F-07-accepted checkpoint may be used"
        )
    checkpoint = directory / str(manifest["checkpoint_path"])
    if not checkpoint.is_file():
        raise InputError(f"checkpoint does not exist: {checkpoint}")
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise InputError(f"checkpoint hash does not match the model manifest: {checkpoint}")
    return manifest


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

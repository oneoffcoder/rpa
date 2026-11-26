"""Run manifests: the reproducibility record for every experiment."""

from __future__ import annotations

import datetime as _dt
import json
import secrets
import subprocess
from pathlib import Path
from typing import Any

from ..config import Config
from ..data.manifest import manifest_summary
from ..errors import InputError
from ..hashing import is_sha256_hex, sha256_file
from ..paths import find_repo_root
from ..schemas import RUN_MANIFEST
from ..schemas import validate as validate_schema

RUN_MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def new_run_id(stage: str, *, suffix: str | None = None, now: _dt.datetime | None = None) -> str:
    moment = now or _dt.datetime.now(_dt.UTC)
    stamp = moment.strftime("%Y%m%dt%H%M%Sz")
    token = suffix or secrets.token_hex(3)
    run_id = f"{stage.lower().replace('-', '')}-{stamp}-{token}"
    if not is_valid_run_id(run_id):
        raise InputError(f"generated run id is not schema-safe: {run_id!r}")
    return run_id


def is_valid_run_id(run_id: str) -> bool:
    import re

    return bool(re.fullmatch(r"[a-z0-9][a-z0-9._-]{3,127}", run_id))


def code_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else find_repo_root()

    def run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def _relative(path: Path, repo_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def hash_outputs(items: list[tuple[Path, str]], repo_root: Path) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for path, kind in items:
        candidate = Path(path)
        if not candidate.is_file():
            raise InputError(f"declared output does not exist: {candidate}")
        outputs.append(
            {
                "path": _relative(candidate, repo_root),
                "sha256": sha256_file(candidate),
                "kind": kind,
            }
        )
    return outputs


def build_run_manifest(
    *,
    config: Config,
    run_id: str,
    stage: str,
    command: str,
    dataset_manifest: Path,
    rows: list[dict[str, Any]],
    environment: dict[str, Any],
    outputs: list[dict[str, str]] | None = None,
    training: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    failures: int = 0,
    failure_notes: str = "",
    notes: str = "",
    status: str = "planned",
    created_at: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else find_repo_root()
    if not is_valid_run_id(run_id):
        raise InputError(f"invalid run id: {run_id!r}")
    summary = manifest_summary(rows)
    training_block = {
        "seed": config.seed,
        "precision": config.precision,
        "effective_batch_size": 1,
        "gradient_accumulation": 1,
        "optimizer_updates": 0,
        "learning_rate": None,
        "max_steps": 0,
    }
    training_block.update(training or {})
    resources_block = {"wall_time_seconds": 0.0, "peak_vram_mb": None}
    resources_block.update(resources or {})
    gpu = environment.get("gpu", {})
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage,
        "created_at": created_at or utc_now(),
        "command": command,
        "status": status,
        "code": code_state(root),
        "config": {
            "path": _relative(config.path, root),
            "sha256": sha256_file(config.path),
        },
        "dataset": {
            "manifest_path": _relative(dataset_manifest, root),
            "manifest_sha256": sha256_file(dataset_manifest),
            "unique_audio_hours": summary["accepted_training_hours"],
            "rows": summary["rows"],
            "split_rows": {
                name: summary["by_split"][name]["rows"] for name in ("train", "dev", "test")
            },
        },
        "environment": {
            "hostname": str(environment.get("hostname", "unknown")),
            "os": f"{environment['platform']['system']} {environment['platform']['release']}",
            "python": str(environment.get("python", "")),
            "device": config.device,
            "gpu_name": gpu.get("name"),
            "gpu_memory_mb": gpu.get("memory_total_mb"),
            "driver": gpu.get("driver_version"),
            "torch": environment.get("packages", {}).get("torch"),
            "cuda": environment.get("packages", {}).get("torch_cuda_runtime"),
            "precision": config.precision,
        },
        "training": training_block,
        "resources": {
            "wall_time_seconds": float(resources_block["wall_time_seconds"]),
            "peak_vram_mb": resources_block["peak_vram_mb"],
        },
        "outputs": outputs or [],
        "failures": {"count": int(failures), "notes": failure_notes},
        "notes": notes,
    }
    validate_schema(manifest, "run-manifest.schema.json")
    return manifest


def write_run_manifest(
    manifest: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise InputError(f"run manifest already exists (append history instead): {destination}")
    validate_schema(manifest, "run-manifest.schema.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise InputError(f"run manifest does not exist: {candidate}")
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"run manifest is not valid JSON: {candidate}: {exc}") from None
    validate_schema(data, "run-manifest.schema.json")
    return data


def validate_output_hashes(manifest: dict[str, Any], repo_root: Path | None = None) -> list[str]:
    """Check recorded outputs still resolve; returns human-readable problems."""
    root = Path(repo_root) if repo_root else find_repo_root()
    problems: list[str] = []
    for output in manifest["outputs"]:
        path = Path(output["path"])
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            problems.append(f"missing output: {output['path']}")
            continue
        if not is_sha256_hex(output["sha256"]) or sha256_file(resolved) != output["sha256"]:
            problems.append(f"output hash changed: {output['path']}")
    return problems


__all__ = [
    "RUN_MANIFEST",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "build_run_manifest",
    "code_state",
    "hash_outputs",
    "load_run_manifest",
    "new_run_id",
    "utc_now",
    "validate_output_hashes",
    "write_run_manifest",
]

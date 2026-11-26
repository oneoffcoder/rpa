"""Repository and artifact path resolution (no hard-coded absolute paths)."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until a directory containing pyproject.toml."""
    candidate = (Path(start) if start else Path(__file__)).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        "could not locate the repository root (no pyproject.toml above "
        f"{candidate}); pass an explicit path"
    )


def recovery_dir(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else find_repo_root()
    return root / "docs" / "tts" / "recovery"


def evidence_index_path(repo_root: Path | None = None) -> Path:
    return recovery_dir(repo_root) / "evidence-index.json"


def config_dir(repo_root: Path | None = None) -> Path:
    root = Path(repo_root) if repo_root else find_repo_root()
    return root / "configs" / "tts"


def default_config_path(repo_root: Path | None = None) -> Path:
    return config_dir(repo_root) / "default.yaml"


def schema_dir(repo_root: Path | None = None) -> Path:
    return config_dir(repo_root) / "schemas"


def workspace_root(config_value: str | Path | None) -> Path:
    """Resolve the artifact workspace, honouring ``RPA_TTS_WORKSPACE``."""
    override = os.environ.get("RPA_TTS_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    if config_value is None:
        return (find_repo_root() / "data").resolve()
    path = Path(config_value).expanduser()
    return path.resolve() if path.is_absolute() else (find_repo_root() / path).resolve()

"""F-04 Magpie runner: diagnosis, packaged-config resolution, load/save/render.

The qualified runtime is the pinned Linux NeMo stack. This module provides the
real runner entry points and refuses with precise defects instead of pretending a
model loaded. The parts that can be exercised without NeMo (archive/config
resolution, environment diagnosis, parameter fingerprints, private WAV writing,
upstream command wrapping) are tested on this host; GPU load/step and native
listening remain pending until the host/data prerequisites exist.

Release synthesis stays gated on F-07 in `inference/synthesize.py`. This module
only produces private evaluation audio.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config import Config
from ..errors import InputError, RpaTtsError
from ..hashing import sha256_file, sha256_json
from ..paths import find_repo_root
from ..text.normalize import normalize_rpa
from . import backend

MAGPIE = {
    "model_repo": "nvidia/magpie_tts_multilingual_357m",
    "model_revision": "446bff55e1fe3d3aa29c771574e7d2b0fa322990",
    "artifact_sha256": "ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953",
    "nemo_commit": "0305e4b15ac2924219536da44d315397d92ee769",
    "codec_repo": "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
    "codec_revision": "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7",
    "codec_sha256": "28c2518de3e3d5a2c7d9bca40a7ebc0644eb76c60b890970365325bdd8e9f099",
    "byt5_repo": "google/byt5-small",
    "byt5_revision": "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
}
PACKAGED_CONFIG = "model_config.yaml"
PRIVATE_OUTPUT_GUARDS = ("source",)


class MagpieRunnerError(RpaTtsError):
    """A runner prerequisite is missing; the message names the exact defect."""


def effective_packaged_config(
    artifact_path: str | Path, *, destination: str | Path | None = None
) -> dict[str, Any]:
    """Resolve the packaged config the way tar extraction does (last member wins).

    The Magpie `.nemo` contains five same-named `model_config.yaml` members. NeMo
    extracts the archive to a directory, so the last member is the effective one.
    Recording that hash avoids silently inspecting the wrong config.
    """
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise InputError(f"model artifact does not exist: {artifact}")
    configs: list[tuple[str, str, bytes]] = []
    with tarfile.open(artifact, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(PACKAGED_CONFIG):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            configs.append((member.name, hashlib.sha256(data).hexdigest(), data))
    if not configs:
        raise InputError(f"no {PACKAGED_CONFIG} found inside {artifact}")
    effective_name, effective_sha, effective_bytes = configs[-1]
    if destination:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(effective_bytes)
    try:
        config = yaml.safe_load(effective_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise InputError(f"effective packaged config is not valid YAML: {exc}") from None
    if not isinstance(config, dict):
        raise InputError("effective packaged config root is not a mapping")
    selected = {
        key: config.get(key)
        for key in (
            "target",
            "model_type",
            "nemo_version",
            "use_text_conditioning_encoder",
            "legacy_text_conditioning",
            "text_conditioning_tokenizer_name",
            "codecmodel_path",
            "frame_stacking_factor",
        )
    }
    return {
        "member_count": len(configs),
        "member_hashes": [digest for _, digest, _ in configs],
        "effective_member": effective_name,
        "effective_sha256": effective_sha,
        "selected": selected,
        "config": config,
    }


def _check_platform() -> dict[str, Any]:
    system = platform.system()
    if sys.platform.startswith("linux"):
        return {
            "name": "platform",
            "status": "passed",
            "detail": f"{system} {platform.release()}",
        }
    return {
        "name": "platform",
        "status": "failed",
        "code": "platform_unsupported",
        "detail": (
            f"{system} {platform.release()} is not the pinned NeMo GPU platform; "
            "NeMo resolves CUDA torch only on Linux"
        ),
        "remediation": (
            "Enable WSL2 (admin command + reboot) or run on a Linux worker; see "
            "docs/tts/recovery/f04-host-path.md"
        ),
    }


def _check_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {
            "name": "torch",
            "status": "failed",
            "code": "torch_missing",
            "detail": f"PyTorch is not installed: {exc}",
            "remediation": "install torch==2.9.1+cu128 (requirements-gpu.txt)",
        }
    cuda_available = bool(torch.cuda.is_available())
    cuda_runtime = torch.version.cuda
    if not cuda_available or not cuda_runtime:
        return {
            "name": "torch",
            "status": "failed",
            "code": "torch_cuda_unavailable",
            "detail": (
                f"torch={torch.__version__} cuda_runtime={cuda_runtime} "
                f"cuda_available={cuda_available}"
            ),
            "remediation": "install the CUDA 12.8 build and verify the GPU driver",
        }
    return {
        "name": "torch",
        "status": "passed",
        "detail": f"torch={torch.__version__} cuda={cuda_runtime}",
    }


def _check_nemo() -> dict[str, Any]:
    if importlib.util.find_spec("nemo") is None:
        return {
            "name": "nemo",
            "status": "failed",
            "code": "nemo_not_installed",
            "detail": "NeMo Speech is not installed in this environment",
            "remediation": (
                "install NeMo at pinned commit "
                + MAGPIE["nemo_commit"]
                + " in a Linux/WSL2 environment"
            ),
        }
    try:
        from nemo.collections.tts.models import MagpieTTSModel  # noqa: F401
    except Exception as exc:  # import-time framework defects are evidence
        return {
            "name": "nemo",
            "status": "failed",
            "code": "magpie_import_failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "remediation": "inspect the pinned NeMo/QT dependencies; do not open-endedly port",
        }
    return {"name": "nemo", "status": "passed", "detail": "MagpieTTSModel imports"}


def _check_artifacts(config: Config, repo_root: Path) -> dict[str, Any]:
    report = backend.check_lane(config, "magpie", repo_root=repo_root)
    if report["ready"]:
        return {
            "name": "artifacts",
            "status": "passed",
            "detail": "model and codec artifacts match their recorded hashes",
            "lane": report,
        }
    return {
        "name": "artifacts",
        "status": "failed",
        "code": "artifacts_not_verified",
        "detail": report.get("next_action") or "lane artifacts are not ready",
        "remediation": (
            "download the pinned artifacts and record their hashes (f04-artifact-inspection.md)"
        ),
        "lane": report,
    }


def _check_tokenizer_pins(config: Config) -> dict[str, Any]:
    upstream = config.data["model_lanes"]["magpie"].get("upstream", {})
    tokenizer = str(upstream.get("text_conditioning_tokenizer") or "")
    revision = str(upstream.get("tokenizer_revision") or "")
    if tokenizer == MAGPIE["byt5_repo"] and revision == MAGPIE["byt5_revision"]:
        return {
            "name": "tokenizer_pin",
            "status": "passed",
            "detail": f"{tokenizer}@{revision}",
        }
    return {
        "name": "tokenizer_pin",
        "status": "failed",
        "code": "tokenizer_pin_missing",
        "detail": (
            f"expected {MAGPIE['byt5_repo']}@{MAGPIE['byt5_revision']}, "
            f"got {tokenizer or '<unset>'}@{revision or '<unset>'}"
        ),
        "remediation": (
            "record the ByT5 text-conditioning tokenizer repo and revision in the lane config"
        ),
    }


def _check_calibration(manifest_path: str | Path | None, repo_root: Path) -> dict[str, Any]:
    if manifest_path is None:
        return {
            "name": "calibration_data",
            "status": "pending",
            "detail": (
                "no manifest supplied; F-04's Hmong step/synthesis requires F-02 calibration speech"
            ),
            "remediation": "supply an accepted calibration manifest once F-02 exists",
        }
    from ..data.manifest import load_manifest, require_valid_rows

    rows = load_manifest(manifest_path)
    require_valid_rows(rows, purpose="training")
    return {
        "name": "calibration_data",
        "status": "passed",
        "detail": f"{len(rows)} accepted rows in {Path(manifest_path).as_posix()}",
    }


def diagnose_environment(
    config: Config,
    *,
    manifest_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return check-by-check readiness plus the first concrete defect."""
    root = Path(repo_root) if repo_root else find_repo_root()
    checks = [
        _check_platform(),
        _check_torch(),
        _check_nemo(),
        _check_artifacts(config, root),
        _check_tokenizer_pins(config),
        _check_calibration(manifest_path, root),
    ]
    artifact_check = next(check for check in checks if check["name"] == "artifacts")
    packaged_config: dict[str, Any] | None = None
    if artifact_check["status"] == "passed":
        upstream_artifacts = config.data["model_lanes"]["magpie"]["artifacts"]
        artifact = Path(str(upstream_artifacts["model_path"]))
        if not artifact.is_absolute():
            artifact = root / artifact
        try:
            with tempfile.TemporaryDirectory(prefix="rpa-magpie-config-") as tmp:
                resolved = effective_packaged_config(
                    artifact, destination=Path(tmp) / PACKAGED_CONFIG
                )
            packaged_config = {
                "member_count": resolved["member_count"],
                "effective_sha256": resolved["effective_sha256"],
                "selected": resolved["selected"],
            }
            checks.append(
                {
                    "name": "packaged_config",
                    "status": "passed",
                    "detail": (
                        f"{resolved['member_count']} config members; effective "
                        f"{resolved['effective_sha256'][:12]}... "
                        f"({resolved['selected'].get('nemo_version')})"
                    ),
                }
            )
        except (InputError, tarfile.TarError) as exc:
            checks.append(
                {
                    "name": "packaged_config",
                    "status": "failed",
                    "code": "packaged_config_unreadable",
                    "detail": str(exc),
                    "remediation": "re-verify the artifact hash and archive contents",
                }
            )
    defects = [check for check in checks if check["status"] == "failed"]
    pending = [check for check in checks if check["status"] == "pending"]
    return {
        "schema_version": 1,
        "ok": not defects,
        "checks": checks,
        "defects": defects,
        "pending": pending,
        "first_defect": defects[0] if defects else None,
        "packaged_config": packaged_config,
        "pins": MAGPIE,
        "note": (
            "Diagnosis only. A passing diagnosis does not prove a training step, save/reload, "
            "free synthesis or native acceptance."
        ),
    }


def load_magpie(
    artifact_path: str | Path,
    *,
    device: str = "cuda:0",
    checkpoint_path: str | Path | None = None,
) -> Any:
    """Restore the pinned Magpie model, optionally loading trained weights.

    Passing ``checkpoint_path`` loads the checkpoint's ``state_dict`` strictly and
    moves every module to ``device`` before returning, so a caller can verify that
    the trained weights (not just the base artifact) are in use.
    """
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise MagpieRunnerError(f"model artifact does not exist: {artifact}")
    try:
        import torch
        from nemo.collections.tts.models import MagpieTTSModel
    except Exception as exc:
        raise MagpieRunnerError(
            f"NeMo/Magpie is not importable in this environment: {type(exc).__name__}: {exc}"
        ) from exc
    model = MagpieTTSModel.restore_from(str(artifact), map_location=torch.device(device))
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise MagpieRunnerError(f"training checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise MagpieRunnerError("checkpoint does not contain a state_dict")
        model.load_state_dict(state, strict=True)
        model.to(device)
    return model


def ensure_hmong_tokenizer_mapping(model: Any) -> bool:
    """Add the Hmong byte-tokenizer mapping to Magpie's (possibly struct) config.

    Returns True when a mapping change was made. The packaged config is a
    structured OmegaConf DictConfig without an ``hmong`` key, so ordinary item
    assignment raises ConfigKeyError; ``open_dict`` is required.
    """
    config = getattr(model, "cfg", None)
    if config is None:
        return False
    mapping = config.get("language_to_tokenizer_mapping", None)
    if mapping is None:
        config["language_to_tokenizer_mapping"] = {"hmong": ["text_ce_tokenizer"]}
        return True
    if "hmong" in mapping:
        return False
    try:
        mapping["hmong"] = ["text_ce_tokenizer"]
    except Exception:
        from omegaconf import open_dict

        with open_dict(mapping):
            mapping["hmong"] = ["text_ce_tokenizer"]
    return True


def parameter_fingerprint(model: Any) -> str:
    """Exact fingerprint: tensor names, dtype, shape and byte content.

    Sums alone are unsafe (two swapped constant blocks can preserve them), so each
    tensor's raw bytes are hashed and the ordered entries are hashed together.
    """
    if not hasattr(model, "state_dict"):
        raise InputError("model does not expose state_dict")
    import torch

    entries: list[dict[str, Any]] = []
    for name, tensor in sorted(model.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        try:
            payload = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        except (RuntimeError, TypeError):
            payload = detached.reshape(-1).numpy().tobytes()
        entries.append(
            {
                "name": name,
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return sha256_json(entries)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python/NumPy/Torch(+CUDA) RNG state for resumable training."""
    import random

    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [],
    }
    if torch.cuda.is_available():
        state["cuda"] = [
            torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())
        ]
    return state


def write_rng_state(path: str | Path) -> dict[str, Any]:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = capture_rng_state()
    torch.save(state, destination)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "python_rng": True,
        "numpy_rng": True,
        "torch_rng": True,
        "cuda_rng_streams": len(state["cuda"]),
    }


def verify_resume_checkpoint(
    checkpoint_path: str | Path, *, rng_state_path: str | Path
) -> dict[str, Any]:
    """Verify a pinned-trainer checkpoint can resume training (not just weights).

    A NeMo `.nemo` artifact contains weights/config only. Resuming requires the
    PTL trainer checkpoint with optimizer/scheduler/step/epoch state plus an
    explicitly captured RNG sidecar.
    """
    import torch

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise MagpieRunnerError(f"resume checkpoint does not exist: {checkpoint}")
    if checkpoint.suffix.lower() == ".nemo":
        raise MagpieRunnerError(
            ".nemo is a weights/config artifact and cannot resume training; "
            "use the pinned trainer's .ckpt checkpoint with optimizer/scheduler state"
        )
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise MagpieRunnerError(f"checkpoint is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise MagpieRunnerError("checkpoint root is not a mapping")
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "epoch"):
        if key not in data:
            raise MagpieRunnerError(f"checkpoint is missing required key {key!r}")
    if not data["optimizer_states"]:
        raise MagpieRunnerError("checkpoint has no optimizer state; it cannot resume training")
    if not data["lr_schedulers"]:
        raise MagpieRunnerError("checkpoint has no scheduler state; it cannot resume training")
    if int(data["global_step"]) <= 0:
        raise MagpieRunnerError("checkpoint global_step is zero; no training progress to resume")

    rng_path = Path(rng_state_path)
    if not rng_path.is_file():
        raise MagpieRunnerError(
            "RNG sidecar is missing; Python/NumPy/Torch RNG state is required to resume exactly"
        )
    try:
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise MagpieRunnerError(f"RNG sidecar is unreadable: {type(exc).__name__}: {exc}") from exc
    missing = [key for key in ("python", "numpy", "torch") if key not in rng]
    if missing:
        raise MagpieRunnerError(f"RNG sidecar is missing: {missing}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "global_step": int(data["global_step"]),
        "epoch": int(data["epoch"]),
        "optimizer_states": len(data["optimizer_states"]),
        "scheduler_states": len(data["lr_schedulers"]),
        "rng_state_path": str(rng_path),
        "rng_state_sha256": sha256_file(rng_path),
        "cuda_rng_streams": len(rng.get("cuda") or []),
        "passed": True,
        "note": "Resume-state structure verified; a full resume run remains required.",
    }


def save_and_reload(
    model: Any,
    path: str | Path,
    *,
    loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Weights-only round trip: save a NeMo artifact and restore it.

    This proves the `.nemo` weights/config path only. It does **not** prove
    training resume; use :func:`verify_resume_checkpoint` for optimizer,
    scheduler, step/epoch and RNG state.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = parameter_fingerprint(model)
    model.save_to(str(destination))
    if not destination.is_file():
        raise MagpieRunnerError(f"model.save_to did not produce {destination}")
    reloaded = loader(destination) if loader else load_magpie(destination)
    after = parameter_fingerprint(reloaded)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "fingerprint_before": before,
        "fingerprint_after": after,
        "match": before == after,
        "scope": "weights_and_config_only",
        "resume_eligible": False,
        "note": "Use verify_resume_checkpoint for optimizer/scheduler/step/RNG resume evidence.",
    }


def _guard_private_output(path: Path, repo_root: Path) -> None:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    if relative.parts and relative.parts[0] in PRIVATE_OUTPUT_GUARDS:
        raise InputError(
            f"private evaluation audio may not be written under {relative.parts[0]}/; "
            "lesson audio requires F-07 acceptance and explicit integration"
        )


def render_private_audio(
    model: Any,
    text: str,
    output_path: str | Path,
    metadata_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    language: str = "en",
    apply_tn: bool = False,
    speaker_index: int = 0,
    seed: int = 3407,
    sample_rate: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
    strict_normalization: bool = True,
) -> dict[str, Any]:
    """Generate private evaluation audio (never release/lesson integration)."""
    import torch

    root = Path(repo_root) if repo_root else find_repo_root()
    normalized = normalize_rpa(text, strict=strict_normalization)
    output = Path(output_path)
    metadata = Path(metadata_path)
    _guard_private_output(output, root)
    _guard_private_output(metadata, root)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ensure_hmong_tokenizer_mapping(model)
    started = time.perf_counter()
    audio, audio_len = model.do_tts(
        normalized.normalized,
        language=language,
        apply_TN=apply_tn,
        speaker_index=speaker_index,
    )
    elapsed = time.perf_counter() - started
    samples = (
        audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    )
    samples = samples.reshape(-1)
    if samples.size == 0:
        raise MagpieRunnerError("model returned empty audio")
    if not np.isfinite(samples).all():
        raise MagpieRunnerError("model returned non-finite audio")
    rate = int(sample_rate or getattr(model, "sample_rate", 0) or 22050)
    pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    report = {
        "schema_version": 1,
        "private_evaluation": True,
        "release_eligible": False,
        "text": normalized.normalized,
        "normalization": {
            "version": normalized.version,
            "strict": strict_normalization,
            "valid": not normalized.issues,
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "position": issue.position,
                    "token": issue.token,
                }
                for issue in normalized.issues
            ],
        },
        "language": language,
        "apply_tn": apply_tn,
        "speaker_index": speaker_index,
        "seed": seed,
        "sample_rate": rate,
        "samples": int(samples.size),
        "audio_seconds": round(samples.size / rate, 4),
        "elapsed_seconds": round(elapsed, 4),
        "audio_path": str(output),
        "audio_sha256": sha256_file(output),
        "model_audio_length": int(audio_len) if audio_len is not None else None,
        "note": (
            "Private evaluation only. Not an F-07 judgment and must never be published as "
            "accepted lesson audio."
        ),
        **(extra_metadata or {}),
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["metadata_path"] = str(metadata)
    report["metadata_sha256"] = sha256_file(metadata)
    return report


def prepare_magpie_manifest(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    *,
    min_context_seconds: float = 3.0,
) -> dict[str, Any]:
    """Convert accepted rows to the pinned Magpie finetuning manifest fields.

    Each row gets target audio/text plus a same-speaker context. A different
    utterance is preferred for context; if none is long enough, the row's own
    audio is used and disclosed as self-context. Transcripts are the normalized
    RPA text; the dataloader tokenizer must be the pinned byte tokenizer.
    """
    if not rows:
        raise InputError("cannot prepare a Magpie manifest from zero rows")
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in ordered:
        speaker = str(row["source"]["speaker_id"])
        by_speaker.setdefault(speaker, []).append(row)
    prepared: list[dict[str, Any]] = []
    self_context = 0
    cross_context = 0
    for row in ordered:
        speaker = str(row["source"]["speaker_id"])
        candidates = [
            other
            for other in by_speaker[speaker]
            if other["id"] != row["id"] and float(other["duration_seconds"]) >= min_context_seconds
        ]
        context = candidates[0] if candidates else row
        if context["id"] == row["id"]:
            self_context += 1
        else:
            cross_context += 1
        prepared.append(
            {
                "audio_filepath": str(row["audio_path"]),
                "text": str(row["transcript_normalized"]),
                "duration": float(row["duration_seconds"]),
                "context_audio_filepath": str(context["audio_path"]),
                "context_text": str(context["transcript_normalized"]),
                "speaker_id": speaker,
                "utterance_id": str(row["id"]),
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in prepared),
        encoding="utf-8",
    )
    return {
        "path": output.as_posix(),
        "sha256": sha256_file(output),
        "rows": len(prepared),
        "speakers": sorted(by_speaker),
        "self_context_rows": self_context,
        "cross_context_rows": cross_context,
        "min_context_seconds": min_context_seconds,
        "note": (
            "Manifest only; audio files are not copied. Tokenizer name must select the pinned "
            "byte-level text tokenizer in the training command."
        ),
    }


def build_finetune_command(
    *,
    repo_root: str | Path,
    init_checkpoint: str | Path,
    exp_dir: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    audio_dir: str | Path,
    feature_dir: str | Path | None = None,
    tokenizer_name: str = "hmong_chartokenizer",
    devices: int = 1,
    batch_size: int = 1,
    max_epochs: int = 1,
    max_steps: int = 1,
    learning_rate: float = 1e-5,
) -> dict[str, Any]:
    """Build the exact pinned upstream Magpie finetuning command.

    The command follows `docs/source/tts/magpietts-finetuning.rst` at NeMo commit
    0305e4b1: Hydra config, PTL checkpoint init, per-language tokenizer, single
    device and a bounded step count for the F-04 gate.
    """
    root = Path(repo_root)
    init = Path(init_checkpoint)
    if not init.is_absolute():
        init = root / init
    if not init.is_file():
        raise InputError(f"init checkpoint does not exist: {init}")
    for label, path in (("train", train_manifest), ("val", val_manifest)):
        if not Path(path).is_file():
            raise InputError(f"{label} manifest does not exist: {path}")
    feature = Path(feature_dir) if feature_dir else Path(audio_dir)
    command = [
        sys.executable,
        "examples/tts/magpietts.py",
        "--config-path=examples/tts/conf/magpietts",
        "--config-name=magpietts",
        f"+init_from_ptl_ckpt={init}",
        f"exp_manager.exp_dir={Path(exp_dir)}",
        f"+model.text_tokenizers.{tokenizer_name}._target_=AutoTokenizer",
        f"+model.text_tokenizers.{tokenizer_name}.pretrained_model={MAGPIE['byt5_repo']}",
        f"+train_ds_meta.whmn.manifest_path={Path(train_manifest)}",
        f"+train_ds_meta.whmn.audio_dir={Path(audio_dir)}",
        f"+train_ds_meta.whmn.feature_dir={feature}",
        "+train_ds_meta.whmn.sample_weight=1.0",
        f'+train_ds_meta.whmn.tokenizer_names=["{tokenizer_name}"]',
        f"+val_ds_meta.whmn_dev.manifest_path={Path(val_manifest)}",
        f"+val_ds_meta.whmn_dev.audio_dir={Path(audio_dir)}",
        f"+val_ds_meta.whmn_dev.feature_dir={feature}",
        "+val_ds_meta.whmn_dev.sample_weight=1.0",
        f'+val_ds_meta.whmn_dev.tokenizer_names=["{tokenizer_name}"]',
        f"model.codecmodel_path={MAGPIE['codec_repo']}",
        "model.context_duration_min=5.0",
        "model.context_duration_max=5.0",
        "model.alignment_loss_scale=0.0",
        "model.prior_scaling_factor=null",
        f"model.optim.lr={learning_rate}",
        "~model.optim.sched",
        "model.load_cached_codes_if_available=true",
        "trainer.precision=32",
        f"trainer.devices={devices}",
        "trainer.num_nodes=1",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=0",
        f"+trainer.max_steps={max_steps}",
        f"batch_size={batch_size}",
        f"max_epochs={max_epochs}",
    ]
    return {
        "command": command,
        "cwd": str(root),
        "tokenizer_name": tokenizer_name,
        "init_checkpoint": str(init),
        "train_manifest": str(Path(train_manifest)),
        "val_manifest": str(Path(val_manifest)),
        "audio_dir": str(Path(audio_dir)),
        "gated_steps": max_steps,
        "note": (
            "Bounded F-04 command. A production run raises max_steps/max_epochs and uses the "
            "recorded dataset version; the RNG/optimizer resume check is separate."
        ),
    }


def read_pcm16_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV without an audio-codec dependency."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise InputError(
                "verse training audio must be mono 16-bit PCM WAV "
                f"({handle.getnchannels()} channel(s), {handle.getsampwidth()} bytes)"
            )
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, rate


def encode_transcript(
    model: Any, text: str, *, tokenizer_name: str = "text_ce_tokenizer"
) -> list[int]:
    """Single training/inference transcript contract: tokens plus EOS, no BOS.

    Pinned NeMo dataset code appends ``model.eos_id`` after tokenization; using
    this helper for both training batches and controlled inference keeps IDs and
    terminal tokens identical.
    """
    eos_id = getattr(model, "eos_id", None)
    if eos_id is None:
        raise InputError("model has no eos_id; cannot build the transcript contract")
    tokens = model.tokenizer.encode(text, tokenizer_name=tokenizer_name)
    return list(tokens) + [int(eos_id)]


def make_verse_batch(model: Any, row: dict[str, Any], *, device: str = "cuda") -> dict[str, Any]:
    """Build the Magpie training batch for one verse row."""
    import torch

    samples, rate = read_pcm16_wav(row["audio_path"])
    audio = torch.from_numpy(samples).unsqueeze(0).to(device)
    token_ids = encode_transcript(model, row["text"])
    return {
        "text": torch.tensor([token_ids], dtype=torch.long, device=device),
        "text_lens": torch.tensor([len(token_ids)], dtype=torch.long, device=device),
        "audio": audio,
        "audio_lens": torch.tensor([audio.shape[-1]], dtype=torch.long, device=device),
        "sample_rate": rate,
    }


def evaluate_verse_loss(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Mean/min/max Magpie loss over held-out verse rows (no gradients)."""
    import torch

    subset = rows[:limit] if limit else rows
    if not subset:
        raise InputError("no rows to evaluate")
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for row in subset:
            batch = make_verse_batch(model, row, device=device)
            values.append(float(model.process_batch(batch)["loss"].detach().cpu()))
    model.train()
    return {
        "rows": len(values),
        "mean": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def train_verse_steps(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    steps: int,
    learning_rate: float = 1e-5,
    clip_grad_norm: float = 1.0,
    device: str = "cuda",
    eval_rows: list[dict[str, Any]] | None = None,
    eval_every: int = 0,
    log_every: int = 0,
    start_step: int = 0,
    start_row: int = 0,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Run bounded private verse finetuning steps and return losses/history.

    ``start_step``/``start_row`` resume the global step and data cursor. Resuming
    must advance the row cursor, otherwise a long run silently repeats its first
    epoch (the bug this API exists to prevent).
    """
    import torch

    if steps <= 0:
        raise InputError("steps must be positive")
    if not rows:
        raise InputError("no training rows")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise InputError("model has no trainable parameters")
    if optimizer is None:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, steps // 2), gamma=0.5
        )
    model.train()
    losses: list[float] = []
    history: list[dict[str, Any]] = []
    for offset in range(steps):
        row = rows[(start_row + offset) % len(rows)]
        global_step = start_step + offset
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(make_verse_batch(model, row, device=device), global_step)
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if eval_rows and eval_every and (offset + 1) % eval_every == 0:
            history.append(
                {
                    "step": global_step + 1,
                    "eval": evaluate_verse_loss(model, eval_rows, limit=None, device=device),
                }
            )
        if log_every and (offset + 1) % log_every == 0:
            history.append({"step": global_step + 1, "train_loss": losses[-1]})
    return {
        "steps": steps,
        "start_step": start_step,
        "start_row": start_row,
        "next_step": start_step + steps,
        "next_row": (start_row + steps) % len(rows),
        "losses": losses,
        "losses_finite": all(float(np.isfinite(value)) for value in losses),
        "history": history,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def save_training_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    path: str | Path,
    *,
    step: int,
    rng_state_path: str | Path,
) -> dict[str, Any]:
    """Save optimizer/scheduler/step/RNG state and verify resume structure."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = write_rng_state(rng_state_path)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_states": [optimizer.state_dict()],
            "lr_schedulers": [scheduler.state_dict()],
            "global_step": int(step),
            "epoch": 0,
        },
        destination,
    )
    report = verify_resume_checkpoint(destination, rng_state_path=rng["path"])
    return {
        "checkpoint": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "rng": rng,
        "resume_verification": report,
    }


def run_upstream(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run a configured upstream NeMo command and capture its outcome."""
    if not command or not all(isinstance(part, str) for part in command):
        raise InputError("upstream command must be a non-empty list of strings")
    merged_env = dict(os.environ)
    merged_env.update(env or {})
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "ok": completed.returncode == 0,
    }


__all__ = [
    "MAGPIE",
    "MagpieRunnerError",
    "build_finetune_command",
    "capture_rng_state",
    "diagnose_environment",
    "effective_packaged_config",
    "ensure_hmong_tokenizer_mapping",
    "evaluate_verse_loss",
    "encode_transcript",
    "load_magpie",
    "make_verse_batch",
    "parameter_fingerprint",
    "prepare_magpie_manifest",
    "read_pcm16_wav",
    "render_private_audio",
    "run_upstream",
    "save_and_reload",
    "save_training_checkpoint",
    "train_verse_steps",
    "verify_resume_checkpoint",
    "write_rng_state",
]

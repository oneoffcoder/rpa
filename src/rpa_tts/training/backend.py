"""Model-lane readiness and gated training entry points.

F-04 is the single live GPU gate. This module can inspect lane metadata and
artifact hashes today; it refuses to train until the gate passes, and it refuses
to invent a wrapper for a lane whose packaged artifacts have not been audited.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import GateNotPassedError, InputError, RpaTtsError
from ..hashing import sha256_file
from ..paths import find_repo_root
from .gates import require_gate_passed

HEX40 = re.compile(r"^[0-9a-f]{40}$")
PINNED_CODEC_SAMPLE_RATE = 22050  # named NanoCodec candidate, issue F-04


def _resolve(path_value: str | None, repo_root: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _upstream_checks(lane_name: str, upstream: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    if lane_name == "magpie":
        required = ("model", "model_version", "nemo_commit", "codec", "codec_sample_rate")
        missing = [key for key in required if upstream.get(key) in (None, "")]
        checks.append(
            {
                "name": "upstream_metadata",
                "status": "failed" if missing else "passed",
                "detail": f"missing: {missing}" if missing else "Magpie model/codec ids recorded",
            }
        )
        commit = upstream.get("nemo_commit")
        checks.append(
            {
                "name": "nemo_commit_immutable",
                "status": "passed"
                if isinstance(commit, str) and HEX40.fullmatch(commit)
                else "failed",
                "detail": f"NeMo commit {commit!r} must be a 40-character commit hash",
            }
        )
        rate = upstream.get("codec_sample_rate")
        checks.append(
            {
                "name": "codec_sample_rate",
                "status": "passed" if rate == PINNED_CODEC_SAMPLE_RATE else "failed",
                "detail": (
                    f"codec sample rate {rate!r} must equal the named candidate's "
                    f"{PINNED_CODEC_SAMPLE_RATE} Hz or a recorded substitution must be justified"
                ),
            }
        )
    elif lane_name == "vits":
        original = upstream.get("original_commit")
        checks.append(
            {
                "name": "original_commit",
                "status": "passed"
                if isinstance(original, str) and HEX40.fullmatch(original)
                else "failed",
                "detail": f"original VITS commit {original!r} must be a 40-character commit hash",
            }
        )
        implementation = upstream.get("implementation")
        checks.append(
            {
                "name": "implementation_pinned",
                "status": "passed"
                if isinstance(implementation, str)
                and implementation
                and HEX40.fullmatch(str(upstream.get("implementation_commit") or ""))
                else "not_started",
                "detail": (
                    f"implementation {implementation!r} is a candidate; "
                    "pin its commit before the gate"
                ),
            }
        )
    else:
        checks.append(
            {"name": "known_lane", "status": "failed", "detail": f"unknown lane {lane_name!r}"}
        )
    return checks


def _artifact_checks(artifacts: dict[str, Any], repo_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    pairs = [("model", artifacts.get("model_path"), artifacts.get("model_sha256"))]
    if "codec_path" in artifacts or "codec_sha256" in artifacts:
        pairs.append(("codec", artifacts.get("codec_path"), artifacts.get("codec_sha256")))
    for label, raw_path, raw_hash in pairs:
        if not raw_path and not raw_hash:
            checks.append(
                {
                    "name": f"{label}_artifact",
                    "status": "not_started",
                    "detail": "resolve the immutable artifact, record its path and SHA-256",
                }
            )
            continue
        if not raw_path or not raw_hash:
            checks.append(
                {
                    "name": f"{label}_artifact",
                    "status": "failed",
                    "detail": "path and SHA-256 must be recorded together",
                }
            )
            continue
        path = _resolve(str(raw_path), repo_root)
        if path is None or not path.is_file():
            checks.append(
                {
                    "name": f"{label}_artifact",
                    "status": "failed",
                    "detail": f"artifact does not exist: {raw_path}",
                }
            )
            continue
        digest = sha256_file(path)
        checks.append(
            {
                "name": f"{label}_artifact",
                "status": "passed" if digest == raw_hash else "failed",
                "detail": (
                    f"{raw_path} sha256 {'matches' if digest == raw_hash else 'does not match'}"
                ),
            }
        )
    return checks


def check_lane(config: Config, lane_name: str, *, repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else find_repo_root()
    lanes = config.lanes
    if lane_name not in lanes:
        raise InputError(f"unknown model lane {lane_name!r}; configured: {sorted(lanes)}")
    lane = lanes[lane_name]
    checks = _upstream_checks(lane_name, lane.upstream)
    checks.extend(_artifact_checks(lane.artifacts, root))
    statuses = [check["status"] for check in checks]
    ready = lane.enabled and all(status in {"passed", "informational"} for status in statuses)
    next_action = None
    if not ready:
        blockers = [
            check["name"] for check in checks if check["status"] not in {"passed", "informational"}
        ]
        next_action = "resolve lane checks: " + ", ".join(blockers)
    return {
        "lane": lane_name,
        "enabled": lane.enabled,
        "priority": lane.priority,
        "recorded_status": lane.status,
        "ready": ready,
        "checks": checks,
        "next_action": next_action,
    }


def check_lanes(config: Config, *, repo_root: Path | None = None) -> dict[str, Any]:
    reports = {name: check_lane(config, name, repo_root=repo_root) for name in config.lanes}
    ready = sorted(name for name, report in reports.items() if report["ready"])
    return {
        "lanes": reports,
        "ready_lanes": ready,
        "gate": {"name": "F-04", "required_for_training": True},
        "note": (
            "Only one GPU gate is active at a time; Magpie has first priority if its packaged "
            "artifacts support a coherent path (issue F-04)."
        ),
    }


def available_training_lanes(config: Config, *, repo_root: Path | None = None) -> list[str]:
    reports = check_lanes(config, repo_root=repo_root)["lanes"]
    return [
        name
        for name, report in sorted(reports.items(), key=lambda item: item[1]["priority"])
        if report["ready"] and config.lanes[name].enabled
    ]


def require_training_backend(config: Config, *, repo_root: Path | None = None) -> str:
    """Return the lane a training run would use, or raise a precise blocker."""
    gate = require_gate_passed("F-04", operation="training", repo_root=repo_root)
    assert gate["status"] == "passed"
    lanes = available_training_lanes(config, repo_root=repo_root)
    if not lanes:
        blockers = {
            name: report["next_action"]
            for name, report in check_lanes(config, repo_root=repo_root)["lanes"].items()
        }
        raise RpaTtsError("F-04 passed but no lane has verified artifacts; " + repr(blockers))
    return lanes[0]


def training_command(config: Config, lane_name: str) -> list[str]:
    """Return the configured wrapper command, or explain what is missing."""
    lane = config.data["model_lanes"][lane_name]
    command = lane.get("training", {}).get("command")
    if not command:
        raise InputError(
            f"lane {lane_name!r} has no configured training wrapper; add a tested command under "
            f"model_lanes.{lane_name}.training.command after the F-04 gate, and record it in the "
            "runbook. Commands are never invented by the CLI."
        )
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise InputError(f"model_lanes.{lane_name}.training.command must be a list of strings")
    return command


def refusal(run: str) -> GateNotPassedError:
    return GateNotPassedError(
        "F-04",
        f"{run} cannot start: the 5090 model gate has not produced recorded evidence",
        "inspect docs/tts/recovery/model-candidates.md, prepare the Magpie lane, and run the "
        "bounded F-04 gate",
    )

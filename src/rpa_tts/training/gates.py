"""Gate enforcement for downstream commands.

A gated command must refuse to run until recorded evidence says its dependency
passed. Missing evidence is reported as an actionable blocker, never as success.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import GateNotPassedError
from ..evidence import gate_status, load_index
from ..paths import evidence_index_path, find_repo_root

GATES = tuple(f"F-0{index}" for index in range(1, 10))
NEXT_ACTIONS = {
    "F-01": "complete docs/tts/recovery/environment.md and record the GPU kernel proof",
    "F-02": "obtain native approval of the language contract and listener calibration",
    "F-03": "clear source/rights review and adopt a pilot manifest",
    "F-04": "run the bounded model gate on the RTX 5090 and append its evidence",
    "F-05": "freeze an accepted, audited corpus with leakage-safe splits",
    "F-06": "complete the declared learning comparison and freeze a candidate",
    "F-07": "run the locked independent native qualification",
    "F-08": "package the accepted checkpoint and restore it in a clean environment",
    "F-09": "record the explicit serving decision (Not required is acceptable)",
}


def gate_statuses(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = repo_root or find_repo_root()
    index = load_index(evidence_index_path(root), repo_root=root)
    return {gate: gate_status(index, gate) for gate in GATES}


def gate_status_for(gate: str, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {GATES}")
    index = load_index(evidence_index_path(root), repo_root=root)
    return gate_status(index, gate)


def require_gate_passed(
    gate: str,
    *,
    operation: str,
    repo_root: Path | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    status = gate_status_for(gate, repo_root)
    if status["status"] != "passed":
        raise GateNotPassedError(
            gate,
            f"{operation} requires a passed gate, current status is {status['status']!r}"
            + (
                f" (latest evidence {status['evidence_id']})"
                if status.get("evidence_id")
                else " (no evidence recorded)"
            ),
            next_action or NEXT_ACTIONS[gate],
        )
    return status

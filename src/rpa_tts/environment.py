"""Host inventory and a real CUDA forward/backward proof.

F-01 requires measured kernels, not a successful GPU-detection call. The proof
trains a tiny MLP for a few optimizer steps, checks finite loss, real parameter
changes and peak VRAM, and records the exact stack.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import InputError
from .hashing import sha256_file

ENVIRONMENT_SCHEMA_VERSION = 1


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024**3), 2)
        else:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def _gpu_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}
    if completed.returncode != 0:
        return {"available": False, "reason": completed.stderr.strip()[:500]}
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return {"available": False, "reason": f"unexpected nvidia-smi output: {line!r}"}
    result: dict[str, Any] = {
        "available": True,
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mb": int(float(parts[2])),
    }
    if len(parts) >= 4 and parts[3]:
        result["compute_capability"] = parts[3]
    return result


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def collect_environment(repo_root: Path, *, workspace: Path | None = None) -> dict[str, Any]:
    disks: dict[str, Any] = {}
    for label, target in (("repo_root", repo_root), ("workspace", workspace or repo_root)):
        try:
            usage = shutil.disk_usage(Path(target).resolve())
            disks[label] = {
                "path": str(Path(target).resolve()),
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        except OSError as exc:
            disks[label] = {"path": str(target), "error": str(exc)}
    torch_version = _package_version("torch")
    cuda_runtime = None
    if torch_version:
        try:
            import torch

            cuda_runtime = torch.version.cuda
        except Exception:  # pragma: no cover - broken installs are reported, not fatal
            cuda_runtime = None
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "collected_at": utc_now(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "disks": disks,
        "gpu": _gpu_inventory(),
        "packages": {
            "torch": torch_version,
            "torch_cuda_runtime": cuda_runtime,
            "nvidia_nemo_toolkit": _package_version("nemo_toolkit"),
            "numpy": _package_version("numpy"),
            "jsonschema": _package_version("jsonschema"),
            "PyYAML": _package_version("PyYAML"),
        },
        "ffmpeg": _ffmpeg_version(),
        "git": _git_state(repo_root),
        "environment_variables": {
            "RPA_TTS_WORKSPACE": os.environ.get("RPA_TTS_WORKSPACE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def _ffmpeg_version() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.splitlines()[0]


def gpu_kernel_proof(*, seed: int = 3407, steps: int = 8) -> dict[str, Any]:
    """Run a real forward/backward/optimizer sequence on the declared CUDA device.

    Returns a record with status in {passed, failed, not_started}. ``not_started``
    means the GPU stack is not installed; that is a missing prerequisite, not a
    pass. The check deliberately uses no training data and no TTS model.
    """
    base: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "check": "tiny-mlp-fp32-forward-backward",
        "seed": seed,
        "steps": steps,
    }
    try:
        import torch
    except ImportError as exc:
        return {**base, "status": "not_started", "reason": f"PyTorch is not installed: {exc}"}

    base.update(
        {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    )
    if not torch.cuda.is_available():
        return {**base, "status": "failed", "reason": "torch.cuda.is_available() is False"}
    base["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda:0")
        capability = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = torch.randn(64, 128, device=device, generator=generator)
        target = torch.randn(64, 128, device=device, generator=generator)

        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        losses: list[float] = []
        start = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            gradients_present = all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not gradients_present:
                return {**base, "status": "failed", "reason": "a parameter has no finite gradient"}
            optimizer.step()
            value = float(loss.detach())
            if not (value == value and value not in (float("inf"), float("-inf"))):
                return {**base, "status": "failed", "reason": "non-finite loss"}
            losses.append(value)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        change = sum(
            float((parameter.detach() - before).norm().item())
            for parameter, before in zip(model.parameters(), initial_parameters, strict=True)
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        free, total = torch.cuda.mem_get_info(device)

        bf16 = None
        if torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                probe = model(inputs) @ inputs.transpose(0, 1)
            bf16 = {
                "supported": True,
                "matmul_finite": bool(torch.isfinite(probe).all()),
                "dtype": str(probe.dtype),
            }

        return {
            **base,
            "status": "passed",
            "device_name": name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "precision": "fp32",
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "losses": losses,
            "parameter_change_l2_sum": change,
            "parameters_changed": change > 0,
            "all_losses_finite": all(value == value for value in losses),
            "elapsed_seconds": round(elapsed, 4),
            "steps_per_second": round(steps / elapsed, 3) if elapsed > 0 else None,
            "peak_vram_allocated_mb": round(peak_allocated / 1024**2, 2),
            "peak_vram_reserved_mb": round(reserved / 1024**2, 2),
            "device_memory_free_mb": round(free / 1024**2, 1),
            "device_memory_total_mb": round(total / 1024**2, 1),
            "bf16": bf16,
            "numeric_sanity": {
                "finite_loss": True,
                "finite_gradients": True,
                "loss_decreased": losses[-1] < losses[0],
            },
        }
    except Exception as exc:  # OOM and driver faults are evidence, not crashes
        return {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def write_json_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination


def describe_shortfall(environment: dict[str, Any]) -> list[str]:
    """Human-readable prerequisites that are missing from an inventory."""
    missing: list[str] = []
    if not environment["gpu"].get("available"):
        missing.append(f"GPU: {environment['gpu'].get('reason', 'unavailable')}")
    if environment["packages"].get("torch") is None:
        missing.append("PyTorch is not installed in this environment")
    if environment["ffmpeg"] is None:
        missing.append("ffmpeg is not on PATH")
    return missing


def require_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise InputError(f"file does not exist: {candidate}")
    return sha256_file(candidate)

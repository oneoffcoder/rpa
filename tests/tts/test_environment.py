from __future__ import annotations

import os

import pytest

from rpa_tts.environment import collect_environment, describe_shortfall, gpu_kernel_proof


def test_environment_inventory_has_required_facts(repo_root):
    environment = collect_environment(repo_root)
    assert environment["schema_version"] == 1
    assert environment["hostname"]
    assert environment["platform"]["system"]
    assert environment["python"].count(".") >= 1
    assert "gpu" in environment
    assert "packages" in environment
    assert environment["git"]["commit"]
    assert "repo_root" in environment["disks"]


def test_shortfall_lists_missing_gpu_and_torch():
    environment = {
        "gpu": {"available": False, "reason": "nvidia-smi not found"},
        "packages": {"torch": None},
        "ffmpeg": None,
    }
    missing = describe_shortfall(environment)
    assert any("GPU" in item for item in missing)
    assert any("PyTorch" in item for item in missing)
    assert any("ffmpeg" in item for item in missing)


@pytest.mark.skipif(
    os.environ.get("RPA_TTS_RUN_GPU_TESTS") != "1",
    reason="set RPA_TTS_RUN_GPU_TESTS=1 to run the real CUDA kernel proof",
)
def test_gpu_kernel_proof_passes_on_the_declared_device():
    proof = gpu_kernel_proof(seed=3407, steps=8)
    assert proof["status"] == "passed", proof.get("reason")
    assert proof["loss_decreased"] is True
    assert proof["parameters_changed"] is True
    assert proof["peak_vram_allocated_mb"] > 0

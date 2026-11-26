# Reproducible TTS environment (F-01)

**Status:** Passed for the host inventory, pinned stack and real-GPU kernel
proof. Not a claim that a TTS model lane has loaded (that is F-04).

Measured inventory evidence:
`docs/tts/recovery/evidence/f01-environment.json` and
`docs/tts/recovery/evidence/f01-gpu-kernel-proof.json` (hashes in
`evidence-index.json`).

## Host inventory (measured 2026-11-26T08:04:42Z)

| Fact | Value |
| --- | --- |
| Hostname | `Desktop-W9QAA` |
| OS | Windows 11 (AMD64, AMD64 Family 25 Model 33) |
| CPU threads | see `f01-environment.json` (`cpu_count`) |
| RAM | 127.9 GB |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0 |
| Driver | 621.35 |
| Disk (repo/workspace) | 556.13 GB free of 1,862.07 GB |
| Python (TTS env) | 3.12.16 (uv-managed `.venv`) |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime | 12.8 |
| ffmpeg / ffprobe | 9.0.2-full_build (gyan.dev) |
| NeMo Speech | not installed yet (F-04 Magpie lane) |

The workspace reservation of 100–200 GB is satisfied by free space; an
**independent backup is still an owner action** and is not established by this
inventory.

## Pinned stack and reproduction

TTS environment (kept separate from the Sphinx book environment):

```text
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
uv pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` pins `torch==2.9.1` from the CUDA 12.8 wheel index;
`requirements-tts.txt` pins the measured core/dev versions. Do not add torch to
the default install or install a CPU wheel silently: the kernel proof records the
actual build string.

Book environment (separate, pinned): `requirements-docs.txt`; build with
`.venv-docs/Scripts/sphinx-build -M html source build` (Windows) or
`make html` on a host with a pinned `sphinx-build` on PATH. Building is not
publishing.

## Real GPU kernel proof

`rpa-tts environment check --gpu-proof` trains a tiny MLP (128→256→128, batch
64) for eight AdamW steps in FP32 with seed 3407, then checks finite loss,
finite gradients, real parameter change, loss decrease, timing and peak VRAM.
A GPU-detection call alone is not accepted.

| Measure | Result |
| --- | --- |
| Status | **passed** |
| Device | NVIDIA GeForce RTX 5090 (compute capability 12.0) |
| Torch / CUDA runtime | 2.9.1+cu128 / 12.8 |
| Initial → final loss | 1.0487 → 0.8429 (monotone decrease) |
| Parameter change (L2 sum) | 2.7503 (parameters changed) |
| Peak VRAM allocated / reserved | 18.88 MB / 24.00 MB |
| Device memory free / total | 30,822.0 / 32,579.8 MB |
| Elapsed | 15.68 s for 8 steps (includes CUDA init; 0.51 steps/s) |
| BF16 supported | yes (finite BF16 matmul probe) |
| Deterministic seed | 3407 |

Interpretation: forward/backward/optimizer kernels execute on the 5090 and
produce finite gradients and real updates. This proves the F-01 kernel path, not
TTS model fit, throughput at scale or F-04 compatibility. The 0.51 steps/s figure
is dominated by first-use initialization and must not be used as a training
throughput estimate.

## Known gaps and next actions

- NeMo Speech is not installed; Magpie artifact versions/hashes and packaged
  codec pairing are unresolved (F-04). No model gate has run.
- The independent backup destination is not established (owner decision).
- Windows is the measured host. If a Linux worker is used later, re-run
  `rpa-tts environment check --gpu-proof` and append new evidence rather than
  editing this file's measurements.
- No training data exists in this environment; `data/` holds manifests and
  ignored artifact directories only.

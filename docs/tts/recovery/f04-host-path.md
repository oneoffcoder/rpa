# F-04 host path: can the pinned Magpie stack run on this 5090 host?

**Status:** Open owner decision. No model has been loaded. This document records
measured facts so the next step is a concrete choice rather than a guess.

Recorded 2026-11-26 on `Desktop-W9QAA`.

## Measured facts

| Fact | Value |
| --- | --- |
| CPU | AMD Ryzen 9 5950X 16-Core |
| Virtualization firmware enabled | True |
| SLAT (second-level address translation) | True |
| Windows feature `Microsoft-Windows-Subsystem-Linux` | Enabled |
| Windows feature `VirtualMachinePlatform` | **Disabled** |
| WSL2 status | `WSL2 is not supported with your current machine configuration` |
| Installed WSL distributions | none |
| Docker | not installed |
| Linux worker/container available | no |
| GPU | RTX 5090, 32,607 MiB, driver 621.35, compute capability 12.0 |
| Working CUDA stack | PyTorch 2.9.1+cu128 (native Windows), real GPU kernel proof passed |

## Why native Windows is not the qualified Magpie path

The pinned NeMo Speech `pyproject.toml`
(`0305e4b15ac2924219536da44d315397d92ee769`) resolves GPU PyTorch only on Linux:

- `torch==2.11.0+cu129 ; sys_platform == 'linux'`
- `torch==2.12.0+cu132 ; sys_platform == 'linux'`
- non-Linux, non-Darwin uses the `pytorch-cpu` index.

So installing NeMo on native Windows would resolve a CPU PyTorch that cannot
exercise the 5090, and NeMo's supported runtime is Linux. The `.nemo` artifact
and config (`nemo_version: 2.8.0rc0`) are inspectable on Windows, but the F-04
GPU gate (load, Hmong forward/backward, free synthesis, save/resume, VRAM) is not
credibly runnable there. This is a platform/compatibility constraint, not a
model-quality finding.

## Options (owner chooses; local 5090 preferred)

1. **Enable WSL2 on this host (recommended, no purchase).** Requires an
   administrator command, enabling the Virtual Machine Platform feature, a
   reboot, installing a Linux distribution, and the NVIDIA WSL CUDA driver
   (already supported by the installed driver). Firmware virtualization and SLAT
   are already enabled, so the remaining blockers are Windows features + reboot.
   Then run NeMo at the pinned commit inside WSL2 with the GPU.
   - Owner commands (admin PowerShell):
     `wsl --install --no-distribution` (or enable
     `VirtualMachinePlatform` and `Microsoft-Windows-Subsystem-Linux`), reboot,
     then `wsl --install -d Ubuntu`, and verify `nvidia-smi` inside WSL.
   - Cost: $0 plus host reboot/administration time. GPU work stays local.
2. **Use a Linux worker on the same or another GPU.** Keeps NeMo's supported
   platform and avoids Windows feature changes; requires a machine, disk, driver
   and transfer of the hash-verified artifacts. Not available today.
3. **Cloud GPU contingency** (Runpod/Modal) only if a measured need and an
   authorized budget exist. The issue treats cloud as a contingency, not an
   initial purchase; no account or spend is authorized now.

Do not start an open-ended Windows port of NeMo. If WSL2/Linux is not available
within the F-04 gate cap (two engineering days, eight active GPU hours, one
named compatibility correction), record the blocker and activate the single VITS
fallback.

## VITS fallback note

The retained compact single-speaker VITS lane uses PyTorch directly and the
native Windows `torch==2.9.1+cu128` stack is already proven for
forward/backward. A pinned maintained implementation
(`idiap/coqui-ai-TTS` `dev` at `7d547eed8580be4fbcd409b7f3082d1d44acbe64`,
MPL-2.0) is the candidate. This is not yet qualified and does not bypass F-02
calibration data, F-04 evidence or licensing review. Activation requires a
recorded Magpie compatibility failure.

## Next executable step

Owner: choose option 1, 2 or 3. If option 1, the agent can then install and
validate the pinned NeMo stack inside WSL2 and continue the F-04 gate checklist
in `model-candidates.md` and `f04-artifact-inspection.md`. Until then, F-04 stays
`In progress` (preparation), not passed.

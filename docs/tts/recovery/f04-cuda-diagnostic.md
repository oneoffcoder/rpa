# F-04 CUDA diagnostic: Magpie load/inference on the RTX 5090

**Scope:** private diagnostic only. This is **not** the F-04 gate, not a training
step, not Hmong adaptation, not native acceptance, and not release audio.

## Result

| Measure | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| Torch / CUDA | 2.9.1+cu128 / 12.8 |
| Free / total VRAM before | 30,994 / 32,579.8 MiB |
| Model load | 22.45 s |
| Parameters on CUDA | 372,919,838; `param_device=cuda:0`, all parameters CUDA |
| Text | `kuv hlub koj` (ByT5 `text_ce_tokenizer`, `apply_TN=False`, speaker 0) |
| Generation | 18.266 s, synchronized |
| Waveform | 19,456 samples @ 22,050 Hz = **0.8824 s** |
| Audio device before host copy | `cuda:0` |
| Peak allocated / reserved VRAM | **3,527.3 / 3,880.0 MiB** |
| Free VRAM after | 26,968 MiB |
| Audio SHA-256 | `697da45aef2a0b1ef5ba4ff389ed77fbb73046e9795f2052062e3a321dbab4a3` |
| Listened | **no**; no quality claim |

The real hash-verified `.nemo` (`ec675fa8…`), NanoCodec and ByT5 tokenizer were
used. The same diagnostic compatibility shims as the CPU run were required
(`nv_one_logger`/`NeptuneLogger` stubs and `hmong → text_ce_tokenizer`); the
production runner still reports `platform_unsupported` on native Windows and
release synthesis stays F-07-gated.

## Still missing for F-04

- A real training/optimizer step (CPU or GPU).
- A resumed PTL checkpoint with optimizer, scheduler, step/epoch and RNG state.
- Permitted Hmong speech and any native listening/acceptance.
- The qualified Linux/WSL2 path; this Windows diagnostic is private and shimmed.

Evidence: `evidence/f04-cuda-inference.json`.

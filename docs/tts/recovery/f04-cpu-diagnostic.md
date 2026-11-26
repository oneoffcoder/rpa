# F-04 CPU diagnostic: real Magpie load, inference and weights round trip

**Scope:** diagnostic only. This is **not** the F-04 5090 gate, not a training
step, not Hmong adaptation and not native acceptance. Release synthesis remains
F-07-gated.

## What ran

- Environment: Python 3.12.16, `torch 2.9.1+cu128`, PyPI `nemo_toolkit 3.0.0`
  (diagnostic; **not** pinned commit `0305e4b1`), `pyopenjtalk-plus 0.4.1.post10`,
  Lightning 2.6.6. Device: **CPU**.
- Artifact: the hash-verified Magpie `.nemo` (`ec675fa8…`), 5 packaged configs,
  **372,919,838 parameters**, NanoCodec restored from HF, 22,050 Hz, 5 baked
  context speakers.
- Compatibility shims (diagnostic only): stub `nv_one_logger` telemetry API/PTL
  integration and `NeptuneLogger` (the global `overrides`/Lightning mismatch), and
  map `hmong` to the packaged `text_ce_tokenizer` (ByT5). The production runner
  does **not** apply these shims and still refuses this Windows host.
- Inference: `kuv hlub koj` → **1.0217 s** audio in **10.63 s** CPU. Audio SHA-256
  `043544b6…`; **not listened to**, no quality claim (the base model has no Hmong
  adaptation and may be unintelligible).
- Weights round trip: saved a `.nemo` (1,459,763,200 bytes), restored it, and
  compared all **260 tensors: keys equal and bit-exact**; SHA-256
  `6de1d7b1…`. Direct `.nemo` saving fails on Windows inside NeMo's `nemo:`
  artifact-path copy; rewiring archived artifact paths to local paths before
  `save_to` worked. This is still **weights/config only**, not optimizer/scheduler/
  RNG resume; `verify_resume_checkpoint` remains the required resume check.

Exact fields and hashes: `evidence/f04-cpu-inference.json`.

## What did not run

- No 5090 GPU load or measurement.
- No training/optimizer step, no finetuning, no PTL resume.
- No F-04 gate, no native listening, no accepted audio, no release.
- The audio file is ignored under `data/runs/`; only its hash is evidence.

## Meaning

The earlier "cannot load on Windows" conclusion is narrowed: the real weights,
tokenizer and codec can construct and generate on CPU behind clearly labelled
compatibility shims. The remaining F-04 blockers are the qualified Linux/WSL2 GPU
stack, a real training step/resume, and permitted Hmong data.

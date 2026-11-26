# F-04 Magpie runner: real path, executed checks and defects

**Status:** Runner implemented. A **CPU diagnostic** now loads the real Magpie
weights, generates 1.02 s of private audio and round-trips a `.nemo` bit-exactly;
the **5090 GPU load, training step, PTL resume and any listening have not been
executed.** No F-04 gate is claimed. Release synthesis remains gated on F-07.

## Intended execution path

1. **Environment** — Linux/WSL2 with the pinned NeMo commit
   `0305e4b15ac2924219536da44d315397d92ee769`, CUDA 12.x and the 5090. Native
   Windows is not that environment (see defects below).
2. **Artifacts** — Magpie `.nemo`
   (`446bff55…`, SHA-256 `ec675fa8…`) and NanoCodec
   (`fc00890b…`, SHA-256 `28c2518d…`); `rpa-tts model check` verifies hashes.
3. **Packaged config** — the archive contains **five** same-named
   `model_config.yaml` members; `effective_packaged_config()` extracts them in
   order and records the last (effective) one. Current result: 5 members,
   effective SHA-256 and NeMo `2.8.0rc0` metadata.
4. **Tokenizer** — text conditioning uses `google/byt5-small`
   (`68377bd…`), an external tokenizer fetched at load. It is pinned in
   `configs/tts/f04-magpie.yaml`.
5. **Finetune command** — `build_finetune_command()` reproduces the pinned
   upstream Hydra command from
   `docs/source/tts/magpietts-finetuning.rst`:
   `python examples/tts/magpietts.py --config-path=examples/tts/conf/magpietts
   --config-name=magpietts +init_from_ptl_ckpt=…`, a Hmong byte-tokenizer entry,
   `train_ds_meta/val_ds_meta`, codec path, FP32, one device and bounded
   `max_steps`. `rpa-tts train finetune` refuses before building the command when
   the environment diagnosis fails.
6. **Private evaluation audio** — `render_private_audio()` loads the restored
   model, normalizes RPA strictly, runs `do_tts(..., apply_TN=False)`, writes a
   PCM16 WAV plus metadata marked `private_evaluation: true` and
   `release_eligible: false`, refuses paths under `source/`, and is exposed as
   `rpa-tts train render-private` (refused on a failing diagnosis).
7. **Save/reload** — `save_and_reload()` proves the weights/config `.nemo` round
   trip with an exact tensor fingerprint (names, dtypes, shapes and SHA-256 of
   raw bytes). `verify_resume_checkpoint()` separately requires the PTL
   trainer checkpoint with optimizer, scheduler, step/epoch and a captured RNG
   sidecar; a `.nemo` alone is explicitly rejected as non-resumable.

## What was executed on this host

| Check | Result |
| --- | --- |
| Artifact hashes / lane readiness | passed (both `.nemo` hashes match) |
| Effective packaged config | 5 members; last member resolved |
| ByT5 tokenizer pin in lane config | passed |
| Environment diagnosis | failed at `platform_unsupported` |
| `train gate` | refused with the platform defect; diagnosis JSON written |
| Native NeMo CPU import attempt | initially failed; fixed with diagnostic shims (see below) |
| CPU Magpie load (shimmed) | passed: 372,919,838 params, 22.3 s, codec restored |
| CPU private generation (`kuv hlub koj`) | passed mechanically: 1.0217 s in 10.63 s; **not listened to**, no quality claim |
| CPU weights `.nemo` round trip | passed: 260 tensors bit-exact (`6de1d7b1…`); weights/config only |
| GPU load / training step / PTL resume | **not executed** |
| 5090 F-04 gate | **not executed** |
| Native listening | **not performed** |

See `f04-cpu-diagnostic.md` and `evidence/f04-cpu-inference.json` for the shims,
hashes and limitations.

## First reproducible integration defect chain (native Windows diagnostic)

A bounded attempt installed PyPI `nemo-toolkit==3.0.0` (not the pinned source
commit; diagnostic only) into `.venv-nemo` with `torch==2.9.1+cu128` and the TTS
dependencies except `nemo_text_processing` (its `pynini` dependency has no
Windows wheel).

1. **Fixed dependency gap:** import failed with
   `ModuleNotFoundError: nv_one_logger.training_telemetry.integration`.
   Installing `nv-one-logger-pytorch-lightning-integration==2.3.1` resolved it.
2. **Blocking version conflict:** import then failed in
   `nv_one_logger/.../pytorch_lightning.py` at the `@override` check:
   `OneLoggerPTLTrainer.save_checkpoint: weights_only must be a supertype of
   typing.Optional[bool] but is <class 'bool'>` (nv-one-logger 2.3.1 vs
   Lightning 2.6.6). Downgrading `overrides` to 7.6.0 did not change it; no
   newer nv-one-logger is published. This needs a pinned upstream patch or the
   qualified Linux stack — not an open-ended Windows port.
3. **Platform constraint remains:** even with that patched, NeMo resolves CUDA
   torch only on Linux, so the pinned GPU stack requires WSL2 or a Linux worker.
   See `f04-host-path.md`.

Conclusion: the first actionable F-04 defect is the missing/incompatible
nv-one-logger dependency on Windows; the underlying blocker is the host platform.
The runner is ready for the qualified environment; it does not pretend the model
loaded here.

## Recording-capacity constraint (2026-11-26)

Jee will not provide new calibrated recordings or complete the native worksheet
(see `capacity-constraint.md`). Therefore F-02 cannot pass as planned and the
dependent F-04 data step (Hmong training/step speech) has no accepted source.
The runner and private-render code paths can still be exercised when a scope
decision and permitted data exist; no agent-only step satisfies that gate.

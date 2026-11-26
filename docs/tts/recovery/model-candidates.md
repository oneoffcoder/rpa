# F-04 model candidates and gate plan

**Status:** Not started. The lane metadata is recorded and checkable with
`rpa-tts model check`; no model artifact has been downloaded, loaded or trained.
**Owner:** ML engineer (agent). **Dependencies:** F-01 environment (passed),
F-02 calibration speech (blocked on native work).

The F-04 gate is a two-working-day engineering gate per activated family,
initially capped at **8 active GPU hours** for compatibility work (learning-curve
runs are separate). One named compatibility correction is allowed inside the
gate. Stop and record the blocker if a lane needs an open-ended framework port.

Only one GPU model gate is active at a time. Magpie has first priority if its
packaged artifacts support a coherent path; a model's resemblance to earlier
failures neither qualifies nor disqualifies it.

## Lane 1 — Magpie (priority 1)

| Item | Pinned value |
| --- | --- |
| Model | `nvidia/magpie_tts_multilingual_357m`, version `v2607` |
| Reference framework | NVIDIA NeMo Speech commit `0305e4b15ac2924219536da44d315397d92ee769` |
| Codec candidate | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`, 22,050 Hz |
| License | NVIDIA Open Model License (read the exact code/weights/codec terms) |

Before any training:

1. Resolve immutable artifact versions and SHA-256 hashes; record paths in
   `configs/tts/default.yaml` (`model_lanes.magpie.artifacts`).
2. Read the packaged configuration before training. Record `.nemo` versus PTL
   checkpoint format, loader, `legacy_text_conditioning`, tokenizer keys/order
   and full token-ID mapping, embedding dimensions, BOS/EOS IDs,
   context-encoder/baked-table presence and which tensors are trainable.
3. Verify the actual packaged model/codec pairing and sample rate; do not
   substitute an unqualified codec. `rpa-tts model check` enforces the recorded
   22,050 Hz value.
4. Prefer a compatible existing target-text vocabulary over surgery, but prove
   that it preserves the full approved RPA inventory and feeds the intended
   embedding table. Equal row counts do not establish equal token meanings.
   A shared byte or Vietnamese character tokenizer remains only a candidate
   until that test passes.
5. `.nemo` restoration and PTL `state_dict` initialization are different paths.
   The pinned loader's `strict=False` branch falls through to strict child
   loads; dropping the text embedding is not a demonstrated workaround. Any
   migration must preserve existing IDs/weights, initialize new rows/BOS/EOS
   coherently and pass complete load/save/reload checks.

## Lane 2 — single-speaker VITS fallback (priority 2, standby)

Activate only after a recorded Magpie compatibility/codec failure. Use one
from-scratch, single-speaker configuration with a pinned maintained
implementation; the original VITS commit
`2e561ba58618d021b5b8323d3765880f7e0ecfdb` is a behavior anchor and
`idiap/coqui-ai-TTS` is a candidate to inspect and pin, not a prequalified Hmong
recipe.

Do **not** add the proposed MMS Vietnamese checkpoint to the default path: its
shipped vocabulary/normalization loses RPA symbols and its non-commercial weight
terms add a product restriction. Transfer learning remains a future bounded
proposal requiring compatible tokens, weights and rights.

## Gate evidence (per activated lane)

- [ ] Official supported-language inference reproduced.
- [ ] Real Hmong forward/backward update: finite loss, intended gradients, real
      parameter changes.
- [ ] Free-running synthesis without target codes/durations.
- [ ] Complete training-state save → resume → reload.
- [ ] Peak VRAM and per-step/epoch timing measured; resource estimates recorded.
- [ ] FP32 reference where it fits; measured mixed-precision comparison if used
      (smaller batches/accumulation are allowed; do not copy multi-GPU guide
      batch/epoch values).
- [ ] For Magpie: at least 30 new native-reviewed Hmong controls reconstructed
      through the actual codec, compared for lexical/phonation damage. A new
      systematic lexical defect blocks that codec path. This is an acoustic
      representation check, not generated-speech acceptance.
- [ ] Deployment voice fixed; record whether reference audio is used (a frozen
      voice vector alone is not a failure when generator weights adapt).

Exit: one coherent training recipe with a passed technical gate and measured
per-step/epoch resource estimates, or a precise blocker record. Architecture
success and final speech quality remain unknown; tiny-batch fitting and a
checkpoint round trip establish pipeline readiness only.

## Explicitly out of scope

- Another broad model sweep.
- Reopening the optional CosyVoice upstream-versus-custom wrapper check unless an
  identified wrapper discrepancy would change the decision (12 development
  prompts, one declared seed, one repeat only for a suspected stochastic
  failure).
- Multi-speaker conditioning or language pretraining as a substitute for hours
  from the selected target speaker.

## Command surface

```text
rpa-tts model check --config configs/tts/default.yaml
rpa-tts train run --config configs/tts/default.yaml --manifest <frozen.jsonl> --output data/runs/<run_id>
```

`train run` refuses to execute (exit 3) until F-04 has recorded passing evidence
and a tested wrapper command is configured. The CLI never invents a training
command.

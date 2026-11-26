# F-04 VITS standby lane: text and environment contract

**Status:** Standby preparation. The lane is **not activated** and no VITS model
has trained. Magpie remains the first live gate; activation requires a recorded
Magpie compatibility/codec failure. This document records the prepared text
contract and environment plan so the fallback can start without reopening the
architecture debate.

## Pinned implementation

| Item | Value |
| --- | --- |
| Candidate implementation | `idiap/coqui-ai-TTS` (`dev`) |
| Pinned commit | `7d547eed8580be4fbcd409b7f3082d1d44acbe64` |
| Package | `coqui-tts==0.28.0` (PyPI, `TTS` import), MPL-2.0 |
| Python | `>=3.10,<3.15` |
| Original VITS anchor | `2e561ba58618d021b5b8323d3765880f7e0ecfdb` |
| Text contract | `rpa-chars-1` (`src/rpa_tts/text/vits_tokens.py`) |
| Gate config | `configs/tts/f04-vits.yaml` (lane status `standby`, artifacts null) |

`rpa-tts model check --config configs/tts/f04-vits.yaml --lane vits` reports the
pin checks passed and the artifact check `not_started`; that is expected before
the lane is activated.

The MMS Vietnamese checkpoint stays outside the default path: its shipped
vocabulary/normalization loses RPA symbols and its non-commercial terms add a
product restriction. Transfer learning would be a separate bounded proposal.

## Text contract `rpa-chars-1`

Character-level, not phoneme-level:

- 37 tokens: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3, space, `.,!?;:`, and
  `a`–`z`.
- All **5,187** documented syllable spellings round-trip exactly; the inventory
  uses all 26 letters, so no tone letter or multigraph onset is lost.
- `encode_rpa(text)` applies strict `rpa-norm-1` normalization first. Digits,
  foreign characters, unapproved quotes and unrecorded spellings raise; there is
  no English cleaner or phonemizer in the path.
- The ten inventory tone/multigraph distinctions survive because every character
  is preserved in order; the model is free to learn tone from context.
- A phoneme/tone representation is a later controlled intervention only if
  observed errors justify it (issue F-02), not an initial guess.

Rationale: RPA tone letters are lexical. A character contract is lossless,
auditable and reversible, and avoids importing an English G2P or an unapproved
Hmong phoneme inventory. It may limit naturalness; that is a measured risk, not
a hidden assumption.

## Environment plan (measured on this host)

- Separate environment `.venv-vits`, isolated from `.venv` and `.venv-docs`.
- CUDA stack: `torch==2.9.1+cu128` (the F-01 kernel proof already passed on this
  exact build), then `coqui-tts==0.28.0` pinned. No NeMo dependency.
- Windows install is feasible on paper: `monotonic-alignment-search` publishes a
  cp312 Windows wheel and coqui's trainer leaves torch at `>=2.2` (no upper pin).
  The actual install/import result is recorded below when measured.
- Training data still depends on F-02/F-03/F-05; a from-scratch single-speaker
  configuration needs accepted connected speech, not the repository syllables.
- First gate run must start from one saved initialized weight set and seed, be
  compared at 1/2 hours, and record peak VRAM/time. Tiny-batch fitting and a
  checkpoint round trip prove pipeline readiness only, not quality.

## F-04 gate checklist for this lane

- [ ] Real Hmong forward/backward update: finite loss, intended gradients, real
      parameter changes.
- [ ] Free-running synthesis without target codes/durations.
- [ ] Complete training-state save → resume → reload.
- [ ] Measured peak VRAM and per-step/epoch timing.
- [ ] Saved initialized weights + seed frozen for subset comparisons.
- [ ] Deployment voice fixed; note that no pretrained speaker/voice exists for a
      from-scratch model.
- [ ] No codec test is required for this lane unless the implementation adds one.

## Activation rule

Do not activate this lane merely because Magpie is inconvenient. Activate when a
recorded Magpie compatibility or codec gate fails (for example, the packaged
loader/tokenizer contract cannot be made to load or train without an open-ended
port). The current host constraint is a platform limitation with an owner path
(enable WSL2 or use a Linux worker); it is documented in `f04-host-path.md` but
does not by itself replace the Magpie attempt.

## Measured environment result (2026-11-26)

A separate `.venv-vits` was created and the stack imported successfully on this
native Windows host:

```text
torchcodec 0.9.1  coqui 0.28.0  torch 2.9.1+cu128  audio 2.9.1+cu128  cuda True
```

Full freeze: `evidence/f04-vits-pip-freeze.txt`; measurement record:
`evidence/f04-vits-environment.txt`.

Corrections required (each is a recorded Windows risk):

1. coqui-tts 0.28.0's post-install check requires `torchaudio`; installed the
   matching `2.9.1+cu128` wheel.
2. coqui-tts imports `torchcodec` on PyTorch >= 2.9; installed it.
3. `torchcodec==0.17.0` requires torch >= 2.11 and failed to load
   `libtorchcodec_image.dll` on torch 2.9.1; pinned `torchcodec==0.9.1`, whose
   compatibility table targets torch 2.9.
4. torchcodec 0.9.1 supports FFmpeg 4–8. The host's static FFmpeg 9 build ships
   no shared DLLs and master/9 is unsupported, so a BtbN shared FFmpeg 8.1 build
   (90,902,330 B, SHA-256
   `89a40d7c3d4aa627844885e29c8cf4d90034cc19ac8eed43863a86fe7a87a400`) is
   extracted under ignored `data/artifacts/` and its `bin/` is placed on PATH at
   runtime. This is a host runtime dependency, not a code change.

```text
uv venv .venv-vits --python 3.12
uv pip install torch==2.9.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install coqui-tts==0.28.0 torchcodec==0.9.1
# prepend the extracted FFmpeg 8.1 shared bin/ to PATH, then:
.venv-vits/Scripts/python -c "import TTS, torch; print(TTS.__version__, torch.__version__, torch.cuda.is_available())"
```

This proves the import environment only. No model was instantiated or trained;
no data exists; the lane remains standby and is **not** an F-04 gate pass.

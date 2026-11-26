# F-04 VITS standby lane: text and environment contract

**Status:** **Activated for a bounded private diagnostic on 2026-11-26** after the
recorded Magpie/NanoCodec failure. This is not an F-04 gate pass and no VITS
quality claim is made. Magpie remains the first live gate; the VITS lane was
activated only because the Magpie codec reconstruction of the confirmed Hmong
source was not recognizable as Hmong.

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

## Activated bounded diagnostic (2026-11-26)

Trigger: recorded Magpie codec failure (LUK 19:22 source confirmed Hmong; both
Magpie and directly restored NanoCodec round-trips were not recognizable as
Hmong). `Vits` additionally required `six==1.17.0`, installed in `.venv-vits`;
the FFmpeg 8.1 shared `bin/` was placed on `PATH`.

Configuration: single-speaker `VitsConfig`, `rpa-chars-1` (`num_chars=37`),
83,046,892 parameters, seed 3407, batch 3/4, AdamW 2e-4 for discriminator and
generator. Six short clean train verses from 1 Corinthians (4.15–5.75 s) were
paired one-to-one with their exact WAVs; LUK 19:22 was held out. There was
**no transcript dropout or unconditional draw**: every supervised example used
its transcript. Quotes in the held-out LUK text were mapped to spaces only
because `rpa-chars-1` has no apostrophe token.

### Initial probe and corrected measurement

- The first 80-update probe reported distinct WAVs, but its "audio-conditioned
  reconstruction" incorrectly decoded `flow(z)` (prior space). The correct VITS
  copy-synthesis path decodes the posterior `z` directly (`decoder(z*mask)`).
- Corrected probe across saved checkpoints: u100 0.984, u200 0.981, u300 0.965,
  u400 **0.936** lag-256. A further 50 updates reached u450 **0.346**, but the
  MBP still measured a 430.66 Hz / 86 Hz hop artifact, so this was not promoted.
- Old incorrect u400 probe: 0.954 lag; corrected u400: 0.936. The old threshold
  and distinct hashes were not treated as speech.

### Same-family positive control

- Coqui registry LJSpeech VITS (22.05 kHz, hop 256; 36,321,264 parameters)
  copy-synthesized LUK 19:22 through its posterior encoder and waveform decoder:
  **16.0798 s**, lag-256 **−0.00984** (source −0.01097), RMS 0.1203.
- Its waveform decoder is key/shape compatible with the Hmong VITS
  (**231/231 keys**); preserved at
  `data/runs/vits-diagnostic/ljspeech/ljspeech_waveform_decoder.pt`
  (SHA `3418c191…`).

### Promoted paired Hmong fit

- Fresh Hmong `rpa-chars-1` text model with the LJSpeech waveform decoder
  initialized; 200 paired generator/discriminator updates, same six exact
  transcript/WAV pairs, no transcript dropout.
- Corrected reconstruction lag-256: u100 **0.365**, u150 **−0.302**,
  u200 **0.062** — the hop artifact is broken.
- Free-running, fixed speaker/seed 3407:

  | Text | Output | Ref | Ratio | Median F0 out/ref | Lag-256 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 1CO 11:1 | 4.319 s | 4.47 s | 0.966 | 185.1/176.7 Hz | 0.156 |
  | 1CO 12:19 | 2.926 s | 4.15 s | 0.705 | 186.1/181.9 Hz | 0.140 |
  | LUK 19:22 | 13.502 s | 16.08 s | 0.840 | 185.1/147.7 Hz | 0.203 |

- Full-state checkpoint:
  `data/runs/vits-diagnostic/hmong-pretrained-decoder/checkpoint.pt`
  (SHA `d3cca137…`), 200 updates, seed 3407, generator/discriminator optimizer
  states present. Promoted decoder:
  `.../hmong_waveform_decoder.pt` (SHA `3234678a…`, 14,337,024 parameters).
- Outputs: `seen_a.wav`, `seen_b.wav`, `heldout_luk.wav`.

This promotes the **waveform path**, not a native or quality result. Training
still needs native-qualified evaluation; no release claim.

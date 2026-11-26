# F-04 artifact inspection and text-contract findings

**Status:** Preparation only. The pinned artifacts are downloaded and
hash-verified; no model has been loaded, no NeMo environment is installed and no
GPU gate has run. This document freezes the inspection facts so the next worker
does not repeat the download or guess the contract.

Recorded 2026-11-26. Artifacts live under ignored `data/artifacts/` and must be
backed up by the owner.

## Verified immutable artifacts

| Item | Revision / SHA-256 | Size |
| --- | --- | --- |
| Magpie repo revision | `446bff55e1fe3d3aa29c771574e7d2b0fa322990` | — |
| `magpie_tts_multilingual_357m.nemo` | `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953` | 1,470,208,000 B |
| Codec repo revision | `fc00890b604aa2de298d2641ffc6c5f6caf8c4d7` | — |
| `nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo` | `28c2518de3e3d5a2c7d9bca40a7ebc0644eb76c60b890970365325bdd8e9f099` | 425,021,440 B |
| Magpie GGUF f16 / q8_0 | `fb8e9297…f618fc` / `318c8f83…50cf9` | 568,663,392 / 331,812,192 B |
| Codec decoder GGUF f16 | `cc86d36d…c07ef2d` | 78,823,104 B |

The `.nemo` hashes match the Hugging Face LFS object ids exactly. GGUF variants
are inference-only; the default training/inspection path uses `.nemo`.

Download commands (exact revisions; a fresh worker must re-verify the hashes):

```text
curl -L -o data/artifacts/magpie-v2607/magpie_tts_multilingual_357m.nemo \
  https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/446bff55e1fe3d3aa29c771574e7d2b0fa322990/magpie_tts_multilingual_357m.nemo
curl -L -o data/artifacts/nano-codec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
  https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps/resolve/fc00890b604aa2de298d2641ffc6c5f6caf8c4d7/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo
sha256sum data/artifacts/.../magpie_tts_multilingual_357m.nemo
```

`configs/tts/f04-magpie.yaml` points at these paths and hashes;
`rpa-tts model check --config configs/tts/f04-magpie.yaml --lane magpie` reports
all five lane checks `passed` on this host.

## Packaged Magpie configuration (read without NeMo)

The `.nemo` is a tar archive containing a 1.44 GB `model_weights.ckpt` and
**five `./model_config.yaml` members with the same name but different bytes**
(SHA-256 prefixes `fda01948`, `7f21112e`, `6b0dd49c`, `a942d5c9`, `9577bfa4`).
Extraction order makes the last copy the effective `model_config.yaml`. All five
share the core architecture; the differences observed are language-tokenizer
mapping keys (`ar_AE` vs `ar-AE` variants). This is a packaging hazard: the
next worker must verify which config NeMo actually restores and record the
effective config hash. Do not assume the first member.

Effective-looking (last) config facts:

| Field | Value |
| --- | --- |
| Model target | `nemo.collections.tts.models.magpietts.MagpieTTSModel` |
| `nemo_version` | `2.8.0rc0` |
| `model_type` | `decoder_ce` |
| `use_text_conditioning_encoder` | `true` |
| `legacy_text_conditioning` | `false` |
| Text-conditioning tokenizer | `text_ce_tokenizer` → `AutoTokenizer`, `google/byt5-small` |
| `embedding_dim` | 768 |
| Encoder / decoder | encoder 6 layers, decoder 12 layers, 768 d_model, 12 heads |
| Text conditioning | 1-layer 768 context encoder over the byte-tokenized text |
| Frame stacking | 2 |
| Audio codec path | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` |
| Context duration | 10.0 s max |
| Speaker map | 5 baked voices: Aria 0, Jason 1, John 2, Leo 3, Sofia 4 |
| Inference | temperature 0.6, top-k 80, CFG 2.5, max 500 decoder steps, EOS `argmax_or_multinomial_any` |
| Speaker map file | `speakers.json` (68 B) inside the archive |

### Weight shapes (state dict, 260 tensors)

| Tensor | Shape | Interpretation |
| --- | --- | --- |
| `text_embedding.weight` | `(3359, 768)` | 3,357 combined tokenizer tokens + 2 BOS/EOS |
| `audio_embeddings.{0..15}.weight` | `(2024, 768)` each | 8 codebooks × frame stacking 2; 2,016 codes + 8 specials |
| `baked_context_embedding.weight` | `(5, 166656)` | 5 speakers × 217 frames × 768 |
| `baked_context_embedding_len` | `(5,)` | per-speaker context lengths |
| `encoder` / `decoder` / `local_transformer` | layers + `position_embeddings` | transformer stacks |

`baked_context_embedding` is the frozen speaker context; the source notes it is
non-trainable at load. A frozen voice vector is not itself a failure when
generator weights adapt.

## Text contract implications (must be proven at load)

The pinned source (commit `0305e4b1…`, `magpietts.py`) does the following:

1. Builds a combined tokenizer from `cfg.text_tokenizers`
   (`setup_tokenizers(...)`), then sets
   `num_tokens = len(self.tokenizer.tokens) + 2`; `bos_id = num_tokens - 2`,
   `eos_id = num_tokens - 1`.
2. Creates `text_embedding = nn.Embedding(num_tokens, 768)`.
3. Calls `check_text_embedding_matches_tokenizer(...)` during
   `load_state_dict`; a tokenizer/embedding mismatch is a load error.
4. For non-legacy text conditioning, token IDs are offset per sub-tokenizer
   (`self.tokenizer.tokenizer_offsets[self.text_conditioning_tokenizer_name]`).

Consequences for Hmong:

- The default text-conditioning tokenizer is **ByT5**, a byte-level tokenizer.
  Its 256 byte tokens can represent every ASCII RPA character (all tone letters,
  multigraph onsets, vowels, space, punctuation) without loss. This is the
  "shared byte tokenizer candidate" named in the issue; it is a candidate until
  the offset/ID mapping and embedding rows are verified under a real NeMo load.
- Adding new tokens (a Hmong character/phoneme table) changes `num_tokens` and
  fails the embedding check unless rows/BOS/EOS are resized coherently. Equal row
  counts are not sufficient: token meanings and offsets must be proven.
- The model card lists 12 languages (ar, de, en, es, fr, hi, it, ja, ko, pt, vi,
  zh) and **not Hmong**. `do_tts(..., language=...)` selects a tokenizer and text
  normalizer; an unsupported language needs an explicit approved path
  (`apply_TN=False` plus the byte tokenizer) or a recorded config extension.
  Silent fallback to an English normalizer is forbidden by the language contract.
- The tokenizer itself (`google/byt5-small`) is an **external dependency fetched
  at load**, not baked into the `.nemo`. Inspected tokenizer revision
  `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`; extra ids 125, specials
  `<pad>`, `</s>`, `<unk>`. Pin this revision in the lane environment and record
  the resolved files; a floating `main` is not reproducible.
- The packaged model requires `text normalization` in its own card; our
  `rpa-norm-1` replaces it. Hmong tone letters are lexical, so no English TN or
  phonemizer may run.

## Codec facts

- 22,050 Hz; `samples_per_frame` 1,024 → 21.53 frames/s (21.5 fps variant).
- Group FSQ, 8 codebooks × 2,016 codes, embed dim 32; 1.89 kbps.
- Codec config has one `model_config.yaml`; decoder is causal HiFiGAN.
- F-04 still requires at least 30 native-reviewed Hmong controls reconstructed
  through this codec and compared for lexical/phonation damage; a systematic
  new lexical defect blocks this codec path. That native work is **not started**.

## Environment and lane blockers

- **NeMo Speech is not installed.** The pinned reference commit is
  `0305e4b15ac2924219536da44d315397d92ee769`. The model config targets NeMo
  `2.8.0rc0` behaviour; a newer NeMo may change tokenizer/loader semantics.
- **Windows host risk.** NeMo Speech's supported platform is Linux; this measured
  host is Windows 11. The F-04 gate must prove a working path (e.g. WSL2/Linux
  container with GPU passthrough) or record the incompatibility and activate the
  single VITS fallback. Do not start an open-ended framework port; the gate cap
  is two engineering days and eight active GPU hours, with one named
  compatibility correction.
- **VITS fallback pinned (inspection):** `idiap/coqui-ai-TTS` `dev` at
  `7d547eed8580be4fbcd409b7f3082d1d44acbe64`, MPL-2.0, active as of
  2026-11-22. It is a maintained-implementation candidate, not a qualified Hmong
  recipe; the original VITS anchor remains
  `2e561ba58618d021b5b8323d3765880f7e0ecfdb`.
- The five-config archive, external tokenizer fetch and unsupported-language path
  are named risks to resolve in the first hours of the gate, before any long
  run.

## Next executable steps (no native dependency)

1. Install NeMo Speech at the pinned commit in a Linux-compatible environment
   (WSL2/container or a Linux worker) with CUDA 12.8 support; record the platform
   and resolved package versions.
2. Restore the Magpie `.nemo`; record which `model_config.yaml` is effective,
   the tokenizer offsets, ByT5 byte-token IDs for every RPA character, BOS/EOS
   ids and the `check_text_embedding_matches_tokenizer` result.
3. Reproduce official supported-language inference (e.g. English), then attempt
   the Hmong raw-text path with `apply_TN=False`.
4. Prove a real Hmong forward/backward update, free-running synthesis without
   target codes, and complete save/resume/reload; measure peak VRAM and timing.
5. Only then run the 30-control codec reconstruction check (native reviewers
   required) and append F-04 evidence.

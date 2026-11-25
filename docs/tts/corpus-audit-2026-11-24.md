# RPA recording audit — 24 November 2026

Source commit: `cb0401e0c38652f998ca1b3bfe49d2084a279be7`.
Audio tree: `d88ccd9e54240e76c347d46d987e64615c5633bd`.
Manifest SHA-256: `42f2b4eea43f1c3aacc2807e8ff00586c58e078086654fc9c95b09e950c234be`.

| Measure | Value |
| --- | ---: |
| Audio files | 5187 |
| Integrity-valid files | 5187 |
| Expected unique syllables | 5187 |
| Present expected syllables | 5187 |
| Missing syllables | 0 |
| Unknown labels | 0 |
| Duplicate labels | 0 |
| Decoded audio, seconds | 2309.580 |
| Decoded audio, minutes | 38.493 |
| Median duration, seconds | 0.445 |
| Byte-identical groups | 0 |
| PCM-identical groups | 0 |
| Files with review flags | 8 |

## Inventory

This repository enumerates (56 consonants + an empty onset) × 13 vowels × 7 tones = **5,187** unique strings. The unmarked vowel-only syllables are already in that product. This inventory is not a complete validator of every Hmong dialect or spelling convention.

## Quality and interpretation

Every candidate file is probed and decoded. Duration measures decoded samples including any recorded silence; it is not speech-only duration. MP3 container duration can be longer because of encoder framing and padding. Flags are review heuristics, not automatic exclusions or proof of bad pronunciation. No trimming, denoising, loudness normalization, or source-file rewriting occurs.

The filenames provide syllable labels. Speaker identity, dialect, transcription accuracy, lexical validity, and tone correctness still need native-speaker review. Isolated syllables alone do not establish natural sentence prosody.

### Review flags

- `near_full_scale_review`: 8

Full filenames, hashes, source metadata, coverage gaps, and errors are in `manifest.jsonl` and `report.json`.

## Verified preparation and implementation

The complete manifest was prepared into 24 kHz mono PCM16 WAV: 4,151 training, 518 validation, and 518 test recordings. All observed CVT components remain in training and no split required a component-coverage repair. Original files are unchanged. The prepared audio totals 2,309.651 seconds after per-file resampling length rounding. PCM16 conversion saturates 60 samples across six files; that conversion is recorded explicitly rather than presented as lossless.

Eight original recordings have near-full-scale review flags: `nyag`, `nyaj`, `nyauj`, `nyeb`, `nyiab`, `nyiaj`, `nyoj`, `nywj`. Decoding MP3 can produce overshoot, so this is not a diagnosis of original clipping or incorrect pronunciation.

Validation completed on Linux using Python 3.12.14 and ffmpeg version 7.1.5-0+deb13u1: full original-file audit, complete preparation, 30 unit/integration tests, lockfile resolution, installed CLI commands, training-manifest preflight, and generation of a recording-concatenation WAV for “Kuv yog Hmoob.” The MLX-specific tests use injected doubles to validate orchestration and artifacts. No Apple Silicon training, neural synthesis, listening assessment, or model-quality claim has been completed.

See [machine-readable audit](corpus-audit-2026-11-24.json), [preparation metrics and hashes](preparation-2026-11-24.json), and the [Mac runbook](README.md).

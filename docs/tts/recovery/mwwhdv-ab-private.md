# Private blinded base-vs-trained A/B (corrected checkpoint)

No recording, worksheet, native claim or release. Authorized 2000 UBS corpus;
audio/weights private. Permission:
<https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## End-to-end check

- Loaded the base `.nemo` and the **corrected** checkpoint `3ede81ea…` through the
  pinned NeMo commit (`0305e4b1…`) using `load_magpie(..., checkpoint_path=...)`.
- Base vs trained exact parameter fingerprints differ
  (`af2ccadd…` vs `75b863c2…`), confirming the trained weights—not just the base
  artifact—were in use; all parameters were on CUDA.
- `train render-private --checkpoint` is now wired to the same loader; a private
  diagnostic flag permits raw unapproved prompt text while recording every
  normalization issue and keeping `release_eligible: false`.

## A/B set

- Four held-out Mark prompts: MRK 1:1, 1:3, 1:6, 1:7; same `seed`, speaker 0,
  `apply_TN=False`, 22,050 Hz and inference settings for both conditions.
- Conditions were randomized per item into blinded `A`/`B` slots. Answer key,
  reference text, per-slot hashes/durations and playback paths:
  `data/runs/ab-private-cursor/answer-key.json`; WAVs in the same directory.
- Base/trained files per item: `item01_A.wav` … `item04_B.wav`; the key is the only
  mapping. Nothing was listened to.
- The earlier A/B on the repeated-row checkpoint (`c382291d…`) is superseded.

## Acoustic sanity check

All four base/trained pairs are byte-distinct and non-silent. RMS spans
0.067–0.211; mean-MFCC distances span 82.7–184.1; trained durations differ from
base (for example 5.0 s vs 9.7 s on MRK 1:6). This only shows the trained weights
changed the generated signal; it does **not** establish which is better. Details:
`data/runs/ab-private-cursor/acoustic-check.json` (private).

## Caveat

This is a private listening aid, not a quality/native evaluation. The corpus is a
multi-voice drama; no intelligibility, WER or preference result is asserted.

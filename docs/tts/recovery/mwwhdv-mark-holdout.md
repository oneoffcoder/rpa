# MWWHDV book-disjoint test: held-out Mark

Private development test. Authorized 2000 UBS Hmong Daw corpus; audio/weights
private. Permission: <https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## Setup

- Train: all NT rows **except Mark** (7,259 rows). Validation: 100 Mark verses,
  never trained on. 2,000 steps at 5e-6 from the base checkpoint.
- Held-out Mark mean loss: **11.9279 base → 11.1840 final** (6.2% relative).
- Validation checks: **11.2795** (500) → **11.2203** (1,000) → **11.1933**
  (1,500) → **11.1840** (2,000).
- Train losses finite (first 11.5533, last 10.7435, min 6.3350, max 12.4975).
- Checkpoint `e69d6018…` with optimizer/scheduler/step/RNG; resume structure
  verified. A 5.619 s private sample of held-out Mark 1:1 was written
  (`Ntawm no yog Txojmoo Zoo…`), unlistened.
- Peak VRAM 6,327.2 MiB; 534.7 s.

## Caveat

This is book-disjoint but **not speaker-disjoint**: the same drama voices appear in
training. It measures text/content generalization under that leakage. No human
listening, WER, native acceptance or release claim.

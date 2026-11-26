# MWWHDV private validation continuation (3,000 steps)

Authorized corpus and permission: see
<https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.
Development only; no native quality, accepted corpus or release claim.

## Run

- Resumed checkpoint `3d475a1f…` (step 1,000) with optimizer, scheduler and RNG,
  then ran **2,000 additional steps** (total 3,000) at 5e-6 on the same
  7,735/200 random verse split.
- Validation (20-row checks): **10.9471** (1,500) → **10.9392** (2,000) →
  **10.9362** (2,500) → **10.9346** (3,000); plateau after the first 500.
- Full 200-row held-out mean: **10.9656** (previous checkpoint: 10.9937; base:
  11.7113).
- Train losses finite (first 11.2096, last 11.1568, min 6.4362, max 12.2797).
- Checkpoint SHA-256 `08fd04cf…`, step 3,000, optimizer/scheduler/RNG verified;
  0.975 s private sample written, unlistened.
- Peak VRAM 7,725.5 MiB; elapsed 493.8 s.

## Caveat

The split is random by verse, so the same chapters and drama voices appear in
train and validation; this is **not** a speaker- or book-disjoint generalisation
test. Loss gains are small and plateauing. No human listening, WER or quality
claim is made.

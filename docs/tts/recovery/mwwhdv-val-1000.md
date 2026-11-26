# MWWHDV private validation run (engineering record)

Authorized corpus: 7,935 verse pairs / 22.523 h, 2000 UBS Hmong Daw, private.
Permission/source record: <https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## Split and result

- 7,735 train / 200 held-out verse rows (seed 3407); 1,000 optimizer steps at
  5e-6, gradient clipping 1.0, single RTX 5090.
- Base held-out mean loss: **11.7113**.
- Validation trajectory: **11.1384** (200) → **11.0455** (400) → **11.0070**
  (600) → **10.9811** (800) → **10.9652** (1,000).
- Final held-out mean: **10.9937** (200 rows), a 6.1% relative reduction from
  base; train losses stayed finite (first 11.9245, last 11.8716, min 6.8105).
- Checkpoint `3d475a1f…` (3.706 GB) with optimizer/scheduler/step/RNG; resume
  structure verified. A 1.161 s private sample was written (unlistened).
- Peak allocated VRAM 7,704.5 MiB; elapsed 267.6 s.

## Boundaries

This is a **development** loss improvement on a multi-voice drama; it is not
intelligibility, native acceptance, an accepted/frozen corpus, or a release. No
human listening or WER/quality claim. Audio and weights remain private. F-06
requires judged learning-curve rows and native review, so this evidence is
recorded as development only.

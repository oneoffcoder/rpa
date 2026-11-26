# MWWHDV clean split training (train → dev)

Private development run. Authorized 2000 UBS corpus; audio/weights private.
Permission: <https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## Setup

- Used the deterministic chapter-grouped split: **5,761 train rows**, **714 dev
  rows**, **1,460 test rows**. Test was not touched.
- Base `.nemo` loaded through the pinned NeMo path; 2,000 steps at 5e-6, batch 1,
  gradient clipping 1.0, on the RTX 5090.
- Dev evaluation on the first 100 dev rows (held-out chapters).

## Result

- Base dev mean: **11.9123** → **11.0115** after 2,000 steps (7.6% relative).
- Dev trajectory: **11.1935** (500) → **11.0946** (1,000) → **11.0504** (1,500)
  → **11.0115** (2,000); monotonic.
- Train losses finite (first 11.3063, last 11.2610, min 7.3559, max 12.5021).
- Checkpoint `a16f4cb3…`, step 2,000, optimizer/scheduler/RNG verified resumable;
  8.127 s private dev sample written, unlistened.
- Peak VRAM 6,303.0 MiB; 531.1 s.

## Caveat

Chapter-disjoint dev, but the same multi-voice drama speakers appear in training.
The first training candidate also includes the ambiguous `REV 22:21` row (shared
WAV with 22:20) from the first 2,000 rows; it is one row among 5,761 and is
disclosed, not silently accepted. No native listening, WER, preference or release
claim. The private test split remains untouched for a future locked comparison.

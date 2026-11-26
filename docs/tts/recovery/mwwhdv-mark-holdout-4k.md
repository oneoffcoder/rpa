# MWWHDV held-out Mark: 4,000-step continuation

> **Correction:** both this and the preceding script restarted the row cursor at 0,
> so the 2,000 additional steps repeated the **first 2,000 non-Mark rows**. These
> 4,000 optimizer steps cover only 2,000 distinct verses; the 11.1643 value is not
> coverage of 4,000 distinct verses. The checkpoint is retained as `c382291d…` but
> labelled accordingly. See `mwwhdv-cursor-resume.md` for the corrected run
> (11.1562, overlap 0).

Private development run; authorized 2000 UBS corpus; audio/weights private.
Permission: <https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## Run

- Resumed the 2,000-step Mark-holdout checkpoint (`e69d6018…`) with optimizer,
  scheduler and RNG; trained 2,000 more steps on the same non-Mark rows
  (7,259), so Mark remains completely unseen.
- Held-out Mark (100 verses): **11.1840 at 2,000 → 11.1643 at 4,000**; total
  reduction from base **11.9279 → 11.1643 (6.4%)**.
- Late validation checks: **11.1758** (2,500) → **11.1728** (3,000) →
  **11.1692** (3,500) → **11.1643** (4,000); clearly plateauing.
- Train losses finite (first 10.7492, last 10.7744, min 6.2298, max 12.3271).
- Checkpoint `c382291d…` with optimizer/scheduler/step/RNG verified; 5.108 s
  private Mark sample written, unlistened.
- Peak VRAM 6,356.1 MiB; 541.0 s.

## Caveat

Book-disjoint but **not speaker-disjoint**; the drama voices still overlap. Gains
after 2,000 steps are marginal. No human listening, WER, native acceptance or
release claim.

# Corrected resume cursor and held-out Mark continuation

Private development record. Authorized 2000 UBS corpus; audio/weights private.
Permission: <https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.

## Bug found

`magpie_bookholdout.py` and `magpie_bookholdout_4k.py` both started their loops at
step 0 with `train[step % len(train)]`. The checkpoint labelled 4,000 steps
(`c382291d…`) therefore took 4,000 optimizer steps but **repeated the first 2,000
non-Mark rows twice**. Its 11.1643 plateau is not coverage of 4,000 distinct
verses; the checkpoint is retained but labelled accordingly.

## Fix

`train_verse_steps` now accepts `start_step`, `start_row`, `optimizer` and
`scheduler`; it advances the row cursor and returns `next_step`/`next_row`.
A focused test trains rows `a,b,c,d` uninterrupted, then interrupted after two
steps and resumed from the returned cursor; the seen-token sequences match and the
first/second halves do not overlap.

## Corrected continuation

- Loaded the 2,000-step checkpoint (`e69d6018…`) through the exact pinned NeMo
  path, restored optimizer/scheduler/RNG, and started at **step 2000 / row 2000**.
  The cumulative checkpoint includes ambiguous `REV 22:21` from the first 2,000
  rows; the continuation block 2,000–3,999 contains neither ambiguous row.
- First-row set and continuation-row set each contain 2,000 IDs with **overlap 0**.
- Trained the next 2,000 distinct rows; checkpoint `3ede81ea…`, global step 4,000,
  optimizer/scheduler/RNG verified resumable.
- Held-out Mark (100 verses): **11.1562** mean (min 8.6837, max 12.9648);
  previous 2,000-step value 11.1840; base 11.9279. Total reduction 6.5%.
- The training-loss trajectory for this corrected segment was not captured after a
  post-run private-sample normalization error; only the checkpoint and held-out
  evaluation are claimed. A 5.573 s private sample was written with
  `strict_normalization=False` and is unlistened.

## Boundaries

Book-disjoint but not speaker-disjoint; multi-voice drama. No native listening,
WER, quality, accepted-corpus or release claim.

# MWWHDV portable candidate: deterministic synthesis check

Private development check; no native listening, accepted package or release.

## Result

- Loaded the portable candidate `data/runs/f04-split-train/model.nemo`
  (SHA `87d16174…`) in **two fresh Python processes** through the pinned NeMo path.
- Same prompt (`kuv hlub koj`), language mapping, speaker 0, seed 3407, no TN.
- Both runs produced exactly:
  - waveform SHA-256 `e6d1f5d3e745339e7b56be6792abdc95388fd59f5f3baf7b0c7507cdcd44860e`
  - **1.2074 s / 26,624 samples** at 22,050 Hz.
- This establishes deterministic synthesis from the exported artifact on this host.
  It does not establish naturalness, intelligibility, native acceptance or release
  readiness; the audio was not listened to.

## Scope

Private candidate only. The corpus remains a multi-voice drama; distribution
rights, F-07 and a second-worker package inspection remain open.

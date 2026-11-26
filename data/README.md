# Data workspace (not tracked)

This directory is a local artifact workspace. Ordinary Git ignores audio,
derived features, checkpoints, run outputs, audit manifests and exports. Tracked
material lives in `data/manifests/` and `configs/tts/data/`.

```
data/
  manifests/        tracked, immutable accepted manifests (JSONL) with versions
  raw/              ignored: untouched source audio masters (48 kHz/24-bit when owned)
  derived/          ignored: resampled/segmented model inputs, features
  checkpoints/      ignored: model checkpoints, optimizer state
  runs/             ignored: run manifests, logs, metrics, generated audio
  audit/            ignored: technical audit outputs (report copies live in docs)
  artifacts/        ignored: downloaded immutable upstream artifacts
  exports/          ignored: packaged inference artifacts
  private/          ignored: consent records and restricted transcripts
```

Rules:

- Every external artifact is referenced by path + SHA-256 in a run manifest or
  evidence entry; the manifest, not the directory listing, is the record.
- Owners keep an independent checksummed backup. Initial reservation 100–200 GB;
  measured free space is in `docs/tts/recovery/environment.md`.
- Never place speaker identities, consent forms or restricted transcripts here
  without access control; they are ignored by Git but still need host security.
- Two workers sharing a host must coordinate writes by run identifier; do not
  overwrite another worker's run directory.

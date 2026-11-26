# Versioned data configs (tracked)

This directory holds approved, versioned dataset configuration and manifest
references used by training runs — for example:

```
configs/tts/data/
  owned-pilot-a.yaml          dataset version, manifest path + SHA-256, hours, split report
  owned-pilot-a.split.json    split report copy
```

Rules:

- Only material permitted for the public repository belongs here. Restricted
  transcripts or private consent records stay in access-controlled storage and
  are referenced by ID.
- The SHA-256 here must match the manifest actually consumed by a run; a run
  manifest records both.
- Changes create a new version directory/file; do not rewrite a frozen version.

Current contents: none. No dataset version has been approved.

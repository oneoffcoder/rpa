# Run manifests, evidence and artifact storage

Every later run records input/config/code/artifact hashes, device, precision,
effective batch, seed, unique audio hours, optimizer updates, wall time, peak
VRAM, complete output paths and failure counts. A run without a valid manifest
is not reproducible and does not count as evidence.

## Schemas

Canonical definitions: `src/rpa_tts/schemas.py`; reviewable copies:
`configs/tts/schemas/`. A test fails if the copies drift.

| Document | Purpose |
| --- | --- |
| `utterance-manifest.schema.json` | One accepted, native-reviewed connected-speech utterance |
| `run-manifest.schema.json` | One experiment's reproducibility record |
| `decision.schema.json` | Product/project decision (`D-01`…, `F-0x-Dnn`) |
| `stage-decision.schema.json` | Pre-run learning-stage decision (F-06) |
| `evidence-entry.schema.json` / `evidence-index.schema.json` | Gate evidence |

## Run manifest fields

- `run_id`: `f06-20261126t080000z-abc123` (stage, UTC timestamp, random token).
- `stage`: `F-04` … `F-08`.
- `command`: the exact command line used.
- `code`: commit, branch, dirty flag. A dirty tree is recorded as dirty; it is
  not hidden.
- `config`: path and SHA-256 of the YAML actually used.
- `dataset`: manifest path and SHA-256, unique accepted training hours, row
  counts per split. Hours are accepted human connected speech, not advertised
  duration, repeated epochs, silence or isolated syllables.
- `environment`: hostname, OS, Python, device, GPU name/memory, driver, torch,
  CUDA runtime, precision.
- `training`: seed, precision, effective batch size, gradient accumulation,
  optimizer updates, learning rate, max steps.
- `resources`: wall time seconds, peak VRAM MB (null when no GPU was used).
- `outputs`: every checkpoint/log/metrics/report with SHA-256.
- `failures`: count and notes; failed outputs remain in denominators.
- `status`: planned / running / completed / failed / aborted.

Validate with:

```text
rpa-tts run-manifest validate --path data/runs/<run_id>/run-manifest.json
```

## Evidence

`docs/tts/recovery/evidence-index.json` is append-only. Each entry states one
falsifiable claim, its gate, status, timestamp, optional run id and the files
that support it (hashed). The entry with the latest `created_at` for a gate is
the gate's effective status; corrections are appended, never deleted.

Hashed evidence paths must be **tracked, non-ignored snapshots** (measurement
files, frozen reports, protocol documents), never living documents such as the
stage table (`README.md`), `decisions.jsonl` or mutable configs. Progress updates
belong in those living documents; the evidence index records point-in-time facts.
Tests enforce existence, non-ignored status and hash agreement.

Status vocabulary: Not started / In progress / Passed / Failed / Inconclusive /
Blocked / Not required. An unavailable listener, GPU, permission or right is a
blocker for that gate, not a pass and not a reason to lower a threshold.

Append an entry:

```text
rpa-tts evidence add --gate F-01 --claim "GPU kernel proof passed" \
  --status passed --path docs/tts/recovery/evidence/f01-gpu-kernel-proof.json
```

## Storage rules

- Large audio, derived features, checkpoints and run outputs stay outside
  ordinary Git (`data/raw/`, `data/derived/`, `data/checkpoints/`, `data/runs/`,
  `data/exports/` are ignored). Only manifests, configs, reports and small
  evidence JSON files are committed.
- Initial workspace reservation: 100–200 GB plus an independent backup. Measured
  free space on the current host: see `environment.md` (561.05 GB free at F-01
  inventory). Resize from measured artifact sizes.
- Every external artifact location and hash must remain resolvable. Keep
  checksummed owner-controlled backups; a checksum without a retrievable backup
  is not a preservation path.
- Private consent records, speaker identities and restricted transcripts stay in
  access-controlled storage referenced by stable IDs (e.g. `consent-01`,
  `spk-01`). They never enter public Git.
- Public Git contains only material permitted for that audience; distribution
  rights for weights and audio are checked before publishing.

## Decisions

- `decisions.jsonl`: append-only product/project decisions.
- `stage-decisions.jsonl`: append-only pre-run learning decisions (dataset
  version, objective, baseline, caps, inference policy, native raters,
  measurement set, next increment). `rpa-tts train plan` writes a draft and can
  append it after review.
- `learning-curve.csv`: one row per judged learning run. Header is fixed by the
  schema module.

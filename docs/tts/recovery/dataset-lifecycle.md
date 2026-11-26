# Dataset freeze, verification and learning-curve records (F-05/F-06)

**Status:** Software implemented and tested. **No dataset version is frozen and no
learning-curve row exists.** No corpus has been accepted; the repository's
isolated syllables are not training data.

## Freezing an accepted dataset version (F-05)

`rpa-tts corpus freeze` refuses to create a version unless all of these hold:

1. Every row validates against the utterance-manifest schema and passes
   training-purpose checks: unique ids, resolved rights and consent, native
   approval, normalized text derived from the raw transcript.
2. `train`, `dev` and `test` are all non-empty.
3. No group/passage/session spans splits (leakage check).
4. The language contract is **approved for training** (status `approved` with
   attested cells), not a draft.
5. The coverage screen passes: at least five distinct training examples per
   attested cell and at least one held-out example.

A version is immutable. It writes:

```text
data/manifests/<version>/accepted.jsonl      canonical sorted manifest
configs/tts/data/<version>.yaml              versioned dataset config
```

The config records the manifest path and SHA-256, split rows/hours, buckets,
rights ids, speaker ids, normalization/preprocessing versions, the contract hash
and coverage outcome. The schema is
`configs/tts/schemas/dataset-version.schema.json`.

```text
rpa-tts corpus freeze \
  --manifest <split.jsonl> --version pilot-a \
  --contract docs/tts/recovery/language-contract.yaml \
  --manifests-dir data/manifests --configs-dir configs/tts/data

rpa-tts corpus verify-dataset --config configs/tts/data/pilot-a.yaml
```

`verify-dataset` re-hashes the frozen manifest and reports rows, splits and
normalization versions. A changed manifest fails; create a new version instead of
editing a frozen one.

### What freeze does not prove

- It relies on the recorded `audio_sha256` values; it does not re-decode every
  audio file. The technical audit (`corpus audit`) and native QA remain the
  evidence for audio integrity and acceptance.
- It does not prove acoustic balance, speaker/session diversity or release
  readiness.
- The coverage screen is a project screening count, not language completeness.

## Learning-curve records (F-06)

`learning-curve.csv` has fixed columns:
`run_id,stage,dataset_version,accepted_training_hours,optimizer_updates,dev_syllable_error,dev_naturalness_mean,decision,evidence_id`.

`rpa-tts train record` appends one row from a completed/failed/aborted run
manifest and validates: unique run id, stage `F-05`–`F-08`, non-empty dataset
version, non-negative hours/updates, syllable error in `[0,1]`, naturalness in
`[1,5]`, a decision from `baseline/promote/hold/reject/inconclusive`, and a
hashed evidence id. Invalid or duplicate rows fail; existing rows are never
rewritten.

```text
rpa-tts train record --run-manifest data/runs/<run_id>/run-manifest.json \
  --dataset-version pilot-a --dev-syllable-error 0.081 \
  --dev-naturalness 3.6 --decision hold --evidence-id ev_0123456789abcdef
```

Before a run, `rpa-tts train plan ... --record-decision` appends the declared
stage decision (objective, baseline, cap, inference policy, raters, measurement
set, next increment). The curve records the judged outcome after it.

## Current state and next actions

- No dataset version is frozen and `learning-curve.csv` has only its header.
- Once F-02/F-03 produce native-approved, rights-cleared pilot data, run the
  audit, split, coverage and freeze commands in that order.
- F-05 also needs a byte-level audio verification step against the audit
  manifest; that is a separate follow-up, not assumed by `freeze`.

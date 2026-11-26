# F-08 local package contract and packaging plan

**Status:** Package inspection is implemented and tested. **No model package
exists**, no checkpoint has been accepted, and no distribution rights are
cleared. `package build` and `synthesize` remain gated (F-08/F-07); this document
defines the package a qualified model must produce and how it will be checked.

## Package contents

```text
<package>/
  model-manifest.json     schema-validated package record (see below)
  checkpoint.bin          exact accepted checkpoint bytes
  config.json             inference configuration used for the frozen benchmark
  benchmark.jsonl         frozen benchmark text/manifest for reproduction
  scores.json             recorded benchmark outputs/scores
```

The manifest records: status (`candidate`/`accepted`/`rejected`), stable
`voice_id`, `model_version`, `normalization_version` (`rpa-norm-N`),
`text_contract` (e.g. `rpa-chars-1`), model sample rate, checkpoint and config
paths/SHA-256, creation time, optional run id, license, rights/consent id, and a
benchmark block (manifest path/hash, scores path/hash, gate verdict).

Schema: `configs/tts/schemas/model-manifest.schema.json`.

## Inspection checks

`rpa-tts package inspect --checkpoint <package>` reports, without producing
audio:

| Check | Fails when |
| --- | --- |
| `manifest` | missing or not schema-valid |
| `status_accepted` | status is candidate/rejected |
| `checkpoint` | file missing or SHA-256 mismatch |
| `config` | file missing or SHA-256 mismatch |
| `benchmark_verdict` | recorded F-07 verdict is not `accepted` |
| `benchmark_manifest` / `benchmark_scores` | missing or hash mismatch |
| `rights_resolved` | rights id is unresolved/pending |
| `normalization_contract` | reports the text front end and contract version |

Exit status is 0 only when every check passes. Inspection is structural and
hash-based; it is **not** a new native judgment and cannot turn a failed
qualification into an accepted package.

## Reproducibility requirement

A second worker must restore the package in a clean environment, synthesize the
frozen benchmark, and reproduce the recorded configuration and pass/fail
conclusions within the declared stochastic variation. Preserve source, config,
checkpoint and output hashes. Distinguish reproducibility evidence from any new
native listening judgment: reproducing an output is not re-qualification.

## Lesson-audio integration (later, authorized)

- Generate fixed lesson audio from the accepted package, have it natively
  reviewed, and integrate only approved files into the existing Sphinx/S3/
  CloudFront publishing path when release is authorized.
- Preserve original recordings and a model/version pointer for rollback.
- Test page/audio references and a fresh installation. A lossless software
  refactor needs contract checks; export, quantization, resampling or
  segmentation changes require targeted listening of the resulting artifact.
- Never redirect existing syllable playback or `generate.html?q=...` links to an
  unqualified neural model.
- If VITS wins, measure CPU/ONNX/browser export on the actual model and intended
  devices (front-end equivalence, size, memory, speed, crashes, audio fidelity,
  weight-redistribution rights); adopt the smallest option that passes.

## Rights and privacy

- Distribution rights for weights, text front end and audio are prerequisites;
  `rights_id` must resolve in the rights ledger, and unresolved ids fail
  inspection.
- Public packages contain only material permitted for public distribution.
  Private consent records and restricted transcripts stay in access-controlled
  storage referenced by IDs.

## Current state and limitations

- No accepted checkpoint, package, benchmark or rights clearance exists; F-08
  remains `In progress` only for preparation.
- The acoustic inference wrapper is not implemented; `synthesize` raises after
  validating the package and the F-07 gate.
- Inspection does not prove the checkpoint actually produces the benchmark audio
  (that is the second-worker reproduction), does not re-run the native panel, and
  does not validate license compatibility beyond the recorded id.
- `package build` remains refused until F-08 passes and distribution rights are
  recorded.

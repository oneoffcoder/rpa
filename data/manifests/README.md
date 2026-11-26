# Accepted manifests (tracked, immutable)

A frozen dataset version is a JSONL file of rows that validate against
`configs/tts/schemas/utterance-manifest.schema.json` and pass
`rpa-tts text validate --manifest <path>`. Freeze by:

1. Native QA sign-off on every admitted pair (F-05).
2. `rpa-tts corpus split --manifest <accepted.jsonl> --output <split.jsonl>`.
3. Copy the accepted/split manifest and its report into this directory under a
   version name such as `owned-pilot-a/`.
4. Record the manifest SHA-256 in the dataset row of the next run manifest and
   in `docs/tts/recovery/decisions.jsonl`.

Rules:

- Never edit a frozen manifest in place; create a new version and record the
  reason.
- Do not include speaker names, consent documents, contact details or
  restricted transcript text. Reference private records by stable IDs
  (`spk-01`, `consent-01`).
- Rejected rows and their reasons stay in the audit ledger; exclusion is not
  deletion of evidence.
- Related audio, passages, prompts and recording sessions share a `group_id` and
  never span splits. A split report with warnings about missing buckets is a
  coverage fact to report, not something to hide.
- Every row records raw and normalized text plus the normalization version; the
  model must use the same front end as inference.

Current contents: none. No corpus has been accepted for this issue.

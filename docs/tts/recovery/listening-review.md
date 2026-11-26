# Post-scoring native review: adjudication and paragraph checks

**Status:** Software implemented and tested. No real adjudication or paragraph
review data exists; native reviewers are not scheduled. This does not claim any
release judgment.

The blind instrument records raw listener responses first. Two native review
steps follow and are merged as separate sheets, so the raw choices, ties and
failures remain auditable and are never rewritten by consensus.

## Order of operations

```text
evaluate ingest       # phase-A/B sheets -> raw scores
evaluate adjudicate   # confirm/reject/uncertain critical-contrast errors
evaluate paragraphs   # missing/repeated clauses and language switches
evaluate metrics      # pooled syllable errors, naturalness interval, contrast accuracy
evaluate gate         # F-07 release verdict
```

Do not hand-edit the scores JSON. Every merge validates and reports.

## Adjudication sheet

CSV columns: `code,verdict,confirmed_by,notes`.

```csv
code,verdict,confirmed_by,notes
A1B2C3D4E5F6,confirmed,A-01;A-02,clear /nees/ response; expected /neeg/
A1B2C3D4E5F7,rejected,A-01,acceptable fast-speech variant
A1B2C3D4E5F8,uncertain,,recording unclear
```

Rules enforced:

- `code` must be a contrast assignment in the scores payload; duplicates and
  unknown codes fail.
- `verdict` is `confirmed`, `rejected` or `uncertain`.
- A **confirmed** error must be a wrong choice on a **critical** item and must
  list at least **two distinct native adjudicators** (adjudicator IDs only;
  identities stay private). Rejected errors must not list confirming
  adjudicators.
- Uncertain and rejected decisions keep the original choice; they do not erase
  it. Confirmed decisions set `adjudicated=true` and `confirmed_by`.
- The merge reports `unadjudicated_wrong_critical` so a coordinator can see
  unreviewed wrong critical responses instead of silently inferring them.
- A confirmed critical error repeated on at least two distinct utterances with
  adjudicator support blocks release through the existing gate rule; a single
  severe error remains counted and is not silently dismissed.

## Paragraph review sheet

CSV columns: `script_id,missing_clauses,repeated_clauses,language_switch,notes`.

```csv
script_id,missing_clauses,repeated_clauses,language_switch,notes
p001,0,0,no,complete
p002,1,0,no,dropped final clause
p003,0,0,yes,switched into English mid-paragraph
```

Rules enforced: non-negative integer clause counts; boolean-like
`language_switch`; unique paragraph ids; when `--pack` is supplied, every id must
be a `kind: paragraph` item and the pack must contain paragraph items. Counts are
human judgments, not inferred by transcription scoring. Any missing clause,
repeated clause or intended-language switch rejects the paragraph gate.

## Commands

```text
rpa-tts evaluate adjudicate \
  --scores scores.json --adjudications adjudications.csv --output adjudicated.json

rpa-tts evaluate paragraphs \
  --scores adjudicated.json --reviews paragraphs.csv [--pack <pack-dir>] \
  --output final-scores.json
```

## Privacy and provenance

- Reviewer identities, consent and payment records remain in access-controlled
  storage; sheets carry IDs such as `A-01`.
- Sheets record verdict, adjudicators, notes and file name; the merged scores
  retain the original response plus the review object.
- A failed or rejected final test remains in the record; corrections are new
  entries/files, not overwritten history.

## Limitations

- The tool does not decide whether a response is an acceptable dialect/variant
  realization; two native adjudicators must make that judgment.
- It does not infer clause completeness or language switches; paragraph review is
  human input.
- Adjudication of non-blind development comparisons follows the same sheet
  format but is not a substitute for the locked qualification.

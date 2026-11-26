# Review import: validated approved scripts (F-02)

**Status:** Software implemented and tested. **No review decisions exist and no
approved script set has been produced** — the worksheet is blank.

`rpa-tts corpus review-import` turns a filled native review sheet into a
validated approved script set for recording. It does not record consent, speaker
selection, audio or a locked test.

## Workflow

```text
corpus scripts                 # draft candidates from book examples
corpus review-sheet            # blank accept/edit/reject worksheet
   (native lead fills decisions, edited text and notes)
corpus review-import           # validate into approved.jsonl + report
```

```text
rpa-tts corpus review-import \
  --draft docs/tts/recovery/calibration-scripts.draft.jsonl \
  --review <filled-review.csv> \
  --output-jsonl <approved.jsonl> \
  --output-report <approved.md> \
  --reviewer-id N-01
```

## Validation rules

- The draft is the frozen source of truth: every review `script_id` must exist,
  be unique, and its `text`/`status` must match the draft exactly.
- Every draft row must appear in the review sheet; no silent additions or drops.
- Every row must be decided `accept`, `edit` or `reject`; blanks fail.
- `accept` uses the draft text and is refused for a blocked row (blocked rows need
  an approved edit or a rejection).
- `edit` requires non-empty `edited_text`; the result must pass strict `rpa-norm-1`
  (no digits, foreign text, unapproved punctuation or unknown spellings).
- `reject` removes the row from the approved set but keeps it in the report.
- `reviewer_id` is a stable non-identifying ID (`N-01`), never a name or contact
  detail; identities/consent remain in private storage.
- Outputs are never overwritten unless `--overwrite` is explicit.

## Approved output

Each approved JSONL row records the accepted normalized `text`, `original_text`,
`decision`, `reviewer_id`, bucket, source line/section, syllable count and
normalization version. The report records draft/review/generated hashes, counts
and the edited/rejected lists.

This is native approval of the script text only. Recording requires a consented
speaker and the language contract; pack building requires stimuli; the locked
test must use fresh material, not these exposed book sentences.

## Current state

- `calibration-scripts.review.csv` has 272 blank rows (270 pending, 2 blocked).
- No approved set exists; `review-import` has only been exercised on synthetic
  review data in tests.
- After a native lead fills the sheet, import the decisions, then record the
  100–150 approved calibration scripts.

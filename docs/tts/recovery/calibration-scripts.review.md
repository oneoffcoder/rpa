# Native review worksheet — draft calibration/development scripts

Draft JSONL: `docs/tts/recovery/calibration-scripts.draft.jsonl`.
Review CSV: `docs/tts/recovery/calibration-scripts.review.csv`.
Rows: 272; pending approval: 270; blocked: 2.

**Draft only. No row is approved or recorded. These public book sentences are
calibration/development material, not the locked 300-item test.**

## Reviewer instructions

1. Set `decision` on every row to `accept`, `edit` or `reject`.
2. For `edit`, put the approved RPA text in `edited_text`; it must pass
   `rpa-norm-1` with no digits, foreign text or unapproved punctuation.
3. Blocked rows need a spelling/product decision (approved spelling or
   explicit rejection); record it in `notes` and, if editing, `edited_text`.
4. Record reviewer identity/consent separately under stable private IDs; do
   not put names or contact details in this worksheet.
5. Selected scripts still need native approval and a consented speaker
   recording before any pack or training use.

## Counts by bucket

- `ordinary_sentences`: 13
- `targeted_contrasts`: 259

## Blocked rows (decision required)

| Script | Line | Text | Issues |
| --- | ---: | --- | --- |
| cal-006 | 131 | Tam sim no, kuv kawm keeb kwm Mekas, lus Eeskiv thiab kam teb kev chaw. | unknown_word:Mekas@30; unknown_word:Eeskiv@41 |
| cal-182 | 321 | Thaum zoo ces hais tias "mog mog", thaum phem ces cuag dev sib tog. | unsupported_character:"@24; unsupported_character:"@32 |

## Blocked tokens

`"` (2), `Eeskiv` (1), `Mekas` (1)

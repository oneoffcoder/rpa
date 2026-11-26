# F-07 locked qualification suite: composition contract

**Status:** Software implemented and tested. **No qualification suite, stimuli,
candidate model or native panel exists.** This document defines the suite
composition enforced by `rpa-tts evaluate pack` for `design: qualification`; it
does not claim a test has been designed with native reviewers or run.

## Required composition (enforced at build)

| Component | Requirement |
| --- | --- |
| Sentence items | exactly **300** |
| `ordinary_sentences` | 100 |
| `targeted_contrasts` | 80, each with contrast options and an expected value |
| `questions_numbers_names` | 60 |
| `continuous_passages` (sentences) | 60 |
| Continuous paragraphs | exactly **20**, each 30–90 seconds |
| Declared listeners | at least **12** |
| Listeners eligible for all three roles | at least **9** |
| Critical contrast items | at least **40** |
| Conditions | at least 3 and must include `candidate` |

A qualification build that misses any of these fails with the concrete
violation list; it cannot silently ship a smaller or unbalanced suite. The
manifest records the full `design_check` report, and `pack-summary.json` carries
it too. Calibration packs are exempt from these minimums.

```text
rpa-tts evaluate pack --definition qualification.json --output <pack-dir>
```

The definition's reference text must pass `rpa-norm-1`; paragraph items need
`duration_seconds` inside 30–90. Each item carries one audio path per condition
(`candidate`, `natural`, `baseline` or the declared equivalents).

## What the validator does not prove

- That scripts are fresh, natural, properly contrasted or native-approved.
- That contrast alternatives are plausible or that the intended answer is valid in
  context; the native lead designs those items.
- That natural/candidate/baseline audio exists, is correctly paired or is
  qualified; `evaluate pack --stage-audio` only stages and hashes files.
- That 12 listeners actually attend or remain eligible; recruitment and
  replacements are separate.

The composition contract is a floor for the locked test, not evidence of its
quality.

## Task order and scoring

Still governed by `listening-instrument.md` and `listening-review.md`:

1. blind transcription + naturalness (one condition per script/listener);
2. contrast identification after phase A is submitted;
3. two-adjudicator review of critical errors;
4. paragraph clause/language-switch review;
5. `evaluate metrics` and `evaluate gate`.

The final suite is administered once per frozen candidate. A failed final test
that influences changes is retired to disclosed regression use, and fresh final
material is frozen before another release attempt.

## Remaining prerequisites

- F-05 accepted corpus and F-06 frozen candidate.
- Native lead authors/approves the 300 sentence and 20 paragraph scripts,
  including at least 40 critical contrast pairs and accepted variants declared
  before generation.
- Candidate, matched-natural and baseline audio is generated and staged with the
  declared seed/policy; failed outputs stay in the denominators.
- Twelve native listeners are recruited with at least nine eligible for the
  three-condition design and at least three eligible ratings per item/condition.

Current state: none of these exist; F-07 remains `In progress` for instrument
preparation only.

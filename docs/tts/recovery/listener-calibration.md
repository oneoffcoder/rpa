# Listener calibration and scoring instrument (F-02) — NOT STARTED

**Status:** Not started. This document defines the protocol and prerequisites;
no listening session has occurred.

**Owner:** Jee and a native White Hmong language lead; an independent native
reviewer for disagreements. An agent cannot substitute for these judgments.

## Purpose

One short listening session establishes that the scoring instrument can
distinguish accepted natural speech from retained failures. It is not a test of
a model and cannot lower a release threshold.

## Participants

- At least three fluent White Hmong listeners, including Jee.
- At least one listener outside Jee's immediate speech community where available.
- Identity, consent and payment records stay in access-controlled storage and
  are referenced here by stable IDs only (`L-01`, `L-02`, ...).

## Stimulus set

- Accepted natural speech from the selected speaker/accent.
- Natural speech from candidate narrators/sources (blind-labelled).
- Three to five retained historical failures (rejected stitched-syllable /
  neural outputs) preserved with provenance.
- No commercial TTS sample is required.

Every item is assigned a blind code before the session. The condition key is
stored separately and not shown during scoring.

## Procedure

1. Confirm quiet room, wired or high-quality headphones, fixed comfortable
   volume, and a short level check.
2. For each item, each listener records a judgment and a typed reason **before**
   discussion.
3. Allowed reason categories (one primary): wrong word/tone; accent/variant;
   rhythm; voice quality; boundary/truncation; acoustic artifact.
4. Replays are capped (at most two complete plays per short item, one extra for a
   paragraph) and logged.
5. Each listener submits independent responses; do not converge on a consensus
   transcript.
6. Only after all responses are captured may the group discuss disagreements.
7. Adjudication: two native adjudicators review repeatable critical errors after
   considering legitimate variants. Disagreement about dialect, pronunciation
   or voice preference is diagnosed, not averaged away.

## Instrument checks (declared screening rules)

- Oral-fluency screening on fresh natural controls: twenty sentences with at
  least 300 reference units, at most 5% normalized transcription error.
- Contrast screening: forty adjudicated neutral-context tokens, at least 95%
  identification.
- Screen oral fluency and RPA transcription separately. A fluent listener who
  cannot transcribe RPA may still contribute other qualified judgments but
  cannot replace a required transcription rater.
- Recalibrate once for instruction/orthography confusion; record the change and
  the version of the instructions used.

## Recording setup for calibration speech

- 10–15 accepted minutes of complete, natural lesson/everyday sentences from the
  selected consented speaker.
- 48 kHz / 24-bit mono masters; derive model rates once from masters.
- Quiet, consistent space; stable microphone/distance/gain; pop filter;
  monitoring; no clipping, music, heavy processing or silent setup changes.
- Log session changes and keep untouched masters. This calibration set is
  development material and may supply useful approved lesson audio even if TTS
  research fails.

## Outputs

- Completed responses (private storage; stable item/listener IDs).
- Summary of instrument checks and any recalibration.
- Signed decisions: intended dialect/accent, narrator acceptance, accepted
  orthographic variants, scoring instructions, adjudication outcomes.
- Approval entry in `language-contract.yaml` (`status: approved`, reviewer IDs,
  attested contrast cells).

## Privacy

Listener names, consent forms and contact details never enter public Git.
Public artifacts reference listener IDs only. Raw response audio/text uses
private access-controlled storage with checksums.

# Blinded listening instrument (F-02 calibration / F-07 qualification)

**Status:** Software implemented, schema-validated and tested. **No pack has been
used with a listener and no responses exist.** Scripts and stimuli still require
native approval and recording; this document does not claim either.

The instrument turns the protocol in `listener-calibration.md` and the F-07 task
order into reproducible artifacts:

- a counterbalanced assignment with **one condition per script per listener**;
- opaque blind codes so listener sheets never reveal condition, source path or
  the intended text;
- **phase A** transcription + naturalness before **phase B** contrast
  alternatives (alternatives never appear in phase A);
- strict ingestion that rejects missing, duplicate, unknown, out-of-range or
  over-replay responses instead of silently dropping them;
- a scores file consumed directly by `rpa-tts evaluate metrics` / `gate`.

## Design rules enforced

| Rule | Implementation |
| --- | --- |
| One condition per script/listener | Assignment `(listener_index + item_index) mod conditions` |
| Counterbalanced | Each listener and each script sees conditions within ±1 count |
| Blind sheets | Listener sheets contain only `code`, blind `audio` path, response columns |
| Blind audio | Staged as `audio/<CODE>.wav`; the code embeds listener+script, not condition |
| Contrast hidden until phase B | Separate `<listener>-phase-b.csv`; `expected` never appears in a listener sheet |
| Replay caps | 2 plays per short item, +1 for a paragraph item, logged per row |
| Failed outputs counted | Empty transcription increments `catastrophic_outputs` and stays in the denominator |
| No silent dropout | Missing/extra/duplicate codes, non-integer or 1–5 violations, over-cap replays, unknown contrast choices all fail ingestion |
| Coordinator-only keys | `coordinator/assignments.csv` and `<listener>-key.csv` carry condition/source map |

Assignment balances are checked and reported in `pack-summary.json`; warnings
appear instead of hiding an unbalanced design.

## Definition

A pack definition is JSON validated against
`configs/tts/schemas/listening-pack.schema.json`. Items carry the reference text
(valid RPA, approved expansions already applied), bucket, optional paragraph kind,
optional critical contrast options and a relative audio path per condition.

Illustrative template (not approved session material):
[`listening-pack.template.json`](listening-pack.template.json).

```json
{
  "schema_version": 1,
  "pack_id": "calibration-2026-12-a",
  "design": "calibration",
  "seed": 3407,
  "conditions": ["candidate", "natural", "baseline"],
  "items": [{
    "script_id": "s001",
    "bucket": "ordinary_sentences",
    "text": "kuv hlub koj",
    "condition_audio": {
      "candidate": "audio/candidate/s001.wav",
      "natural": "audio/natural/s001.wav",
      "baseline": "audio/baseline/s001.wav"
    }
  }],
  "listeners": [
    {"listener_id": "L-01", "roles": ["transcription", "naturalness", "contrast"]}
  ]
}
```

Validation enforces: unique pack/script/listener ids; audio keys exactly match
the condition list; critical items declare contrast options containing the
expected value; item text passes `rpa-norm-1`; qualification packs exclude the
`calibration` bucket and include at least one critical contrast item. A
qualification pack will have 300 sentence items plus 20 paragraph items and at
least 12 declared listeners; the pack builder is size-agnostic.

## Commands

```text
# 1. Build pack (manifest, coordinator sheet, listener sheets, summary)
rpa-tts evaluate pack --definition <definition.json> --output <pack-dir>

# 2. Stage blind audio once the files exist (hardlinks when possible)
rpa-tts evaluate pack --definition <definition.json> --output <pack-dir> \
  --stage-audio <audio-root>

# 3. During the session: distribute only <listener>-phase-a.csv first
#    After phase A is submitted: distribute <listener>-phase-b.csv

# 4. Ingest completed sheets into a scores payload
rpa-tts evaluate ingest --pack <pack-dir> --responses <responses-dir> \
  --output scores.json

# 5. Score and apply the release gate
rpa-tts evaluate metrics --scores scores.json --output metrics.json
rpa-tts evaluate gate --scores scores.json --output gate.json
```

`scores.json` contains `transcriptions`, `naturalness`, `contrasts`,
`catastrophic_outputs` and the pack manifest hash. `evaluate gate` still needs
the coordinator-supplied `paragraphs` judgement and adjudicated critical-error
confirmations before a real verdict; those are human steps, not inferred by the
tool.

## Pack contents

```
<pack-dir>/
  pack-manifest.json        full assignment map, condition, source audio, expected contrast
  pack-summary.json         balance table and warnings
  PACK-README.md            distribution and task-order instructions
  coordinator/assignments.csv
  listener-sheets/<L>-phase-a.csv
  listener-sheets/<L>-phase-b.csv   (contrast items only)
  listener-sheets/<L>-key.csv       (coordinator-only, reveals condition)
  audio/<CODE>.wav                  (after staging)
```

## Privacy and provenance

- Listener identities and consent records stay in access-controlled storage,
  referenced by `L-01`… only. Response sheets and pack manifests for a real
  session are private unless separately approved for publication.
- The pack manifest records the definition SHA-256, seed, blind-code algorithm,
  staged-audio hashes and item text, so a scoring run is reproducible.
- Do not reuse a script across calibration, development and the locked test. A
  script exposed in phase A cannot be used blind again; retain prior exposure in
  the pack metadata/notes.

## Known limitations

- No HTML player is generated; sheets and staged audio are the interface. The
  coordinator distributes phase A and phase B at the required times.
- Contrast-adjudication (`confirmed_by`, adjudicator count) is supplied by the
  native review step, not by ingestion.
- Paragraph clause completeness and unintended language switch are human
  judgements; ingestion does not infer them.
- The template's sentences are illustrative and require native approval before
  any recording or session.

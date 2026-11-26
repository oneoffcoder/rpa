# Byte-level audio verification (F-05)

**Status:** Software implemented and tested against synthetic WAVs. **No real
corpus audio has been verified**, because no accepted corpus exists.

`corpus freeze` trusts the `audio_sha256` recorded in a manifest. This command
re-checks the actual files before a manifest is treated as reproducible:

```text
rpa-tts corpus verify-audio \
  --manifest data/manifests/<version>/accepted.jsonl \
  --audio-root <directory containing each row's audio_path> \
  [--workers 4] [--tolerance 0.05] [--output report.json]
```

## Checks

| Check | Failure token |
| --- | --- |
| File exists under the audio root | `missing_file` |
| SHA-256 equals the manifest's `audio_sha256` | `sha256_mismatch` |
| Decoded duration within `--tolerance` seconds of the recorded duration | `duration_mismatch` |
| Probed sample rate equals the recorded rate | `sample_rate_mismatch` |
| Probed channel count equals the recorded count | `channels_mismatch` |
| File probes and decodes, with finite samples | `unreadable` |

The report lists rows checked, passed/failed counts, per-problem counts,
failures (id, path, details), measured hours, sample rates and codecs. Exit
status is 0 only when every row passes.

## What this does not establish

- Native acceptance, transcription/tone correctness or speaker/dialect identity.
- Rights or consent (the manifest's rights/consent ids are checked by
  `text validate`/`freeze`, not here).
- Acoustic quality, music/noise, clipping acceptability or boundary correctness;
  the technical audit flags candidates, native QA decides.
- That the recorded duration is music/silence-free or speech-only.

A technically intact file can still be rejected by native review. Passing
`verify-audio` is an integrity check, not an acceptance or release claim.

## Ordering

```text
corpus audit              # probe/hash/flags candidates
native QA                  # accept/reject, boundaries, text
corpus split               # grouped train/dev/test
corpus coverage            # attested-cell screen
corpus verify-audio        # re-hash/decode every admitted file
corpus freeze              # immutable accepted version
```

Freeze currently relies on the recorded hashes; run `verify-audio` immediately
before freezing and keep its report with the dataset evidence. A follow-up could
make audio verification a required freeze precondition once a real pilot exists;
that decision should be recorded rather than silently changing the gate.

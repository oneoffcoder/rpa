# MWWHDV full NT corpus and private training (engineering record)

Authoritative permission/source record:
<https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.
Training permission covers the existing 2000 UBS Hmong Daw narration; audio and
weights stay private. This is a compact engineering record, not a duplicate.

## Corpus

- **7,935 verse pairs / 22.523 hours**, all 27 NT books, 22,050 Hz mono.
- Source: live API audio fileset `MWWHDVN2DA`, text fileset `MWWHDV`, and
  `/api/timestamps/...` verse timings. Edition: **Hmong Daw — United Bible
  Societies, 2000** (not the 2004 text).
- Manifest SHA-256
  `3a21824fc43c98fe2fcdcd857ab9ec0eb3002cebd7686dfd218ecf118c1dc17d`;
  max cut-duration error 0.0161 s; no empty texts, no missing audio, no duplicate
  ids, two clips shorter than 1 s.
- Known gaps: 22 NT verses had no timestamp/usable span; the narration is a
  multi-voice drama, so this is not a single-speaker corpus. Native review,
  coverage audit, and freeze remain open.
- Audio under ignored `data/raw/mwwhdv-nt/`; private.

## Private training on RTX 5090

| Run | Steps | Loss | Checkpoint |
| --- | ---: | --- | --- |
| MAT 1:1 single step | 1 | 10.5202 finite; 140/144 sampled tensors changed | `7a0bd9cb…`, 3.706 GB; resumed step loss 10.2875 |
| MAT 1–10 pilot | 100 | 12.1236 → 11.4437; min 10.0270 | `547caf9e…` |
| NT snapshot | 300 | 10.8905 → 9.8866; min 9.3818 | `abd5b0ae…` |
| Full NT | 500 | 11.2907 → 11.6555; min 6.5577, max 12.4151 | `84b93e67…` |

All losses were finite; optimizer, scheduler, step and RNG were saved and the
resume structure verified. The 500-step run is private and not a quality claim;
the multi-voice drama and small step count mean no intelligibility/quality result is
asserted. Full-resume execution of every checkpoint and native listening remain
open.

## Boundaries

Training permission is recorded; audio and weights are private. No owner
recordings/worksheets, no 2004 text, no public release. F-02 native approval,
F-05 acceptance/freeze, F-07 listening and F-08 distribution rights remain
unmet.

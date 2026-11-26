# MWWHDV private pilot — engineering record

Authoritative permission/source record:
<https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>
(training permission, all sources, hashed pairs). This is a compact engineering
record, not a duplicate.

## Access verified

- CMU index: 7,382 segments, 260 NT chapters.
- Modern live API: audio fileset `MWWHDVN2DA`, text fileset `MWWHDV`, and
  `/api/timestamps/MWWHDVN2DA/MAT/<chapter>` verse timings.
- Edition: **Hmong Daw — United Bible Societies, 2000** (audio and text); the
  different 2004 text was not used.
- One verified pair: MAT 1:1, 3.48–13.06 s (9.580 s), text read from the 2000
  verse API; chapter MP3 SHA-256 `51f88f1c…`.

## Pilot corpus

- Matthew 1–10, **314 verse pairs**, **54.517 minutes**, 22,050 Hz mono.
- Manifest SHA-256 `e05bf913f7e2183267934c1b6e0748b37307cf53eb653e90edc4bb0a26acf18c`;
  max cut-duration error 0.0001 s.
- Audio and segment WAVs live under ignored `data/raw/mwwhdv-pilot/`; private.

## Real training steps (RTX 5090, private)

- MAT 1:1 single step: loss **10.5202** (finite), 140/144 sampled tensors changed,
  checkpoint `7a0bd9cb…` (3.706 GB) with optimizer/scheduler/step/RNG; a resumed
  second step produced finite loss **10.2875**.
- Pilot finetune: 10 sequential verse steps, losses
  `10.481 / 11.477 / 11.509 / 12.077 / 11.743 / 11.584 / 11.162 / 11.438 /
  10.630 / 11.493`; checkpoint `85cc1154…` with resume structure verified.
- Not an accepted corpus, not native-reviewed, no release/weight publication.

## Boundaries

Training permission is recorded on the Confluence page; audio and weights stay
private. F-02 native approval/reviewer remains outstanding; F-05 acceptance,
F-07 listening and F-08 distribution rights are not met. No owner recordings or
worksheets and no new license forms were used.

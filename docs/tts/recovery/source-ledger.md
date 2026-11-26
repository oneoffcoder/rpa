# Source and rights ledger (F-03)

**Status:** Preparation only. No source has cleared the access + rights + target-
speaker contract; no corpus has been retrieved, heard or reconstructed for this
issue. Access to a web player, repository or API is not permission to train,
publish weights or redistribute recordings.

Rules:

- Record source URL, edition, file hash, retrieval date, speaker/session/dialect,
  transcript origin and permitted uses for every source.
- Distinguish permissions actually granted from unresolved requests.
- Use private consent IDs in manifests; store identities and signed records in
  access-controlled storage, never public Git.
- Do not pool unrelated voices into a single-speaker training set.
- CMU and other retrieval candidates are not accepted corpora.

## CMU Wilderness — White Hmong (MWWHDV)

| Fact | Value |
| --- | --- |
| Alignment archive | 116,131 bytes, SHA-256 `10528fc5528bdfa0f6ae3dc7e2bd80e08e00d30464f0e15581664f502f2ebc8f` |
| Archive contents | indices/scores/metadata only; **no audio and no transcript text** |
| Segment file | 7,382 rows, approximately 15.40 indexed hours |
| Catalog claim | 8,688 Pass 1 utterances, 18:06:46 |
| Mismatch | unresolved; neither duration is accepted usable speech |
| Downloader inspected | `festvox/datasets-CMU_Wilderness` commit `a2102dcd050bf09d4b8aeea5babc6470b5b9376f` |
| Downloader defect | `bin/do_found` exits immediately after its first page download in `get_from_bibleis` |
| Reconstruction defect | applies F0-based internal-silence pruning before the supplied offsets |
| Rights | Bible Brain access/API terms do not themselves grant training or weight-distribution rights |
| Status | **Blocked on source/text/rights match** (timeboxed investigation not yet run) |

Next bounded investigation (one engineer day): establish a permitted current
source for the exact edition and matching text, retrieve one representative
chapter, reconstruct at most 20 pairs, inspect the old downloader rather than
executing its dependency chain, preserve raw masters and avoid pitch-based
silence deletion. One additional repair day only for a concrete compatibility
defect.

## Other acquisition leads

| Source | Lead | Rights status | Usable speech recorded | Next action |
| --- | --- | --- | --- | --- |
| University of Minnesota Hmong corpus | `https://hmong-corpus.umn.edu/about` | Unresolved: exports and access terms | Unknown; descriptions are not an accepted corpus | Request exports and terms; inspect audio/labels |
| Garellek / Esposito White Hmong materials | `https://pages.ucsd.edu/~mgarellek/files/SupplementaryFiles_Hmong.html` | Paper availability is separate from corpus rights | Folktale recordings described; duration unknown | Request permission and file inventory |
| Pangeanic Hmong data | `https://pangeanic.com/hmong-datasets-for-ai-training` | No license or quote | Per-speaker White Hmong hours unverified | Request representative sample, speaker hours and concrete license/quote |
| Yuhalu | `https://www.yuhalu.org/download.php?file=Yuhalu_Voice.zip` | Unaudited | Bounded eight-byte check returned HTTP 200 and a ZIP signature; advertised Content-Length 255,561,989 | Full archive, labels, duration and rights audit before any use |
| Pakorn2112 broadcast data | Broadcast recordings | Documented label conflicts | Unknown | **Quarantined**; every excerpt would need fresh native verification and provenance |
| RPA book syllable MP3s | `source/_static/mp3/` in this repository | CC BY 4.0 (repository README) | 5,187 isolated syllables; speaker identity/dialect unverified | Internal lexicon/reference only; isolated syllables are not connected-speech training data |
| Owned recordings (planned) | Selected consented speaker | Not yet granted | None | F-02 speaker/accent selection and consent |

## Recording plan (F-02/F-03)

- 10–15 accepted minutes of calibration speech from the selected speaker.
- 30–60 minute pilot, then staged 1/2/5/10 hour budgets with at most 20 hours as
  a costed contingency. These are project allocations, not language minimums.
- Preferred mixture by accepted duration: 60% ordinary sentences/educational
  prose; 20% targeted lexical/tone/phonation in context; 10% questions,
  negation, numbers and names; 10% continuous paragraph/dialogue turns.
- Native editor approves scripts before recording; automatic segmentation tools
  only propose flags.

## Required audit when a source clears rights

Audit 200 pairs (or all if fewer), stratified by speaker, chapter/session,
length and recording condition. Native reviewers assess exact RPA/transcript
match, dialect/voice acceptability, tone/phonation, music/noise, speaker
changes and boundaries. Publish defect/rejection counts and accepted unique
hours by speaker. One reviewer verifies every admitted training pair; a second
reviews all evaluation pairs, disputes and a stratified 10% of training,
expanding review when defects cluster.

Coverage screening then applies (see `coverage-ledger.md`): at least five
distinct training examples per attested prioritized cell and a held-out context.

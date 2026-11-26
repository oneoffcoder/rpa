# F-03 source investigation — 2026-11-26

**Status:** Verification pass only. No audio was retrieved for training, no
rights were cleared, no corpus hours are accepted, and no supplier was
contacted. This pass checks the facts the controlling issue relies on and
records what is now stale or blocked.

## CMU Wilderness White Hmong (MWWHDV)

Verified locally from the pinned raw file:

| Fact | Result |
| --- | --- |
| Index archive | 116,131 bytes |
| SHA-256 | `10528fc5528bdfa0f6ae3dc7e2bd80e08e00d30464f0e15581664f502f2ebc8f` (matches the issue) |
| Archive members | `info`, `MWWHDV.segments`, `.scores`, `.rfs`, `languageid`, `languagename`, `starturl` |
| Segment rows | 7,382 across 260 chapter parents |
| Sum of segment durations | 55,435.827 s = **15.3988 indexed hours** |
| `info` Pass 0 | 7,716 utterances (7,196 without NaN); 15:52:52 = 15.88 h |
| `info` Pass 1 | 9,233 utterances (8,688 without NaN); 18:06:46 = 18.11 h |
| `info` RFS | 2,000 utterances; 3:24:45 = 3.41 h |
| Archive contents | indices/scores/metadata only — **no audio, no transcript text** |

The 15.40 h figure is the aligned-segment subset (7,382 rows), while 18:06:46 is
Pass 1's 8,688-utterance total. The remaining difference is not explained by the
archive itself (2,688 Pass 1 utterances have no segment rows); neither duration
is accepted usable speech.

### Downloader defect confirmed

Pinned source `festvox/datasets-CMU_Wilderness` commit
`a2102dcd050bf09d4b8aeea5babc6470b5b9376f`, `bin/do_found`:

- `get_from_bibleis` runs `wget -O xxx.html $uuu` (line 339) and then an
  unconditional `exit` (line 340) before parsing the page. Everything after it
  is unreachable. The earlier three-command recipe cannot work.
- The recorded start URL `http://listen.bible.is/MWWHDV/Matt/1/D` no longer
  resolves from this host (`curl: (6) Could not resolve host: listen.bible.is`),
  so the recipe is stale at the source as well as in the script.

### Reconstruction defect confirmed

- `prune_silences` (line 418 onward) removes leading, trailing and internal
  silences using an **F0 extractor**, then replaces `wav/` with the pruned files
  (lines 1273–1279).
- Segment reconstruction then applies the supplied start/end offsets to the
  pruned `wav/` (lines 1291–1292). The offsets no longer correspond to the
  original timeline, so blind application would mis-cut audio. F0-based pruning
  can also remove Hmong phonation cues.

### Access and rights

- Bible Brain core concepts and license pages return HTTP 200.
- The API endpoint `https://4.dbt.io/api/bibles?language_code=hmv` returns HTTP
  422 without credentials; no API key is present in this environment.
- Access/API terms do not themselves grant training, weight distribution or
  redistribution rights. No permission request has been made.

**Status: blocked** on a permitted current source for the exact edition, matching
text, and training/distribution permission. Do not run the old download chain or
apply the indices to current chapter audio. The issue's one-engineer-day
investigation remains available once the owner authorizes the outreach/access
request.

## Garellek / Esposito White Hmong materials

- Supplementary page returns HTTP 200 (1.1 MB) and contains White Hmong RPA
  story texts with broad phonemic transcriptions and English translations.
- The page states the study corpus is read speech from **eight native White
  Hmong speakers (three male, five female)** reading three folk tales, with
  audio available for viewing through the **D.C. Everest Oral History Project**.
- No explicit license, copyright grant or training permission statement was
  found on the page.
- Useful as a source for attested vowel/tone discussion and potential
  pronunciation reference; it is multi-speaker and not the selected target
  speaker. Audio rights/permission are unresolved; nothing was downloaded.

## University of Minnesota Hmong corpus

- `https://hmong-corpus.umn.edu/about` fails TLS validation from this host
  (`SEC_E_UNTRUSTED_ROOT`; curl error 60). No export, terms or audio obtained.
- Not an accepted corpus; contact/terms request still required.

## Pangeanic Hmong datasets

- Marketing page returns HTTP 200; it advertises Hmong speech/text datasets but
  exposes no per-speaker White Hmong hours, sample, license price or quote.
- No sample or license/quote exists in hand; a representative-sample and
  per-speaker-hours request is still required.

## Yuhalu

- Recheck today: `HEAD` returns HTTP 405; a bounded `Range: bytes=0-7` request
  returns HTTP 406 blocked by ModSecurity.
- The earlier finalization check (HTTP 200 with a ZIP signature and advertised
  Content-Length 255,561,989) could not be reproduced from this host today.
- Full archive, labels, duration and rights remain unaudited. Do not use.

## Pakorn2112 broadcast data

- Unchanged: quarantined for documented label conflicts. Every admitted excerpt
  would need fresh native verification and provenance; none was retrieved.

## Repository syllable recordings

- 5,187 isolated MP3s audited without modification (0.6416 decoded hours, 0
  decode errors). Reference/lexicon material only; speaker identity and dialect
  are unverified and it is not connected-speech training data.

## What would unblock F-03

1. Owner authorization and account/terms access for Bible Brain or another
   permitted source of the exact edition and matching text.
2. Outreach requests (authors/oral-history project, UMN, Pangeanic) that state
   the intended uses: training, synthetic reproduction, weight distribution and
   public/commercial use.
3. A representative sample with per-speaker hours and a concrete license/quote,
   before any purchase or corpus commitment.

Waiting on these must not stop F-02 preparation or owned-recording planning.

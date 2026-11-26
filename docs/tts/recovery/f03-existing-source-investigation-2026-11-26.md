# F-03 existing-source acquisition investigation — 2026-11-26 (mww)

**Status:** Existing-source only; **no new recordings**. No audio was retrieved
for training and no chapter was reconstructed, because no source currently grants
training/derivative rights or exposes permitted audio without an account. The
corrected source identity and actual/known minutes are below.

## Correction: ISO and fileset

The prior probe used ISO `hmv`. The CMU `LangList.txt` row and segment filenames
identify **MWWHDV, ISO `mww` (Hmong Daw), fileset `MWWHDVN2DA`**:

```text
MWWHDV mww https://en.wikipedia.org/wiki/Hmong_language \
  http://listen.bible.is/MWWHDV/Matt/1/D 21.5461 103.164 \
  7196 7.26 8688 7.24 18:06:46 6.00 5.85 Hmong Daw
```

| Field (README columns) | Value |
| --- | --- |
| Pass 0 utterances / MCD | 7,196 / 7.26 |
| Pass 1 utterances / MCD | 8,688 / 7.24 |
| Pass 1 duration | 18:06:46 (1,086.77 min) |
| v_rfs MCD values | 6.00 / 5.85 |
| Index archive | 116,131 B, SHA-256 `10528fc5…`; 7,382 segment rows, 260 chapter parents, 15.3988 h (923.93 min) |

The 260 parents are exactly the New Testament chapter count, so the index covers
Matthew–Revelation. The 15.3988 h segment subset is smaller than the 18:06:46
Pass 1 total; 2,688 Pass 1 utterances have no segment rows.

## Source identity: FCBH Hmong Daw Bible New Testament

Evidence from public catalogs:

- Google Play app `org.fcbh.mwwhdv.n2`, publisher **Faith Comes By Hearing**,
  titled **Hmong Daw Bible**, offers “Audio Bible (New Testament) in Hmong Daw”
  with verse-synchronized text. The package/fileset name contains `mwwhdv.n2`,
  matching CMU’s `MWWHDVN2DA`.
- Bible.com’s `mww` language page lists Hmong Daw editions including
  *Vajtswv Txojlus – lus Hmoob Dawb – txhais tshiab xyoo 2004* (HMOWSV, Thailand
  Bible Society, with audio) and *Vajluskub* (MWW2022, Bible Society Vietnam).
- The CMU start URL `/MWWHDV/Matt/1/D` and the `N2DA` fileset, plus a public
  listing describing a “Hmong Daw Audio Drama New Testament”, indicate the FCBH
  NT audio is likely a **dramatized, multi-voice recording**, not one narrator.
  This must be confirmed from FCBH metadata before any single-speaker training
  plan; if confirmed, it does not satisfy the target-speaker contract by itself.

The Bible narration and the CMU index therefore describe the same FCBH Hmong Daw
New Testament source. The exact Bible Society edition/text behind the FCBH audio
(likely HMOWSV 2004) still needs FCBH metadata confirmation.

## Access and training terms (verified today)

- `https://4.dbt.io/api/bibles?language_code=mww&v=4` and `?iso=mww&v=4` both return
  HTTP 422: “You need to provide the missing parameter 'key'.” The earlier `hmv`
  probe was wrong but failed for the same missing-key reason; credentials are
  required for either.
- FCBH **Bible Brain/DBP API License** text (retrieved 2026-11-26) limits content
  to access “solely through DBP … personal, non-commercial use on a single device”
  and states that other uses are not permitted without further consent, with all
  rights reserved. It grants no training, derivative-model, weight-publication or
  redistribution rights.
- No API key exists in this workspace and no written permission has been
  requested. **Do not download/retrieve the audio or train on it.**

## Actual/known minutes and remaining needs by source

| Source | Existing content | Actual/known minutes | Access/rights status | Remaining need |
| --- | --- | ---: | --- | --- |
| CMU MWWHDV index / FCBH Hmong Daw NT | 260 NT chapters; 7,382 aligned segments; Pass 1 8,688 utterances | **923.93 min** indexed (15.3988 h); Pass 1 **1,086.77 min** (18:06:46); **0 min retrieved** | index public; audio needs DBP key; personal-use-only terms, no training grant | DBP developer key + written FCBH training/derivative/weights permission; confirm edition/fileset and whether audio is multi-voice drama |
| FCBH Hmong Daw Bible (app/DBP) | same narration/text, NT audio + verse text | 0 permitted (same 18.1 h inventory) | free app for personal use; API key required; no training rights | same as above; also confirm text edition (likely HMOWSV 2004) |
| Pangeanic Hmong datasets | advertised Hmong speech WAV/MP3, transcribed, metadata | **2,580 min** (43 audio hours) advertised; **0 accessed** | enterprise licensing; no sample, quote, per-speaker hours or dialect detail | request sample + per-speaker White Hmong hours + concrete license/quote for training, derivatives, weight publication |
| Garellek / Esposito folktales | 8 native speakers × 3 folktales; 11,282 vowel tokens analyzed | duration not stated; **0 accessed** | audio “freely available” via the D.C. Everest Oral History Project, whose sites did not resolve today; no explicit training permission | author/oral-history permission, working audio URLs, total duration; multi-speaker |
| University of Minnesota Hmong Corpus | project building a searchable annotated corpus; no public export found | unknown; **0 accessed** | site TLS chain invalid from this host; project page HTTP 403; no terms | working access + corpus inventory/hours + export/licence terms |
| Yuhalu Voice archive | 6,282 WAVs, one folder `Kong`, 44.1 kHz | **51.38 min** (0.8564 h) isolated words/syllables; downloaded archive SHA-256 `b161393b…` | no README/licence/labels found beyond filenames; not sentences | licence/terms and labels; even if cleared it is not connected sentence speech |
| Repository syllable MP3s | 5,187 isolated syllables | **38.49 min** (0.6416 h) isolated syllables | book CC BY 4.0; speaker identity/dialect unverified | reference/technical use only; not sentence training data |

No new recordings are assumed or planned.

## One-chapter reconstruction (not performed)

The task allowed reconstructing one chapter with at most 20 transcript/audio pairs
**if permitted**. It was not performed:

- FCBH audio is not permitted for training/derivatives without written consent,
  and no API key exists.
- Pangeanic has not supplied a sample or license.
- Garellek’s audio host did not resolve and no permission is on file.
- UMN is inaccessible and has no terms.
- Yuhalu is word clips without labels or a license.

Reconstructing under the old `do_found` chain would also be invalid: it exits
after the first page, applies F0 pruning before supplied offsets, and its start
host is dead (`f03-source-investigation-2026-11-26.md`).

## Immediate data task and smallest owner decision

Acquire existing **sentence** audio with permission, not new recordings. The
smallest decision is to pursue one concrete source of the two viable sentence
leads:

1. **FCBH:** obtain a DBP developer key **and** written permission covering
   training, derivative models, weight publication and redistribution; confirm
   edition/fileset `MWWHDVN2DA` and whether the narration is multi-voice drama.
2. **Pangeanic:** approve requesting a sample, per-speaker White Hmong hours and a
   concrete training/derivative/weight-publication license/quote.

In parallel (no permission implied): request author/oral-history permission for
the Garellek folktales and working access/terms for the UMN corpus. Yuhalu and the
repository syllables remain non-sentence references. No source is accepted until
rights and native review are recorded.

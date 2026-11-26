# F-03 downloadable pairs vs indexes and their training terms (2026-11-26)

This supplements `f03-existing-source-investigation-2026-11-26.md`. It answers
one question: **is there an existing White Hmong sentence audio + text pair that
is downloadable now under terms permitting training?** Short answer: not yet.

| Source | What is actually available | Audio+text pair? | Training terms | Usable now? |
| --- | --- | --- | --- | --- |
| CMU MWWHDV | an **index** (7,382 rows, 15.3988 h) and segment filenames; no audio | no audio | none (index only) | **No** |
| FCBH Hmong Daw NT | same recordings as CMU; audio + verse text via DBP/app | yes, behind API | personal, non-commercial, single-device; no training/derivative/weights grant | **No** |
| Pangeanic | **43 transcribed audio hours**, WAV/MP3, metadata | yes, after contract | commercial data licence; needs quote | **No, until licensed** |
| Yuhalu | **downloadable** 247.8 MB archive, 6,282 clips, 51.38 min | filename “text” + clip, **word-level** | site says free non-commercial redistribution of the software; no explicit training grant | **Not for sentences; clarify terms** |
| Garellek/Esposito | eight-speaker folktales with RPA texts; audio via third-party site | yes in principle | no permission on file; audio host unresolved | **No** |
| UMN Hmong Corpus | project site; no export found | unknown | none | **No** |

The CMU index and the FCBH Hmong Daw Bible fileset `MWWHDVN2DA` refer to the same
recordings; the CMU archive provides **no waveforms**. No owner recordings or
worksheets are assumed.

## Consequence

The only sentence-level, transcribed alternative with a clear licensing path is
Pangeanic (commercial quote). Yuhalu is downloadable but word-level and has no
explicit training grant. The next data action is therefore to request the
Pangeanic sample/per-speaker hours/quote and, separately, clarify Yuhalu training
terms; neither is done. No audio was downloaded for training and no chapter was
reconstructed.

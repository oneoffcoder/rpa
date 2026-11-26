# Book syllable-playback reference check

Verifies the existing `generate.html?q=...` links against the documented
5,187-syllable inventory and `_static/mp3` files. It does not modify the book.

Pages: `source`; audio: `source/_static/mp3`.

| Measure | Value |
| --- | ---: |
| Playback links | 281 |
| Query tokens | 3690 |
| Playable tokens | 3688 |
| Expected syllables | 5187 |
| Audio files | 5187 |
| Missing audio for expected syllables | 0 |
| Unrecognized audio files | 0 |
| Playable tokens with no audio | 0 |
| Links with zero playable tokens | 0 |

## Result

**OK**

## Non-inventory tokens (disclosed as pre-existing)

- `dc` (1)
- `washington` (1)

The existing player filters these out, so their audio is not heard. This branch does not change that behavior; a later authorized content fix may replace the text or add an approved recording.


## Interpretation

Static reference integrity only. It does not prove browser playback, pronunciation, native acceptance or that stitching sounds natural.

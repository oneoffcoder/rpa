# Built HTML reference check

Verifies that the built Sphinx site's local `href`/`src` targets exist,
that the built syllable MP3 inventory matches, and that no
`generate.html?q=...` link has zero playable tokens.

Build: `build/html`.

| Measure | Value |
| --- | ---: |
| HTML pages | 12 |
| Checked local references | 6063 |
| Missing references | 0 |
| References escaping the build | 0 |
| Generate links | 281 |
| Generate tokens | 3690 |
| Playable tokens | 3688 |
| Built audio files | 5187 |
| Missing expected audio | 0 |
| Unexpected audio | 0 |
| Zero-playable links | 0 |
| Sitemap present | True |

## Result

**OK**

### Non-inventory tokens (pre-existing player behavior)

`dc` (1), `washington` (1)

## Interpretation

Static built-HTML integrity only. It does not execute JavaScript, verify audible playback, pronunciation or native acceptance.

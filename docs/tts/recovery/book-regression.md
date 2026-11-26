# Book regression check (verification layer 1)

**Scope:** the recovery branch adds `src/rpa_tts/`, `configs/tts/`, `tests/tts/`,
`docs/tts/recovery/`, packaging and requirements files. It does **not** modify
`source/`, `Makefile`, the original syllable MP3s or the explorer JavaScript:

```text
git diff --stat origin/main -- source Makefile
# (empty)
```

Building is not publishing. No S3/CloudFront upload, no DNS change and no merge
to `publish` occurred.

## Pinned documentation environment

Kept separate from the TTS environment so research pins cannot break the book.
`requirements-docs.txt`:

| Package | Pin |
| --- | --- |
| Sphinx | 7.4.7 |
| sphinx-rtd-theme | 2.0.0 |
| sphinxcontrib-bibtex | 2.6.2 |
| sphinx-sitemap | 2.6.0 |
| sphinxcontrib-blockdiag | 3.0.0 |
| blockdiag | 3.0.0 |
| setuptools | 80.9.0 (provides `pkg_resources` needed by blockdiag) |
| Pillow | 9.5.0 (blockdiag uses `ImageDraw.textsize`, removed in Pillow 10) |

## Pre-existing failures found and disposition

1. `sphinxcontrib.blockdiag` failed to import: `No module named 'pkg_resources'`
   with modern setuptools. **Resolved** by pinning `setuptools==80.9.0`.
2. blockdiag diagrams failed with `'ImageDraw' object has no attribute
   'textsize'` under Pillow 13. **Resolved** by pinning `Pillow==9.5.0`.
3. Remaining warnings are pre-existing or extension-version warnings (below);
   they do not fail the build and are not caused by this branch.

## Command and result

```text
uv venv .venv-docs --python 3.12
uv pip install -r requirements-docs.txt
.venv-docs/Scripts/sphinx-build -M html source build -E
```

Result: **build succeeded, 10 warnings**, 0 errors. Output in `build/html`
(ignored by Git). The sitemap is generated for `https://rpa.oneoffcoder.com/`.

Remaining warning classes:

| Warning | Classification |
| --- | --- |
| `pkg_resources is deprecated` (blockdiag import) | Environment note; pinned setuptools < 81 |
| `edge_layout is very experimental` | blockdiag extension notice |
| Duplicate explicit target name `"listen"` / `"[listen]"` | Pre-existing `source/enunciation.rst` / `source/speaking.rst` |
| `mathjax_config` does not work with current MathJax | Pre-existing `source/conf.py` setting |

## Reference checks

- 5,187 original syllable MP3s remain in `source/_static/mp3/` and were not
  modified (see the audit in `audit/repo-syllables/report.md`).
- `source/_static/js/custom.js` still drives the syllable playback and existing
  `generate.html?q=...` links; no redirect to an unqualified neural model was
  added.
- `source/_static/js/viz.js` and `source/viz.rst` are untouched.
- 10 content pages build: `index`, `counting`, `cvt`, `enunciation`, `grammar`,
  `speaking`, `spelling`, `syllables`, `viz`, `vocabulary` (plus `genindex` and
  `search`).
- No lesson audio or recording was replaced, so no native listening is claimed.

## Not run

- Publishing to S3/CloudFront, CodeBuild, or the `publish` branch.
- Browser-level playback testing (headless browser) — the audio references and
  JavaScript are unchanged by this branch, but a rendered-page check should run
  before any lesson-audio integration in F-08.

# Retired PR #1 archive: what was reused and what was not

PR #1 (`feat/hmong-tts-bootstrap`) is closed and **unmerged**. Its branch is no
longer advertised, but the archived commit remains accessible at
[`96698f0689a489b16868ee85046cac6604dfbb77`](https://github.com/oneoffcoder/rpa/tree/96698f0689a489b16868ee85046cac6604dfbb77).
A local tag `archive/pr1-96698f0` points at it in the recovery clone for
inspection; it is not merged and not used as the product baseline.

## Selectively reused (rewritten for the new package)

| Archive file | Reuse |
| --- | --- |
| `src/rpa_tts/orthography.py` | The 56 consonant + empty onset × 13 vowel × 7 tone inventory and `parse_syllable` contract, adapted into `src/rpa_tts/text/orthography.py` |
| `src/rpa_tts/audit.py` | Non-destructive probe/decode/hash/flag design and duplicate-group reporting, adapted into `src/rpa_tts/data/audio.py` and `src/rpa_tts/data/audit.py` |
| `tests/test_orthography.py` | Test ideas for inventory size, unmarked tone and multigraph parsing, rewritten in `tests/tts/test_orthography.py` |

## Deliberately not reused

- MLX/macOS backend and joint MLX experiments (`mlx_backend.py`,
  `joint_mlx.py`): not a qualified Linux/5090 environment.
- Syllable-stitching baseline and the old syllable/CVT split unit: historical
  approach that did not deliver accepted connected speech; connected-speech
  groups are the new split unit.
- The memorization-six audio files and reports: kept accessible in the archive
  commit as historical rejected/control material; not copied into the new
  branch and not presented as accepted speech.
- Any MLX dependency lock, model sweep or Apple Silicon configuration.

## Historical evidence kept accessible

- `docs/tts/corpus-audit-2026-11-24.{json,md}` — isolated-syllable corpus audit.
- `docs/tts/memorization-six/` — six personal test sentences with
  original/base/adapted/reconstructed WAVs and measurements. Development and
  regression material only; never a fresh final test.
- `docs/tts/orpheus-segmentation/` — protocol, run index, primary results and
  codec control.

These remain historical evidence under their archive commit. The new branch
records their provenance instead of merging the archive wholesale.

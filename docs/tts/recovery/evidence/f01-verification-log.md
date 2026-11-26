# F-01 verification log

Recorded 2026-11-26 (America/Los_Angeles) on `Desktop-W9QAA`, branch
`feat/white-hmong-tts-recovery`, base `origin/main`
`5919bb0650737a1534dc5746e191714820039b7d`. Exact commands and exit status:

| # | Command | Exit | Result |
| --- | --- | --- | --- |
| 1 | `uv venv .venv --python 3.12 && uv pip install -e ".[dev]"` | 0 | Python 3.12.16; core/dev pins installed |
| 2 | `uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` | 0 | `torch==2.9.1+cu128`, CUDA runtime 12.8, `torch.cuda.is_available() == True` |
| 3 | `python -m rpa_tts environment check --output docs/tts/recovery/evidence/f01-environment.json --gpu-proof --proof-output docs/tts/recovery/evidence/f01-gpu-kernel-proof.json` | 0 | Inventory recorded; kernel proof status `passed` |
| 4 | `python -m rpa_tts corpus audit --audio-dir source/_static/mp3 --output data/audit/repo-syllables --workers 8` | 1 | 5,187 files audited, 0 decode errors; `ok=false` because no transcripts (expected); reports copied to `docs/tts/recovery/audit/repo-syllables/` |
| 5 | `python -m pytest -q --junitxml=test-results.xml` | 0 | 179 tests, 0 failures, 0 errors, 1 skipped (GPU test behind `RPA_TTS_RUN_GPU_TESTS=1`) |
| 6 | `ruff check src tests` | 0 | All checks passed |
| 7 | `ruff format --check src tests` | 0 | 49 files already formatted |
| 8 | `uv venv .venv-docs --python 3.12 && uv pip install -r requirements-docs.txt` | 0 | Pinned book environment (Sphinx 7.4.7 etc.) |
| 9 | `.venv-docs/Scripts/sphinx-build -M html source build -E` | 0 | Build succeeded, 10 warnings, 0 errors (see `book-regression.md`) |
| 10 | `git diff --stat origin/main -- source Makefile` | 0 | Empty: lesson sources, audio and Makefile untouched |

Environment facts, kernel measurements and limitations are in `environment.md`
and `evidence/f01-environment.json` / `evidence/f01-gpu-kernel-proof.json`.
Installed package versions: `evidence/f01-pip-freeze.txt`.

Not run: any TTS model load or training (F-04), any native listening (F-02/F-07),
any publishing step. The GPU proof is pipeline readiness only.

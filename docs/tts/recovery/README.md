# Owned White Hmong TTS — recovery runbook

**Issue:** <https://github.com/oneoffcoder/rpa/issues/3> — *Implement owned White
Hmong TTS on RTX 5090 — final recovery plan v1.2*.
**Branch:** `feat/white-hmong-tts-recovery` (base `origin/main`
`5919bb0650737a1534dc5746e191714820039b7d`; PR target `main`).
**Status:** F-01 preparation is executed; no model has been trained, no native
listening has occurred, no source rights have cleared and no TTS release exists.
This issue must not be closed as a completed TTS product at this stage.

Goal: read unfamiliar RPA sentences and paragraphs with correct words,
tone/phonation and natural connected speech, without a recording of the
requested sentence. One clear neutral White Hmong / Hmong Daw / Hmoob Dawb voice
for RPA lessons and everyday reading; hosting is not mandatory for local/static
delivery.

## Stage status

State vocabulary: Not started / In progress / Passed / Failed / Inconclusive /
Blocked / Not required. Evidence is append-only in
[`evidence-index.json`](evidence-index.json).

| Stage | State | Dependencies | Responsible | Evidence | Next action |
| --- | --- | --- | --- | --- | --- |
| **F-01** workspace, environment, evidence, compute | **In progress** | none | implementation agent | [`environment.md`](environment.md), [`evidence/f01-environment.json`](evidence/f01-environment.json), [`evidence/f01-gpu-kernel-proof.json`](evidence/f01-gpu-kernel-proof.json), [`run-manifest.md`](run-manifest.md), [`decisions.jsonl`](decisions.jsonl) | Establish the independent owner-controlled backup destination (100–200 GB workspace is measured); then append the F-01 completion entry |
| **F-02** language/voice contract, calibration, scoring | **In progress** | native participants | Jee + native White Hmong lead | [`language-contract.yaml`](language-contract.yaml) (draft), [`listener-calibration.md`](listener-calibration.md) (Not started) | Hold the three-listener calibration session, approve dialect/voice, enumerate attested contrast cells, adopt approved expansions |
| **F-03** source acquisition, rights, pilot manifest | **In progress** | rights/outreach; owner decisions | data engineer + native QA; Jee for permissions | [`source-ledger.md`](source-ledger.md), [`rights-matrix.csv`](rights-matrix.csv), repository syllable audit (below) | One-day bounded CMU source/text/rights investigation; request UMN/Garellek/Pangeanic terms; no corpus purchase without a prepared scope |
| **F-04** one model proves 5090 load/train/generate | **Not started** | F-01 environment, F-02 calibration | ML engineer (agent) | [`model-candidates.md`](model-candidates.md), `rpa-tts model check` | Resolve Magpie artifact versions/hashes; install NeMo Speech; run the bounded two-day/8-GPU-hour gate |
| **F-05** accepted corpus, coverage, frozen splits | **Not started** | F-02, F-03; expansion needs F-04 | data engineer + native QA | [`coverage-ledger.md`](coverage-ledger.md), `rpa-tts corpus split` | Admit a 30–60 minute native-confirmed pilot after rights clear |
| **F-06** bounded learning comparisons, frozen candidate | **Not started** | F-04, F-05 | ML engineer + native development panel | [`stage-decisions.jsonl`](stage-decisions.jsonl) (empty), [`learning-curve.csv`](learning-curve.csv) | Write the first stage decision before any 1/2-hour comparison |
| **F-07** independent native qualification | **Not started** | F-05, F-06 frozen candidate; approved rights | evaluation lead (separate from checkpoint selection) | `rpa-tts evaluate gate` (implemented, no data) | Staff the 12-listener plan only after a candidate and protocol are frozen |
| **F-08** reproducible package + approved lesson audio | **Not started** | F-07 passed + distribution rights | ML/application engineer + native content reviewer | `rpa-tts package build` (gated) | Not actionable before F-07 |
| **F-09** online serving | **Not required** (deferred) | F-08 acceptance + concrete hosting requirement | operations | decision D-05 in [`decisions.jsonl`](decisions.jsonl) | Becomes required only if arbitrary-text serving is authorized |

Keyboard rule: an unavailable native participant, right, GPU or recording is a
blocker for that gate, never a pass and never a reason to lower a threshold.
Corrections are appended, not deleted.

## Executed evidence

- **Environment and kernel proof:** Windows 11, RTX 5090 (32,607 MiB, compute
  capability 12.0), driver 621.35, Python 3.12.16, PyTorch 2.9.1+cu128, CUDA 12.8.
  A real FP32 forward/backward/AdamW sequence produced finite loss and gradients,
  a loss decrease from 1.0487 to 0.8429, real parameter change and 18.88 MB peak
  allocated VRAM. See [`environment.md`](environment.md).
- **Repository syllable audit:** the existing 5,187 isolated-syllable MP3s were
  probed, decoded and hashed without modifying them; report in
  [`audit/repo-syllables/report.md`](audit/repo-syllables/report.md). This is
  lexicon/reference material with unverified speaker identity and dialect, not
  connected-speech training hours.
- **Book regression:** the pinned Sphinx environment builds `source/` with no
  lesson/audio regressions (see [`book-regression.md`](book-regression.md)).
  Building is not publishing.
- **Retired PR #1 archive** inspected selectively; see
  [`archive-pr1.md`](archive-pr1.md).

## Command surface

Stable CLI (`rpa-tts`, also `python -m rpa_tts`). Exit status: 0 success,
1 not-ok/finding, 2 bad input, 3 gate blocked.

| Command | Works today | Notes |
| --- | --- | --- |
| `rpa-tts environment check [--gpu-proof]` | yes | Inventory and real CUDA kernel proof |
| `rpa-tts text normalize --text ...` | yes | Lossless RPA normalization report |
| `rpa-tts text validate --manifest ...` | yes | Schema + leakage + normalization validation |
| `rpa-tts corpus audit --audio-dir ... --output ...` | yes | Non-destructive technical audit |
| `rpa-tts corpus split --manifest ... --output ...` | yes | Deterministic grouped split |
| `rpa-tts corpus coverage --manifest ... --contract ...` | yes | Coverage ledger |
| `rpa-tts model check` | yes | Lane metadata/hash readiness (not a GPU gate) |
| `rpa-tts train plan ...` | yes | Pre-run plan/stage-decision draft; no GPU work |
| `rpa-tts train run/resume` | **gated (F-04)** | Refuses until F-04 passes and a wrapper is tested |
| `rpa-tts synthesize` | **gated (F-07)** | Refuses until F-07 passes and a package is accepted |
| `rpa-tts evaluate metrics/gate` | yes | Scoring and F-07 gate; needs real listening data |
| `rpa-tts evidence add` | yes | Hash-checked append |
| `rpa-tts package build` | **gated (F-08)** | Refuses before F-08 |

Tests: `.venv/Scripts/python -m pytest` (TTS) — the book environment is separate.
The TTS suite exercises normalization preservation, manifest/schema/leakage
rules, split determinism, coverage accounting, pooled syllable errors, empty
outputs in denominators, crossed listener/script bootstrap arithmetic, gate
outcomes, evidence append-only behavior and CLI/gate refusals.

## Runbook index

| Document | Contents |
| --- | --- |
| [`environment.md`](environment.md) | Measured host, pinned stack, GPU proof, reproduction commands |
| [`language-contract.yaml`](language-contract.yaml) | Draft dialect, normalization, scoring, buckets, attested cells (native work) |
| [`listener-calibration.md`](listener-calibration.md) | F-02 session protocol, instrument checks, privacy |
| [`source-ledger.md`](source-ledger.md) | Source facts, CMU findings, acquisition leads, audit rules |
| [`rights-matrix.csv`](rights-matrix.csv) | Per-source permission status and next actions |
| [`model-candidates.md`](model-candidates.md) | F-04 Magpie/VITS lane contracts, gate checklist, budgets |
| [`run-manifest.md`](run-manifest.md) | Schemas, reproducibility fields, evidence and storage rules |
| [`coverage-ledger.md`](coverage-ledger.md) | Coverage screen rule and current (empty) data |
| [`decisions.jsonl`](decisions.jsonl) | Product/project decisions, append-only |
| [`stage-decisions.jsonl`](stage-decisions.jsonl) | Pre-run learning decisions (empty) |
| [`learning-curve.csv`](learning-curve.csv) | Learning-curve columns (no rows) |
| [`handoff.md`](handoff.md) | Exact resumable state for the next worker |

## Not claimed

- No TTS model has been loaded, fine-tuned or evaluated on Hmong text.
- No native listening, calibration, approval or qualification has occurred.
- No source rights, consent or distribution permission has been cleared.
- No corpus hours are accepted; the repository's isolated syllables are not a
  connected-speech corpus.
- No lesson audio is integrated, no weights are published and no online service
  exists.
- The issue remains open and product completion is unresolved.

## Progress reporting

The agent session that produced this branch has no authenticated GitHub session
(`gh` returns HTTP 401), so the owner must post the issue progress update that
links this README and the evidence index. Keep this table live; do not replace
failed or rejected evidence.

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
| **F-02** language/voice contract, calibration, scoring | **In progress** | native participants | Jee + native White Hmong lead | [`language-contract.yaml`](language-contract.yaml) (draft), [`listener-calibration.md`](listener-calibration.md) (Not started), [`listening-instrument.md`](listening-instrument.md) (software ready), [`calibration-scripts.review.csv`](calibration-scripts.review.csv) (272 rows, blank decisions), [`review-import.md`](review-import.md) (validation ready) | Native lead fills the review sheet (270 pending, 2 blocked), imports decisions, approves 100–150 scripts, then records and runs the calibration pack |
| **F-03** source acquisition, rights, pilot manifest | **In progress** | rights/outreach; owner decisions | data engineer + native QA; Jee for permissions | [`source-ledger.md`](source-ledger.md), [`rights-matrix.csv`](rights-matrix.csv), [`f03-source-investigation-2026-11-26.md`](f03-source-investigation-2026-11-26.md), repository syllable audit (below) | CMU is blocked (dead start URL, broken downloader, offsets invalidated by F0 pruning, no rights); owner must authorize outreach/access to Bible Brain, Garellek/oral-history, UMN or Pangeanic; continue owned-recording planning |
| **F-04** one model proves 5090 load/train/generate | **In progress** | F-01 environment, F-02 calibration | ML engineer (agent) | [`model-candidates.md`](model-candidates.md), [`f04-artifact-inspection.md`](f04-artifact-inspection.md), [`f04-host-path.md`](f04-host-path.md), [`f04-vits-standby.md`](f04-vits-standby.md), `configs/tts/f04-magpie.yaml`, `configs/tts/f04-vits.yaml`, `rpa-tts model check` | Owner chooses the Linux path for Magpie (enable WSL2 or a Linux worker; native Windows NeMo resolves CPU torch); the VITS standby environment now imports on Windows and its `rpa-chars-1` text contract is verified, but no lane has run |
| **F-05** accepted corpus, coverage, frozen splits | **In progress** | F-02, F-03; expansion needs F-04 | data engineer + native QA | [`coverage-ledger.md`](coverage-ledger.md), [`dataset-lifecycle.md`](dataset-lifecycle.md), [`audio-verification.md`](audio-verification.md), `rpa-tts corpus split/freeze/verify-audio` | Freeze, verify and byte-level audio checks are ready; no accepted corpus exists. Admit a 30–60 minute native-confirmed pilot after rights clear |
| **F-06** bounded learning comparisons, frozen candidate | **In progress** | F-04, F-05 | ML engineer + native development panel | [`stage-decisions.jsonl`](stage-decisions.jsonl) (empty), [`learning-curve.csv`](learning-curve.csv) (header), [`dataset-lifecycle.md`](dataset-lifecycle.md) | Record a stage decision before any comparison; `train record` now enforces unique, bounded learning-curve rows |
| **F-07** independent native qualification | **In progress** | F-05, F-06 frozen candidate; approved rights | evaluation lead (separate from checkpoint selection) | [`listening-instrument.md`](listening-instrument.md), [`listening-review.md`](listening-review.md), [`qualification-suite.md`](qualification-suite.md), `rpa-tts evaluate gate` (implemented, no data) | Build the 300-sentence + 20-paragraph suite with the enforced composition contract once scripts, contrast pairs and stimuli are frozen; recruit 12 listeners |
| **F-08** reproducible package + approved lesson audio | **In progress** | F-07 passed + distribution rights | ML/application engineer + native content reviewer | [`f08-packaging.md`](f08-packaging.md), `rpa-tts package inspect` | Package contract and inspection are ready; no accepted checkpoint, package, benchmark or distribution rights exists. Not actionable before F-07 |
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
- **Pinned model/codec artifacts:** Magpie `446bff55…` (`.nemo` SHA-256
  `ec675fa8…`, 1.47 GB) and NanoCodec `fc00890b…` (`.nemo` SHA-256 `28c2518d…`,
  425 MB) downloaded and hash-verified against the published LFS ids. The
  packaged config, weight shapes and ByT5 byte-token conditioning are recorded
  in [`f04-artifact-inspection.md`](f04-artifact-inspection.md);
  `rpa-tts model check --config configs/tts/f04-magpie.yaml --lane magpie`
  passes all lane checks. No model has loaded.
- **F-02 native review worksheet:** the 272-candidate draft pool now has a
  blank-decision review CSV (accept/edit/reject, edited text, notes) and
  instructions; 270 rows await approval and 2 are blocked on loanword/quote
  decisions. `review-import` validates a filled sheet into an approved set
  (matching draft ids/text, all rows decided, strict normalization, no silent
  drops). No row is approved, imported or recorded. See
  [`calibration-scripts.review.md`](calibration-scripts.review.md) and
  [`review-import.md`](review-import.md).
- **F-03 source verification:** the CMU index archive was re-verified
  (116,131 B, SHA-256 `10528f…`, 7,382 segments = 15.3988 h), the downloader's
  unconditional exit and F0-pruning-before-offsets defects were confirmed, and
  the old start host no longer resolves. Garellek materials expose RPA texts and
  eight-speaker read folk tales with audio via a third party; UMN fails TLS;
  Yuhalu blocks range requests; no rights are cleared. See
  [`f03-source-investigation-2026-11-26.md`](f03-source-investigation-2026-11-26.md).
- **F-04 host constraint:** WSL2 is not enabled (`VirtualMachinePlatform`
  disabled, no distributions, no Docker) and the pinned NeMo stack resolves GPU
  PyTorch only on Linux. The owner must choose WSL2 or a Linux worker before the
  Magpie GPU gate; evidence and options are in
  [`f04-host-path.md`](f04-host-path.md).
- **Book regression:** the pinned Sphinx environment builds with no lesson/audio
  regressions (see [`book-regression.md`](book-regression.md)). A playback
  reference check found 281 links / 3,688 playable tokens, an exact 5,187-file
  inventory match, no zero-playable links and two disclosed non-inventory tokens
  (`washington`, `dc`) in one pre-existing link; see
  [`book-audio-references.md`](book-audio-references.md). Building is not
  publishing, and static references are not browser playback.
- **Draft script pool:** 272 native-authored book sentences extracted with source
  line/section, normalized under `rpa-norm-1` (270 pending native approval, 2
  blocked on loanword/quote decisions), totaling 3,633 syllables ≈18 minutes at
  4 s/script. Drafts only; no recording or approval exists. See
  [`calibration-scripts.draft.md`](calibration-scripts.draft.md).
- **F-05/F-06 data lifecycle:** dataset freeze/verify enforces native approval,
  resolved rights, non-empty splits, no group leakage, an approved contract and a
  passing coverage screen before writing an immutable version; learning-curve
  append enforces fixed columns, unique run ids and bounded values. Byte-level
  `verify-audio` re-hashes and re-decodes manifest files (synthetic tests only).
  No dataset version or learning row exists. See
  [`dataset-lifecycle.md`](dataset-lifecycle.md) and
  [`audio-verification.md`](audio-verification.md).
- **F-07 suite composition:** qualification packs must contain exactly 300
  sentences (100/80/60/60), 20 paragraphs of 30–90 s, ≥12 listeners (≥9
  all-role), ≥40 critical contrast items and a `candidate` condition; the
  manifest records the design check. No suite, stimuli or candidate exists.
  See [`qualification-suite.md`](qualification-suite.md).
- **F-08 package contract:** a schema-validated `model-manifest.json` records the
  accepted checkpoint/config hashes, text front end, voice policy, benchmark
  verdict and rights id; `package inspect` verifies all of them, while
  `package build` and synthesis remain gated. No package or accepted model
  exists. See [`f08-packaging.md`](f08-packaging.md).
- **VITS standby preparation:** the `rpa-chars-1` character contract (37 tokens)
  round-trips all 5,187 documented syllables, and a separate `.venv-vits` imports
  `coqui-tts 0.28.0` with `torch 2.9.1+cu128` on this Windows host after pinning
  `torchcodec 0.9.1` and supplying shared FFmpeg 8.1 DLLs. Import only; no model
  instantiated or trained, lane standby. See
  [`f04-vits-standby.md`](f04-vits-standby.md).
- **Listening instrument:** blinded, counterbalanced pack builder and response
  ingestion implemented and tested (335 TTS tests), including two-adjudicator
  critical-error merging and human paragraph clause/language-switch review; see
  [`listening-instrument.md`](listening-instrument.md) and
  [`listening-review.md`](listening-review.md). No session, stimuli or
  responses exist, and the template sentences are not native-approved.
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
| `rpa-tts corpus scripts --rst source/speaking.rst ...` | yes | Drafts candidate scripts from book examples (not approved) |
| `rpa-tts corpus book-audio` | yes | Verifies existing syllable playback links against the 5,187 MP3 inventory |
| `rpa-tts corpus review-sheet --draft ...` | yes | Writes a blank-decision native review CSV and instructions |
| `rpa-tts corpus review-import --review ...` | yes | Validates a filled review sheet into an approved script set (none exists) |
| `rpa-tts corpus freeze --manifest ... --version ...` | yes | Freezes an approved, coverage-passing dataset version (none exists) |
| `rpa-tts corpus verify-dataset --config ...` | yes | Re-hashes a frozen dataset config |
| `rpa-tts corpus verify-audio --manifest ... --audio-root ...` | yes | Re-hashes and re-decodes every manifest audio file (no real corpus yet) |
| `rpa-tts train record --run-manifest ...` | yes | Appends one validated learning-curve row (none exists) |
| `rpa-tts model check` | yes | Lane metadata/hash readiness (not a GPU gate) |
| `rpa-tts train plan ...` | yes | Pre-run plan/stage-decision draft; no GPU work |
| `rpa-tts train run/resume` | **gated (F-04)** | Refuses until F-04 passes and a wrapper is tested |
| `rpa-tts synthesize` | **gated (F-07)** | Refuses until F-07 passes and a package is accepted |
| `rpa-tts evaluate metrics/gate` | yes | Scoring and F-07 gate; needs real listening data |
| `rpa-tts evaluate pack` | yes | Builds a blinded pack; qualification packs must meet the 300/80/60/60 + 20-paragraph/12-listener composition |
| `rpa-tts evaluate ingest` | yes | Validates completed sheets into a scores payload |
| `rpa-tts evaluate adjudicate` | yes | Merges two-adjudicator critical-error decisions; preserves raw choices |
| `rpa-tts evaluate paragraphs` | yes | Merges human clause/language-switch paragraph review |
| `rpa-tts evidence add` | yes | Hash-checked append |
| `rpa-tts package build` | **gated (F-08)** | Refuses before F-08 |
| `rpa-tts package inspect` | yes | Validates package manifest, hashes, rights and benchmark verdict (no package exists) |

Tests: `.venv/Scripts/python -m pytest` (TTS) — the book environment is separate.
The TTS suite exercises normalization preservation, manifest/schema/leakage
rules, split determinism, coverage accounting, pooled syllable errors, empty
outputs in denominators, crossed listener/script bootstrap arithmetic, gate
outcomes, evidence append-only behavior and CLI/gate refusals.
`.github/workflows/tts.yml` runs the CPU test/lint suite and the pinned book
build on pushes and PRs; this session could not read GitHub check status (no
authenticated API session), so the workflow is added but not claimed green.

## Runbook index

| Document | Contents |
| --- | --- |
| [`environment.md`](environment.md) | Measured host, pinned stack, GPU proof, reproduction commands |
| [`language-contract.yaml`](language-contract.yaml) | Draft dialect, normalization, scoring, buckets, attested cells (native work) |
| [`listener-calibration.md`](listener-calibration.md) | F-02 session protocol, instrument checks, privacy |
| [`listening-instrument.md`](listening-instrument.md) | Blinded pack builder, task order, ingestion rules and commands |
| [`listening-review.md`](listening-review.md) | Two-adjudicator critical-error review and paragraph clause/language-switch review |
| [`qualification-suite.md`](qualification-suite.md) | Enforced F-07 suite composition and remaining prerequisites |
| [`f08-packaging.md`](f08-packaging.md) | Package contents, manifest schema, inspection checks, reproduction and release rules |
| [`calibration-scripts.draft.md`](calibration-scripts.draft.md) | Draft script pool extracted from the book (not approved, not recorded) |
| [`calibration-scripts.draft.jsonl`](calibration-scripts.draft.jsonl) | Per-candidate normalized text, source line/section and issue status |
| [`calibration-scripts.review.csv`](calibration-scripts.review.csv) | Native review worksheet with blank decisions and edit columns |
| [`calibration-scripts.review.md`](calibration-scripts.review.md) | Review instructions, counts and blocked rows |
| [`review-import.md`](review-import.md) | Validation and provenance for importing filled review decisions |
| [`source-ledger.md`](source-ledger.md) | Source facts, CMU findings, acquisition leads, audit rules |
| [`f03-source-investigation-2026-11-26.md`](f03-source-investigation-2026-11-26.md) | Verified CMU defects, rechecked lead access and rights status |
| [`rights-matrix.csv`](rights-matrix.csv) | Per-source permission status and next actions |
| [`model-candidates.md`](model-candidates.md) | F-04 Magpie/VITS lane contracts, gate checklist, budgets |
| [`f04-artifact-inspection.md`](f04-artifact-inspection.md) | Verified artifact hashes, packaged config, text-tokenizer contract and risks |
| [`f04-host-path.md`](f04-host-path.md) | WSL2/Linux requirement, measured host facts and owner options |
| [`f04-vits-standby.md`](f04-vits-standby.md) | Standby VITS pin, `rpa-chars-1` text contract and measured import environment |
| [`book-regression.md`](book-regression.md) | Pinned Sphinx build, warnings and reference checks |
| [`book-audio-references.md`](book-audio-references.md) | Static syllable playback link/inventory verification |
| [`audit/repo-syllables/report.md`](audit/repo-syllables/report.md) | Technical audit of the 5,187 isolated-syllable recordings |
| [`run-manifest.md`](run-manifest.md) | Schemas, reproducibility fields, evidence and storage rules |
| [`dataset-lifecycle.md`](dataset-lifecycle.md) | Immutable dataset freeze/verify and validated learning-curve records |
| [`audio-verification.md`](audio-verification.md) | Byte-level manifest audio re-hash/decode checks and their limits |
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

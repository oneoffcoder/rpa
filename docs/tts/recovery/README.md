# Owned White Hmong TTS — recovery runbook

**Issue:** <https://github.com/oneoffcoder/rpa/issues/3> — *Implement owned White
Hmong TTS on RTX 5090 — final recovery plan v1.2*.
**Branch:** `feat/white-hmong-tts-recovery` (base `origin/main`
`5919bb0650737a1534dc5746e191714820039b7d`; PR target `main`).
**Status:** F-01 preparation is executed; F-02 is blocked by the lasting
no-new-recording constraint; a real F-04 runner exists but no model has been
loaded or trained, no native listening has occurred, and no release exists. F-09
is `not_required` for local/static delivery. No source rights have cleared. This
issue must not be closed as a completed TTS product at this stage.

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
| **F-02** language/voice contract, calibration, scoring | **Blocked** | native participants; Jee capacity | independent native reviewer + rights | [`language-contract.yaml`](language-contract.yaml) (draft), [`listener-calibration.md`](listener-calibration.md) (Not started), [`listening-instrument.md`](listening-instrument.md) (software ready), [`calibration-scripts.review.csv`](calibration-scripts.review.csv) (272 rows, blank decisions), [`review-import.md`](review-import.md) (validation ready), [`capacity-constraint.md`](capacity-constraint.md) | Jee will not record, recruit, or fill the worksheet. Engage an independent native reviewer/rights holder or adopt an explicit scope change; do not expect calibration recordings |
| **F-03** source acquisition, rights, pilot manifest | **In progress** | native QA; distribution rights | data engineer + native QA; owner for permissions | [`mwwhdv-nt.md`](mwwhdv-nt.md), [`mwwhdv-pilot.md`](mwwhdv-pilot.md), [`source-ledger.md`](source-ledger.md), [`rights-matrix.csv`](rights-matrix.csv) | First-party training permission for the 2000 UBS Hmong Daw narration is recorded on the durable Confluence page; live audio/text/timestamp access is verified. Audio/weights stay private; distribution/weight-publication rights and native QA remain open |
| **F-04** one model proves 5090 load/train/generate | **In progress** | F-01 environment, F-02 data | ML engineer (agent) | [`f04-vits-standby.md`](f04-vits-standby.md), [`f04-runner.md`](f04-runner.md), [`f04-cuda-diagnostic.md`](f04-cuda-diagnostic.md) | Magpie bounded by its 1.89 kbps codec. VITS activated: LJSpeech decoder copy-synthesis control works (lag −0.0098, 231/231 compatible keys); a fresh Hmong text model with that decoder + 200 paired updates breaks the hop artifact (recon lag 0.062; outputs 0.14–0.20, F0 185–186 Hz). Waveform path promoted; no native/quality/release result |
| **F-05** accepted corpus, coverage, frozen splits | **In progress** | F-02, F-03; native acceptance | data engineer + native QA | [`mwwhdv-negative-diagnosis.md`](mwwhdv-negative-diagnosis.md), [`mwwhdv-nt.md`](mwwhdv-nt.md), [`mwwhdv-splits.md`](mwwhdv-splits.md) | Full authorized NT: **7,935 pairs / 22.523 h**; all WAVs verified. Duplicate audit found REV 22:20/21 share one WAV/interval; both quarantined, split now 5,759/714/1,460. Multi-voice, 22 missing-timestamp verses; not native-accepted/frozen |
| **F-06** bounded learning comparisons, frozen candidate | **In progress** | F-04, F-05; native panel | ML engineer + native development panel | [`mwwhdv-negative-diagnosis.md`](mwwhdv-negative-diagnosis.md), [`mwwhdv-split-train.md`](mwwhdv-split-train.md) | Clean train/dev dev mean 11.9123 → 11.0115, but owner heard all 8 A/B clips and none sounded Hmong. Codec audit: 1.89 kbps NanoCodec round-trip is itself not recognizable Hmong; Magpie and standalone codec have 713/713 equal functional weights and exactly equal codes, so TTS is bounded by the codec. EOS contract now matches pinned dataset (LUK 276, MRK 78); aligned probe remains codec-limited. No frozen candidate |
| **F-07** independent native qualification | **In progress** | F-05, F-06 frozen candidate; approved rights | evaluation lead (separate from checkpoint selection) | [`listening-instrument.md`](listening-instrument.md), [`listening-review.md`](listening-review.md), [`qualification-suite.md`](qualification-suite.md) | Negative owner listening (A/B, LUK codec and matched-token Luke); English native generation transcribed correctly. No native-authored suite, contrast pairs, stimuli or 12-listener panel |
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
- **2026-11-26 capacity constraint:** Jee will not record, recruit a speaker or fill
  the native worksheet. F-02 is blocked and F-04–F-08 cannot pass without a
  permitted existing sentence corpus and an independent native reviewer, or an
  explicitly adopted scope change. No acceptance criterion is lowered. See
  [`capacity-constraint.md`](capacity-constraint.md).
- **F-03 correction and investigation:** the CMU lookup is ISO `mww`, fileset
  `MWWHDVN2DA`; the CMU index and the FCBH Hmong Daw New Testament narration are
  the same source. The index is 116,131 B (SHA-256 `10528f…`, 7,382 segments =
  15.3988 h). FCBH terms are personal-use-only and require a DBP key; no audio
  was downloaded or reconstructed. Garellek materials expose RPA texts and
  eight-speaker read folk tales with audio via a third party; UMN fails TLS;
  Yuhalu serves a 51.38-minute isolated-word archive under an unverified license;
  Pangeanic advertises 43 h with no sample or quote; no rights are cleared. See
  [`f03-existing-source-investigation-2026-11-26.md`](f03-existing-source-investigation-2026-11-26.md).
- **F-04 CPU diagnostic:** behind clearly labelled compatibility shims, the real
  Magpie weights load on CPU (372.9 M params, 22.3 s) and generate 1.02 s of
  private audio (10.63 s, **unlistened**, no quality claim); a `.nemo` save/reload
  is bit-exact across 260 tensors. **No 5090 GPU load, training step, resume or
  native listening has run, and release synthesis stays F-07-gated.** See
  [`f04-cpu-diagnostic.md`](f04-cpu-diagnostic.md).
- **F-03/F-04/F-05 authorized pilot:** first-party training permission for the
  existing Hmong Daw UBS 2000 narration is recorded on the durable Confluence page
  (link in [`mwwhdv-pilot.md`](mwwhdv-pilot.md)). A private MAT 1–10 pilot has 314
  verse pairs / 54.517 min; a real 5090 step, a resumed second step and a 10-step
  finetune all produced finite losses and resumable checkpoints. Audio and weights
  stay private; native review, corpus acceptance/freeze, listening and distribution
  rights remain open.
- **F-04 CUDA diagnostic:** the same real checkpoint on the RTX 5090 (all
  372.9 M parameters `cuda:0`, 22.45 s load) generated 0.8824 s in 18.27 s with
  **3,527.3 MiB peak allocated** and a verified waveform hash. Audio was
  **unlistened**; no training step, resume, native listening or release occurred.
  See [`f04-cuda-diagnostic.md`](f04-cuda-diagnostic.md).
- **F-03 downloadable vs index:** CMU supplies only an index; FCBH audio is the
  same recordings behind a key with personal-use terms; Pangeanic is the only
  sentence-level transcribed alternative with a commercial licensing path; Yuhalu
  is downloadable word-level audio without an explicit training grant. No sentence
  pair is training-usable now. See
  [`f03-downloadable-pairs-2026-11-26.md`](f03-downloadable-pairs-2026-11-26.md).
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
  [`book-audio-references.md`](book-audio-references.md). The built-HTML check
  then verified 12 pages, 5,152 audio references, 737 page links, all styles/
  scripts/images and the sitemap with 0 missing local references; see
  [`book-html-references.md`](book-html-references.md). Building is not
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
  ingestion implemented and tested (382 TTS tests), including two-adjudicator
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
| `rpa-tts corpus book-html` | yes | Verifies built HTML local references, assets and generated-syllable links |
| `rpa-tts corpus mwwhdv-chapter` | yes | Cuts a local authorized MWWHDV chapter into hashed verse rows |
| `rpa-tts corpus review-sheet --draft ...` | yes | Writes a blank-decision native review CSV and instructions |
| `rpa-tts corpus review-import --review ...` | yes | Validates a filled review sheet into an approved script set (none exists) |
| `rpa-tts corpus freeze --manifest ... --version ...` | yes | Freezes an approved, coverage-passing dataset version (none exists) |
| `rpa-tts corpus verify-dataset --config ...` | yes | Re-hashes a frozen dataset config |
| `rpa-tts corpus verify-audio --manifest ... --audio-root ...` | yes | Re-hashes and re-decodes every manifest audio file (no real corpus yet) |
| `rpa-tts train record --run-manifest ...` | yes | Appends one validated learning-curve row (none exists) |
| `rpa-tts model check` | yes | Lane metadata/hash readiness (not a GPU gate) |
| `rpa-tts model diagnose` | yes | Exact F-04 environment defect chain (platform, CUDA, NeMo, artifacts, tokenizer) |
| `rpa-tts train gate` | yes | Attempts F-04; writes diagnosis and exits 3 with the first defect when blocked |
| `rpa-tts train finetune` | yes | Builds/runs the pinned Magpie Hydra command in a qualified environment |
| `rpa-tts train render-private` | yes | Private evaluation audio only; refuses on a failing environment diagnosis; never release |
| `rpa-tts train plan ...` | yes | Pre-run plan/stage-decision draft; no GPU work |
| `rpa-tts train run/resume` | **gated (F-04)** | Refuses until F-04 passes and a wrapper is tested |
| `rpa-tts synthesize` | **gated (F-07)** | Refuses until F-07 passes and a package is accepted |
| `rpa-tts evaluate metrics/gate` | yes | Scoring and F-07 gate; needs real listening data |
| `rpa-tts evaluate pack` | yes | Builds a blinded pack; qualification packs must meet the 300/80/60/60 + 20-paragraph/12-listener composition |
| `rpa-tts evaluate ab-player` | yes | Writes a local blinded A/B listening page from an answer key |
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
build on pushes and PRs. GitHub Actions run **45789357474 passed on `9add501`**
(the torch-optional test boundary fix); the owner earlier verified run
45785823770 on `cfc2872`. Commits after `9add501` were not observed from this
session, so do not claim green beyond that run.

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
| [`f03-existing-source-investigation-2026-11-26.md`](f03-existing-source-investigation-2026-11-26.md) | Corrected `mww`/MWWHDVN2DA identity, FCBH terms, minutes by source |
| [`capacity-constraint.md`](capacity-constraint.md) | Lasting no-new-recordings constraint and owner options |
| [`rights-matrix.csv`](rights-matrix.csv) | Per-source permission status and next actions |
| [`model-candidates.md`](model-candidates.md) | F-04 Magpie/VITS lane contracts, gate checklist, budgets |
| [`f04-artifact-inspection.md`](f04-artifact-inspection.md) | Verified artifact hashes, packaged config, text-tokenizer contract and risks |
| [`f04-host-path.md`](f04-host-path.md) | WSL2/Linux requirement, measured host facts and owner options |
| [`f04-runner.md`](f04-runner.md) | Real Magpie load/finetune/private-render path, defect chain and pending GPU/listening checks |
| [`f04-cpu-diagnostic.md`](f04-cpu-diagnostic.md) | CPU-only load, private generation and bit-exact `.nemo` round trip (not the gate) |
| [`f04-cuda-diagnostic.md`](f04-cuda-diagnostic.md) | One bounded 5090 load/inference with VRAM/time (not the gate) |
| [`mwwhdv-pilot.md`](mwwhdv-pilot.md) | Authorized 2000 drama source, 314-pair pilot, real steps/checkpoints |
| [`mwwhdv-nt.md`](mwwhdv-nt.md) | Full 22.5-hour authorized NT corpus and private training runs |
| [`mwwhdv-val-1000.md`](mwwhdv-val-1000.md) | Private held-out validation improvement (development only) |
| [`mwwhdv-val-3000.md`](mwwhdv-val-3000.md) | Private 3,000-step validation continuation (development only) |
| [`mwwhdv-mark-holdout.md`](mwwhdv-mark-holdout.md) | Book-disjoint held-out Mark test (development only) |
| [`mwwhdv-mark-holdout-4k.md`](mwwhdv-mark-holdout-4k.md) | Superseded 4,000-step run; repeated first 2,000 rows |
| [`mwwhdv-cursor-resume.md`](mwwhdv-cursor-resume.md) | Cursor bug, corrected distinct-row continuation, Mark metric |
| [`mwwhdv-ab-private.md`](mwwhdv-ab-private.md) | Blinded base-vs-trained private A/B (unlistened) |
| [`mwwhdv-corpus-verification.md`](mwwhdv-corpus-verification.md) | Audio integrity and syllable-coverage audit |
| [`mwwhdv-splits.md`](mwwhdv-splits.md) | Private chapter-grouped train/dev/test split |
| [`mwwhdv-split-train.md`](mwwhdv-split-train.md) | Clean train/dev candidate, dev loss 11.9123 → 11.0115 |
| [`mwwhdv-portable-artifact.md`](mwwhdv-portable-artifact.md) | Private portable candidate `.nemo`, fingerprint-exact |
| [`mwwhdv-synth-repro.md`](mwwhdv-synth-repro.md) | Cross-process deterministic synthesis from the candidate |
| [`mwwhdv-negative-diagnosis.md`](mwwhdv-negative-diagnosis.md) | Negative listening, duplicate quarantine and pipeline diagnosis |
| [`f03-downloadable-pairs-2026-11-26.md`](f03-downloadable-pairs-2026-11-26.md) | Downloadable pair vs index and training-term status |
| [`f04-vits-standby.md`](f04-vits-standby.md) | Standby VITS pin, `rpa-chars-1` text contract and measured import environment |
| [`book-regression.md`](book-regression.md) | Pinned Sphinx build, warnings and reference checks |
| [`book-audio-references.md`](book-audio-references.md) | Static syllable playback link/inventory verification |
| [`book-html-references.md`](book-html-references.md) | Built-site local reference/asset verification |
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

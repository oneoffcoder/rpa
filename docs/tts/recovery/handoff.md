# Handoff — owned White Hmong TTS recovery

Written 2026-11-26 for the next worker. Read
[`README.md`](README.md) (stage table) and
[`evidence-index.json`](evidence-index.json) (hashed evidence) first.

## Branch and repository state

- **Repository:** `oneoffcoder/rpa`
- **Branch:** `feat/white-hmong-tts-recovery`
- **Base:** `origin/main` `5919bb0650737a1534dc5746e191714820039b7d` (verified at
  branch creation; no later main changes were present)
- **Commit / working tree:** see the "Checkpoint" note at the end of this file
  and `git status --short --branch` on the checkout. The implementation commit
  contains code, configs, tests and this runbook; one follow-up commit records
  this handoff.
- **PR to `main`:** not opened. This session had no authenticated GitHub API
  session (`gh` returned HTTP 401), so branch push succeeded over SSH but the PR
  must be opened by the owner. Do not auto-merge.
- **CI:** `.github/workflows/tts.yml` runs the CPU TTS tests/lint and the pinned
  book build on pushes and PRs. GitHub Actions run **45791014812 passed on
  `639a6dd`** (both jobs); run 45789357474 passed on `9add501`. Commits after
  `639a6dd` were not observed from this session.
- **Retired archive:** `archive/pr1-96698f0` locally tags the unmerged PR #1
  commit `96698f0689a489b16868ee85046cac6604dfbb77`; it is inspected, not merged.

## Host and environment

- **Host:** `Desktop-W9QAA` (Windows 11, AMD64, 127.9 GB RAM, Ryzen 9 5950X),
  RTX 5090 with 32,607 MiB and compute capability 12.0, driver 621.35.
  Firmware virtualization and SLAT are enabled, but the `VirtualMachinePlatform`
  feature is disabled, no WSL distribution or Docker is installed, and the pinned
  NeMo stack resolves GPU PyTorch only on Linux. F-04 therefore needs an owner
  host-path choice (WSL2 or Linux worker); see `f04-host-path.md`.
- **TTS env:** `.venv` (uv, Python 3.12.16), `torch==2.9.1+cu128`, CUDA runtime
  12.8, ffmpeg 9.0.2. Full inventory:
  `evidence/f01-environment.json`; versions: `evidence/f01-pip-freeze.txt`.
- **Book env:** `.venv-docs` (pinned Sphinx 7.4.7 stack, setuptools 80.9.0,
  Pillow 9.5.0). Keep the two environments separate.
- **VITS standby env (measured import only):** `.venv-vits`, Python 3.12.16,
  `torch==2.9.1+cu128`, `torchaudio==2.9.1+cu128`, `coqui-tts==0.28.0`,
  `torchcodec==0.9.1`; CUDA available. `torchcodec` needs shared FFmpeg 4–8
  DLLs: the BtbN FFmpeg 8.1 shared build is extracted under ignored
  `data/artifacts/ffmpeg-shared/` and its `bin/` must be on PATH at runtime.
  Full pins: `evidence/f04-vits-pip-freeze.txt`. No model was instantiated or
  trained; the lane is standby.
- **Disk:** 556.13 GB free at recording (workspace reservation 100–200 GB met).
- **Backups:** an independent owner-controlled artifact backup destination is
  **not yet established** (the remaining F-01 item). Do not claim preservation
  without it; keep checksums and owner-controlled copies outside the repo.

## Data, splits, checkpoints and runs

- **Dataset:** none accepted. `data/manifests/` contains policy only; no frozen
  manifest or dataset config exists. `learning-curve.csv` has only its header;
  `stage-decisions.jsonl` is empty. `corpus freeze`/`verify-dataset` and
  `train record` are implemented and tested but have never run on real data.
- **Repository syllable audit:** 5,187 isolated MP3s audited without
  modification; decoded 0.6416 hours; 0 decode errors. Manifest SHA-256
  `d87289cbc97aef73594e5fcc8939e87f4a20afec988c57a7458ae09d3e5177ce`
  (manifest itself is under ignored `data/audit/`; reports are committed under
  `audit/repo-syllables/`). Speaker identity and dialect are unverified and this
  is **not** connected-speech training data.
- **Downloaded lane artifacts (ignored by Git; owner backup required):**
  - `data/artifacts/magpie-v2607/magpie_tts_multilingual_357m.nemo`,
    1,470,208,000 B, SHA-256 `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953`,
    repo revision `446bff55e1fe3d3aa29c771574e7d2b0fa322990`.
  - `data/artifacts/nano-codec/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo`,
    425,021,440 B, SHA-256 `28c2518de3e3d5a2c7d9bca40a7ebc0644eb76c60b890970365325bdd8e9f099`,
    repo revision `fc00890b604aa2de298d2641ffc6c5f6caf8c4d7`.
  - `configs/tts/f04-magpie.yaml` points at these paths/hashes;
    `rpa-tts model check --config configs/tts/f04-magpie.yaml --lane magpie`
    reports five `passed` checks on this host. A fresh worker must re-download and
    re-verify (commands in `f04-artifact-inspection.md`).
- **Last completed run:** no model run. The only executed GPU work is the F-01
  kernel proof (8 AdamW steps, 15.68 s, peak 18.88 MB allocated).
- **Resumable checkpoint:** none. Nothing is resumable because no training run
  started. Do not present a resumable command for a run that never existed.
- **F-04 contract findings:** the `.nemo` contains five same-named
  `model_config.yaml` members (last wins) and a 3359×768 `text_embedding`;
  text conditioning uses the external `google/byt5-small` byte tokenizer
  (inspected revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`); config targets
  NeMo `2.8.0rc0`; Hmong is not a supported language; the codec is 22,050 Hz
  with 8 codebooks × 2,016 codes. Details in `f04-artifact-inspection.md`.

## Active processes and GPU budget

- **No active process or background job** from this session remains. Earlier
  audit/build jobs finished; their logs were transient.
- **F-04 allowance:** 8 active GPU hours and two engineering days, unspent.
  Record any future GPU job's id/host/command/log path/elapsed time here.

## Completed evidence (see `evidence-index.json`)

| Gate | Evidence |
| --- | --- |
| F-01 | Environment/stack (`ev_…0001`), GPU kernel proof (`…0002`), book regression (`…0003`), tests/pins (`…0004`); playback reference check (`…0018`); built-HTML reference check (`…0019`); backup item open (`…0005`) |
| F-02 | Draft `rpa-norm-1` contract and calibration protocol (`…0006`); blinded listening instrument (`…000c`); 272-candidate draft script pool (`…000e`); native review worksheet (`…0015`); validated review import (`…0017`); two-adjudicator error review and paragraph review (`…000f`); **blocked on no-new-recording constraint** (`…001a`) |
| F-03 | Source/rights ledgers + syllable audit (`…0007`); independent CMU/lead re-verification (`…000d`); corrected mww/MWWHDVN2DA identity, FCBH terms and minutes-by-source investigation (`…001b`); downloadable-pair vs index/training-terms distinction (`…0020`); first-party training permission + verified live source (`…0021`); full authorized NT acquired (`…0024`) |
| F-04 | Lane metadata and readiness check (`…0008`); pinned artifacts and packaged config (`…000a`); host-path constraint (`…000b`); VITS standby text/environment (`…0010`); real runner and defect chain (`…001c`); CPU load/inference/bit-exact `.nemo` round trip (`…001e`); private 5090 load/inference (`…001f`); real MAT 1:1 optimizer/resume steps and pilot finetune (`…0022`); 100/300/500-step private finetunes (`…0025`) |
| F-05 | Data lifecycle tooling: freeze/verify (`…0011`), byte-level audio verification (`…0013`); private 314-pair MAT 1–10 pilot (`…0023`); full 7,935-pair / 22.523 h authorized NT corpus (`…0026`); all-WAV verification and syllable coverage (`…0030`); chapter-grouped private splits 5,761/714/1,460 (`…0035`); no accepted/frozen dataset |
| F-06 | Learning-curve recording (`…0012`); private 200-verse validation 11.7113 → 10.9937 after 1,000 steps (`…0027`); 3,000-step resume 10.9656 (`…0028`); book-disjoint Mark 11.9279 → 11.1840 (`…0029`); cursor bug: 4,000-step 11.1643 repeated first 2,000 rows (`…0031`); corrected distinct-row resume 11.1562 (`…0032`); blinded A/B (`…0033`/`…0034`); clean chapter-split dev 11.9123 → 11.0115 (`…0036`); no stage decision, judged row or native comparison |
| F-07 | Blinded instrument (`…000c`, `…000f`); qualification composition enforced (`…0014`); no candidate, stimuli or panel |
| F-08 | Package contract and inspection (`…0016`); no accepted checkpoint, package, benchmark or rights |
| F-09 | Not required/deferred for local/static delivery (`…0009`) |

Verification commands and results: `evidence/f01-verification-log.md`
(the F-01 checkpoint recorded 179 tests then; the current suite is 377 passed,
1 skipped, ruff clean). Book build: 0 errors / 10 warnings; playback reference
check: 281 links, 3,688 playable tokens, exact 5,187-file inventory match, no
zero-playable links (`book-audio-references.md`); built HTML: 12 pages, 5,152
audio references, 737 page links, 0 missing local references, sitemap present
(`book-html-references.md`). F-04 diagnosis on this host: `platform_unsupported`.
Private 5090 work then completed a real MAT 1:1 optimizer step (loss 10.5202
finite, 140/144 sampled tensors changed), a resumed second step (loss 10.2875
finite) and 100/300/500-step finetunes on the authorized corpus, all with finite
losses and resumable checkpoints. The full corpus is **7,935 verse pairs /
22.523 h** across 27 NT books (multi-voice drama; 22 verses lack usable
timestamps). Audio/weights are private and unlistened. Permission/source record:
<https://jeepower.atlassian.net/wiki/spaces/RPA/pages/2467102728>.
The book's `source/`, original audio and explorer are unchanged
(`git diff --stat origin/main -- source Makefile` is empty).

## Remaining blockers

1. **Native participants (F-02, blocked):** Jee will not record, recruit a
   speaker, or fill the worksheet. Three fluent listeners, a native language lead
   and adjudicators are not scheduled; no approvals, attested contrast cells or
   approved expansions exist. The instrument and blank 272-row worksheet remain
   available for an independent reviewer, but **no row is approved, no pack has
   been run and no stimuli exist**. See `capacity-constraint.md`.
2. **Rights/people (F-03):** first-party training permission for the existing
   Hmong Daw UBS 2000 narration is recorded on the durable Confluence page
   (link in `mwwhdv-pilot.md`); audio/text/timestamps were verified from the live
   API. Audio and weights stay private. Distribution/weight-publication rights and
   independent native QA remain open; no new recordings/forms.
3. **Model runtime (F-04):** private 5090 step on MAT 1:1: loss 10.5202 finite,
   140/144 sampled tensors changed, 3.706 GB checkpoint with optimizer/scheduler/
   step/RNG; a resumed second step gave finite loss 10.2875. Private 100/300/500-step
   finetunes on the authorized corpus also produced finite losses and resumable
   checkpoints. Still missing: a clean reproducible production environment, native
   listening and release; `train run`/`resume` and release synthesis still refuse
   (exit 3).
4. **Owner backup destination (F-01 completion).**
5. **F-08 packaging:** `package inspect` exists, but packaging and synthesis
   need an F-07-accepted model, distribution rights and a frozen benchmark; no
   package exists and `package build` correctly refuses (exit 3).
6. **GitHub PR/auth:** branch can be pushed over SSH; PR creation needs owner
   authentication.

## Approvals, money and test-set status

- **Native approvals:** none. Identities and consents will live in private
  storage referenced by IDs (`spk-01`, `consent-01`, `L-01`); none exist yet.
- **Money:** $0 spent; no purchase, cloud rental or paid native work consumed.
  No spending authority was used. A corpus/QA/evaluation quote is not prepared
  yet because source options and speaker availability are unresolved.
- **Test material:** the six personal regression sentences remain
  development/regression material only. No fresh final evaluation material has
  been recorded; the locked 300-item suite does not exist.

## Owner/data unblock brief (round 12)

### Lasting constraint

Jee will **not** record new audio, recruit a speaker, or complete the native
worksheet. F-02 is **blocked**; F-04–F-08 cannot pass without a permitted existing
sentence corpus and an independent native reviewer, or an explicitly adopted
scope change. Acceptance criteria are unchanged; no rejected output is relabeled.
See `capacity-constraint.md`.

### Immediate data task: existing White Hmong sentence audio

- Corrected identity: **MWWHDV, ISO `mww`, fileset `MWWHDVN2DA`** (the prior `hmv`
  probe was wrong). CMU index and the **FCBH Hmong Daw New Testament** narration
  are the same source: 260 NT chapters, 7,382 aligned segments, 15.3988 h indexed
  (Pass 1 18:06:46), likely a **multi-voice audio drama**.
- **FCBH/DBP:** the API now requires a key (`language_code=mww` returns 422);
  the license permits personal, non-commercial use through DBP on a single device
  and grants no training/derivative/weights/redistribution rights. No audio was
  downloaded and no chapter was reconstructed.
- **Pangeanic:** advertises 43 audio hours (2,580 min), WAV/MP3, transcribed with
  metadata; no sample, quote, per-speaker White Hmong hours or licence terms.
- **Garellek/Esposito:** eight speakers read three folktales; 11,282 vowel tokens
  analysed; total duration not stated; audio host did not resolve; no training
  permission on file.
- **UMN Hmong Corpus:** site TLS invalid and project page 403; no inventory/export
  or terms verified.
- **Yuhalu:** 6,282 WAVs in one `Kong` folder, 51.38 min of isolated words at
  44.1 kHz; no labels/licence; not sentence speech.
- **Repository:** 5,187 isolated syllables, 38.49 min, reference/technical use
  only; speaker identity/dialect unverified.
- Full table, hashes and remaining access needs:
  `f03-existing-source-investigation-2026-11-26.md`.

### Native work still required

The worksheet (`calibration-scripts.review.csv`, 272 rows; 270 pending, 2 blocked)
and `review-import` remain valid for **any independent native reviewer**, but Jee
will not fill them and no reviewer is currently assigned. Native qualification
(F-07) still needs an independent speaker/listener panel; it cannot be waived.

### 5090 host options (measured)

1. **Enable WSL2 (recommended; no purchase).** CPU virtualization and SLAT are
   enabled and the WSL feature is on, but `VirtualMachinePlatform` is disabled and
   no distro exists. Requires an administrator command, a reboot, a distro and the
   WSL CUDA driver; then the pinned Linux NeMo/Magpie stack can use the 5090.
2. **Linux worker** on the 5090 or another GPU (none available today); move the
   hash-verified artifacts and NeMo environment.
3. **VITS standby on native Windows** (`coqui-tts 0.28.0` + `torch 2.9.1+cu128`)
   imports with CUDA. It remains standby: activation requires a recorded Magpie
   compatibility/codec failure per F-04 and the same permitted data.
4. **Cloud GPU** only with an authorized budget; none exists.

### Authorized pilot data available

**None.** No consent, rights manifest, accepted corpus or dataset version exists.
The FCBH narration is not permitted for training; Pangeanic is a paid-licence
lead; Garellek/UMN access is unresolved; Yuhalu/repository are isolated words.
No new recordings will be made.

### Smallest owner decision that unlocks progress

**Pursue one permitted existing sentence source now:** obtain FCBH DBP access
**plus written training/derivative/weights permission** (and confirm the edition/
fileset and whether the narration is multi-voice drama), **or** approve a
Pangeanic sample/per-speaker-hours/licence request. Host enablement (WSL2) can run
in parallel but is insufficient without data. The research-milestone scope change
using existing paired recordings is still **not adopted** (`decisions.jsonl` D-06).

### Paused

Remaining critical work requires permissions/rights (FCBH/Pangeanic/Garellek/UMN),
an independent native reviewer, or infrastructure (WSL2/Linux worker, backup).
The runner is implemented and ready; no agent-only step advances a gate until data
and permission exist.

## Next smallest executable steps

1. Owner: establish the independent backup destination, then append the F-01
   completion evidence (example):
   `rpa-tts evidence add --gate F-01 --claim "independent backup destination established at <owner path>" --status passed --path docs/tts/recovery/environment.md`
2. Owner: open the PR for `feat/white-hmong-tts-recovery` → `main` and post the
   issue progress update linking `docs/tts/recovery/README.md`.
3. Owner (**data, first**): pursue FCBH DBP access + written training permission,
   or approve the Pangeanic sample/licence request. Do not download/train without
   permission; do not reconstruct from the old downloader.
4. Owner: choose the F-04 host path in `f04-host-path.md` (recommended: enable
   WSL2; CPU/BIOS virtualization flags are already on), or provide a Linux worker.
5. Agent (after host + permitted data): install NeMo at pinned commit
   `0305e4b15ac2924219536da44d315397d92ee769`, run
   `rpa-tts model diagnose` and `rpa-tts train gate`; then build the manifest
   (`prepare_magpie_manifest`), run the bounded `train finetune`, verify the
   `.nemo` weights round trip with exact-byte fingerprints, verify the PTL
   checkpoint + RNG sidecar with `verify_resume_checkpoint`, and render private
   evaluation audio only. Do not touch release synthesis.
6. Before any learning run, write the stage decision with
   `.venv/Scripts/python -m rpa_tts train plan ...` (see `--help`), then append it
   to `stage-decisions.jsonl`.
7. If an independent native reviewer is engaged: fill
   `calibration-scripts.review.csv` for the 272 candidates, validate with
   `corpus review-import --draft ... --review ... --reviewer-id N-01`, and build
   the calibration pack. Jee will not do this.
8. Once a permitted, native-approved pilot exists: audit → split → coverage →
   `corpus verify-audio` → freeze → `corpus verify-dataset`; then `train record`.
9. After a frozen candidate: author/approve the 300 sentence + 20 paragraph
   qualification scripts, stage three conditions, recruit 12 listeners, then run
   ingest → adjudicate → paragraphs → metrics → gate. No suite exists.
10. After F-07 acceptance and distribution rights: assemble the package, run
    `package inspect`, have a second worker reproduce the benchmark, and only then
    consider authorized lesson-audio integration.

## Checkpoint

- **Implementation checkpoints:**
  - `47c18d11a3a4c3514bfa785f02d5c638a5401526` — F-01 workspace, package,
    CLI, tests and runbook.
  - `e52b7f61c09b522719ed6ba7ac306d5b22f91481` — pinned artifact download,
    packaged-config inspection, F-04 lane config and refreshed handoff.
  - `ccd004820ce717e2906b0c4790c2e34740e121c4` — measured F-04 host-path
    constraint (WSL2/Linux requirement) and owner options.
  - `07ff18f19d802a801345e0a881227bfa2ed02447` — blinded listening instrument,
    schema and tests for F-02/F-07.
  - `268a3f55abf5a5d6d37a2065efc388445742679a` — F-03 source re-verification and
    access blockers.
  - `0b397b5f6242e4d812c02dcbe85b1298589a4a23` — draft calibration/development
    script pool extracted from the book.
  - `87e97217aff98817e022906ee7f2c18295737a7a` — CI workflow for the CPU TTS
    suite and the pinned book build.
  - `d0904f09c7ade2abea9819ae375f20647b7d67dc` — two-adjudicator critical-error
    review and paragraph clause/language-switch review.
  - `037352a4afa8ebf741d904c218e7740c70ec97b0` — VITS standby `rpa-chars-1`
    text contract, pinned environment and import proof.
  - `bb8c625010c40338a17f18f4c5d0d2d61dc328ae` — immutable dataset freeze/verify
    and validated learning-curve recording.
  - `90d3840abb482a58c19a2ccb2d25ac8ae711c669` — byte-level manifest audio
    verification.
  - `3832b83006902d43b6cb1bc24164ea1c0d91d1aa` — enforced F-07 qualification
    suite composition.
  - `1225de9b3105720a25a5750cd1d867c70f123c24` — native review worksheet for
    the 272-candidate draft pool.
  - `177644251a9dcb2f983cbb18c16a587201e67631` — F-08 package manifest schema
    and inspection.
  - `95be0b3a85b196cad48b546f7dc217713f20dbad` — validated native review import
    into an approved script set.
  - `698111d636b521d0880917668623af163d1f0fa2` — existing syllable playback
    reference verification.
  - `b0ec3380d1c94fd9984847e3df29ef80692e83b0` — built-HTML reference check and
    paused owner-unblock brief.
  - `6a316c9d2e82100aeac304c25bf8ffc4ba4b4956` — real F-04 runner, capacity
    constraint and corrected `mww` existing-source investigation.
  - `9add5016784f2000b7faf9ff79981b1f12fe9b13` — torch-optional test boundary;
    CI run 45789357474 passed.
  - `5491e4e1b80b00f86610d1014d277d595ff97435` — real CPU Magpie load,
    private generation and bit-exact `.nemo` round trip.
  - `639a6ddea916620958f87df3d83163c056377d56` — remove accidental JUnit output;
    CI run 45791014812 passed.
- **Pushed branch tip at the time of writing:** the child of the checkpoints
  above that records this handoff (verify with `git log -1 --format=%H` and
  `git ls-remote --heads origin feat/white-hmong-tts-recovery`).
- **Push:** `git push -u origin feat/white-hmong-tts-recovery`. The PR to `main`
  still needs owner authentication (see above).
- Any later worker must verify `git status --short --branch`, `git worktree list`
  and `git ls-remote --heads origin main feat/white-hmong-tts-recovery` before
  changing files, and must not force-push or reset.

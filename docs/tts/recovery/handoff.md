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
  book build on pushes and PRs. The owner independently verified GitHub Actions
  run **45785823770 passed on `cfc2872`**. Commits after `cfc2872` (review
  import, playback/built-HTML checks) have not been observed; do not claim green
  beyond the verified run.
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
| F-02 | Draft `rpa-norm-1` contract and calibration protocol (`…0006`); blinded listening instrument (`…000c`); 272-candidate draft script pool (`…000e`); native review worksheet (`…0015`); validated review import (`…0017`); two-adjudicator error review and paragraph review (`…000f`) |
| F-03 | Source/rights ledgers + syllable audit (`…0007`); independent CMU/lead re-verification (`…000d`) |
| F-04 | Lane metadata and readiness check (`…0008`); pinned artifacts and packaged config (`…000a`); host-path constraint (`…000b`); VITS standby text/environment (`…0010`) |
| F-05 | Data lifecycle tooling: freeze/verify (`…0011`), byte-level audio verification (`…0013`); no dataset version exists |
| F-06 | Learning-curve recording (`…0012`); no learning row or stage decision exists |
| F-07 | Blinded instrument (`…000c`, `…000f`); qualification composition enforced (`…0014`); no candidate, stimuli or panel |
| F-08 | Package contract and inspection (`…0016`); no accepted checkpoint, package, benchmark or rights |
| F-09 | Not required/deferred for local/static delivery (`…0009`) |

Verification commands and results: `evidence/f01-verification-log.md`
(the F-01 checkpoint recorded 179 tests then; the current suite is 345 passed,
1 skipped, ruff clean). Book build: 0 errors / 10 warnings; playback reference
check: 281 links, 3,688 playable tokens, exact 5,187-file inventory match, no
zero-playable links (`book-audio-references.md`); built HTML: 12 pages, 5,152
audio references, 737 page links, 0 missing local references, sitemap present
(`book-html-references.md`).
The book's `source/`, original audio and explorer are unchanged
(`git diff --stat origin/main -- source Makefile` is empty).

## Remaining blockers

1. **Native participants (F-02):** three fluent listeners, native language lead
   and adjudicators are not scheduled; no approvals, no attested contrast cells,
   no approved expansions. The blinded instrument is ready
   (`listening-instrument.md`), a 272-candidate draft pool exists, and a
   blank-decision review worksheet awaits the native lead
   (`calibration-scripts.review.csv`); **no row is approved, no pack has been
   run**, no stimuli exist and nothing has been recorded. The public book
   sentences are calibration/development material, not the locked test.
2. **Rights/people (F-03):** no source has cleared training/weight-distribution
   rights. CMU is blocked: the index hash/rows were re-verified, the old
   downloader exits after its first page, it prunes with an F0 extractor before
   applying offsets, and the recorded `listen.bible.is` host no longer resolves;
   Bible Brain's API requires credentials and its terms do not grant training.
   Garellek/oral-history, UMN and Pangeanic outreach still needs owner
   authorization. See `f03-source-investigation-2026-11-26.md`.
3. **Model runtime/host (F-04):** artifacts are resolved and hash-verified, but
   NeMo Speech is not installed and the host cannot run the pinned Linux GPU
   stack natively (no WSL2, no Docker). The owner must enable WSL2 or provide a
   Linux worker before the Magpie GPU gate. The VITS standby environment *does*
   import with CUDA on native Windows after the recorded torchcodec/FFmpeg
   fixes, but activation still requires a recorded Magpie failure and accepted
   F-02 data. `train run` and `resume` correctly refuse (exit 3).
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

## Owner unblock brief (round 11, paused here)

### Exact native review return

- Worksheet: `docs/tts/recovery/calibration-scripts.review.csv` (272 rows: 270
  `pending_native_approval`, 2 `blocked_contract_review`).
- For **every** row the reviewer sets `decision` to `accept`, `edit` or `reject`.
  For `edit`, fill `edited_text` with approved RPA (must pass strict `rpa-norm-1`);
  use `notes` for rationale. Blocked rows are `cal-006` (`Mekas`, `Eeskiv`) and
  `cal-182` (ASCII double quotes); they need a spelling/product decision, not a
  silent accept. Do not put names or contact details in the sheet.
- Return: the filled CSV and a stable non-identifying reviewer id (e.g. `N-01`).
  Then validate and produce the approved set:

  ```text
  rpa-tts corpus review-import \
    --draft docs/tts/recovery/calibration-scripts.draft.jsonl \
    --review <filled-review.csv> \
    --output-jsonl <approved.jsonl> --output-report <approved.md> \
    --reviewer-id N-01
  ```

- The native lead then approves/selects the 100–150 calibration scripts and
  records 10–15 minutes at 48 kHz/24-bit mono from the consented selected
  speaker (quiet space, stable setup, logged changes), with private consent/rights
  IDs. Attested onset/vowel-by-tone cells and approved number expansions remain
  separate native deliverables in `language-contract.yaml`.

### 5090 host options (measured)

1. **Enable WSL2 (recommended; no purchase).** CPU virtualization and SLAT are
   enabled and the WSL feature is on, but `VirtualMachinePlatform` is disabled and
   no distro exists. Requires an administrator command (`wsl --install
   --no-distribution` or enabling the feature), a reboot, a distro, and the WSL
   CUDA driver; then the pinned Linux NeMo/Magpie stack can use the 5090.
2. **Linux worker** on the 5090 or another GPU (none available today); move the
   hash-verified artifacts and NeMo environment.
3. **VITS standby on native Windows** (`coqui-tts 0.28.0` + `torch 2.9.1+cu128`)
   imports successfully with CUDA. It is still standby: activation requires a
   recorded Magpie compatibility/codec failure per F-04 and the same F-02 data.
4. **Cloud GPU** only as a contingency with an authorized budget; no account or
   spend exists.

### Authorized pilot data available

**None.** The repository's 5,187 isolated syllable MP3s (0.6416 decoded hours)
are reference/lexicon material with unverified speaker identity and dialect, not
connected-speech training data. The CMU archive is metadata only with no rights
clearance; the other sources are unresolved/unavailable. No consent, rights
manifest, accepted corpus or dataset version exists.

### Smallest owner decision that unlocks progress

**Assign the native White Hmong lead and authorize the calibration script review
plus the 10–15 minute recording session.** That is the earliest real dependency:
F-04's gate needs F-02 calibration speech, and F-05–F-08 depend on accepted data;
no agent-only step can substitute for it. Host enablement (WSL2) is the smallest
*infrastructure* action and can run in parallel, but it is not sufficient alone.
Rights/outreach authorization is also required before any external corpus use.

### Paused

Remaining critical work requires people (native reviewer, listeners, speaker),
permission (rights/outreach), or infrastructure (WSL2/Linux worker, independent
backup). No further agent-only change advances a gate; work is paused here per
the steering request, with rights and native acceptance gates intact.

## Next smallest executable steps

1. Owner: establish the independent backup destination, then append the F-01
   completion evidence (example):
   `rpa-tts evidence add --gate F-01 --claim "independent backup destination established at <owner path>" --status passed --path docs/tts/recovery/environment.md`
2. Owner: open the PR for `feat/white-hmong-tts-recovery` → `main` and post the
   issue progress update linking `docs/tts/recovery/README.md`.
3. Owner: choose the F-04 host path in `f04-host-path.md` (recommended: enable
   WSL2; the CPU/BIOS virtualization flags are already on), or provide a Linux
   worker. This blocks the Magpie GPU gate, not F-02/F-03 preparation.
4. Agent (after the host path exists): install NeMo Speech at pinned commit
   `0305e4b15ac2924219536da44d315397d92ee769` with CUDA 12.8, restore the
   hash-verified Magpie `.nemo`, record which of the five packaged configs is
   effective, the tokenizer offsets, ByT5 byte-token IDs for every RPA character
   and the `check_text_embedding_matches_tokenizer` result. Re-run
   `rpa-tts model check --config configs/tts/f04-magpie.yaml --lane magpie` to
   confirm hashes before loading.
5. Before any learning run, write the stage decision with
   `.venv/Scripts/python -m rpa_tts train plan ...` (see `--help`), then append it
   to `stage-decisions.jsonl`.
6. Native lead: fill `calibration-scripts.review.csv` (accept/edit/reject,
   edited text, notes) for the 272 candidates — 270 pending, 2 blocked on
   `Mekas`/`Eeskiv`/ASCII quotes — then validate it with
   `corpus review-import --draft ... --review ... --reviewer-id N-01`, which
   writes the approved script set and report. Record the approved 100–150
   calibration scripts. After stimuli exist, build the pack
   with `.venv/Scripts/python -m rpa_tts evaluate pack --definition <definition.json> --output <pack-dir> --stage-audio <root>`,
   distribute phase A then phase B, ingest with
   `.venv/Scripts/python -m rpa_tts evaluate ingest --pack <pack-dir> --responses <responses-dir> --output scores.json`,
   then merge two-adjudicator critical-error decisions with `evaluate adjudicate`
   and human paragraph findings with `evaluate paragraphs`, then run
   `evaluate metrics`/`gate`. Do not reuse exposed scripts for the locked test.
7. Owner: authorize outreach/access requests for a permitted CMU/Bible Brain
   current source and matching text, the Garellek/oral-history recordings, UMN
   and Pangeanic. No purchase, account or audio retrieval is authorized yet; CMU
   must not be re-attempted with the broken downloader or blind indices.
8. Once a native-approved pilot exists: audit → split → coverage →
   `corpus verify-audio --manifest <split.jsonl> --audio-root <root>` → freeze
   (`corpus freeze --version <v> --contract ...`) and then
   `corpus verify-dataset`. After each learning run, append a judged row with
   `train record`; do not edit the curve by hand.
9. After a frozen candidate: author/approve the 300 sentence + 20 paragraph
   qualification scripts and contrast pairs, generate and stage the three
   conditions, build the pack (composition is enforced), recruit 12 listeners,
   then run ingest → adjudicate → paragraphs → metrics → gate. No suite exists.
10. After F-07 acceptance and distribution rights: assemble the package
    (`model-manifest.json`, checkpoint, config, benchmark hashes), run
    `package inspect`, have a second worker reproduce the benchmark, and only then
    consider authorized lesson-audio integration. No package exists.

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
- **Pushed branch tip at the time of writing:** the child of the checkpoints
  above that records this handoff (verify with `git log -1 --format=%H` and
  `git ls-remote --heads origin feat/white-hmong-tts-recovery`).
- **Push:** `git push -u origin feat/white-hmong-tts-recovery`. The PR to `main`
  still needs owner authentication (see above).
- Any later worker must verify `git status --short --branch`, `git worktree list`
  and `git ls-remote --heads origin main feat/white-hmong-tts-recovery` before
  changing files, and must not force-push or reset.

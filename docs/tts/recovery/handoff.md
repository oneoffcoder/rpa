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
- **Disk:** 556.13 GB free at recording (workspace reservation 100–200 GB met).
- **Backups:** an independent owner-controlled artifact backup destination is
  **not yet established** (the remaining F-01 item). Do not claim preservation
  without it; keep checksums and owner-controlled copies outside the repo.

## Data, splits, checkpoints and runs

- **Dataset:** none accepted. `data/manifests/` contains policy only; no frozen
  manifest exists. No split, config dataset version or checkpoint exists.
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
| F-01 | Environment/stack (`ev_…0001`), GPU kernel proof (`…0002`), book regression (`…0003`), tests/pins (`…0004`); backup item open (`…0005`) |
| F-02 | Draft `rpa-norm-1` contract and calibration protocol (`…0006`) |
| F-03 | Source/rights ledgers + syllable audit (`…0007`) |
| F-04 | Lane metadata and readiness check (`…0008`); pinned artifacts downloaded, hash-verified and packaged config inspected (`…000a`) |
| F-09 | Not required/deferred for local/static delivery (`…0009`) |

Verification commands and results: `evidence/f01-verification-log.md`
(179 tests passed, 1 skipped; ruff clean; book build 0 errors / 10 warnings).
The book's `source/`, original audio and explorer are unchanged
(`git diff --stat origin/main -- source Makefile` is empty).

## Remaining blockers

1. **Native participants (F-02):** three fluent listeners, native language lead
   and adjudicators are not scheduled; no approvals, no attested contrast cells,
   no approved expansions.
2. **Rights/people (F-03):** no source has cleared training/weight-distribution
   rights; CMU is blocked on source/text/rights match; outreach requires owner
   authorization and money where quoted.
3. **Model runtime/host (F-04):** artifacts are resolved and hash-verified, but
   NeMo Speech is not installed and the host cannot run the pinned Linux GPU
   stack natively (no WSL2, no Docker). The owner must enable WSL2 or provide a
   Linux worker before the GPU gate. `train run` and `resume` correctly refuse
   (exit 3) until the gate passes and a tested wrapper exists.
4. **Owner backup destination (F-01 completion).**
5. **GitHub PR/auth:** branch can be pushed over SSH; PR creation needs owner
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

## Checkpoint

- **Implementation checkpoints:**
  - `47c18d11a3a4c3514bfa785f02d5c638a5401526` — F-01 workspace, package,
    CLI, tests and runbook.
  - `e52b7f61c09b522719ed6ba7ac306d5b22f91481` — pinned artifact download,
    packaged-config inspection, F-04 lane config and refreshed handoff.
- **Pushed branch tip at the time of writing:** the child of the checkpoints
  above that records this handoff (verify with `git log -1 --format=%H` and
  `git ls-remote --heads origin feat/white-hmong-tts-recovery`).
- **Push:** `git push -u origin feat/white-hmong-tts-recovery`. The PR to `main`
  still needs owner authentication (see above).
- Any later worker must verify `git status --short --branch`, `git worktree list`
  and `git ls-remote --heads origin main feat/white-hmong-tts-recovery` before
  changing files, and must not force-push or reset.

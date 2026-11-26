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

- **Host:** `Desktop-W9QAA` (Windows 11, AMD64, 127.9 GB RAM), RTX 5090 with
  32,607 MiB and compute capability 12.0, driver 621.35.
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
- **Last completed run:** no model run. The only executed GPU work is the F-01
  kernel proof (8 AdamW steps, 15.68 s, peak 18.88 MB allocated).
- **Resumable checkpoint:** none. Nothing is resumable because no training run
  started. Do not present a resumable command for a run that never existed.

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
| F-04 | Lane metadata and readiness check only (`…0008`) |
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
3. **Model artifacts (F-04):** Magpie artifacts unresolved (paths/hashes null),
   NeMo Speech not installed, VITS implementation not pinned. `train run` and
   `resume` correctly refuse (exit 3).
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
3. Agent (no native dependency): prepare the F-04 Magpie inspection — resolve
   the immutable model/codec artifacts and hashes, install NeMo Speech in the
   TTS environment (or a separate lane environment), then run:
   `.venv/Scripts/python -m rpa_tts model check --config configs/tts/default.yaml`
   (current expected exit 1 with artifact checks `not_started`).
4. Before any learning run, write the stage decision with
   `.venv/Scripts/python -m rpa_tts train plan ...` (see `--help`), then append it
   to `stage-decisions.jsonl`.

## Checkpoint

- **Implementation commit:** `47c18d11a3a4c3514bfa785f02d5c638a5401526` (the
  follow-up commit that records this handoff is its child; run
  `git log -1 --format=%H` on the branch for the current tip).
- **Push:** `git push -u origin feat/white-hmong-tts-recovery`; verify with
  `git ls-remote --heads origin feat/white-hmong-tts-recovery`. The PR to `main`
  still needs owner authentication (see above).
- Any later worker must verify `git status --short --branch`, `git worktree list`
  and `git ls-remote --heads origin main feat/white-hmong-tts-recovery` before
  changing files, and must not force-push or reset.

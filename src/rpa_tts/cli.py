"""Stable ``rpa-tts`` command line interface.

Commands that depend on an unpassed recovery gate are present and discoverable
but refuse to run (exit status 3) with the next concrete action. Commands that
only inspect, normalize, audit or score data run today.

Exit status: 0 success, 1 not-ok/validation-finding, 2 bad input, 3 gate blocked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config
from .data.audit import audit_corpus
from .data.book_audio import write_reference_report
from .data.book_html import write_html_reference_report
from .data.book_scripts import draft_scripts, import_review, write_review_sheet
from .data.contract import LanguageContract
from .data.coverage import coverage_from_contract, render_coverage_markdown
from .data.manifest import (
    load_manifest,
    manifest_summary,
    require_valid_rows,
    validate_rows,
    write_manifest,
)
from .data.mwwhdv import audit_audio_uniqueness, build_chapter_manifest
from .data.mwwhdv import write_manifest as write_verse_manifest
from .data.split import assign_splits
from .data.verify import verify_audio
from .data.versioning import freeze_dataset, verify_dataset_config
from .environment import collect_environment, gpu_kernel_proof, write_json_report
from .errors import GateNotPassedError, InputError, RpaTtsError
from .evaluation.ab_player import write_ab_player
from .evaluation.gate import evaluate_release
from .evaluation.metrics import contrast_summary, naturalness_summary, score_transcriptions
from .evaluation.pack import (
    build_pack,
    load_definition,
    load_responses,
    qualification_evidence,
    stage_audio,
)
from .evaluation.review import apply_adjudications, apply_paragraph_reviews
from .evaluation.uncertainty import mean_with_interval
from .evidence import append_entry, make_entry
from .evidence import gate_status as evidence_gate_status
from .hashing import sha256_file
from .inference.synthesize import inspect_package, synthesize
from .inference.synthesize import package as build_package
from .paths import evidence_index_path, find_repo_root, recovery_dir, workspace_root
from .schemas import STAGE_DECISION
from .text.normalize import normalize_rpa
from .training import backend
from .training.gates import gate_statuses
from .training.learning_curve import append_row, row_from_run_manifest
from .training.magpie_runner import (
    build_finetune_command,
    diagnose_environment,
    render_private_audio,
    run_upstream,
)
from .training.run_manifest import (
    build_run_manifest,
    load_run_manifest,
    new_run_id,
    write_run_manifest,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="rpa-tts",
        description="Owned White Hmong (Hmong Daw) TTS toolkit; see docs/tts/recovery/.",
    )
    result.add_argument("--version", action="version", version=f"rpa-tts {__version__}")
    commands = result.add_subparsers(dest="command", required=True)

    # environment check -----------------------------------------------------
    environment = commands.add_parser("environment", help="host inventory and GPU kernel proof")
    environment_commands = environment.add_subparsers(dest="environment_command", required=True)
    check = environment_commands.add_parser("check", help="collect the host/environment inventory")
    check.add_argument("--repo-root", type=Path, default=None)
    check.add_argument("--config", type=Path, default=None)
    check.add_argument("--output", type=Path, help="write the JSON inventory here")
    check.add_argument(
        "--gpu-proof",
        action="store_true",
        help="run a real CUDA forward/backward proof (requires the installed GPU stack)",
    )
    check.add_argument("--proof-output", type=Path, help="write the GPU proof JSON here")

    # text ------------------------------------------------------------------
    text = commands.add_parser("text", help="RPA normalization and validation")
    text_commands = text.add_subparsers(dest="text_command", required=True)
    normalize = text_commands.add_parser("normalize", help="normalize one RPA string")
    normalize.add_argument("--text", required=False)
    normalize.add_argument("--file", type=Path, required=False)
    validate = text_commands.add_parser("validate", help="validate an utterance manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument(
        "--purpose", choices=["audit", "training", "evaluation"], default="training"
    )

    # corpus ----------------------------------------------------------------
    corpus = commands.add_parser("corpus", help="audit, split and measure a corpus")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    audit = corpus_commands.add_parser("audit", help="decode/hash/flag every candidate recording")
    audit.add_argument("--audio-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--labels", type=Path, default=None)
    audit.add_argument("--workers", type=int, default=4)
    audit.add_argument("--repo-root", type=Path, default=None)
    split = corpus_commands.add_parser("split", help="leakage-safe grouped train/dev/test split")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--report", type=Path, default=None)
    split.add_argument("--seed", type=int, default=3407)
    split.add_argument("--dev-fraction", type=float, default=0.1)
    split.add_argument("--test-fraction", type=float, default=0.2)
    coverage = corpus_commands.add_parser("coverage", help="coverage ledger for declared cells")
    coverage.add_argument("--manifest", type=Path, required=True)
    coverage.add_argument("--contract", type=Path, required=True)
    coverage.add_argument("--output-md", type=Path, required=True)
    coverage.add_argument("--output-json", type=Path, default=None)
    coverage.add_argument("--min-examples", type=int, default=None)
    scripts = corpus_commands.add_parser(
        "scripts", help="draft calibration/development scripts from the book's examples"
    )
    scripts.add_argument("--rst", type=Path, default=Path("source/speaking.rst"))
    scripts.add_argument("--output-jsonl", type=Path, required=True)
    scripts.add_argument("--output-report", type=Path, required=True)
    scripts.add_argument("--min-words", type=int, default=3)
    review_sheet = corpus_commands.add_parser(
        "review-sheet", help="write a native-review CSV and instructions from a draft script pool"
    )
    review_sheet.add_argument("--draft", type=Path, required=True)
    review_sheet.add_argument("--csv", type=Path, required=True)
    review_sheet.add_argument("--report", type=Path, required=True)
    review_sheet.add_argument("--overwrite", action="store_true")
    review_import = corpus_commands.add_parser(
        "review-import", help="validate a filled review sheet into an approved script set"
    )
    review_import.add_argument("--draft", type=Path, required=True)
    review_import.add_argument("--review", type=Path, required=True)
    review_import.add_argument("--output-jsonl", type=Path, required=True)
    review_import.add_argument("--output-report", type=Path, required=True)
    review_import.add_argument("--reviewer-id", required=True)
    review_import.add_argument("--overwrite", action="store_true")
    book_audio = corpus_commands.add_parser(
        "book-audio", help="check existing generate.html syllable playback references"
    )
    book_audio.add_argument("--page-dir", type=Path, default=Path("source"))
    book_audio.add_argument("--audio-dir", type=Path, default=Path("source/_static/mp3"))
    book_audio.add_argument("--output-json", type=Path, default=None)
    book_audio.add_argument("--output-md", type=Path, default=None)
    book_html = corpus_commands.add_parser(
        "book-html", help="check built HTML local references and generated-syllable links"
    )
    book_html.add_argument("--build-dir", type=Path, default=Path("build/html"))
    book_html.add_argument("--output-json", type=Path, default=None)
    book_html.add_argument("--output-md", type=Path, default=None)
    mwwhdv = corpus_commands.add_parser(
        "mwwhdv-chapter",
        help="cut a local MWWHDV chapter audio/text/timestamps into verse rows",
    )
    mwwhdv.add_argument("--audio", type=Path, required=True)
    mwwhdv.add_argument("--text-json", type=Path, required=True)
    mwwhdv.add_argument("--timestamps-json", type=Path, required=True)
    mwwhdv.add_argument("--book", required=True)
    mwwhdv.add_argument("--chapter", type=int, required=True)
    mwwhdv.add_argument("--chapter-duration", type=float, required=True)
    mwwhdv.add_argument("--output-dir", type=Path, required=True)
    mwwhdv.add_argument("--manifest", type=Path, required=True)
    mwwhdv_audit = corpus_commands.add_parser(
        "mwwhdv-audit", help="flag duplicate WAV hashes in a verse manifest"
    )
    mwwhdv_audit.add_argument("--manifest", type=Path, required=True)
    mwwhdv_audit.add_argument("--output", type=Path, required=True)
    freeze = corpus_commands.add_parser(
        "freeze", help="freeze an approved accepted dataset version with hashes and coverage"
    )
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--contract", type=Path, default=None)
    freeze.add_argument("--manifests-dir", type=Path, default=None)
    freeze.add_argument("--configs-dir", type=Path, default=None)
    freeze.add_argument("--min-examples", type=int, default=5)
    freeze.add_argument("--notes", default="")
    verify_dataset = corpus_commands.add_parser(
        "verify-dataset", help="re-hash a frozen dataset configuration"
    )
    verify_dataset.add_argument("--config", type=Path, required=True)
    verify_audio = corpus_commands.add_parser(
        "verify-audio", help="re-hash and re-decode manifest audio against source files"
    )
    verify_audio.add_argument("--manifest", type=Path, required=True)
    verify_audio.add_argument("--audio-root", type=Path, required=True)
    verify_audio.add_argument("--workers", type=int, default=4)
    verify_audio.add_argument("--tolerance", type=float, default=0.05)
    verify_audio.add_argument("--output", type=Path, default=None)

    # model -----------------------------------------------------------------
    model = commands.add_parser("model", help="model-lane and artifact readiness")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_check = model_commands.add_parser("check", help="inspect lane metadata and hashes")
    model_check.add_argument("--config", type=Path, default=None)
    model_check.add_argument("--lane", choices=["magpie", "vits"], default=None)
    model_check.add_argument("--repo-root", type=Path, default=None)
    model_diagnose = model_commands.add_parser(
        "diagnose", help="report the exact F-04 environment defect chain"
    )
    model_diagnose.add_argument("--config", type=Path, default=None)
    model_diagnose.add_argument("--manifest", type=Path, default=None)
    model_diagnose.add_argument("--repo-root", type=Path, default=None)

    # train -----------------------------------------------------------------
    train = commands.add_parser("train", help="gated training entry points")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    plan = train_commands.add_parser(
        "plan", help="write a pre-run plan and stage-decision draft (no GPU work)"
    )
    plan.add_argument("--config", type=Path, default=None)
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--stage", default="F-06", choices=[f"F-0{index}" for index in range(5, 9)])
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--objective", required=True)
    plan.add_argument("--baseline", required=True)
    plan.add_argument("--max-gpu-hours", type=float, required=True)
    plan.add_argument("--measurement-set", required=True)
    plan.add_argument("--inference-policy", required=True)
    plan.add_argument("--raters", required=True)
    plan.add_argument("--next-increment", required=True)
    plan.add_argument("--record-decision", type=Path, default=None)
    record = train_commands.add_parser(
        "record", help="append one judged learning run to learning-curve.csv"
    )
    record.add_argument("--run-manifest", type=Path, required=True)
    record.add_argument("--dataset-version", required=True)
    record.add_argument("--dev-syllable-error", type=float, required=True)
    record.add_argument("--dev-naturalness", type=float, required=True)
    record.add_argument(
        "--decision",
        required=True,
        choices=["baseline", "promote", "hold", "reject", "inconclusive"],
    )
    record.add_argument("--evidence-id", required=True)
    record.add_argument("--curve", type=Path, default=None)
    run = train_commands.add_parser("run", help="start a training run (requires F-04)")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--stage", default="F-06", choices=[f"F-0{index}" for index in range(5, 9)])
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--lane", choices=["magpie", "vits"], default=None)
    gate = train_commands.add_parser(
        "gate", help="attempt the F-04 gate or record its first blocking defect"
    )
    gate.add_argument("--config", type=Path, default=None)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--manifest", type=Path, default=None)
    gate.add_argument("--lane", choices=["magpie", "vits"], default="magpie")
    gate.add_argument("--dry-run", action="store_true")
    finetune = train_commands.add_parser(
        "finetune", help="run the pinned Magpie finetuning command in a qualified environment"
    )
    finetune.add_argument("--config", type=Path, default=None)
    finetune.add_argument("--init-checkpoint", type=Path, required=True)
    finetune.add_argument("--exp-dir", type=Path, required=True)
    finetune.add_argument("--train-manifest", type=Path, required=True)
    finetune.add_argument("--val-manifest", type=Path, required=True)
    finetune.add_argument("--audio-dir", type=Path, required=True)
    finetune.add_argument("--feature-dir", type=Path, default=None)
    finetune.add_argument("--tokenizer-name", default="hmong_chartokenizer")
    finetune.add_argument("--devices", type=int, default=1)
    finetune.add_argument("--batch-size", type=int, default=1)
    finetune.add_argument("--max-epochs", type=int, default=1)
    finetune.add_argument("--max-steps", type=int, default=1)
    finetune.add_argument("--learning-rate", type=float, default=1e-5)
    finetune.add_argument("--dry-run", action="store_true")
    render_private = train_commands.add_parser(
        "render-private", help="generate private evaluation audio (never release audio)"
    )
    render_private.add_argument("--config", type=Path, default=None)
    render_private.add_argument("--artifact", type=Path, required=True)
    render_private.add_argument("--checkpoint", type=Path, default=None)
    render_private.add_argument("--text", required=True)
    render_private.add_argument("--output", type=Path, required=True)
    render_private.add_argument("--metadata", type=Path, required=True)
    render_private.add_argument("--device", default="cuda:0")
    render_private.add_argument("--language", default="en")
    render_private.add_argument("--speaker-index", type=int, default=0)
    render_private.add_argument("--seed", type=int, default=3407)
    render_private.add_argument("--sample-rate", type=int, default=None)
    render_private.add_argument(
        "--allow-unapproved-text",
        action="store_true",
        help="private diagnostics only: render raw text with normalization issues recorded",
    )
    resume = train_commands.add_parser("resume", help="resume a run (requires F-04)")
    resume.add_argument("--config", type=Path, default=None)
    resume.add_argument("--run-dir", type=Path, required=True)

    # synthesize ------------------------------------------------------------
    synth = commands.add_parser("synthesize", help="local inference (requires F-07)")
    synth.add_argument("--checkpoint", type=Path, required=True)
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--config", type=Path, default=None)

    # evaluate --------------------------------------------------------------
    evaluate = commands.add_parser("evaluate", help="score human listening data and apply gates")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    metrics = evaluate_commands.add_parser(
        "metrics", help="compute syllable/naturalness/contrast scores"
    )
    metrics.add_argument("--scores", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--config", type=Path, default=None)
    gate = evaluate_commands.add_parser("gate", help="apply the F-07 release gate")
    gate.add_argument("--scores", type=Path, required=True)
    gate.add_argument("--pack", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--config", type=Path, default=None)
    pack_build = evaluate_commands.add_parser(
        "pack", help="build a blinded, counterbalanced listening pack"
    )
    pack_build.add_argument("--definition", type=Path, required=True)
    pack_build.add_argument("--output", type=Path, required=True)
    pack_build.add_argument(
        "--stage-audio",
        type=Path,
        default=None,
        help="source root for the definition's audio paths; stages blind audio into the pack",
    )
    pack_build.add_argument("--overwrite", action="store_true")
    ingest = evaluate_commands.add_parser(
        "ingest", help="validate completed response sheets into a scores payload"
    )
    ingest.add_argument("--pack", type=Path, required=True)
    ingest.add_argument("--responses", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    adjudicate = evaluate_commands.add_parser(
        "adjudicate", help="merge native adjudications of critical-contrast errors"
    )
    adjudicate.add_argument("--scores", type=Path, required=True)
    adjudicate.add_argument("--adjudications", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    paragraphs = evaluate_commands.add_parser(
        "paragraphs", help="merge human paragraph clause/language-switch review"
    )
    paragraphs.add_argument("--scores", type=Path, required=True)
    paragraphs.add_argument("--reviews", type=Path, required=True)
    paragraphs.add_argument("--pack", type=Path, default=None)
    paragraphs.add_argument("--output", type=Path, required=True)
    ab_player = evaluate_commands.add_parser(
        "ab-player", help="write a local blinded A/B listening page from an answer key"
    )
    ab_player.add_argument("--answer-key", type=Path, required=True)
    ab_player.add_argument("--output", type=Path, required=True)
    ab_player.add_argument("--title", default="Private blinded A/B")

    # evidence --------------------------------------------------------------
    evidence = commands.add_parser("evidence", help="append to the evidence index")
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    add = evidence_commands.add_parser("add", help="append one hash-checked evidence entry")
    add.add_argument("--gate", required=True)
    add.add_argument("--claim", required=True)
    add.add_argument(
        "--status",
        required=True,
        choices=[
            "not_started",
            "in_progress",
            "passed",
            "failed",
            "inconclusive",
            "blocked",
            "not_required",
        ],
    )
    add.add_argument("--path", type=Path, action="append", default=[])
    add.add_argument("--run-id", default=None)
    add.add_argument("--notes", default="")
    add.add_argument("--index", type=Path, default=None)
    add.add_argument("--repo-root", type=Path, default=None)

    # package ---------------------------------------------------------------
    package = commands.add_parser("package", help="build a distributable package (requires F-08)")
    package_commands = package.add_subparsers(dest="package_command", required=True)
    package_build = package_commands.add_parser("build", help="build the accepted package")
    package_build.add_argument("--checkpoint", type=Path, required=True)
    package_build.add_argument("--output", type=Path, required=True)
    package_inspect = package_commands.add_parser(
        "inspect", help="validate a package's manifest, hashes, rights and benchmark verdict"
    )
    package_inspect.add_argument("--checkpoint", type=Path, required=True)

    # run manifest ----------------------------------------------------------
    run_manifest = commands.add_parser("run-manifest", help="inspect run manifests")
    run_manifest_commands = run_manifest.add_subparsers(dest="run_manifest_command", required=True)
    run_validate = run_manifest_commands.add_parser("validate", help="validate a run manifest")
    run_validate.add_argument("--path", type=Path, required=True)
    return result


def _repo_root(args: argparse.Namespace) -> Path:
    configured = getattr(args, "repo_root", None)
    return Path(configured).resolve() if configured else find_repo_root()


def _config(args: argparse.Namespace):
    return load_config(getattr(args, "config", None))


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_environment(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    root = _repo_root(args)
    environment = collect_environment(root, workspace=workspace_root(config.workspace_config))
    payload: dict[str, Any] = {"environment": environment}
    if args.output:
        write_json_report(environment, args.output)
        payload["environment_output"] = {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        }
    exit_code = 0
    if args.gpu_proof:
        proof = gpu_kernel_proof(seed=config.seed)
        payload["gpu_proof"] = proof
        if args.proof_output:
            write_json_report(proof, args.proof_output)
            payload["gpu_proof_output"] = {
                "path": str(args.proof_output),
                "sha256": sha256_file(args.proof_output),
            }
        if proof["status"] != "passed":
            exit_code = 1
    return payload, exit_code


def _cmd_text_normalize(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if (args.text is None) == (args.file is None):
        raise InputError("provide exactly one of --text or --file")
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    normalized = normalize_rpa(text)
    payload = {
        "raw": normalized.raw,
        "surface": normalized.surface,
        "normalized": normalized.normalized,
        "tokens": list(normalized.tokens),
        "syllables": list(normalized.syllables),
        "issues": [issue.__dict__ for issue in normalized.issues],
        "transformations": list(normalized.transformations),
        "version": normalized.version,
        "valid": normalized.is_valid,
    }
    return payload, 0 if normalized.is_valid else 1


def _cmd_text_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = load_manifest(args.manifest)
    problems = validate_rows(rows, purpose=args.purpose)
    errors = [problem for problem in problems if problem.level == "error"]
    payload = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
        "purpose": args.purpose,
        "errors": [problem.as_dict() for problem in errors],
        "warnings": [problem.as_dict() for problem in problems if problem.level == "warning"],
        "summary": manifest_summary(rows),
    }
    return payload, 0 if not errors else 1


def _cmd_corpus_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = audit_corpus(
        args.audio_dir,
        _repo_root(args),
        args.output,
        labels_path=args.labels,
        workers=args.workers,
    )
    return report, 0 if report["ok"] else 1


def _cmd_corpus_split(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = load_manifest(args.manifest)
    # Structural + leakage checks only; native approval is a separate F-05 decision.
    require_valid_rows(rows, purpose="audit", require_native_approval=False)
    assigned, summary = assign_splits(
        rows,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
    )
    manifest_hash = write_manifest(assigned, args.output)
    report_path = args.report or args.output.with_suffix(".report.json")
    report = {
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": sha256_file(args.manifest),
        "output_manifest": str(args.output),
        "output_manifest_sha256": manifest_hash,
        "split": summary,
        "summary": manifest_summary(assigned),
    }
    write_json_report(report, report_path)
    report["report_path"] = str(report_path)
    return report, 0


def _cmd_corpus_coverage(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = load_manifest(args.manifest)
    contract = LanguageContract.load(args.contract)
    report = coverage_from_contract(
        rows, contract, min_examples=args.min_examples if args.min_examples is not None else 5
    )
    report["contract"] = str(args.contract)
    report["contract_status"] = contract.status
    report["dialect"] = contract.dialect_name
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_coverage_markdown(report), encoding="utf-8")
    if args.output_json:
        write_json_report(report, args.output_json)
    exit_code = 0 if report["screening_passed"] else 1
    return report, exit_code


def _cmd_corpus_scripts(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = draft_scripts(
        args.rst,
        args.output_jsonl,
        args.output_report,
        min_words=args.min_words,
    )
    return report, 0


def _cmd_corpus_review_sheet(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return (
        write_review_sheet(
            args.draft,
            args.csv,
            args.report,
            overwrite=args.overwrite,
        ),
        0,
    )


def _cmd_corpus_review_import(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return (
        import_review(
            args.draft,
            args.review,
            args.output_jsonl,
            args.output_report,
            reviewer_id=args.reviewer_id,
            overwrite=args.overwrite,
        ),
        0,
    )


def _cmd_corpus_book_audio(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = write_reference_report(
        args.page_dir,
        args.audio_dir,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    return report, 0 if report["ok"] else 1


def _cmd_corpus_book_html(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = write_html_reference_report(
        args.build_dir,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    return report, 0 if report["ok"] else 1


def _cmd_corpus_mwwhdv_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_audio_uniqueness(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report, 0 if report["ok"] else 1


def _cmd_corpus_mwwhdv_chapter(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    text_rows = json.loads(args.text_json.read_text(encoding="utf-8"))
    timestamp_rows = json.loads(args.timestamps_json.read_text(encoding="utf-8"))
    rows = build_chapter_manifest(
        args.audio,
        text_rows,
        timestamp_rows,
        args.output_dir,
        book=args.book,
        chapter=args.chapter,
        chapter_duration=args.chapter_duration,
    )
    summary = write_verse_manifest(rows, args.manifest)
    return {
        "book": args.book,
        "chapter": args.chapter,
        **summary,
        "private_training_data": True,
    }, 0


def _cmd_corpus_freeze(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = find_repo_root()
    contract = args.contract or recovery_dir(root) / "language-contract.yaml"
    result = freeze_dataset(
        args.manifest,
        args.version,
        contract_path=contract,
        manifests_dir=args.manifests_dir,
        configs_dir=args.configs_dir,
        repo_root=root,
        min_examples=args.min_examples,
        notes=args.notes,
    )
    return result, 0


def _cmd_corpus_verify_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    result = verify_dataset_config(args.config)
    return result, 0 if result["ok"] else 1


def _cmd_corpus_verify_audio(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    rows = load_manifest(args.manifest)
    require_valid_rows(rows, purpose="audit", require_native_approval=False)
    report = verify_audio(
        rows,
        args.audio_root,
        workers=args.workers,
        duration_tolerance_seconds=args.tolerance,
    )
    if args.output:
        write_json_report(report, args.output)
        report["output"] = str(args.output)
    return report, 0 if report["ok"] else 1


def _cmd_train_record(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = load_run_manifest(args.run_manifest)
    row = row_from_run_manifest(
        manifest,
        dataset_version=args.dataset_version,
        dev_syllable_error=args.dev_syllable_error,
        dev_naturalness_mean=args.dev_naturalness,
        decision=args.decision,
        evidence_id=args.evidence_id,
    )
    curve = args.curve or recovery_dir(find_repo_root()) / "learning-curve.csv"
    return append_row(curve, row), 0


def _cmd_model_diagnose(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    report = diagnose_environment(
        config,
        manifest_path=args.manifest,
        repo_root=_repo_root(args),
    )
    return report, 0 if report["ok"] else 1


def _cmd_train_gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    root = find_repo_root()
    lane = args.lane
    diagnosis = diagnose_environment(config, manifest_path=args.manifest, repo_root=root)
    args.output.mkdir(parents=True, exist_ok=True)
    diagnosis_path = args.output / "f04-diagnosis.json"
    write_json_report(diagnosis, diagnosis_path)
    if not diagnosis["ok"]:
        first = diagnosis["defects"][0]
        raise GateNotPassedError(
            "F-04",
            f"environment diagnosis failed: {first['code']} — {first['detail']}",
            first.get("remediation", "resolve the reported defect and rerun train gate"),
        )
    if args.manifest is None:
        raise InputError(
            "F-04 requires F-02 calibration speech; supply --manifest with accepted rows"
        )
    training = config.data["model_lanes"][lane].get("training", {})
    command = training.get("gate_command")
    if not command or not all(isinstance(part, str) for part in command):
        raise InputError(
            f"lane {lane!r} has no configured gate_command; add the tested pinned upstream "
            "command to the lane config and rerun; commands are never invented"
        )
    if args.dry_run:
        return {
            "diagnosis": diagnosis,
            "diagnosis_path": str(diagnosis_path),
            "command": command,
            "dry_run": True,
        }, 0
    result = run_upstream(command, cwd=root)
    write_json_report(result, args.output / "gate-command.json")
    if not result["ok"]:
        raise RpaTtsError(
            f"F-04 gate command failed with exit {result['returncode']}; see "
            f"{args.output / 'gate-command.json'}"
        )
    return {
        "diagnosis": diagnosis,
        "diagnosis_path": str(diagnosis_path),
        "command_result": result,
        "note": (
            "Review outputs and append F-04 evidence manually; success here is not "
            "native acceptance."
        ),
    }, 0


def _cmd_train_finetune(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    root = find_repo_root()
    diagnosis = diagnose_environment(config, repo_root=root)
    if not diagnosis["ok"]:
        first = diagnosis["defects"][0]
        raise GateNotPassedError(
            "F-04",
            f"finetuning cannot start: {first['code']} — {first['detail']}",
            first.get("remediation", "resolve the reported defect and rerun"),
        )
    built = build_finetune_command(
        repo_root=root,
        init_checkpoint=args.init_checkpoint,
        exp_dir=args.exp_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        audio_dir=args.audio_dir,
        feature_dir=args.feature_dir,
        tokenizer_name=args.tokenizer_name,
        devices=args.devices,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    if args.dry_run:
        return {"diagnosis": diagnosis, "finetune": built, "dry_run": True}, 0
    result = run_upstream(built["command"], cwd=root)
    return {"diagnosis": diagnosis, "finetune": built, "command_result": result}, (
        0 if result["ok"] else 1
    )


def _cmd_train_render_private(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    root = find_repo_root()
    diagnosis = diagnose_environment(config, repo_root=root)
    if not diagnosis["ok"]:
        first = diagnosis["defects"][0]
        raise GateNotPassedError(
            "F-04",
            f"private rendering requires a qualified runner: {first['code']} — {first['detail']}",
            first.get("remediation", "resolve the reported defect and rerun"),
        )
    from .training.magpie_runner import load_magpie

    model = load_magpie(args.artifact, device=args.device, checkpoint_path=args.checkpoint)
    report = render_private_audio(
        model,
        args.text,
        args.output,
        args.metadata,
        repo_root=root,
        language=args.language,
        speaker_index=args.speaker_index,
        seed=args.seed,
        sample_rate=args.sample_rate,
        strict_normalization=not args.allow_unapproved_text,
        extra_metadata={
            "artifact": str(args.artifact),
            "artifact_sha256": sha256_file(args.artifact),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        },
    )
    return report, 0


def _cmd_model_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    root = _repo_root(args)
    if args.lane:
        report = {
            "lane": backend.check_lane(config, args.lane, repo_root=root),
            "gates": gate_statuses(root),
        }
        exit_code = 0 if report["lane"]["ready"] else 1
    else:
        report = backend.check_lanes(config, repo_root=root)
        report["gates"] = gate_statuses(root)
        exit_code = 0 if report["ready_lanes"] else 1
    return report, exit_code


def _stage_decision_draft(args: argparse.Namespace) -> dict[str, Any]:
    from .evidence import utc_now

    return {
        "id": "S-01",
        "timestamp": utc_now(),
        "stage": args.stage,
        "dataset_version": str(args.manifest),
        "objective": args.objective,
        "baseline": args.baseline,
        "max_gpu_hours": args.max_gpu_hours,
        "max_optimizer_updates": None,
        "inference_policy": args.inference_policy,
        "native_raters": args.raters,
        "measurement_set": args.measurement_set,
        "next_increment": args.next_increment,
        "decision": "DRAFT: review and append before starting the run",
    }


def _cmd_train_plan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    import jsonschema

    config = _config(args)
    rows = load_manifest(args.manifest)
    require_valid_rows(rows, purpose="training")
    root = find_repo_root()
    draft = _stage_decision_draft(args)
    jsonschema.Draft202012Validator(STAGE_DECISION).validate(draft)
    gates = gate_statuses(root)
    payload = {
        "stage_decision": draft,
        "dataset": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "summary": manifest_summary(rows),
        },
        "lanes": backend.check_lanes(config, repo_root=root),
        "gates": gates,
        "blockers": [
            f"{name}: {status['status']}"
            for name, status in gates.items()
            if name in {"F-04", "F-05"} and status["status"] != "passed"
        ],
        "note": (
            "This plan performs no GPU work. A learning run still requires a passed F-04 gate and "
            "an appended stage decision."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    write_json_report(payload, plan_path)
    payload["plan_path"] = str(plan_path)
    if args.record_decision:
        existing = []
        if args.record_decision.is_file():
            existing = [
                json.loads(line)
                for line in args.record_decision.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        draft = dict(draft)
        draft["id"] = f"S-{len(existing) + 1:02d}"
        args.record_decision.parent.mkdir(parents=True, exist_ok=True)
        with args.record_decision.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(draft, sort_keys=True) + "\n")
        payload["recorded_decision"] = {"path": str(args.record_decision), "id": draft["id"]}
    return payload, 0


def _cmd_train_run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    rows = load_manifest(args.manifest)
    require_valid_rows(rows, purpose="training")
    lane = args.lane or backend.require_training_backend(config)
    backend.training_command(config, lane)  # raises a precise input error if missing
    # Reached only after F-04 passed and a tested wrapper exists.
    environment = collect_environment(find_repo_root())
    run_id = new_run_id(args.stage)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config=config,
        run_id=run_id,
        stage=args.stage,
        command=" ".join(sys.argv) if sys.argv else "rpa-tts train run",
        dataset_manifest=args.manifest,
        rows=rows,
        environment=environment,
        notes="planned run manifest; training wrapper invocation is recorded in the runbook",
    )
    path = write_run_manifest(manifest, args.output / "run-manifest.json")
    return {"run_manifest": str(path), "run_id": run_id, "lane": lane}, 0


def _cmd_train_resume(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = load_run_manifest(args.run_dir / "run-manifest.json")
    backend.require_training_backend(_config(args))
    raise RpaTtsError(
        f"run {manifest['run_id']} is recorded, but no resume wrapper is configured; the original "
        "checkpoint and optimizer state must be declared in the run manifest before resuming"
    )


def _cmd_synthesize(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    result = synthesize(
        args.text,
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        config=config,
        repo_root=find_repo_root(),
    )
    return result, 0


def _scores_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"scores file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise InputError("scores file root must be an object")
    return data


def _compute_metrics(
    scores: dict[str, Any], config, *, with_intervals: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def candidate_rows(key):
        records = scores.get(key) or []
        if any("condition" in row for row in records):
            return [row for row in records if row.get("condition") == "candidate"]
        return records

    transcriptions = candidate_rows("transcriptions")
    if transcriptions:
        result["syllable"] = score_transcriptions(transcriptions)
    if scores.get("naturalness"):
        result["naturalness"] = naturalness_summary(scores["naturalness"], condition="candidate")
        anchors = [row for row in scores["naturalness"] if row.get("condition") == "natural"]
        result["natural"] = naturalness_summary(
            anchors or scores.get("natural") or [], condition="natural"
        )
        if with_intervals:
            candidate_records = [
                row for row in scores["naturalness"] if row.get("condition") == "candidate"
            ]
            if candidate_records:
                result["naturalness"]["interval"] = mean_with_interval(
                    candidate_records,
                    resamples=int(config.data["gates"]["f06"]["bootstrap_resamples"]),
                    seed=config.seed,
                )
    contrasts = candidate_rows("contrasts")
    if contrasts:
        result["contrast"] = contrast_summary(contrasts)
    if "paragraphs" in scores:
        result["paragraphs"] = scores["paragraphs"]
    if any("condition" in row for row in scores.get("transcriptions") or []):
        result["catastrophic_outputs"] = sum(
            isinstance(row["hypothesis"], str) and not row["hypothesis"].strip()
            for row in transcriptions
        )
    else:
        result["catastrophic_outputs"] = int(scores.get("catastrophic_outputs", 0))
    result["panel_acceptance"] = scores.get("panel_acceptance")
    return result


def _cmd_evaluate_metrics(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    scores = _scores_payload(args.scores)
    result = _compute_metrics(scores, config, with_intervals=True)
    write_json_report(result, args.output)
    result["output"] = str(args.output)
    return result, 0


def _cmd_evaluate_gate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    config = _config(args)
    scores = _scores_payload(args.scores)
    evidence = qualification_evidence(args.pack, scores)
    computed = _compute_metrics(scores, config, with_intervals=True)
    syllable = computed.get("syllable")
    if syllable is None:
        raise InputError("F-07 gate requires transcription records")
    if "naturalness" not in computed:
        raise InputError("F-07 gate requires candidate naturalness records")
    naturalness = computed["naturalness"]
    interval = naturalness.get("interval")
    naturalness_block = {
        "count": naturalness["count"],
        "mean": naturalness["mean"],
        "lower": interval["lower"] if interval else None,
    }
    natural = computed.get("natural", {})
    contrast = computed.get("contrast")
    if contrast is None:
        raise InputError("F-07 gate requires contrast records")
    reviews = scores["paragraphs"]["per_item"]
    paragraphs = {
        "items": len(reviews),
        "missing_clauses": sum(row["missing_clauses"] for row in reviews),
        "repeated_clauses": sum(row["repeated_clauses"] for row in reviews),
        "language_switch_items": sum(row["language_switch"] for row in reviews),
    }
    verdict = evaluate_release(
        thresholds=config.gate_thresholds,
        syllable=syllable,
        contrast=contrast,
        naturalness=naturalness_block,
        natural=natural,
        paragraphs=paragraphs,
        catastrophic_outputs=computed["catastrophic_outputs"],
        panel_acceptance=computed["panel_acceptance"],
        required_buckets=list(config.data["gates"]["f07"]["required_buckets"]),
    )
    report = {"verdict": verdict, "computed": computed, "qualification_evidence": evidence}
    write_json_report(report, args.output)
    report["output"] = str(args.output)
    exit_code = 0 if verdict["verdict"] == "accepted" else 1
    return report, exit_code


def _cmd_evaluate_pack(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    definition = load_definition(args.definition)
    built = build_pack(definition, args.output)
    pack_manifest = args.output / "pack-manifest.json"
    payload: dict[str, Any] = {
        "output": str(args.output),
        "summary": built["summary"],
        "pack_manifest_sha256": sha256_file(pack_manifest),
        "audio_staged": False,
    }
    if args.stage_audio:
        staged = stage_audio(args.output, args.stage_audio, overwrite=args.overwrite)
        payload["audio_staged"] = True
        payload["staged_audio_sha256"] = staged["manifest"]["staged_audio_sha256"]
    return payload, 0


def _cmd_evaluate_ingest(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    scores = load_responses(args.pack, args.responses)
    write_json_report(scores, args.output)
    payload = {
        "output": str(args.output),
        "pack_id": scores["pack_id"],
        "transcriptions": len(scores["transcriptions"]),
        "naturalness": len(scores["naturalness"]),
        "contrasts": len(scores["contrasts"]),
        "catastrophic_outputs": scores["catastrophic_outputs"],
        "note": (
            "Run evaluate metrics/gate next; adjudication of critical errors remains a native step."
        ),
    }
    return payload, 0


def _cmd_evaluate_adjudicate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    scores = _scores_payload(args.scores)
    merged = apply_adjudications(scores, args.adjudications)
    write_json_report(merged, args.output)
    summary = merged["adjudications"]
    return {"output": str(args.output), **summary}, 0


def _cmd_evaluate_paragraphs(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    scores = _scores_payload(args.scores)
    merged = apply_paragraph_reviews(scores, args.reviews, pack_dir=args.pack)
    write_json_report(merged, args.output)
    summary = merged["paragraphs"]
    return {"output": str(args.output), **summary}, 0


def _cmd_evaluate_ab_player(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return (
        write_ab_player(args.answer_key, args.output, title=args.title),
        0,
    )


def _cmd_evidence_add(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = _repo_root(args)
    index_path = args.index or evidence_index_path(root)
    entry = make_entry(
        gate=args.gate,
        claim=args.claim,
        status=args.status,
        paths=args.path,
        run_id=args.run_id,
        notes=args.notes,
        repo_root=root,
    )
    index = append_entry(entry, index_path, repo_root=root)
    return {
        "entry": entry,
        "index": str(index_path),
        "entries": len(index["entries"]),
        "gate_status": evidence_gate_status(index, args.gate),
    }, 0


def _cmd_package(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return build_package(args.checkpoint, args.output, repo_root=find_repo_root()), 0


def _cmd_package_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = inspect_package(args.checkpoint)
    return report, 0 if report["ok"] else 1


def _cmd_run_manifest_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest = load_run_manifest(args.path)
    return manifest, 0


_DISPATCH = {
    ("environment", "check"): _cmd_environment,
    ("text", "normalize"): _cmd_text_normalize,
    ("text", "validate"): _cmd_text_validate,
    ("corpus", "audit"): _cmd_corpus_audit,
    ("corpus", "split"): _cmd_corpus_split,
    ("corpus", "coverage"): _cmd_corpus_coverage,
    ("corpus", "scripts"): _cmd_corpus_scripts,
    ("corpus", "review-sheet"): _cmd_corpus_review_sheet,
    ("corpus", "review-import"): _cmd_corpus_review_import,
    ("corpus", "book-audio"): _cmd_corpus_book_audio,
    ("corpus", "book-html"): _cmd_corpus_book_html,
    ("corpus", "mwwhdv-chapter"): _cmd_corpus_mwwhdv_chapter,
    ("corpus", "mwwhdv-audit"): _cmd_corpus_mwwhdv_audit,
    ("corpus", "freeze"): _cmd_corpus_freeze,
    ("corpus", "verify-dataset"): _cmd_corpus_verify_dataset,
    ("corpus", "verify-audio"): _cmd_corpus_verify_audio,
    ("model", "check"): _cmd_model_check,
    ("model", "diagnose"): _cmd_model_diagnose,
    ("train", "plan"): _cmd_train_plan,
    ("train", "gate"): _cmd_train_gate,
    ("train", "finetune"): _cmd_train_finetune,
    ("train", "render-private"): _cmd_train_render_private,
    ("train", "record"): _cmd_train_record,
    ("train", "run"): _cmd_train_run,
    ("train", "resume"): _cmd_train_resume,
    ("synthesize", None): _cmd_synthesize,
    ("evaluate", "metrics"): _cmd_evaluate_metrics,
    ("evaluate", "gate"): _cmd_evaluate_gate,
    ("evaluate", "pack"): _cmd_evaluate_pack,
    ("evaluate", "ingest"): _cmd_evaluate_ingest,
    ("evaluate", "adjudicate"): _cmd_evaluate_adjudicate,
    ("evaluate", "paragraphs"): _cmd_evaluate_paragraphs,
    ("evaluate", "ab-player"): _cmd_evaluate_ab_player,
    ("evidence", "add"): _cmd_evidence_add,
    ("package", "build"): _cmd_package,
    ("package", "inspect"): _cmd_package_inspect,
    ("run-manifest", "validate"): _cmd_run_manifest_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    key = (
        args.command,
        getattr(args, "environment_command", None)
        or getattr(args, "text_command", None)
        or getattr(args, "corpus_command", None)
        or getattr(args, "model_command", None)
        or getattr(args, "train_command", None)
        or getattr(args, "evaluate_command", None)
        or getattr(args, "evidence_command", None)
        or getattr(args, "package_command", None)
        or getattr(args, "run_manifest_command", None),
    )
    handler = _DISPATCH.get(key)
    if handler is None:  # pragma: no cover - argparse enforces the tree
        print(f"rpa-tts: unhandled command {key}", file=sys.stderr)
        return 2
    try:
        payload, exit_code = handler(args)
    except GateNotPassedError as exc:
        print(f"rpa-tts: {exc}", file=sys.stderr)
        return 3
    except InputError as exc:
        print(f"rpa-tts: {exc}", file=sys.stderr)
        return 2
    except RpaTtsError as exc:
        print(f"rpa-tts: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"rpa-tts: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    _print(payload)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

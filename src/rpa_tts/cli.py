"""Command line workflow; all commands report machine-readable JSON."""

import argparse
import json
from pathlib import Path
import sys


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="rpa-tts", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="decode and inventory every original recording")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--output", type=Path, default=Path("build/tts/audit"))
    audit.add_argument("--workers", type=int, default=4)

    prep = commands.add_parser("prepare", help="verify hashes and create 24kHz WAV train/valid/test sets")
    prep.add_argument("--repo-root", type=Path, default=Path("."))
    prep.add_argument("--manifest", type=Path, default=Path("build/tts/audit/manifest.jsonl"))
    prep.add_argument("--output", type=Path, default=Path("build/tts/data"))
    prep.add_argument("--workers", type=int, default=4)
    prep.add_argument("--seed", type=int, default=3407)
    prep.add_argument("--valid-fraction", type=float, default=0.1)
    prep.add_argument("--test-fraction", type=float, default=0.1)
    prep.add_argument("--split-unit", choices=["syllable", "cv"], default="syllable")

    baseline = commands.add_parser("baseline", help="join existing syllable recordings into a reference WAV")
    baseline.add_argument("--text", required=True)
    baseline.add_argument("--repo-root", type=Path, default=Path("."))
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--gap-ms", type=int, default=40)
    baseline.add_argument("--punctuation-ms", type=int, default=180)

    doctor = commands.add_parser("doctor", help="check Mac/MLX readiness without downloading models")
    doctor.add_argument("--manifest", type=Path)

    train = commands.add_parser("train", help="experimental MLX Qwen3-TTS LoRA canary on Apple Silicon")
    train.add_argument("--manifest", type=Path, default=Path("build/tts/data/train.jsonl"))
    train.add_argument("--output", type=Path, default=Path("build/tts/runs/canary-001"))
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--max-samples", type=int, default=64, help="0 uses every training row; default is a small canary")
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--alpha", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train.add_argument("--seed", type=int, default=3407)
    train.add_argument("--model", default=None, help="Hugging Face MLX Qwen3-TTS model ID; defaults to 0.6B Base BF16")
    train.add_argument("--model-revision", default="main", help="model commit or ref; resolved commit is recorded for inference")

    speech = commands.add_parser("synthesize", help="generate with a saved experimental MLX adapter")
    speech.add_argument("--adapter", type=Path, required=True)
    speech.add_argument("--text", required=True)
    speech.add_argument("--output", type=Path, required=True)
    speech.add_argument("--max-tokens", type=int, default=128)
    speech.add_argument("--seed", type=int, default=3407)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        exit_code = 0
        if args.command == "audit":
            from .audit import audit_corpus
            report = audit_corpus(args.repo_root, args.output, workers=args.workers)
            exit_code = 0 if report["ok"] else 1
        elif args.command == "prepare":
            from .prepare import prepare_corpus
            report = prepare_corpus(args.manifest, args.repo_root, args.output,
                workers=args.workers, seed=args.seed, valid_fraction=args.valid_fraction,
                test_fraction=args.test_fraction, split_unit=args.split_unit)
        elif args.command == "baseline":
            from .baseline import synthesize_baseline
            report = synthesize_baseline(args.text, args.output, repo_root=args.repo_root,
                gap_ms=args.gap_ms, punctuation_ms=args.punctuation_ms)
        elif args.command == "doctor":
            from .mlx_backend import preflight
            report = preflight(args.manifest)
        elif args.command == "train":
            from .mlx_backend import DEFAULT_MODEL, train
            report = train(args.manifest, args.output, steps=args.steps,
                max_samples=args.max_samples, learning_rate=args.learning_rate,
                rank=args.rank, alpha=args.alpha,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed, model_name=args.model or DEFAULT_MODEL,
                model_revision=args.model_revision)
        else:
            from .mlx_backend import synthesize
            report = synthesize(args.adapter, args.text, args.output,
                max_tokens=args.max_tokens, seed=args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return exit_code
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"rpa-tts: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

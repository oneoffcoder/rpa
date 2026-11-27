"""Validate length_scale=1.2 on fresh material for White Hmong TTS.

Generates A/B pairs (1.0 vs 1.2), regression set, and additional seed tests.
Records all metadata: settings, checkpoint/map hashes, WAV hashes, durations.

Usage:
    python scripts/vits_validation_1.2.py validate \
        --bundle data/artifacts/rpa-vits-step35461-listener-verified \
        --output data/artifacts/validation-length-scale-1.2 \
        --prompts data/raw/mwwhdv-nt/prompts-validation-fresh.txt \
        --regression data/raw/mwwhdv-nt/prompts-development.txt \
        --cuda
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard.summary import audio as encode_audio
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vits_coqui_setup import project_text, sha256

# Configuration
LENGTH_SCALE_A = 1.0  # baseline
LENGTH_SCALE_B = 1.2  # candidate (owner preferred)
REGRESSION_SEEDS = [42, 12345]  # additional seeds for frozen 1.2
REGRESSION_PROMPTS_FILE = "data/raw/mwwhdv-nt/prompts-development.txt"


def load_bundle(bundle: Path, *, cuda: bool):
    """Load the verified recovery bundle."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = bundle / "config.json"
    checkpoint = bundle / manifest["checkpoint"]
    config = load_config(str(config_path))
    model = Synthesizer(
        tts_checkpoint=str(checkpoint),
        tts_config_path=str(config_path),
        use_cuda=cuda,
    ).tts_model
    return model, manifest


def render_with_length_scale(
    model,
    prompts: list[str],
    output_dir: Path,
    length_scale: float,
    seed: int = 42,
    cuda: bool = False,
) -> list[dict[str, Any]]:
    """Render prompts with a specific length_scale setting."""
    orig_length_scale = model.length_scale
    model.length_scale = length_scale

    rows = []
    for number, sentence in enumerate(prompts, start=1):
        torch.manual_seed(seed)
        projected, _ = project_text(sentence)
        with torch.inference_mode():
            result = model.synthesize(
                projected,
                speaker=None,
                language=None,
                use_griffin_lim=True,
            )
        wav = result["wav"]
        if hasattr(wav, 'dtype') and wav.dtype.name == 'float16':
            wav = wav.astype('float32')
        sample = encode_audio("audio", wav, sample_rate=model.ap.sample_rate)
        path = output_dir / f"{number:02d}.wav"
        path.write_bytes(sample.value[0].audio.encoded_audio_string)
        with wave.open(str(path)) as clip:
            rows.append({
                "file": path.name,
                "sentence": sentence,
                "model_text": projected,
                "words": len(projected.split()),
                "seconds": clip.getnframes() / clip.getframerate(),
                "sha256": sha256(path),
                "length_scale": length_scale,
                "seed": seed,
            })

    model.length_scale = orig_length_scale
    return rows


def validate(
    bundle: Path,
    prompts: Path,
    output: Path,
    *,
    cuda: bool = False,
    regression_file: Path | None = None,
):
    """Run the complete validation: A/B pairs, regression, additional seeds."""
    model, manifest = load_bundle(bundle, cuda=cuda)
    output.mkdir(parents=True, exist_ok=True)

    # Load fresh prompts
    fresh_prompts = [
        line.strip()
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Generate A/B pairs
    print("Generating A/B pairs...")
    ab_dir = output / "ab-pairs"
    ab_dir.mkdir(parents=True, exist_ok=True)

    # Render A (1.0)
    a_results = render_with_length_scale(
        model, fresh_prompts, ab_dir,
        length_scale=LENGTH_SCALE_A,
        seed=42,
        cuda=cuda,
    )
    # Rename to 1a, 2a, etc.
    for row in a_results:
        old_path = ab_dir / row["file"]
        new_name = row["file"].replace(".wav", f"_{LENGTH_SCALE_A}.wav")
        new_path = ab_dir / new_name
        old_path.rename(new_path)
        row["file"] = new_name

    # Render B (1.2)
    b_results = render_with_length_scale(
        model, fresh_prompts, ab_dir,
        length_scale=LENGTH_SCALE_B,
        seed=42,
        cuda=cuda,
    )
    # Rename to 1b, 2b, etc.
    for row in b_results:
        old_path = ab_dir / row["file"]
        new_name = row["file"].replace(".wav", f"_{LENGTH_SCALE_B}.wav")
        new_path = ab_dir / new_name
        old_path.rename(new_path)
        row["file"] = new_name

    # Build A/B manifest
    ab_manifest = {
        "type": "ab-pairs",
        "length_scale_a": LENGTH_SCALE_A,
        "length_scale_b": LENGTH_SCALE_B,
        "torch_seed": 42,
        "prompts": fresh_prompts,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "vocabulary_sha256": manifest.get("vocabulary_sha256", ""),
        "clips": a_results + b_results,
        "notes": "A=1.0 baseline, B=1.2 candidate (owner preferred B)",
    }
    (ab_dir / "ab-manifest.json").write_text(
        json.dumps(ab_manifest, indent=2), encoding="utf-8"
    )

    # Regression set (original 20-sentence)
    if regression_file:
        print("Generating regression set...")
        reg_dir = output / "regression"
        reg_dir.mkdir(parents=True, exist_ok=True)

        reg_prompts = [
            line.strip()
            for line in regression_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reg_results = render_with_length_scale(
            model, reg_prompts, reg_dir,
            length_scale=LENGTH_SCALE_B,
            seed=42,
            cuda=cuda,
        )

        reg_manifest = {
            "type": "regression",
            "length_scale": LENGTH_SCALE_B,
            "torch_seed": 42,
            "prompts": reg_prompts,
            "clips": reg_results,
            "notes": "Original 20-sentence set at length_scale=1.2",
        }
        (reg_dir / "regression-manifest.json").write_text(
            json.dumps(reg_manifest, indent=2), encoding="utf-8"
        )

    # Additional seeds for 1.2
    print("Testing additional seeds...")
    seed_dir = output / "additional-seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for seed in REGRESSION_SEEDS:
        seed_results = render_with_length_scale(
            model, fresh_prompts, seed_dir,
            length_scale=LENGTH_SCALE_B,
            seed=seed,
            cuda=cuda,
        )
        seed_manifest = {
            "type": "additional-seed",
            "length_scale": LENGTH_SCALE_B,
            "seed": seed,
            "prompts": fresh_prompts,
            "clips": seed_results,
        }
        seed_file = seed_dir / f"seed-{seed:05d}-manifest.json"
        seed_file.write_text(
            json.dumps(seed_manifest, indent=2), encoding="utf-8"
        )

    # Write overall summary
    overall_manifest = {
        "schema": "rpa-validation-length-scale-1.2",
        "objective": "Validate length_scale=1.2 on fresh material per issue 3",
        "owner_feedback": "Owner listened to six emailed A/B pairs, preferred B (1.2) over A (1.0)",
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "vocabulary_sha256": manifest.get("vocabulary_sha256", ""),
        "torch_seed_default": 42,
        "additional_seeds_tested": REGRESSION_SEEDS,
        "fresh_prompts": fresh_prompts,
        "ab_pairs": "data/artifacts/validation-length-scale-1.2/ab-pairs/ab-manifest.json",
        "regression": f"data/artifacts/validation-length-scale-1.2/regression/regression-manifest.json" if regression_file else None,
        "additional_seeds": [f"data/artifacts/validation-length-scale-1.2/additional-seeds/seed-{s:05d}-manifest.json" for s in REGRESSION_SEEDS],
        "notes": "1.0 remains default until fresh native feedback supports adopting 1.2",
    }
    (output / "validation-manifest.json").write_text(
        json.dumps(overall_manifest, indent=2), encoding="utf-8"
    )

    return overall_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate_cmd = commands.add_parser("validate")
    validate_cmd.add_argument("--bundle", type=Path, required=True)
    validate_cmd.add_argument("--output", type=Path, required=True)
    validate_cmd.add_argument("--prompts", type=Path, required=True)
    validate_cmd.add_argument("--regression", type=Path, default=None)
    validate_cmd.add_argument("--cuda", action="store_true")

    args = parser.parse_args()
    if args.command == "validate":
        result = validate(
            bundle=args.bundle,
            prompts=args.prompts,
            output=args.output,
            cuda=args.cuda,
            regression_file=args.regression,
        )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

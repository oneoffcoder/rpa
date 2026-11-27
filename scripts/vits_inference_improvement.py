"""Test inference-time controls to improve recovered White Hmong TTS quality.

Tests length_scale, inference_noise_scale, and inference_noise_scale_dp
on the recovered best_model_35461.pth bundle without retraining.

Usage:
    python scripts/vits_inference_improvement.py render \
        --bundle data/artifacts/rpa-vits-step35461-listener-verified \
        --output data/runs/vits-inference-improvement \
        --prompts data/raw/mwwhdv-nt/prompts-development.txt \
        --cuda

The script follows the improvement cycle from issue #3 comment 7212728391:
- Test one factor at a time on a representative set
- Reset Torch seed 42 before each render
- Record settings, text, tokens, durations, WAV hashes
- Produce a comparison manifest for listening evaluation
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

# Also import the vits_character_map helpers for project_text and sha256
import sys
sys.path.insert(0, str(Path(__file__).parent))
from vits_coqui_setup import project_text, sha256

SCHEMA = "rpa-coqui-v1"

# Representative development prompts (from PREVIEW_TEXTS + common short speech)
DEFAULT_PROMPTS = [
    "peb mus tsev.",
    "kuv yog hmoob.",
    "koj nyob li cas?",
    "uv yav dhau los no.",
    "txiv neng mus ha?",
    "peb yuav mus nploog ntoo.",
]

# Inference controls to test
LENGTH_SCALES = [0.8, 1.0, 1.2]
NOISE_SCALES = [0.5, 0.667, 0.8]
NOISE_SCALES_DP = [0.5, 1.0]


def load_bundle(bundle: Path, *, cuda: bool):
    """Load the verified recovery bundle and return model + manifest."""
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


def render_with_settings(
    model,
    prompts: list[str],
    output: Path,
    *,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_scale_dp: float = 1.0,
    cuda: bool = False,
) -> list[dict[str, Any]]:
    """Render prompts with specific inference settings.

    Modifies model attributes directly for the test, then restores them.
    """
    # Save original settings
    orig_length_scale = model.length_scale
    orig_noise_scale = model.noise_scale_dp if hasattr(model, 'noise_scale_dp') else getattr(model, 'inference_noise_scale_dp', 1.0)
    orig_noise_scale_val = getattr(model, 'inference_noise_scale', 0.667)
    orig_noise_scale_dp = getattr(model, 'inference_noise_scale_dp', 1.0)

    # Apply test settings
    model.length_scale = length_scale
    model.inference_noise_scale = noise_scale
    model.inference_noise_scale_dp = noise_scale_dp

    rows = []
    for number, sentence in enumerate(prompts, start=1):
        torch.manual_seed(42)
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
        sample = encode_audio(
            "audio", wav, sample_rate=model.ap.sample_rate
        )
        path = output / f"{number:02d}.wav"
        path.write_bytes(sample.value[0].audio.encoded_audio_string)
        with wave.open(str(path)) as clip:
            rows.append({
                "file": path.name,
                "sentence": sentence,
                "model_text": projected,
                "words": len(projected.split()),
                "seconds": clip.getnframes() / clip.getframerate(),
                "sha256": sha256(path),
            })

    # Restore original settings
    model.length_scale = orig_length_scale
    model.inference_noise_scale = orig_noise_scale_val
    model.inference_noise_scale_dp = orig_noise_scale_dp

    return rows


def render_experiment(
    bundle: Path,
    prompts: Path,
    output: Path,
    *,
    cuda: bool = False,
    test_length_scale: bool = True,
    test_noise_scale: bool = True,
    test_noise_scale_dp: bool = True,
) -> dict[str, Any]:
    """Run the full improvement experiment."""
    model, manifest = load_bundle(bundle, cuda=cuda)
    output.mkdir(parents=True, exist_ok=True)

    prompts_list = [
        line.strip()
        for line in prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts_list:
        prompts_list = DEFAULT_PROMPTS

    results: dict[str, list[dict]] = {}
    all_clips: list[dict] = []

    # Test length_scale variations
    if test_length_scale:
        for ls in [0.8, 1.0, 1.2]:
            setting = f"length_scale={ls}"
            clip_dir = output / "length_scale" / f"{ls}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            rows = render_with_settings(
                model, prompts_list, clip_dir,
                length_scale=ls,
                cuda=cuda,
            )
            results[setting] = rows
            for row in rows:
                row["setting"] = setting
                row["type"] = "length_scale"
                all_clips.append(row)

    # Test noise_scale variations
    if test_noise_scale:
        for ns in [0.5, 0.667, 0.8]:
            setting = f"noise_scale={ns}"
            clip_dir = output / "noise_scale" / f"{ns}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            rows = render_with_settings(
                model, prompts_list, clip_dir,
                noise_scale=ns,
                cuda=cuda,
            )
            results[setting] = rows
            for row in rows:
                row["setting"] = setting
                row["type"] = "noise_scale"
                all_clips.append(row)

    # Test noise_scale_dp variations
    if test_noise_scale_dp:
        for nsdp in [0.5, 1.0]:
            setting = f"noise_scale_dp={nsdp}"
            clip_dir = output / "noise_scale_dp" / f"{nsdp}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            rows = render_with_settings(
                model, prompts_list, clip_dir,
                noise_scale_dp=nsdp,
                cuda=cuda,
            )
            results[setting] = rows
            for row in rows:
                row["setting"] = setting
                row["type"] = "noise_scale_dp"
                all_clips.append(row)

    # Write manifest
    manifest_out = {
        "schema": SCHEMA,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "vocabulary_sha256": manifest.get("vocabulary_sha256", ""),
        "torch_seed": 42,
        "prompts": prompts_list,
        "settings_tested": {
            "length_scale": [0.8, 1.0, 1.2],
            "noise_scale": [0.5, 0.667, 0.8],
            "noise_scale_dp": [0.5, 1.0],
        },
        "clips": all_clips,
        "notes": (
            "Inference-time controls tested on recovered model. "
            "No retraining. Each setting tested independently. "
            "Torch seed 42 reset before each render."
        ),
    }
    (output / "experiment-manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    render_cmd = commands.add_parser("render")
    render_cmd.add_argument("--bundle", type=Path, required=True)
    render_cmd.add_argument("--output", type=Path, required=True)
    render_cmd.add_argument("--prompts", type=Path, required=True)
    render_cmd.add_argument("--cuda", action="store_true")
    render_cmd.add_argument(
        "--length-scale-values", type=float, nargs="+", default=[0.8, 1.0, 1.2]
    )
    render_cmd.add_argument(
        "--noise-scale-values", type=float, nargs="+", default=[0.5, 0.667, 0.8]
    )
    render_cmd.add_argument(
        "--noise-scale-dp-values", type=float, nargs="+", default=[0.5, 1.0]
    )

    args = parser.parse_args()
    if args.command == "render":
        result = render_experiment(
            bundle=args.bundle,
            prompts=args.prompts,
            output=args.output,
            cuda=args.cuda,
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

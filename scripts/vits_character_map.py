"""Preserve a verified RPA VITS checkpoint and render from its bound character map."""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path

import torch
from torch.utils.tensorboard.summary import audio as encode_audio
from TTS.config import load_config
from TTS.utils.synthesizer import Synthesizer
from vits_coqui_setup import project_text, sha256

from rpa_tts.text.coqui_characters import (
    RpaCharacters,
    ordered_vocabulary,
    verify_checkpoint_tokenizer,
    vocabulary_sha256,
)

SCHEMA = "rpa-coqui-vits-character-map-v1"


def preserve(checkpoint: Path, evidence: Path, destination: Path) -> None:
    receipt = json.loads(evidence.read_text(encoding="utf-8"))
    original_config = checkpoint.with_name("config.json")
    tokens = ordered_vocabulary(receipt["B_candidate_vocab"])
    if (
        sha256(checkpoint) != receipt["checkpoint_sha256"]
        or sha256(original_config) != receipt["config_sha256"]
        or vocabulary_sha256(tokens) != receipt["B_candidate_vocab_sha256"]
    ):
        raise ValueError("Checkpoint, config, or map differs from the listening comparison")

    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    shutil.copy2(original_config, destination / "original-config.json")
    shutil.copy2(evidence, destination / "aba-provenance.json")
    config = load_config(str(original_config))
    config.characters = RpaCharacters(tokens).to_config()
    config.save_json(str(destination / "config.json"))
    if sha256(destination / checkpoint.name) != receipt["checkpoint_sha256"]:
        raise ValueError("Preserved checkpoint copy failed its integrity check")
    manifest = {
        "schema": SCHEMA,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "config_sha256": sha256(destination / "config.json"),
        "original_config_sha256": receipt["config_sha256"],
        "vocabulary": list(tokens),
        "vocabulary_sha256": vocabulary_sha256(tokens),
        "aba_provenance_sha256": sha256(destination / "aba-provenance.json"),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_bundle(bundle: Path, *, cuda: bool):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or Path(manifest["checkpoint"]).name != manifest["checkpoint"]:
        raise ValueError("Invalid RPA VITS bundle manifest")
    config_path = bundle / "config.json"
    checkpoint = bundle / manifest["checkpoint"]
    for name, digest in (
        ("config.json", manifest["config_sha256"]),
        ("original-config.json", manifest["original_config_sha256"]),
        ("aba-provenance.json", manifest["aba_provenance_sha256"]),
        (checkpoint.name, manifest["checkpoint_sha256"]),
    ):
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle checksum mismatch for {name}")
    config = load_config(str(config_path))
    tokens = ordered_vocabulary(config.characters.vocab_dict)
    if (
        list(tokens) != manifest["vocabulary"]
        or vocabulary_sha256(tokens) != manifest["vocabulary_sha256"]
    ):
        raise ValueError("Configured character map differs from the checkpoint manifest")
    evidence = json.loads((bundle / "aba-provenance.json").read_bytes())
    if (
        evidence["checkpoint_sha256"] != manifest["checkpoint_sha256"]
        or evidence["config_sha256"] != manifest["original_config_sha256"]
        or evidence["B_candidate_vocab_sha256"] != manifest["vocabulary_sha256"]
        or evidence["B_candidate_vocab"] != manifest["vocabulary"]
    ):
        raise ValueError("Legacy tokenizer map differs from its recovery evidence")
    model = load_checked_model(checkpoint, config_path, cuda=cuda, legacy_settings=config.to_dict())
    return model, manifest


def load_checked_model(checkpoint: Path, config_path: Path, *, cuda: bool, legacy_settings=None):
    config = load_config(str(config_path))
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    verify_checkpoint_tokenizer(checkpoint_data, config, verified_legacy_settings=legacy_settings)
    del checkpoint_data
    model = Synthesizer(
        tts_checkpoint=checkpoint, tts_config_path=config_path, use_cuda=cuda
    ).tts_model
    if tuple(model.tokenizer.characters.vocab) != ordered_vocabulary(config.characters.vocab_dict):
        raise ValueError("Loaded tokenizer differs from the verified configuration")
    return model


def render(bundle: Path, prompts: Path, output: Path, *, cuda: bool) -> None:
    sentences = [
        line.strip() for line in prompts.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    projected = [project_text(sentence)[0] for sentence in sentences]
    if not projected or len(set(projected)) != len(projected):
        raise ValueError("Provide distinct, nonempty prompts")
    if bundle.is_file():
        model = load_checked_model(bundle, bundle.with_name("config.json"), cuda=cuda)
        manifest = {
            "checkpoint_sha256": sha256(bundle),
            "vocabulary_sha256": vocabulary_sha256(tuple(model.tokenizer.characters.vocab)),
        }
    else:
        model, manifest = load_bundle(bundle, cuda=cuda)
    output.mkdir(parents=True)
    rows = []
    for number, (sentence, text) in enumerate(zip(sentences, projected, strict=True), start=1):
        torch.manual_seed(42)
        with torch.inference_mode():
            result = model.synthesize(text, speaker=None, language=None, use_griffin_lim=True)
        wav = result["wav"]
        if wav.dtype.name == "float16":
            wav = wav.astype("float32")
        sample = encode_audio("audio", wav, sample_rate=model.ap.sample_rate)
        path = output / f"{number:02d}.wav"
        path.write_bytes(sample.value[0].audio.encoded_audio_string)
        with wave.open(str(path)) as clip:
            rows.append(
                {
                    "file": path.name,
                    "sentence": sentence,
                    "model_text": text,
                    "words": len(text.split()),
                    "seconds": clip.getnframes() / clip.getframerate(),
                    "sha256": sha256(path),
                }
            )
    (output / "sentences.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "vocabulary_sha256": manifest["vocabulary_sha256"],
                "torch_seed": 42,
                "clips": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("preserve")
    for name in ("checkpoint", "evidence", "destination"):
        save.add_argument(name, type=Path)
    generate = commands.add_parser("render")
    for name in ("bundle", "prompts", "output"):
        generate.add_argument(name, type=Path)
    generate.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.command == "preserve":
        preserve(args.checkpoint, args.evidence, args.destination)
    else:
        render(args.bundle, args.prompts, args.output, cuda=args.cuda)


if __name__ == "__main__":
    main()

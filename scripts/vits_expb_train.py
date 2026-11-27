"""Prepare and run a weights-only VITS fine-tune on cleaned White Hmong audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import wave
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from rpa_tts.text.coqui_characters import (
    tokenizer_contract,
    verify_checkpoint_tokenizer,
    vocabulary_sha256,
)
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, load_audio
from vits_coqui_setup import project_text

CLEANED_SOURCE = Path("data/derived/mwwhdv-nt/cleaned-v001")
RECOVERED_BUNDLE = Path("data/artifacts/rpa-vits-step35461-listener-verified")
BASELINE_CHECKPOINT = "best_model_35461.pth"
BASELINE_SHA256 = "2919b47c677d103b8487462024efb468289335ae243d8de3fe18a4b1ed5e593e"
BASELINE_CONFIG_SHA256 = "60fabfd4bab30bc407ae23ff387b396d1b173cc9abd71856f738837443426195"
SAMPLE_RATE = 22050
PINNED = {"coqui-tts": "0.28.0", "coqui-tts-trainer": "0.4.0"}
PREVIEW_TEXTS = ("kuv yog hmoob.", "kuv noj mov.", "peb mus tsev.")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare_cleaned_dataset(run: Path) -> dict:
    """Prepare cleaned dataset from cleaned-v001 directory."""
    manifest_path = CLEANED_SOURCE / "manifest.jsonl"
    quality_path = CLEANED_SOURCE / "quality-report.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cleaned manifest not found: {manifest_path}. "
            "Run cleaning first: vits_clean_corpus.py clean"
        )

    # Read cleaned manifest
    cleaned_manifest: dict[str, dict] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cleaned_manifest[row["id"]] = row

    # Read quality report
    quality = {}
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))

    # Prepare dataset directory
    dataset = run / "dataset"
    (dataset / "wavs").mkdir(parents=True, exist_ok=True)

    metadata: dict[str, list[str]] = {"train": [], "dev": []}
    hashes: set[str] = set()
    missing: list[str] = []
    frames: Counter[str] = Counter()
    projected_counts: Counter[str] = Counter()

    for uid, row in cleaned_manifest.items():
        audio_file = CLEANED_SOURCE / row["audio_path"]
        if not audio_file.exists():
            missing.append(uid)
            continue

        split = row.get("split", "unknown")
        if split not in metadata:
            continue

        with wave.open(str(audio_file), "rb") as stream:
            sr, nch, sampw = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
            if (sr, nch, sampw) != (SAMPLE_RATE, 1, 2):
                missing.append(uid)
                continue
            frame_count = stream.getnframes()

        digest = sha256(audio_file)
        if digest in hashes:
            continue
        hashes.add(digest)
        frames[split] += frame_count

        destination = dataset / "wavs" / f"{uid}.wav"
        try:
            os.link(audio_file, destination)
        except OSError:
            shutil.copy2(audio_file, destination)

        text = row["text"]
        projected, _ = project_text(text)
        projected_counts[split] += int(text != projected)
        metadata[split].append(f"{uid}|{projected}|{projected}\n")

    if missing or not all(metadata.values()):
        raise ValueError(
            f"Cleaned dataset is incomplete: {len(missing)} unusable WAVs; {missing[:5]}"
        )

    # Write metadata files
    for split, lines in metadata.items():
        (dataset / f"metadata_{split}.csv").write_text("".join(lines), encoding="utf-8")

    return {
        "cleaned_manifest_sha256": sha256(manifest_path),
        "cleaned_count": len(cleaned_manifest),
        "used_count": sum(len(v) for v in metadata.values()),
        "missing_count": len(missing),
        "missing_ids": missing[:20],
        "splits": {split: len(lines) for split, lines in metadata.items()},
        "projected_transcripts": dict(projected_counts),
        "metadata_sha256": {split: sha256(dataset / f"metadata_{split}.csv") for split in metadata},
        "hours": {split: frames[split] / SAMPLE_RATE / 3600 for split in frames},
        "quality_report": quality.get("cleaned_count", 0),
    }


def recovered_config(bundle: Path) -> tuple[Path, VitsConfig, str]:
    """Authenticate the published recovery pair and its ordered vocabulary."""
    manifest = read_json(bundle / "manifest.json")
    checkpoint = bundle / BASELINE_CHECKPOINT
    config_path = bundle / "config.json"
    for path, field, expected in (
        (checkpoint, "checkpoint_sha256", BASELINE_SHA256),
        (config_path, "config_sha256", BASELINE_CONFIG_SHA256),
    ):
        if manifest[field] != expected or sha256(path) != expected:
            raise ValueError(f"Recovery bundle checksum mismatch: {path}")
    config = load_config(str(config_path))
    tokens = tokenizer_contract(config.to_dict())["tokens"]
    digest = vocabulary_sha256(tokens)
    if tuple(manifest["vocabulary"]) != tokens or manifest["vocabulary_sha256"] != digest:
        raise ValueError("Recovery bundle has an inconsistent character map")
    return checkpoint, config, digest


def training_config(
    run: Path, config: VitsConfig, epochs: int, batch_size: int, lr: float
) -> VitsConfig:
    """Keep the recovered model and frontend settings and override the run settings."""
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    config.run_name = "vits_expb_cleaned"
    config.output_path = str(run / "output")
    config.epochs = epochs
    config.batch_size = batch_size
    config.eval_batch_size = min(2, batch_size)
    config.batch_group_size = 4
    config.num_loader_workers = config.num_eval_loader_workers = 0
    config.save_step = 250
    config.print_step = 20
    config.plot_step = 50
    config.print_eval = config.run_eval = True
    config.test_delay_epochs = -1
    config.test_sentences = list(PREVIEW_TEXTS)
    config.compute_input_seq_cache = False
    config.lr_gen = config.lr_disc = lr
    config.datasets = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="mwwhdv_cleaned",
            language="mww",
            path=str(run / "dataset"),
            meta_file_train="metadata_train.csv",
            meta_file_val="metadata_dev.csv",
        )
    ]
    return config


def initialize_model(checkpoint: Path, recovered: VitsConfig, config: VitsConfig) -> Vits:
    """Check the legacy contract before loading weights. Keep trainer state fresh."""
    trusted = recovered.to_dict()
    if tokenizer_contract(trusted) != tokenizer_contract(config.to_dict()):
        raise ValueError("Launch tokenizer differs from the verified recovery bundle")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    verify_checkpoint_tokenizer(saved, config, verified_legacy_settings=trusted)
    model = Vits(config)
    if tuple(model.tokenizer.characters.vocab) != tokenizer_contract(trusted)["tokens"]:
        raise ValueError("Instantiated VITS tokenizer changed the recovered character IDs")
    model.load_state_dict(saved["model"], strict=True)
    return model


def prepare(run: Path, bundle: Path, epochs: int, batch_size: int, lr: float) -> None:
    _, recovered, vocab_digest = recovered_config(bundle)
    config = training_config(run, recovered, epochs, batch_size, lr)
    run.mkdir(parents=True, exist_ok=False)
    summary = prepare_cleaned_dataset(run)
    config_path = run / "config.json"
    config.save_json(str(config_path))
    write_json(
        run / "run-manifest.json",
        {
            "experiment": "B",
            "initialization": "weights-only",
            "baseline_bundle": str(bundle),
            "baseline_checkpoint": BASELINE_CHECKPOINT,
            "baseline_sha256": BASELINE_SHA256,
            "baseline_config_sha256": BASELINE_CONFIG_SHA256,
            "vocabulary_sha256": vocab_digest,
            "config_sha256": sha256(config_path),
            "dataset_version": CLEANED_SOURCE.name,
            "dataset_summary": summary,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": {"generator": config.lr_gen, "discriminator": config.lr_disc},
            "optimizer": config.optimizer,
            "scheduler": {
                "generator": config.lr_scheduler_gen,
                "discriminator": config.lr_scheduler_disc,
            },
            "initial_step": 0,
            "initial_epoch_index": 0,
            "save_step": config.save_step,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def assert_fresh_trainer(trainer: Trainer, config: VitsConfig) -> None:
    if trainer.epochs_done != 0 or trainer.total_steps_done != 0:
        raise RuntimeError("Fine-tune must start at epoch index 0 and step 0")
    if len(trainer.optimizer) != 2:
        raise RuntimeError("Expected discriminator and generator optimizers")
    for optimizer, expected_lr in zip(
        trainer.optimizer, (config.lr_disc, config.lr_gen), strict=True
    ):
        if optimizer.state or any(group["lr"] != expected_lr for group in optimizer.param_groups):
            raise RuntimeError(
                "Fine-tune must start with fresh optimizers at configured learning rates"
            )


def train(run: Path, *, check_only: bool) -> None:
    manifest = read_json(run / "run-manifest.json")
    config_path = run / "config.json"
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Prepared config checksum changed")
    config = load_config(str(config_path))
    checkpoint, recovered, vocab_digest = recovered_config(Path(manifest["baseline_bundle"]))
    if manifest["vocabulary_sha256"] != vocab_digest:
        raise ValueError("Prepared manifest does not match recovery vocabulary")
    if (
        sha256(CLEANED_SOURCE / "manifest.jsonl")
        != manifest["dataset_summary"]["cleaned_manifest_sha256"]
    ):
        raise ValueError("Cleaned corpus manifest changed after preparation")
    if Path(config.output_path) != run / "output":
        raise ValueError("Output path does not match the prepared run directory")
    for split, expected_hash in manifest["dataset_summary"]["metadata_sha256"].items():
        if sha256(run / "dataset" / f"metadata_{split}.csv") != expected_hash:
            raise ValueError(f"Prepared {split} transcripts changed")
    model = initialize_model(checkpoint, recovered, config)
    training, validation = load_tts_samples(config, eval_split=True)
    expected = manifest["dataset_summary"]["splits"]
    if (len(training), len(validation)) != (expected["train"], expected["dev"]):
        raise ValueError("Coqui loaded different split counts from the prepared dataset")
    allowed_tokens = set(model.tokenizer.characters.vocab)
    for samples in (training, validation):
        for sample in samples:
            # The pinned LJSpeech formatter includes the metadata line ending.
            sample["text"] = sample["text"].rstrip("\r\n")
            if not sample["text"] or set(sample["text"]) - allowed_tokens:
                raise ValueError(f"Unprojected or empty transcript: {sample['audio_file']}")
    for sample in (training[0], validation[0]):
        audio, rate = load_audio(sample["audio_file"])
        if rate != config.audio.sample_rate or audio.numel() == 0:
            raise ValueError(
                f"Coqui could not decode a valid training clip: {sample['audio_file']}"
            )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the Radiance training run")
    print(
        json.dumps(
            {
                "status": "verified",
                "weights_only": checkpoint.name,
                "vocabulary_sha256": vocab_digest,
                "epochs": config.epochs,
                "initial_epoch_index": 0,
                "initial_step": 0,
                "train": len(training),
                "dev": len(validation),
                "gpu": torch.cuda.get_device_name(0),
                "learning_rate": manifest["learning_rate"],
            }
        ),
        flush=True,
    )
    if check_only:
        return

    state_path = run / "training-state.json"
    completed = 0

    def write_state(status: str, trainer: Trainer) -> None:
        write_json(
            state_path,
            {
                "status": status,
                "process_id": os.getpid(),
                "output_path": str(trainer.output_path),
                "completed_epochs": completed,
                "requested_epochs": config.epochs,
                "step": trainer.total_steps_done,
            },
        )

    def epoch_finished(trainer: Trainer) -> None:
        nonlocal completed
        completed = trainer.epochs_done + 1
        if completed == config.epochs:
            trainer.save_checkpoint()
        write_state("training", trainer)

    # A new output root is an exclusive claim. An interrupted run cannot
    # silently start again in the same directory.
    Path(config.output_path).mkdir()
    trainer = Trainer(
        TrainerArgs(),
        config,
        model=model,
        train_samples=training,
        eval_samples=validation,
        parse_command_line_args=False,
        callbacks={"on_epoch_end": epoch_finished},
    )
    assert_fresh_trainer(trainer, config)
    write_state("training", trainer)
    try:
        trainer.fit()
        if completed != config.epochs:
            raise RuntimeError(f"Trainer returned after {completed}/{config.epochs} epochs")
    except BaseException:
        write_state("interrupted", trainer)
        raise
    write_state("complete", trainer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--bundle", type=Path, default=RECOVERED_BUNDLE)
    prep.add_argument("--epochs", type=int, default=100)
    prep.add_argument("--batch-size", type=int, default=16)
    prep.add_argument("--lr", type=float, default=1e-4)
    launch = commands.add_parser("train")
    launch.add_argument("--run", type=Path, required=True)
    launch.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for package, expected in PINNED.items():
        installed = version(package)
        if installed != expected:
            raise ValueError(f"{package}: expected {expected}, found {installed}")
    if args.command == "prepare":
        prepare(args.run.resolve(), args.bundle.resolve(), args.epochs, args.batch_size, args.lr)
    else:
        train(args.run.resolve(), check_only=args.check)


if __name__ == "__main__":
    main()

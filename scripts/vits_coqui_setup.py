"""Prepare and run the pinned upstream Coqui VITS recipe on the RPA corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import wave
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
from trainer import Trainer, TrainerArgs
from trainer.io import get_last_checkpoint
from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits

from rpa_tts.text.coqui_characters import RpaCharacters, verify_checkpoint_tokenizer
from rpa_tts.text.normalize import normalize_rpa

SOURCE = Path("data/raw/mwwhdv-nt")
MANIFEST = SOURCE / "manifest-quarantine.jsonl"
SPLIT = SOURCE / "split-quarantine.jsonl"
SAMPLE_RATE = 22050
EXPECTED_SPLITS = {"train": 5759, "dev": 714, "test": 1460}
EDITORIAL = "\"'“”-—()[]^/#"
EDITORIAL_MAP = str.maketrans({mark: " " for mark in EDITORIAL})
PREVIEW_TEXTS = ("kuv yog hmoob.", "koj nyob li cas?", "peb mus tsev.")
PINNED = {"coqui-tts": "0.28.0", "coqui-tts-trainer": "0.4.0"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_index(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = row["id"]
        if key in records:
            raise ValueError(f"Duplicate utterance ID in {path}: {key}")
        records[key] = row
    return records


def project_text(text: str) -> tuple[str, Counter[str]]:
    """Keep RPA letters and supported punctuation, mapping documented editorial marks."""
    normalized = normalize_rpa(text).normalized
    projected = " ".join(normalized.translate(EDITORIAL_MAP).split())
    vocabulary = set(RpaCharacters().vocab)
    bad = set(projected) - vocabulary
    if not projected or bad:
        raise ValueError(f"Text outside the Coqui RPA vocabulary: {sorted(bad)}")
    if re.findall("[a-z]", projected) != re.findall("[a-z]", normalized):
        raise ValueError("Transcript projection changed a lexical letter")
    return projected, Counter(mark for mark in normalized if mark in EDITORIAL)


def read_corpus(run: Path) -> dict:
    rows, splits = read_index(MANIFEST), read_index(SPLIT)
    if rows.keys() != splits.keys():
        raise ValueError("Corpus and split manifest IDs differ")
    counts = Counter(entry["split"] for entry in splits.values())
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"Unexpected split counts: {dict(counts)}")

    dataset = run / "dataset"
    (dataset / "wavs").mkdir(parents=True)
    metadata: dict[str, list[str]] = {"train": [], "dev": []}
    hashes: set[str] = set()
    frames: Counter[str] = Counter()
    changes = {}
    for uid, row in sorted(rows.items()):
        if not re.fullmatch(r"[A-Za-z0-9-]+", uid):
            raise ValueError(f"Invalid utterance ID for metadata: {uid}")
        split = splits[uid]["split"]
        audio = Path(row["audio_path"]).resolve(strict=True)
        if not audio.is_relative_to(SOURCE.resolve()):
            raise ValueError(f"{uid}: audio is outside the approved corpus")
        digest = sha256(audio)
        if digest != row["audio_sha256"] or digest in hashes:
            raise ValueError(f"{uid}: audio digest mismatch or duplicate audio across splits")
        hashes.add(digest)
        with wave.open(str(audio), "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (
                SAMPLE_RATE,
                1,
                2,
            ):
                raise ValueError(f"{uid}: expected mono 22,050 Hz PCM16 audio")
            frames[split] += stream.getnframes()
        text, edits = project_text(row["text"])
        if edits:
            changes[uid] = dict(edits)
        if split in metadata:
            destination = dataset / "wavs" / f"{uid}.wav"
            try:
                os.link(audio, destination)
            except OSError:
                shutil.copy2(audio, destination)
            metadata[split].append(f"{uid}|{text}|{text}\n")
    for split, lines in metadata.items():
        (dataset / f"metadata_{split}.csv").write_text("".join(lines), encoding="utf-8")
    return {
        "source_manifest_sha256": sha256(MANIFEST),
        "source_split_sha256": sha256(SPLIT),
        "splits": dict(counts),
        "hours": {split: frames[split] / SAMPLE_RATE / 3600 for split in counts},
        "editorial_changes": changes,
    }


def settings(run: Path, *, epochs: int, batch_size: int) -> VitsConfig:
    config = VitsConfig()
    config.audio.sample_rate = SAMPLE_RATE
    config.audio.fft_size = config.audio.win_length = 1024
    config.audio.hop_length = 256
    config.audio.num_mels = 80
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None
    config.run_name = "vits_coqui_rpa"
    config.output_path = str(run / "output")
    config.batch_size = batch_size
    config.eval_batch_size = min(2, batch_size)
    config.batch_group_size = 5
    config.num_loader_workers = config.num_eval_loader_workers = 0
    config.epochs = epochs
    config.save_step = 250
    config.print_eval = config.run_eval = True
    config.test_delay_epochs = -1
    config.test_sentences = [project_text(text)[0] for text in PREVIEW_TEXTS]
    config.use_phonemes = False
    config.text_cleaner = None
    config.mixed_precision = False
    config.compute_input_seq_cache = False
    characters = RpaCharacters()
    config.characters = characters.to_config()
    config.model_args.num_chars = len(characters.vocab)
    config.datasets = [
        BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="mwwhdv_rpa",
            language="mww",
            path=str(run / "dataset"),
            meta_file_train="metadata_train.csv",
            meta_file_val="metadata_dev.csv",
        )
    ]
    return config


def load_recipe(config_path: Path) -> tuple:
    config = load_config(str(config_path))
    train, dev = load_tts_samples(config, eval_split=True)
    model = Vits(config)
    if (len(train), len(dev)) != (EXPECTED_SPLITS["train"], EXPECTED_SPLITS["dev"]):
        raise ValueError("The upstream loader did not retain the audited train/dev split")
    for sample in [*train, *dev]:
        # The stock LJSpeech formatter includes the record delimiter in its text field.
        text = sample["text"].strip()
        ids = model.tokenizer.text_to_ids(text)
        plain_ids = [index for index in ids if index != model.tokenizer.characters.blank_id]
        if not plain_ids or model.tokenizer.ids_to_text(plain_ids) != text:
            raise ValueError(f"Non-reversible tokenizer input: {sample['audio_file']}")
        sample["text"] = text
    return config, model, train, dev


def validate_resume_tokenizers(resume: Path, config: VitsConfig) -> None:
    """Check both the restore checkpoint and historical best selected by Trainer."""
    restore, best = get_last_checkpoint(str(resume))
    for path in dict.fromkeys((restore, best)):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        try:
            verify_checkpoint_tokenizer(saved, config)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        finally:
            del saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run", type=Path, required=True)
    prepare.add_argument("--epochs", type=int, default=100)
    prepare.add_argument("--batch-size", type=int, default=4)
    launch = commands.add_parser("train")
    mode = launch.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--continue-path", type=Path)
    launch.add_argument("--check", action="store_true", help="Verify launch without training")
    args = parser.parse_args()
    for package, expected in PINNED.items():
        installed = version(package)
        if installed != expected:
            raise ValueError(f"{package}: expected {expected}, found {installed}")

    if args.command == "prepare":
        if min(args.epochs, args.batch_size) < 1:
            parser.error("epochs and batch size must be positive")
        run = args.run.resolve()
        run.mkdir(parents=True)
        summary = read_corpus(run)
        config_path = run / "config.json"
        settings(run, epochs=args.epochs, batch_size=args.batch_size).save_json(str(config_path))
        _, model, train, dev = load_recipe(config_path)
        optimizers = model.get_optimizer()
        summary.update(
            config_sha256=sha256(config_path),
            package_versions=PINNED,
            metadata_sha256={
                split: sha256(run / "dataset" / f"metadata_{split}.csv")
                for split in ("train", "dev")
            },
            train_samples=len(train),
            dev_samples=len(dev),
            vocab_size=len(model.tokenizer.characters.vocab),
            optimizers=[type(item).__name__ for item in optimizers],
            schedulers=[type(item).__name__ for item in model.get_scheduler(optimizers)],
            initialization="fresh",
        )
        (run / "readiness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        details = {key: value for key, value in summary.items() if key != "editorial_changes"}
        print(json.dumps(details))
        return

    resume = args.continue_path.resolve(strict=True) if args.continue_path else None
    config_path = resume / "config.json" if resume else args.config.resolve(strict=True)
    config, model, train, dev = load_recipe(config_path)
    if resume is not None:
        validate_resume_tokenizers(resume, config)
    dataset = Path(config.datasets[0].path)
    receipt = json.loads(dataset.parent.joinpath("readiness.json").read_text(encoding="utf-8"))
    if any(
        sha256(dataset / f"metadata_{split}.csv") != receipt["metadata_sha256"][split]
        for split in ("train", "dev")
    ):
        raise ValueError("Dataset metadata changed after the recipe probe")
    if resume is None:
        if receipt["config_sha256"] != sha256(config_path):
            raise ValueError("Configuration changed after the recipe probe")
        output = Path(config.output_path)
        if output.exists() and any(output.iterdir()):
            raise ValueError("This run already has training output; use --continue-path")
    if not torch.cuda.is_available():
        raise ValueError("The pinned CUDA runtime must see a GPU for training")
    if args.check:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "language": "mww",
                    "config": str(config_path),
                    "train_samples": len(train),
                    "dev_samples": len(dev),
                    "preview_prompts": list(config.test_sentences),
                }
            )
        )
        return
    Trainer(
        TrainerArgs(continue_path=str(resume) if resume else ""),
        config,
        config.output_path,
        model=model,
        train_samples=train,
        eval_samples=dev,
        parse_command_line_args=False,
    ).fit()


if __name__ == "__main__":
    main()

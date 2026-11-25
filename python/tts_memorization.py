"""Run the fixed train-only Qwen memorization and listening experiment on Apple Silicon."""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rpa_tts import mlx_backend as backend


SYLLABLES = ("kuv", "yog", "hmoob", "po", "poj", "pos")
SEED = 3407
MAX_TOKENS = 128


def select_training_rows(source, manifest):
    selected = {}
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["text"] in SYLLABLES:
            if row["text"] in selected or row["split"] != "train" or row["id"] != row["text"]:
                raise ValueError(f"Expected one training recording for {row['text']}")
            row["audio_path"] = os.path.relpath(source.parent / row["audio_path"], manifest.parent)
            selected[row["text"]] = row
    if set(selected) != set(SYLLABLES):
        raise ValueError(f"Missing training recordings: {set(SYLLABLES) - set(selected)}")
    with manifest.open("x") as stream:
        for text in SYLLABLES:
            stream.write(json.dumps(selected[text], sort_keys=True) + "\n")
    return backend.read_training_manifest(manifest)


def first_logits(rt, model, text):
    embeddings, _, _ = model.full_model._prepare_generation_inputs(text, language="auto")
    logits, _ = model.full_model.talker(embeddings, cache=model.full_model.talker.make_cache())
    return np.asarray(logits[0, -1].astype(rt.mx.float32))


def check_adapter(rt, model, adapter, base_logits):
    backend._apply_adapter(rt, model, adapter)
    changes = {}
    for text, before in base_logits.items():
        after = first_logits(rt, model, text)
        delta = float(np.max(np.abs(after - before)))
        if not np.isfinite(delta) or delta == 0:
            raise RuntimeError(f"No finite change in prediction scores for {text}")
        changes[text] = {"base_top1": int(before.argmax()), "adapted_top1": int(after.argmax()),
                         "max_abs_logit_delta": delta}
    return changes


def record_audio(path):
    info = backend._wav_info(path)
    return {"file": "audio/" + path.name, "sha256": backend._digest(path), "frames": info["frames"]}


def reconstruct(model, sample, row, folder):
    text = row["text"]
    original = folder / f"{text}-original.wav"
    shutil.copyfile(row["audio_path"], original)
    waveform = sample["audio"]["array"]
    codes = model.codec_adapter.encode_all_codebooks(waveform, backend.SAMPLE_RATE)
    if codes.shape[0] != 16 or codes.shape[1] < 1:
        raise RuntimeError(f"Unexpected codebook dimensions for {text}: {codes.shape}")
    decoded, _ = model.full_model.speech_tokenizer.decode(codes.T[None])
    # Unpadded single item. Retain the complete final codec frame in the WAV.
    audio = np.asarray(decoded[0], dtype=np.float32)
    if audio.size < waveform.size:
        raise RuntimeError(f"Codec truncated the original for {text}")
    path = folder / f"{text}-reconstructed.wav"
    clipped = backend._write_waveform(path, audio, np)
    overlap = audio[:waveform.size]
    correlation = float(np.corrcoef(waveform, overlap)[0, 1])
    if not np.isfinite(correlation):
        raise RuntimeError(f"Degenerate codec reconstruction for {text}")
    return {"original": record_audio(original), "reconstructed": record_audio(path),
            "same_time_waveform_correlation": correlation, "reconstruction_clipped_samples": clipped,
            "reference_code0": np.asarray(codes[0]).tolist(), "codebooks": int(codes.shape[0])}


def next_token_after_prefix(rt, model, text, frames):
    """Replay a fixed prefix through the normal text prompt and incremental talker cache."""
    talker = model.full_model.talker
    embed, trailing, pad = model.full_model._prepare_generation_inputs(text, language="auto")
    cache = talker.make_cache()
    logits, _ = talker(embed, cache=cache)
    for i, frame in enumerate(frames):
        codes = rt.mx.array(frame[None])
        audio = talker.get_input_embeddings()(codes[:, :1])
        for j, layer in enumerate(talker.code_predictor.codec_embedding):
            audio = audio + layer(codes[:, j + 1:j + 2])
        text_embed = trailing[:, i:i + 1] if i < trailing.shape[1] else pad
        logits, _ = talker(text_embed + audio, cache=cache)
    return int(rt.mx.argmax(logits[0, -1]).item())


def assess(rt, model, sample, entry, variant, folder):
    text = sample["text"]
    collator = rt.TTSDataCollator(model=model, tokenizer=model.tokenizer, max_seq_length=2048,
                                  text_column="text", audio_column="audio")
    batch = collator([sample])
    logits, _ = model.full_model.talker(batch["inputs_embeds"])
    labels = np.asarray(batch["labels"][0, 1:])
    predicted = np.asarray(rt.mx.argmax(logits[0, :-1], axis=-1))
    targets = labels[labels != -100]
    truth = entry["reference_code0"]
    eos = model.full_model.config.talker_config.codec_eos_token_id
    if targets.tolist() != [*truth, eos]:
        raise RuntimeError("Shifted training targets disagree with independently encoded audio")

    # Observe the tokens passed to the real decoder, without changing decoding
    # or using reference tokens for generation. Each prompt is a single syllable.
    codec = model.full_model.speech_tokenizer
    with patch.object(codec, "decode", wraps=codec.decode) as decode:
        audio = backend._generate_audio(rt, model, text, MAX_TOKENS, SEED)
    if decode.call_count != 1:
        raise RuntimeError(f"Expected one decoded utterance for {text}")
    codes = np.asarray(decode.call_args.args[0])
    if codes.shape[0] != 1 or codes.shape[2] != 16:
        raise RuntimeError(f"Unexpected generated codebook dimensions: {codes.shape}")
    path = folder / f"{text}-{variant}.wav"
    clipped = backend._write_waveform(path, audio, np)
    divergence = None
    if variant == "adapted":
        # Before the first mismatch, all preceding code-0 tokens agree. Swap only
        # their other 15 codes to isolate their effect on the next prediction.
        reference = np.asarray(model.codec_adapter.encode_all_codebooks(
            sample["audio"]["array"], backend.SAMPLE_RATE)).T
        generated = codes[0]
        observed = generated[:, 0].tolist()
        if len(observed) < MAX_TOKENS:
            observed.append(eos)
        for i, (expected, actual) in enumerate(zip(targets.tolist(), observed)):
            if expected != actual:
                divergence = {
                    "frame_index": i, "reference": expected, "sampled": actual,
                    "top1_reference_history": next_token_after_prefix(rt, model, text, reference[:i]),
                    "top1_generated_history": next_token_after_prefix(rt, model, text, generated[:i]),
                }
                break
    return {**record_audio(path), "clipped_samples": clipped,
            "first_divergence_replay": divergence,
            "teacher_forced_code0_and_eos_correct": int((predicted[labels != -100] == targets).sum()),
            "teacher_forced_targets": int(targets.size), "generated_code0": codes[0, :, 0].tolist()}


def experiment(manifest, canary, output):
    canary_run = backend._read_adapter_run(canary)
    output.mkdir(parents=True, exist_ok=False)
    folder = output / "audio"
    folder.mkdir()
    subset = output / "train-six.jsonl"
    rows = select_training_rows(manifest, subset)
    rt = backend._load_runtime()
    name, revision = canary_run["model_name"], canary_run["model_revision"]
    model, _, _, _ = backend._model(rt, name, revision)
    model.full_model.eval()
    samples = backend._AudioDataset(rows, np)[:]
    results = {r["text"]: reconstruct(model, s, r, folder) for s, r in zip(samples, rows)}
    base_logits = {}
    for sample in samples:
        text = sample["text"]
        base_logits[text] = first_logits(rt, model, text)
        if not np.array_equal(base_logits[text], first_logits(rt, model, text)):
            raise RuntimeError(f"Unstable base logits for {text}")
        results[text]["base"] = assess(rt, model, sample, results[text], "base", folder)
    canary_probe = check_adapter(rt, model, canary, base_logits)
    del model
    gc.collect()
    rt.mx.clear_cache()

    run = backend.train(subset, output / "model", steps=150, max_samples=0, seed=SEED,
                        model_name=name, model_revision=revision)
    if set(run["training_ids"]) != set(SYLLABLES) or run["training_samples"] != len(SYLLABLES):
        raise RuntimeError("Training run does not match the six selected recordings")
    adapter = output / "model" / "adapters"
    backend._read_adapter_run(adapter)
    model, _, _, _ = backend._model(rt, name, revision)
    model.full_model.eval()
    for text in SYLLABLES:
        if not np.array_equal(first_logits(rt, model, text), base_logits[text]):
            raise RuntimeError(f"Reloaded base model differs on {text}")
    adapted_probe = check_adapter(rt, model, adapter, base_logits)
    for sample in samples:
        entry = results[sample["text"]]
        entry["adapted"] = assess(rt, model, sample, entry, "adapted", folder)

    report = {k: run[k] for k in ("model_name", "model_revision", "packages", "source_revisions",
              "config", "training_ids", "training_audio_sha256", "training_duration_seconds",
              "optimizer_updates", "train_loss", "training_runtime_seconds", "adapter_sha256")}
    report.update(source_train_manifest_sha256=backend._digest(manifest),
                  subset_manifest_sha256=backend._digest(subset),
                  synthesis_seed=SEED, max_tokens=MAX_TOKENS, sample_rate=backend.SAMPLE_RATE,
                  canary_adapter_sha256=canary_run["adapter_sha256"]["adapters.safetensors"],
                  canary_probe=canary_probe, memorization_probe=adapted_probe, syllables=results)
    backend._write_json(output / "measurements.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("build/tts/data/train.jsonl"))
    parser.add_argument("--canary-adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new experiment directory")
    args = parser.parse_args()
    experiment(args.train_manifest.resolve(), args.canary_adapter.resolve(), args.output.resolve())

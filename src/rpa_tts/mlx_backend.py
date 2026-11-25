"""Experimental Hmong Qwen3-TTS LoRA canary on Apple Silicon.

The data tools and this module import without MLX. The measured Apple Silicon runs
and listening results are recorded in docs/tts/README.md.

APIs inspected at:
  https://github.com/ARahim3/mlx-tune/commit/9690fe1fed19c72f0810902b9c188d9a2625eb5b
  https://github.com/Neywa-Labs/mlx-audio/commit/91caeb9ed95b3839c7c08bfc58ee8e7d6216dea8

The upstream Qwen collator trains the talker's first codec stream, with frozen
text embeddings, code predictor and speech tokenizer. It uses an automatic
language prefix without reference-speaker conditioning. Hmong is not an
officially supported Qwen3-TTS language; no invented language ID is introduced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
SAMPLE_RATE = 24000
MLX_TUNE_REVISION = "9690fe1fed19c72f0810902b9c188d9a2625eb5b"
MLX_AUDIO_REVISION = "91caeb9ed95b3839c7c08bfc58ee8e7d6216dea8"
INSTALL_HINT = "On an Apple Silicon Mac, run: uv sync --extra mlx"
LIMITATIONS = [
    "Hmong is outside the model's officially supported languages; this is an experiment.",
    "The recordings train isolated syllables, not natural connected-speech prosody.",
    "Only talker LoRA parameters are trained; code predictor and codec stay frozen.",
    "Training and inference use auto language without reference-speaker conditioning.",
    "The upstream trainer has no validation, early stopping or optimizer-state resume.",
    "Loss is training loss, not a measure of Hmong pronunciation or tone correctness.",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() == "arm64"


def _require_platform() -> None:
    if not _is_apple_silicon():
        raise RuntimeError(
            "MLX training and synthesis require macOS on Apple Silicon with a native "
            f"arm64 Python (found {platform.system()} {platform.machine()}). "
            "Dataset preparation can run on this machine. " + INSTALL_HINT
        )


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("mlx", "mlx-tune", "mlx-audio", "mlx-lm", "numpy", "huggingface-hub"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _wav_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as audio:
            info = {
                "sample_rate": audio.getframerate(),
                "channels": audio.getnchannels(),
                "sample_width": audio.getsampwidth(),
                "frames": audio.getnframes(),
                "compression": audio.getcomptype(),
            }
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"Cannot read PCM WAV {path}: {exc}") from exc
    if (info["sample_rate"], info["channels"], info["sample_width"], info["compression"]) != (
        SAMPLE_RATE, 1, 2, "NONE"
    ):
        raise ValueError(f"Expected mono PCM16 WAV at {SAMPLE_RATE} Hz: {path}; got {info}")
    if info["frames"] < 1:
        raise ValueError(f"Empty recording: {path}")
    info["duration_seconds"] = info["frames"] / SAMPLE_RATE
    return info


def read_training_manifest(manifest: Path | str) -> list[dict[str, Any]]:
    """Validate training rows and WAV headers, without importing MLX/numpy.

    Paths are relative to the manifest directory (absolute paths also work).
    An explicit train split is required, so passing valid/test manifests fails.
    No duration filter or VAD is applied to the short syllable recordings.
    """
    manifest = Path(manifest).resolve()
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {manifest}:{line_number}")
            for key in ("id", "text", "audio_path"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing/non-string {key} at {manifest}:{line_number}")
            if row.get("split") != "train":
                raise ValueError(
                    f"Training accepts only split='train'; row {row['id']} has {row.get('split')!r}"
                )
            path = (manifest.parent / row["audio_path"]).resolve()
            if row["id"] in seen_ids or path in seen_paths:
                raise ValueError(f"Duplicate training id or audio path at {manifest}:{line_number}")
            seen_ids.add(row["id"])
            seen_paths.add(path)
            info = _wav_info(path)
            audio_sha256 = _digest(path)
            if "audio_sha256" in row and row["audio_sha256"] != audio_sha256:
                raise ValueError(f"Audio SHA-256 disagrees with manifest for {row['id']}")
            declared = row.get("duration_seconds")
            if declared is not None and (
                not isinstance(declared, (int, float))
                or isinstance(declared, bool)
                or not math.isfinite(declared)
                or abs(declared - info["duration_seconds"]) > 0.001
            ):
                raise ValueError(f"Duration metadata disagrees with WAV header for {row['id']}")
            rows.append({**row, "audio_path": str(path), "duration_seconds": info["duration_seconds"],
                         "audio_sha256": audio_sha256})
    if not rows:
        raise ValueError(f"Training manifest is empty: {manifest}")
    return rows


def preflight(manifest: Path | str | None = None) -> dict[str, Any]:
    """Inspect platform, optional packages and Metal without downloading models."""
    versions = _versions()
    result: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "apple_silicon": _is_apple_silicon(),
        "packages": versions,
        "metal_available": False,
        "model_downloaded": False,
        "install_hint": INSTALL_HINT,
        "limitations": LIMITATIONS,
    }
    if result["apple_silicon"] and versions["mlx"]:
        try:
            import mlx.core as mx
            result["metal_available"] = bool(mx.metal.is_available())
        except Exception as exc:
            result["metal_error"] = str(exc)
    if manifest is not None:
        rows = read_training_manifest(manifest)
        result["manifest"] = {
            "path": str(Path(manifest).resolve()),
            "samples": len(rows),
            "duration_seconds": sum(row["duration_seconds"] for row in rows),
            "sha256": _digest(Path(manifest)),
        }
    result["ready"] = bool(result["metal_available"] and all(versions.values()))
    return result


def _load_runtime() -> SimpleNamespace:
    _require_platform()
    try:
        import mlx.core as mx
        import numpy as np
        from huggingface_hub import snapshot_download
        from mlx_tune import FastTTSModel, TTSDataCollator, TTSSFTConfig, TTSSFTTrainer
    except ImportError as exc:
        raise RuntimeError(f"MLX dependencies could not be imported. {INSTALL_HINT}. {exc}") from exc
    if not mx.metal.is_available():
        raise RuntimeError("MLX could not find a Metal GPU. Use native arm64 Python on an Apple Silicon Mac.")
    return SimpleNamespace(
        mx=mx, np=np, snapshot_download=snapshot_download,
        FastTTSModel=FastTTSModel, TTSDataCollator=TTSDataCollator,
        TTSSFTConfig=TTSSFTConfig, TTSSFTTrainer=TTSSFTTrainer,
    )


class _AudioDataset:
    """Decode only the current microbatch; upstream needs arrays, not paths."""

    def __init__(self, rows: list[dict[str, Any]], np: Any):
        self.rows, self.np = rows, np

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        row = self.rows[index]
        with wave.open(row["audio_path"], "rb") as audio:
            frames = audio.getnframes()
            raw = audio.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError(f"Truncated PCM audio: {row['audio_path']}")
        waveform = self.np.frombuffer(raw, dtype="<i2").astype(self.np.float32) / 32768.0
        return {"text": row["text"], "audio": {"array": waveform, "sampling_rate": SAMPLE_RATE}}


def _model(rt: SimpleNamespace, name: str, revision: str) -> tuple[Any, Any, str, str]:
    if "qwen3-tts" not in name.lower() or "-base-" not in name.lower():
        raise ValueError("This backend requires a Qwen3-TTS Base checkpoint including its speech encoder.")
    snapshot = Path(rt.snapshot_download(repo_id=name, revision=revision)).resolve()
    # Resolve mutable 'main' once, record the immutable revision, and reuse it in inference.
    resolved_revision = snapshot.name
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_revision):
        raise RuntimeError(f"Expected a Hugging Face cache snapshot commit, got {snapshot}")
    wrapper, tokenizer = rt.FastTTSModel.from_pretrained(model_name=str(snapshot), max_seq_length=2048)
    full_model = getattr(wrapper, "full_model", None)
    codec = getattr(full_model, "speech_tokenizer", None)
    if getattr(getattr(full_model, "config", None), "tts_model_type", None) != "base":
        raise RuntimeError("Loaded checkpoint is not a Qwen3-TTS Base model.")
    if codec is None or not getattr(codec, "has_encoder", False):
        raise RuntimeError("The checkpoint has no working speech tokenizer encoder; use the Base BF16 checkpoint.")
    if getattr(wrapper, "sample_rate", None) != SAMPLE_RATE:
        raise RuntimeError(f"Expected a {SAMPLE_RATE} Hz speech tokenizer.")
    return wrapper, tokenizer, resolved_revision, str(snapshot)


def _memory(rt: SimpleNamespace) -> int | None:
    get_peak = getattr(rt.mx, "get_peak_memory", None)
    if get_peak is None:
        get_peak = getattr(rt.mx.metal, "get_peak_memory", None)
    return int(get_peak()) if get_peak else None


def train(
    manifest: Path | str,
    output_dir: Path | str,
    *,
    steps: int = 100,
    max_samples: int = 64,
    learning_rate: float = 1e-4,
    rank: int = 16,
    alpha: int = 16,
    gradient_accumulation_steps: int = 4,
    seed: int = 3407,
    model_name: str = DEFAULT_MODEL,
    model_revision: str = "main",
) -> dict[str, Any]:
    """Train a real LoRA canary. max_samples=0 uses every training row.

    steps counts optimizer updates, each containing gradient_accumulation_steps
    microbatches of one sample. The upstream trainer uses constant-rate Adam;
    unsupported warmup, weight-decay and resume options are deliberately absent.
    """
    for name, value in (("steps", steps), ("rank", rank), ("alpha", alpha),
                        ("gradient_accumulation_steps", gradient_accumulation_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 0:
        raise ValueError("max_samples must be zero (all) or a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    rows = read_training_manifest(manifest)
    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {output_dir}")
    rt = _load_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {
        "status": "running", "started_at": _now(), "backend": "mlx-tune-qwen3-tts-lora",
        "source_revisions": {"mlx-tune": MLX_TUNE_REVISION, "mlx-audio": MLX_AUDIO_REVISION},
        "packages": _versions(), "python": sys.version.split()[0],
        "platform": platform.platform(), "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": _digest(Path(manifest)), "training_ids": [r["id"] for r in rows],
        "training_samples": len(rows), "training_duration_seconds": sum(r["duration_seconds"] for r in rows),
        "training_audio_sha256": {r["id"]: r["audio_sha256"] for r in rows},
        "model_name": model_name, "requested_model_revision": model_revision,
        "conditioning": "auto_without_reference", "sample_rate": SAMPLE_RATE,
        "config": {"steps": steps, "max_samples": max_samples, "learning_rate": learning_rate,
                   "rank": rank, "alpha": alpha, "gradient_accumulation_steps": gradient_accumulation_steps,
                   "seed": seed, "batch_size": 1, "max_seq_length": 2048},
        "limitations": LIMITATIONS,
    }
    _write_json(output_dir / "run.json", run)
    start = time.perf_counter()
    try:
        rt.mx.random.seed(seed)
        model, tokenizer, revision, snapshot = _model(rt, model_name, model_revision)
        run.update(model_revision=revision, model_snapshot=snapshot)
        model = rt.FastTTSModel.get_peft_model(model, r=rank, lora_alpha=alpha)
        collator = rt.TTSDataCollator(model=model, tokenizer=tokenizer, max_seq_length=2048,
                                      text_column="text", audio_column="audio")
        trainer = rt.TTSSFTTrainer(
            model=model, tokenizer=tokenizer, data_collator=collator,
            train_dataset=_AudioDataset(rows, rt.np),
            args=rt.TTSSFTConfig(output_dir=str(output_dir), per_device_train_batch_size=1,
                gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate,
                max_steps=steps, logging_steps=1, train_on_completions=True),
        )
        training_start = time.perf_counter()
        result = trainer.train()
        run["training_runtime_seconds"] = time.perf_counter() - training_start
        loss = float(result.metrics["train_loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"Training returned non-finite loss: {loss}")
        adapter_dir = output_dir / "adapters"
        # The pinned trainer saves these files; fail instead of claiming a checkpoint exists.
        artifacts = {name: _digest(adapter_dir / name) for name in
                     ("adapter_config.json", "adapters.safetensors")}
        run.update(status="trained", train_loss=loss, adapter_path="adapters", adapter_sha256=artifacts,
                   optimizer_updates=steps, completed_at=_now(),
                   runtime_seconds=time.perf_counter() - start, peak_memory_bytes=_memory(rt))
        _write_json(output_dir / "run.json", run)
        return run
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=_now(),
                   runtime_seconds=time.perf_counter() - start)
        _write_json(output_dir / "run.json", run)
        raise


def _read_adapter_run(adapter_dir: Path) -> dict[str, Any]:
    run = json.loads((adapter_dir.parent / "run.json").read_text())
    if run.get("status") != "trained":
        raise ValueError("The adapter does not belong to a completed training run")
    for name in ("adapter_config.json", "adapters.safetensors"):
        if _digest(adapter_dir / name) != run.get("adapter_sha256", {}).get(name):
            raise ValueError(f"Adapter {name} does not match the completed run")
    return run


def _apply_adapter(rt: SimpleNamespace, model: Any, adapter_dir: Path) -> None:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    lora = config["lora_parameters"]
    rt.FastTTSModel.get_peft_model(
        model, r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("keys") or None,
    )
    model.load_adapter(str(adapter_dir))
    rt.FastTTSModel.for_inference(model)
    model.full_model.eval()


def _generate_audio(rt: SimpleNamespace, model: Any, text: str, max_tokens: int, seed: int) -> Any:
    # LoRA initialization consumes random numbers, even when loading saved weights.
    # Start the sampling stream here so the same seed pairs base and adapted renders.
    rt.mx.random.seed(seed)
    chunks = []
    for result in model.full_model.generate(text=text, lang_code="auto", max_tokens=max_tokens,
                                            stream=False):
        if result.sample_rate != SAMPLE_RATE:
            raise RuntimeError("Unexpected generated sample rate")
        chunks.append(rt.np.asarray(result.audio, dtype=rt.np.float32).reshape(-1))
    if not chunks or not sum(chunk.size for chunk in chunks):
        raise RuntimeError("The model generated no audio")
    return rt.np.concatenate(chunks)


def _write_waveform(output: Path, audio: Any, np: Any) -> int:
    if not np.isfinite(audio).all():
        raise RuntimeError("Waveform contains non-finite audio")
    clipped = int(np.count_nonzero((audio < -1.0) | (audio > 1.0)))
    pcm = np.rint(np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    return clipped


def synthesize(
    adapter_dir: Path | str, text: str, output: Path | str, *, max_tokens: int = 128, seed: int = 3407
) -> dict[str, Any]:
    """Reload this project's saved adapter and synthesize one utterance.

    Call the complete Qwen model, not mlx-tune's generic Orpheus generation
    wrapper. No held-out audio is used as a speaker reference or conditioning.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be nonempty")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    adapter_dir, output = Path(adapter_dir).resolve(), Path(output).resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Choose a new output filename: {output}")
    run = _read_adapter_run(adapter_dir)
    rt = _load_runtime()
    start = time.perf_counter()
    model, _, revision, _ = _model(rt, run["model_name"], run["model_revision"])
    _apply_adapter(rt, model, adapter_dir)
    audio = _generate_audio(rt, model, text, max_tokens, seed)
    clipped = _write_waveform(output, audio, rt.np)
    metadata = {
        "text": text, "adapter_dir": str(adapter_dir), "model_name": run["model_name"],
        "model_revision": revision, "conditioning": "auto_without_reference", "seed": seed,
        "max_tokens": max_tokens, "sample_rate": SAMPLE_RATE,
        "duration_seconds": int(audio.size) / SAMPLE_RATE, "clipped_samples": clipped,
        "runtime_seconds": time.perf_counter() - start, "created_at": _now(),
        "audio_sha256": _digest(output), "limitations": LIMITATIONS,
    }
    _write_json(sidecar, metadata)
    return metadata

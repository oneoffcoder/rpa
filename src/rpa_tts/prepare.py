"""Prepare measured recordings without losing quiet tone or onset information.

The split unit is an original syllable (or optionally a consonant/vowel pair).
Identical decoded recordings are grouped before the split. No examples are
created by joining clips or by adding synthetic silence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import wave

import numpy as np

from .audit import decode_audio
from .orthography import parse_syllable


SAMPLE_RATE = 24_000


def load_jsonl(path: Path) -> list[dict]:
    """Read a manifest, reporting the location of malformed input."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def assign_splits(
    rows: list[dict], *, seed: int = 3407, valid_fraction: float = 0.1,
    test_fraction: float = 0.1, split_unit: str = "syllable",
) -> tuple[list[dict], dict]:
    """Deterministically split groups and retain observed CVT components in train.

    Exact hashes of the source and native-rate decoded PCM link repetitions,
    as does normalized text. Fractions allocate groups, not waveform frames.
    Evaluation syllables are unseen; this is not a speaker or session holdout.
    """
    if split_unit not in {"syllable", "cv"}:
        raise ValueError("split_unit must be 'syllable' or 'cv'")
    if not (0 < valid_fraction < 1 and 0 < test_fraction < 1
            and valid_fraction + test_fraction < 1):
        raise ValueError("valid/test fractions must be positive and sum to less than 1")
    if not rows:
        raise ValueError("cannot split an empty corpus")
    rows = sorted(rows, key=lambda row: row["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("recording IDs must be unique")
    parents = list(range(len(rows)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    keys: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        values = [("text", row["text"].strip().lower())]
        for field in ("source_sha256", "pcm_sha256"):
            if row.get(field):
                values.append((field, row[field]))
        if split_unit == "cv":
            values.append(("cv", row["consonant"], row["vowel"]))
        for key in values:
            if key in keys:
                parents[find(i)] = find(keys[key])
            else:
                keys[key] = i

    grouped: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        grouped[find(i)].append(i)
    groups = list(grouped.values())
    groups.sort(key=lambda group: hashlib.sha256(
        (str(seed) + "\0" + "\0".join(rows[i]["id"] for i in group)).encode()
    ).hexdigest())
    if len(groups) < 3:
        raise ValueError("need at least three independent recording groups")
    n_valid = max(1, math.floor(len(groups) * valid_fraction))
    n_test = max(1, math.floor(len(groups) * test_fraction))
    if n_valid + n_test >= len(groups):
        raise ValueError("fractions leave no training groups")
    labels = ["valid"] * n_valid + ["test"] * n_test
    labels += ["train"] * (len(groups) - n_valid - n_test)

    def components(group: list[int]) -> set[tuple[str, str]]:
        return {(field, rows[i][field]) for i in group
                for field in ("consonant", "vowel", "tone")}

    observed = set().union(*(components(g) for g in groups))
    trained = set().union(*(components(g) for g, s in zip(groups, labels) if s == "train"))
    moved = []
    for index, group in enumerate(groups):
        if labels[index] != "train" and components(group) & (observed - trained):
            moved.append({"ids": [rows[i]["id"] for i in group], "from": labels[index]})
            labels[index] = "train"
            trained.update(components(group))
    if not {"train", "valid", "test"}.issubset(labels):
        raise ValueError("corpus is too small to hold out groups while retaining all CVT components in train")

    assigned = []
    for group, label in zip(groups, labels):
        for i in group:
            assigned.append({**rows[i], "split": label})
    assigned.sort(key=lambda row: row["id"])
    summary = {
        "seed": seed, "split_unit": split_unit,
        "requested_valid_fraction": valid_fraction, "requested_test_fraction": test_fraction,
        "group_counts": dict(sorted(Counter(labels).items())),
        "recording_counts": dict(sorted(Counter(row["split"] for row in assigned).items())),
        "component_coverage_repairs": moved,
        "all_observed_components_in_train": observed <= trained,
        "grouping": ["normalized_text", "source_sha256", "native_pcm_sha256"]
                    + (["consonant_vowel_pair"] if split_unit == "cv" else []),
    }
    return assigned, summary


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono PCM16, preserving level (only constrain integer representation)."""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("waveform must be a nonempty finite mono array")
    pcm = np.clip(np.rint(values * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def prepare_corpus(
    manifest: Path, repo_root: Path, output_dir: Path, *, workers: int = 4,
    seed: int = 3407, valid_fraction: float = 0.1, test_fraction: float = 0.1,
    split_unit: str = "syllable",
) -> dict:
    """Verify audit hashes, resample to 24 kHz and atomically publish split files."""
    manifest, repo_root, output_dir = manifest.resolve(), repo_root.resolve(), output_dir.resolve()
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}; use a new directory for a new preparation")
    manifest_bytes = manifest.read_bytes()
    rows = load_jsonl(manifest)
    if not rows:
        raise ValueError("audit manifest contains no recordings")
    required = {"id", "text", "consonant", "vowel", "tone", "audio_path", "source_sha256", "pcm_sha256", "status"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"audit row is missing required fields: {sorted(missing)}")
        if row["status"] != "ok":
            raise ValueError(f"resolve audit errors before preparation: {row['id']}")
        syllable = parse_syllable(row["text"])
        if any(row[key] != getattr(syllable, key) for key in ("text", "consonant", "vowel", "tone")):
            raise ValueError(f"audit text/CVT labels disagree: {row['id']}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", row["id"]):
            raise ValueError(f"unsafe recording ID: {row['id']!r}")
        source = (repo_root / row["audio_path"]).resolve()
        if not source.is_relative_to(repo_root):
            raise ValueError(f"recording is outside repository: {row['audio_path']}")
    assigned, split_summary = assign_splits(
        rows, seed=seed, valid_fraction=valid_fraction,
        test_fraction=test_fraction, split_unit=split_unit,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (temp_dir / "audio").mkdir()

        def convert(row: dict) -> dict:
            source = (repo_root / row["audio_path"]).resolve()
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_sha != row["source_sha256"]:
                raise ValueError(f"recording changed since audit: {row['audio_path']}; rerun audit")
            waveform, rate = decode_audio(source, sample_rate=SAMPLE_RATE)
            if rate != SAMPLE_RATE:
                raise ValueError(f"decoder returned unexpected sample rate: {rate}")
            destination = temp_dir / "audio" / f"{row['id']}.wav"
            write_wav(destination, waveform, rate)
            return {
                **{key: row[key] for key in ("id", "text", "consonant", "vowel", "tone", "split")},
                "audio_path": f"audio/{row['id']}.wav",
                "sample_rate": rate, "channels": 1,
                "num_samples": len(waveform), "duration_seconds": len(waveform) / rate,
                "pcm16_saturated_samples": int(np.count_nonzero((waveform < -1) | (waveform > 32767 / 32768))),
                "source_audio_path": row["audio_path"], "source_sha256": source_sha,
                "source_pcm_sha256": row["pcm_sha256"],
                "audio_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "quality_flags": row.get("quality_flags", []),
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            prepared = list(pool.map(convert, assigned))
        for split in ("train", "valid", "test"):
            with (temp_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                for row in prepared:
                    if row["split"] == split:
                        output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        summary = {
            "schema_version": 1, "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "sample_rate": SAMPLE_RATE, "channels": 1, "format": "PCM_16",
            "processing": "ffmpeg mono resampling and PCM16 quantization with saturation; no trimming, denoising, gain normalization, or duration filter",
            "pcm16_saturated_samples": sum(row["pcm16_saturated_samples"] for row in prepared),
            "pcm16_saturated_files": [row["id"] for row in prepared if row["pcm16_saturated_samples"]],
            "split": split_summary,
            "duration_seconds_by_split": {split: round(sum(row["duration_seconds"] for row in prepared
                if row["split"] == split), 6) for split in ("train", "valid", "test")},
            "manifest_sha256": {split: hashlib.sha256((temp_dir / f"{split}.jsonl").read_bytes()).hexdigest()
                for split in ("train", "valid", "test")},
            "interpretation": "Unseen recorded syllable combinations; no evidence about unseen speakers or natural sentence prosody.",
        }
        (temp_dir / "dataset.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.rename(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

"""JSON Schema definitions for every persisted record.

The same documents are mirrored under ``configs/tts/schemas/`` for reviewers;
``tests/tts/test_schemas.py`` fails if the two copies drift. Keeping the
canonical copy importable means the CLI works from an installed package without
assuming a repository layout.
"""

from __future__ import annotations

from typing import Any

SHA256_PATTERN = r"^[0-9a-f]{64}$"
BUCKETS = [
    "ordinary_sentences",
    "targeted_contrasts",
    "questions_numbers_names",
    "continuous_passages",
    "calibration",
]
SPLITS = ["train", "dev", "test"]
STATUSES = [
    "not_started",
    "in_progress",
    "passed",
    "failed",
    "inconclusive",
    "blocked",
    "not_required",
]

UTTERANCE_MANIFEST: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/utterance-manifest.schema.json",
    "title": "RPA TTS accepted-utterance manifest row",
    "description": (
        "One audited, native-reviewed connected-speech utterance. Related audio, passages and "
        "recording sessions share a group_id and are never split across train/dev/test."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "audio_path",
        "audio_sha256",
        "duration_seconds",
        "sample_rate",
        "channels",
        "group_id",
        "bucket",
        "transcript_raw",
        "transcript_normalized",
        "normalization_version",
        "split",
        "rights_id",
        "consent_id",
        "source",
        "reviewer",
        "quality_tags",
        "preprocessing_version",
    ],
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_-]{0,63}$"},
        "audio_path": {"type": "string", "minLength": 1},
        "audio_sha256": {"type": "string", "pattern": SHA256_PATTERN},
        "duration_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 600},
        "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 192000},
        "channels": {"type": "integer", "enum": [1, 2]},
        "group_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "bucket": {"type": "string", "enum": BUCKETS},
        "transcript_raw": {"type": "string", "minLength": 1},
        "transcript_normalized": {"type": "string", "minLength": 1},
        "normalization_version": {"type": "string", "pattern": r"^rpa-norm-[0-9]+$"},
        "expansion_ids": {"type": "array", "items": {"type": "string"}, "default": []},
        "split": {"type": "string", "enum": SPLITS},
        "rights_id": {"type": "string", "minLength": 1},
        "consent_id": {"type": "string", "minLength": 1},
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_id", "speaker_id"],
            "properties": {
                "source_id": {"type": "string", "minLength": 1},
                "speaker_id": {"type": "string", "minLength": 1},
                "edition": {"type": ["string", "null"]},
                "url": {"type": ["string", "null"]},
                "retrieval_date": {"type": ["string", "null"]},
                "dialect": {"type": ["string", "null"]},
                "session_id": {"type": ["string", "null"]},
                "parent_passage_id": {"type": ["string", "null"]},
                "start_seconds": {"type": ["number", "null"], "minimum": 0},
                "end_seconds": {"type": ["number", "null"], "minimum": 0},
            },
        },
        "reviewer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["native_approved"],
            "properties": {
                "native_approved": {"type": "boolean"},
                "reviewer_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                "reviewed_at": {"type": ["string", "null"]},
                "decisions": {"type": "array", "items": {"type": "string"}, "default": []},
                "notes": {"type": ["string", "null"]},
            },
        },
        "quality_tags": {"type": "array", "items": {"type": "string"}},
        "preprocessing_version": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
}

RUN_MANIFEST: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/run-manifest.schema.json",
    "title": "RPA TTS run manifest",
    "description": (
        "Reproducibility record for one experiment. Every later run records input/config/code/"
        "artifact hashes, device, precision, effective batch, seed, unique audio hours, optimizer "
        "updates, wall time, peak VRAM, complete output paths and failure counts."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "run_id",
        "stage",
        "created_at",
        "command",
        "status",
        "code",
        "config",
        "dataset",
        "environment",
        "training",
        "resources",
        "outputs",
        "failures",
        "notes",
    ],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "run_id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]{3,127}$"},
        "stage": {"type": "string", "pattern": r"^F-0[1-9]$"},
        "created_at": {"type": "string", "format": "date-time"},
        "command": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": ["planned", "running", "completed", "failed", "aborted"],
        },
        "code": {
            "type": "object",
            "additionalProperties": False,
            "required": ["commit", "branch", "dirty"],
            "properties": {
                "commit": {"type": ["string", "null"], "pattern": r"^[0-9a-f]{7,40}$"},
                "branch": {"type": ["string", "null"]},
                "dirty": {"type": "boolean"},
            },
        },
        "config": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "sha256"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "sha256": {"type": "string", "pattern": SHA256_PATTERN},
            },
        },
        "dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "manifest_path",
                "manifest_sha256",
                "unique_audio_hours",
                "rows",
                "split_rows",
            ],
            "properties": {
                "manifest_path": {"type": "string", "minLength": 1},
                "manifest_sha256": {"type": "string", "pattern": SHA256_PATTERN},
                "unique_audio_hours": {"type": "number", "minimum": 0},
                "rows": {"type": "integer", "minimum": 0},
                "split_rows": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": SPLITS,
                    "properties": {name: {"type": "integer", "minimum": 0} for name in SPLITS},
                },
            },
        },
        "environment": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "hostname",
                "os",
                "python",
                "device",
                "gpu_name",
                "gpu_memory_mb",
                "driver",
                "torch",
                "cuda",
                "precision",
            ],
            "properties": {
                "hostname": {"type": "string"},
                "os": {"type": "string"},
                "python": {"type": "string"},
                "device": {"type": "string"},
                "gpu_name": {"type": ["string", "null"]},
                "gpu_memory_mb": {"type": ["integer", "null"]},
                "driver": {"type": ["string", "null"]},
                "torch": {"type": ["string", "null"]},
                "cuda": {"type": ["string", "null"]},
                "precision": {"type": "string", "enum": ["fp32", "bf16", "fp16"]},
            },
        },
        "training": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "seed",
                "precision",
                "effective_batch_size",
                "gradient_accumulation",
                "optimizer_updates",
                "learning_rate",
                "max_steps",
            ],
            "properties": {
                "seed": {"type": "integer", "minimum": 0},
                "precision": {"type": "string", "enum": ["fp32", "bf16", "fp16"]},
                "effective_batch_size": {"type": "integer", "minimum": 1},
                "gradient_accumulation": {"type": "integer", "minimum": 1},
                "optimizer_updates": {"type": "integer", "minimum": 0},
                "learning_rate": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "max_steps": {"type": "integer", "minimum": 0},
            },
        },
        "resources": {
            "type": "object",
            "additionalProperties": False,
            "required": ["wall_time_seconds", "peak_vram_mb"],
            "properties": {
                "wall_time_seconds": {"type": "number", "minimum": 0},
                "peak_vram_mb": {"type": ["integer", "null"], "minimum": 0},
            },
        },
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "sha256", "kind"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": SHA256_PATTERN},
                    "kind": {
                        "type": "string",
                        "enum": ["checkpoint", "log", "audio", "metrics", "report", "other"],
                    },
                },
            },
        },
        "failures": {
            "type": "object",
            "additionalProperties": False,
            "required": ["count", "notes"],
            "properties": {
                "count": {"type": "integer", "minimum": 0},
                "notes": {"type": "string"},
            },
        },
        "notes": {"type": "string"},
    },
}

DECISION: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/decision.schema.json",
    "title": "RPA TTS decision record",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "timestamp",
        "stage",
        "title",
        "decision",
        "status",
        "rationale",
        "evidence",
        "owner",
    ],
    "properties": {
        "id": {"type": "string", "pattern": r"^(D-0[1-9]|F-0[1-9]-D[0-9]{2})$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "stage": {"type": "string", "pattern": r"^(D-0[1-9]|F-0[1-9])$"},
        "title": {"type": "string", "minLength": 1},
        "decision": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": ["settled", "open", "deferred", "rejected", "superseded"],
        },
        "rationale": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "owner": {"type": "string", "minLength": 1},
        "supersedes": {"type": ["string", "null"]},
        "notes": {"type": "string"},
    },
}

STAGE_DECISION: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/stage-decision.schema.json",
    "title": "RPA TTS learning-stage decision record",
    "description": (
        "Written before a learning run (issue F-06) so a run cannot start without a declared "
        "objective, baseline, cap, measurement set and next affordable increment."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "timestamp",
        "stage",
        "dataset_version",
        "objective",
        "baseline",
        "max_gpu_hours",
        "inference_policy",
        "native_raters",
        "measurement_set",
        "next_increment",
        "decision",
    ],
    "properties": {
        "id": {"type": "string", "pattern": r"^S-0[1-9][0-9]*$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "stage": {"type": "string", "pattern": r"^F-0[1-9]$"},
        "dataset_version": {"type": "string", "minLength": 1},
        "objective": {"type": "string", "minLength": 1},
        "baseline": {"type": "string", "minLength": 1},
        "max_gpu_hours": {"type": "number", "exclusiveMinimum": 0},
        "max_optimizer_updates": {"type": ["integer", "null"], "minimum": 0},
        "inference_policy": {"type": "string", "minLength": 1},
        "native_raters": {"type": "string", "minLength": 1},
        "measurement_set": {"type": "string", "minLength": 1},
        "next_increment": {"type": "string", "minLength": 1},
        "decision": {"type": "string", "minLength": 1},
        "notes": {"type": "string"},
    },
}

EVIDENCE_ENTRY: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/evidence-entry.schema.json",
    "title": "RPA TTS evidence index entry",
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "gate", "claim", "status", "created_at", "paths", "notes"],
    "properties": {
        "id": {"type": "string", "pattern": r"^ev_[0-9a-f]{16}$"},
        "gate": {"type": "string", "pattern": r"^F-0[1-9]$"},
        "claim": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": STATUSES},
        "created_at": {"type": "string", "format": "date-time"},
        "run_id": {"type": ["string", "null"]},
        "paths": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "sha256"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": SHA256_PATTERN},
                    "description": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

EVIDENCE_INDEX: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/evidence-index.schema.json",
    "title": "RPA TTS evidence index",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "entries"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "entries": {"type": "array", "items": EVIDENCE_ENTRY},
    },
}

LISTENING_PACK: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://rpa.oneoffcoder.com/schemas/listening-pack.schema.json",
    "title": "RPA TTS blinded listening-pack definition",
    "description": (
        "Input definition for a counterbalanced listening pack. One condition is assigned per "
        "listener and script; contrast alternatives appear only in the separately distributed "
        "phase-B sheet."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "pack_id", "design", "seed", "conditions", "items", "listeners"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "pack_id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]{2,63}$"},
        "design": {"type": "string", "enum": ["calibration", "qualification"]},
        "seed": {"type": "integer", "minimum": 0},
        "conditions": {
            "type": "array",
            "minItems": 2,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": r"^[a-z][a-z0-9_-]{1,31}$"},
        },
        "plays": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "short_max": {"type": "integer", "minimum": 1, "maximum": 5},
                "paragraph_extra": {"type": "integer", "minimum": 0, "maximum": 2},
            },
        },
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["script_id", "bucket", "text", "condition_audio"],
                "properties": {
                    "script_id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9_-]{0,63}$"},
                    "bucket": {"type": "string", "enum": BUCKETS},
                    "kind": {"type": "string", "enum": ["sentence", "paragraph"]},
                    "text": {"type": "string", "minLength": 1},
                    "critical": {"type": "boolean"},
                    "contrast": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["options", "expected"],
                        "properties": {
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "expected": {"type": "string", "minLength": 1},
                        },
                    },
                    "condition_audio": {
                        "type": "object",
                        "minProperties": 2,
                        "additionalProperties": {"type": "string", "minLength": 1},
                    },
                    "expansion_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "listeners": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["listener_id", "roles"],
                "properties": {
                    "listener_id": {"type": "string", "pattern": r"^L-[0-9]{2,3}$"},
                    "roles": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": ["transcription", "naturalness", "contrast"],
                        },
                    },
                },
            },
        },
    },
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "utterance-manifest.schema.json": UTTERANCE_MANIFEST,
    "run-manifest.schema.json": RUN_MANIFEST,
    "decision.schema.json": DECISION,
    "stage-decision.schema.json": STAGE_DECISION,
    "evidence-entry.schema.json": EVIDENCE_ENTRY,
    "evidence-index.schema.json": EVIDENCE_INDEX,
    "listening-pack.schema.json": LISTENING_PACK,
}

LEARNING_CURVE_COLUMNS = [
    "run_id",
    "stage",
    "dataset_version",
    "accepted_training_hours",
    "optimizer_updates",
    "dev_syllable_error",
    "dev_naturalness_mean",
    "decision",
    "evidence_id",
]


def validate(instance: object, name: str) -> None:
    """Validate ``instance`` against a named schema, raising ValidationError."""
    import jsonschema

    from .errors import ValidationError

    schema = SCHEMAS[name]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        if len(errors) > 8:
            details += f"; and {len(errors) - 8} more"
        raise ValidationError(f"{name} validation failed: {details}")

"""Blinded, counterbalanced listening packs and response ingestion.

The instrument enforces the protocol decisions from the issue:

* one condition per script per listener (blind transcription/naturalness);
* counterbalanced assignment so each listener and each script sees conditions
  evenly, without the listener seeing the condition or source path;
* contrast alternatives are never on the phase-A sheet; they appear only in a
  separately distributed phase-B sheet, after phase A is submitted;
* every submitted rating is validated against the assignment; missing, extra,
  duplicate or out-of-range responses fail rather than being silently dropped;
* failed/empty transcriptions stay in the pooled denominator.

The pack manifest is the coordinator record. Listener sheets contain only opaque
codes, audio paths and response fields. See `listening-instrument.md`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from ..errors import InputError, ValidationError
from ..hashing import sha256_file, sha256_json
from ..schemas import validate as validate_schema
from ..text.normalize import normalize_rpa

PACK_SCHEMA_VERSION = 1
TIE_TOKENS = {"", "uncertain", "tie", "unsure", "no_response"}
PHASE_A_COLUMNS = ["code", "audio", "transcription", "naturalness", "replay_count", "comments"]
PHASE_B_COLUMNS = ["code", "option_a", "option_b", "chosen", "confidence", "comments"]

# F-07 locked-test composition: 300 sentence items split 100/80/60/60, plus 20
# continuous paragraphs of 30-90 seconds. Twelve listeners are declared and at
# least nine must be eligible for all three roles.
QUALIFICATION_SENTENCE_TOTAL = 300
QUALIFICATION_PARAGRAPH_TOTAL = 20
QUALIFICATION_LISTENERS = 12
QUALIFICATION_ELIGIBLE_ALL_ROLES = 9
QUALIFICATION_MIN_CRITICAL = 40
QUALIFICATION_PARAGRAPH_SECONDS = (30.0, 90.0)
QUALIFICATION_BUCKETS = {
    "ordinary_sentences": 100,
    "targeted_contrasts": 80,
    "questions_numbers_names": 60,
    "continuous_passages": 60,
}
ALL_ROLES = {"transcription", "naturalness", "contrast"}


def qualification_design_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Report F-07 suite composition and every shortfall; no silent acceptance."""
    items = payload["items"]
    sentences = [item for item in items if item.get("kind", "sentence") == "sentence"]
    paragraphs = [item for item in items if item.get("kind") == "paragraph"]
    by_bucket: dict[str, int] = {}
    for item in sentences:
        by_bucket[item["bucket"]] = by_bucket.get(item["bucket"], 0) + 1
    listeners = payload["listeners"]
    eligible_all_roles = sum(1 for listener in listeners if ALL_ROLES <= set(listener["roles"]))
    critical = sum(1 for item in sentences if item.get("critical"))
    targeted_without_contrast = [
        item["script_id"]
        for item in sentences
        if item["bucket"] == "targeted_contrasts" and not item.get("contrast")
    ]
    violations: list[str] = []
    if len(sentences) != QUALIFICATION_SENTENCE_TOTAL:
        violations.append(f"sentences: {len(sentences)} != {QUALIFICATION_SENTENCE_TOTAL}")
    for bucket, required in QUALIFICATION_BUCKETS.items():
        actual = by_bucket.get(bucket, 0)
        if actual != required:
            violations.append(f"bucket {bucket}: {actual} != {required}")
    if len(paragraphs) != QUALIFICATION_PARAGRAPH_TOTAL:
        violations.append(f"paragraphs: {len(paragraphs)} != {QUALIFICATION_PARAGRAPH_TOTAL}")
    for paragraph in paragraphs:
        duration = paragraph.get("duration_seconds")
        if duration is None:
            violations.append(f"paragraph {paragraph['script_id']}: missing duration_seconds")
        elif not (
            QUALIFICATION_PARAGRAPH_SECONDS[0]
            <= float(duration)
            <= QUALIFICATION_PARAGRAPH_SECONDS[1]
        ):
            violations.append(
                f"paragraph {paragraph['script_id']}: duration {duration}s outside "
                f"{QUALIFICATION_PARAGRAPH_SECONDS[0]}-{QUALIFICATION_PARAGRAPH_SECONDS[1]}s"
            )
    if len(listeners) < QUALIFICATION_LISTENERS:
        violations.append(f"listeners: {len(listeners)} < {QUALIFICATION_LISTENERS}")
    if eligible_all_roles < QUALIFICATION_ELIGIBLE_ALL_ROLES:
        violations.append(
            f"listeners eligible for all three roles: {eligible_all_roles} < "
            f"{QUALIFICATION_ELIGIBLE_ALL_ROLES}"
        )
    if critical < QUALIFICATION_MIN_CRITICAL:
        violations.append(f"critical contrast items: {critical} < {QUALIFICATION_MIN_CRITICAL}")
    if targeted_without_contrast:
        violations.append(
            "targeted_contrasts items without contrast options: "
            + ", ".join(sorted(targeted_without_contrast)[:8])
        )
    conditions = [str(condition) for condition in payload["conditions"]]
    if len(conditions) < 3:
        violations.append(f"conditions: {len(conditions)} < 3")
    if "candidate" not in conditions:
        violations.append("conditions must include 'candidate' for gate scoring")
    return {
        "design": payload["design"],
        "passed": not violations,
        "violations": violations,
        "counts": {
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
            "listeners": len(listeners),
            "eligible_all_roles": eligible_all_roles,
            "critical": critical,
            "by_bucket": dict(sorted(by_bucket.items())),
            "conditions": conditions,
        },
        "requirements": {
            "sentences": QUALIFICATION_SENTENCE_TOTAL,
            "paragraphs": QUALIFICATION_PARAGRAPH_TOTAL,
            "listeners": QUALIFICATION_LISTENERS,
            "eligible_all_roles": QUALIFICATION_ELIGIBLE_ALL_ROLES,
            "min_critical": QUALIFICATION_MIN_CRITICAL,
            "paragraph_seconds": list(QUALIFICATION_PARAGRAPH_SECONDS),
            "buckets": dict(sorted(QUALIFICATION_BUCKETS.items())),
        },
    }


def design_check(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["design"] != "qualification":
        return {
            "design": payload["design"],
            "passed": True,
            "applicable": False,
            "note": "Qualification composition requirements apply only to qualification packs.",
        }
    return {"applicable": True, **qualification_design_report(payload)}


def _canonical_definition(data: dict[str, Any]) -> dict[str, Any]:
    # Underscore-prefixed keys are local metadata added by load_definition and
    # are not part of the validated definition payload.
    payload = {key: value for key, value in data.items() if not key.startswith("_")}
    validate_schema(payload, "listening-pack.schema.json")
    conditions = [str(condition) for condition in payload["conditions"]]
    if len(set(conditions)) != len(conditions):
        raise ValidationError("listening pack conditions must be unique")
    listener_ids = [str(listener["listener_id"]) for listener in payload["listeners"]]
    if len(set(listener_ids)) != len(listener_ids):
        raise ValidationError("listening pack listener ids must be unique")
    script_ids = [str(item["script_id"]) for item in payload["items"]]
    if len(set(script_ids)) != len(script_ids):
        raise ValidationError("listening pack script ids must be unique")
    for item in payload["items"]:
        audio = item["condition_audio"]
        missing = sorted(set(conditions) - set(audio))
        extra = sorted(set(audio) - set(conditions))
        if missing or extra:
            raise ValidationError(
                f"item {item['script_id']!r} audio keys must exactly match conditions; "
                f"missing={missing} extra={extra}"
            )
        if item.get("critical") and not item.get("contrast"):
            raise ValidationError(
                f"critical item {item['script_id']!r} must declare contrast options"
            )
        if item.get("contrast"):
            options = item["contrast"]["options"]
            if item["contrast"]["expected"] not in options:
                raise ValidationError(
                    f"item {item['script_id']!r} expected contrast value is not among its options"
                )
        normalized = normalize_rpa(item["text"])
        if normalized.issues:
            details = "; ".join(f"{issue.code}:{issue.token}" for issue in normalized.issues)
            raise ValidationError(
                f"item {item['script_id']!r} text is outside the RPA contract ({details}); "
                "apply approved expansions before defining the pack"
            )
    if payload["design"] == "qualification":
        if any(item["bucket"] == "calibration" for item in payload["items"]):
            raise ValidationError("qualification packs cannot use the calibration bucket")
        if not any(item.get("critical") for item in payload["items"]):
            raise ValidationError(
                "qualification packs must declare at least one critical contrast item"
            )
    return payload


def load_definition(path: str | Path) -> dict[str, Any]:
    definition_path = Path(path)
    if not definition_path.is_file():
        raise InputError(f"listening pack definition does not exist: {definition_path}")
    try:
        data = json.loads(definition_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{definition_path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise InputError("listening pack definition root must be an object")
    _canonical_definition(data)
    data["_definition_sha256"] = sha256_file(definition_path)
    return data


def _blind_code(seed: int, pack_id: str, listener_id: str, script_id: str) -> str:
    digest = hashlib.sha256(f"{seed}\0{pack_id}\0{listener_id}\0{script_id}".encode())
    return digest.hexdigest()[:12].upper()


def _condition_for(conditions: list[str], listener_index: int, item_index: int, offset: int) -> str:
    return conditions[(listener_index + item_index + offset) % len(conditions)]


def _replay_cap(item: dict[str, Any], plays: dict[str, int]) -> int:
    cap = int(plays["short_max"])
    if item.get("kind") == "paragraph":
        cap += int(plays["paragraph_extra"])
    return cap


def build_pack(definition: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Materialize a pack directory: manifest, coordinator sheet, listener sheets."""
    payload = _canonical_definition(definition)
    check = design_check(payload)
    if check.get("applicable") and not check["passed"]:
        raise ValidationError(
            "qualification suite composition failed: " + "; ".join(check["violations"][:8])
        )
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise InputError(f"pack output directory already exists and is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    seed = int(definition["seed"])
    pack_id = str(definition["pack_id"])
    conditions = [str(condition) for condition in definition["conditions"]]
    items = sorted(definition["items"], key=lambda item: item["script_id"])
    listeners = sorted(definition["listeners"], key=lambda listener: listener["listener_id"])
    plays = {"short_max": 2, "paragraph_extra": 1}
    plays.update(definition.get("plays") or {})
    rng = random.Random(seed)

    assignments: list[dict[str, Any]] = []
    used_codes: set[str] = set()
    for listener_index, listener in enumerate(listeners):
        listener_id = str(listener["listener_id"])
        for item_index, item in enumerate(items):
            condition = _condition_for(conditions, listener_index, item_index, 0)
            code = _blind_code(seed, pack_id, listener_id, str(item["script_id"]))
            while code in used_codes:  # pragma: no cover - 48-bit collision guard
                code = hashlib.sha256(f"{code}\0{seed}".encode()).hexdigest()[:12].upper()
            used_codes.add(code)
            source = str(item["condition_audio"][condition])
            assignments.append(
                {
                    "listener_id": listener_id,
                    "script_id": str(item["script_id"]),
                    "condition": condition,
                    "code": code,
                    "source_audio": source,
                    "blind_audio": f"audio/{code}{Path(source).suffix or '.wav'}",
                    "bucket": str(item["bucket"]),
                    "kind": str(item.get("kind", "sentence")),
                    "critical": bool(item.get("critical", False)),
                    "replay_cap": _replay_cap(item, plays),
                }
            )

    phase_b: list[dict[str, Any]] = []
    for assignment in assignments:
        item = next(item for item in items if item["script_id"] == assignment["script_id"])
        contrast = item.get("contrast")
        if not contrast:
            continue
        options = list(contrast["options"])
        rng.shuffle(options)
        phase_b.append(
            {
                "code": assignment["code"],
                "listener_id": assignment["listener_id"],
                "script_id": assignment["script_id"],
                "options": options,
                "expected": str(contrast["expected"]),
                "critical": bool(item.get("critical", False)),
            }
        )

    manifest = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pack_id": pack_id,
        "design": str(definition["design"]),
        "seed": seed,
        "conditions": conditions,
        "plays": plays,
        "created_at": _utc_now(),
        "definition_sha256": definition.get("_definition_sha256"),
        "items": [
            {
                "script_id": str(item["script_id"]),
                "bucket": str(item["bucket"]),
                "kind": str(item.get("kind", "sentence")),
                "text": str(item["text"]),
                "critical": bool(item.get("critical", False)),
                "contrast": item.get("contrast"),
                "condition_audio": item["condition_audio"],
            }
            for item in items
        ],
        "listeners": [
            {
                "listener_id": str(listener["listener_id"]),
                "roles": list(listener["roles"]),
            }
            for listener in listeners
        ],
        "assignments": assignments,
        "contrast_assignments": phase_b,
        "design_check": check,
        "audio_staged": False,
    }
    _write_json(destination / "pack-manifest.json", manifest)
    _write_coordinator_sheet(destination / "coordinator" / "assignments.csv", manifest)
    _write_listener_sheets(destination, manifest)
    _write_pack_readme(destination, manifest)
    summary = pack_summary(manifest)
    _write_json(destination / "pack-summary.json", summary)
    return {"manifest": manifest, "summary": summary, "output": str(destination)}


def _utc_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_coordinator_sheet(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "listener_id",
                "script_id",
                "condition",
                "code",
                "source_audio",
                "bucket",
                "kind",
                "critical",
            ]
        )
        for assignment in manifest["assignments"]:
            writer.writerow(
                [
                    assignment["listener_id"],
                    assignment["script_id"],
                    assignment["condition"],
                    assignment["code"],
                    assignment["source_audio"],
                    assignment["bucket"],
                    assignment["kind"],
                    assignment["critical"],
                ]
            )


def _write_listener_sheets(destination: Path, manifest: dict[str, Any]) -> None:
    items = {item["script_id"]: item for item in manifest["items"]}
    for listener in manifest["listeners"]:
        listener_id = listener["listener_id"]
        rows = [a for a in manifest["assignments"] if a["listener_id"] == listener_id]
        phase_a = destination / "listener-sheets" / f"{listener_id}-phase-a.csv"
        phase_a.parent.mkdir(parents=True, exist_ok=True)
        with phase_a.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(PHASE_A_COLUMNS)
            for row in rows:
                writer.writerow([row["code"], row["blind_audio"], "", "", "0", ""])
        contrast_codes = {
            entry["code"]
            for entry in manifest["contrast_assignments"]
            if entry["listener_id"] == listener_id
        }
        if contrast_codes:
            phase_b = destination / "listener-sheets" / f"{listener_id}-phase-b.csv"
            with phase_b.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(PHASE_B_COLUMNS)
                for entry in manifest["contrast_assignments"]:
                    if entry["listener_id"] != listener_id:
                        continue
                    writer.writerow(
                        [entry["code"], entry["options"][0], entry["options"][1], "", "", ""]
                    )
        coordinator = destination / "listener-sheets" / f"{listener_id}-key.csv"
        with coordinator.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["code", "script_id", "condition", "replay_cap", "text"])
            for row in rows:
                writer.writerow(
                    [
                        row["code"],
                        row["script_id"],
                        row["condition"],
                        row["replay_cap"],
                        items[row["script_id"]]["text"],
                    ]
                )


def _write_pack_readme(destination: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Listening pack `{manifest['pack_id']}`",
        "",
        f"Design: **{manifest['design']}**; seed `{manifest['seed']}`; "
        f"{len(manifest['assignments'])} assignments for {len(manifest['listeners'])} listeners.",
        "",
        "## Distribution rules",
        "",
        "1. The listener receives only their `<listener>-phase-a.csv` and staged blind audio.",
        "2. Phase A collects transcription and naturalness first. Do not show contrast "
        "options yet.",
        "3. After phase A is submitted, give the same listener their `<listener>-phase-b.csv`.",
        "4. `coordinator/assignments.csv` and `<listener>-key.csv` are coordinator-only; "
        "they reveal",
        "   conditions and must not be shared with listeners before scoring is locked.",
        "",
        "## Unknown or empty transcriptions",
        "",
        "Write what you hear. Leave the cell empty only if no speech was produced; "
        "empty outputs are",
        "scored as full deletions and remain catastrophic failures.",
        "",
        "## Audio",
        "",
        "Staged audio is at the `audio/` paths in the sheets. Run",
        "`rpa-tts evaluate pack --stage-audio <root>` (or copy the files) before the session.",
        "",
    ]
    (destination / "PACK-README.md").write_text("\n".join(lines), encoding="utf-8")


def stage_audio(
    pack_dir: str | Path, source_root: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Stage blind audio files named by assignment code.

    Hardlinks are used when possible so repeated audio does not multiply disk use;
    otherwise files are copied. Staged hashes are recorded in the manifest.
    """
    destination = Path(pack_dir)
    manifest_path = destination / "pack-manifest.json"
    if not manifest_path.is_file():
        raise InputError(f"pack manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(source_root)
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for assignment in manifest["assignments"]:
        source = root / assignment["source_audio"]
        if not source.is_file():
            raise InputError(f"source audio does not exist: {source}")
        target = destination / assignment["blind_audio"]
        if target.exists() and not overwrite:
            raise InputError(f"staged audio already exists (use --overwrite): {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        staged.append(
            {
                "code": assignment["code"],
                "source_audio": assignment["source_audio"],
                "blind_audio": assignment["blind_audio"],
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )
    manifest["audio_staged"] = True
    manifest["staged_audio"] = staged
    manifest["staged_audio_sha256"] = sha256_json(staged)
    _write_json(destination / "pack-manifest.json", manifest)
    summary = pack_summary(manifest)
    _write_json(destination / "pack-summary.json", summary)
    return {"manifest": manifest, "summary": summary, "output": str(destination)}


def pack_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    per_listener: dict[str, dict[str, int]] = {}
    per_script: dict[str, dict[str, int]] = {}
    for assignment in manifest["assignments"]:
        condition = assignment["condition"]
        by_condition[condition] = by_condition.get(condition, 0) + 1
        by_bucket[assignment["bucket"]] = by_bucket.get(assignment["bucket"], 0) + 1
        per_listener.setdefault(assignment["listener_id"], {})
        per_listener[assignment["listener_id"]][condition] = (
            per_listener[assignment["listener_id"]].get(condition, 0) + 1
        )
        per_script.setdefault(assignment["script_id"], {})
        per_script[assignment["script_id"]][condition] = (
            per_script[assignment["script_id"]].get(condition, 0) + 1
        )
    warnings = []
    conditions = list(manifest["conditions"])
    for listener_id, counts in sorted(per_listener.items()):
        values = [counts.get(condition, 0) for condition in conditions]
        if values and max(values) - min(values) > 1:
            warnings.append(f"listener {listener_id} condition counts are unbalanced: {counts}")
    for script_id, counts in sorted(per_script.items()):
        values = [counts.get(condition, 0) for condition in conditions]
        if values and max(values) - min(values) > 1:
            warnings.append(f"script {script_id} condition counts are unbalanced: {counts}")
    return {
        "pack_id": manifest["pack_id"],
        "assignments": len(manifest["assignments"]),
        "listeners": len(manifest["listeners"]),
        "items": len(manifest["items"]),
        "contrast_items": len(manifest["contrast_assignments"]),
        "by_condition": dict(sorted(by_condition.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "per_listener": {
            key: dict(sorted(value.items())) for key, value in sorted(per_listener.items())
        },
        "per_script": {
            key: dict(sorted(value.items())) for key, value in sorted(per_script.items())
        },
        "design_check": manifest.get("design_check"),
        "warnings": warnings,
    }


def _read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputError(f"response sheet does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError(f"response sheet has no header: {path}")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise InputError(f"{path} is missing columns: {missing}")
        rows: list[dict[str, str]] = []
        for row in reader:
            if any((value or "").strip() for value in row.values()):
                rows.append({key: (value or "").strip() for key, value in row.items()})
        return rows


def load_responses(pack_dir: str | Path, responses_dir: str | Path) -> dict[str, Any]:
    """Validate completed phase-A/B sheets and build the gate scores payload."""
    pack_directory = Path(pack_dir)
    manifest_path = pack_directory / "pack-manifest.json"
    if not manifest_path.is_file():
        raise InputError(f"pack manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    responses = Path(responses_dir)
    items = {item["script_id"]: item for item in manifest["items"]}
    assignments = {
        (entry["listener_id"], entry["code"]): entry for entry in manifest["assignments"]
    }
    contrast_by_code = {entry["code"]: entry for entry in manifest["contrast_assignments"]}
    listeners = {listener["listener_id"]: listener for listener in manifest["listeners"]}

    transcriptions: list[dict[str, Any]] = []
    naturalness: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    problems: list[str] = []
    empty_outputs = 0

    for listener_id, listener in sorted(listeners.items()):
        phase_a_path = responses / f"{listener_id}-phase-a.csv"
        rows = _read_csv(phase_a_path, ["code", "transcription", "naturalness", "replay_count"])
        expected_codes = {
            entry["code"]
            for entry in manifest["assignments"]
            if entry["listener_id"] == listener_id
        }
        seen: set[str] = set()
        for row in rows:
            code = row["code"]
            if (listener_id, code) not in assignments:
                problems.append(f"{listener_id}: unknown code {code!r}")
                continue
            if code in seen:
                problems.append(f"{listener_id}: duplicate code {code!r}")
                continue
            seen.add(code)
            assignment = assignments[(listener_id, code)]
            item = items[assignment["script_id"]]
            raw_score = row["naturalness"]
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                problems.append(
                    f"{listener_id}/{code}: naturalness {raw_score!r} is not an integer"
                )
                continue
            if not 1 <= score <= 5:
                problems.append(f"{listener_id}/{code}: naturalness {score} is outside 1-5")
                continue
            try:
                replays = int(row["replay_count"] or 0)
            except ValueError:
                problems.append(
                    f"{listener_id}/{code}: replay_count {row['replay_count']!r} is not an integer"
                )
                continue
            if replays < 0 or replays > assignment["replay_cap"]:
                problems.append(
                    f"{listener_id}/{code}: replay_count {replays} "
                    f"exceeds cap {assignment['replay_cap']}"
                )
                continue
            transcription = row["transcription"]
            if not transcription:
                empty_outputs += 1
            role_flags = set(listener["roles"])
            if "transcription" in role_flags:
                transcriptions.append(
                    {
                        "utterance_id": assignment["script_id"],
                        "listener_id": listener_id,
                        "bucket": assignment["bucket"],
                        "reference": item["text"],
                        "hypothesis": transcription,
                        "condition": assignment["condition"],
                        "code": code,
                    }
                )
            if "naturalness" in role_flags:
                naturalness.append(
                    {
                        "utterance_id": assignment["script_id"],
                        "listener_id": listener_id,
                        "bucket": assignment["bucket"],
                        "condition": assignment["condition"],
                        "score": score,
                        "replays": replays,
                        "code": code,
                    }
                )
        missing_codes = sorted(expected_codes - seen)
        if missing_codes:
            problems.append(f"{listener_id}: missing phase-A codes {missing_codes}")

        if "contrast" in set(listener["roles"]):
            phase_b_path = responses / f"{listener_id}-phase-b.csv"
            expected_contrasts = {
                entry["code"]
                for entry in manifest["contrast_assignments"]
                if entry["listener_id"] == listener_id
            }
            if expected_contrasts:
                phase_b_rows = _read_csv(phase_b_path, ["code", "chosen"])
                seen_b: set[str] = set()
                for row in phase_b_rows:
                    code = row["code"]
                    if code not in contrast_by_code:
                        problems.append(f"{listener_id}: unknown contrast code {code!r}")
                        continue
                    if code in seen_b:
                        problems.append(f"{listener_id}: duplicate contrast code {code!r}")
                        continue
                    seen_b.add(code)
                    entry = contrast_by_code[code]
                    chosen = row["chosen"].strip().lower()
                    if (
                        chosen not in {option.lower() for option in entry["options"]}
                        and chosen not in TIE_TOKENS
                    ):
                        problems.append(
                            f"{listener_id}/{code}: chosen {chosen!r} is not one of "
                            f"{entry['options']} or a tie"
                        )
                        continue
                    contrasts.append(
                        {
                            "utterance_id": entry["script_id"],
                            "listener_id": listener_id,
                            "expected": entry["expected"],
                            "chosen": chosen or None,
                            "critical": entry["critical"],
                            "adjudicated": False,
                            "confirmed_by": [],
                            "code": code,
                        }
                    )
                missing_contrasts = sorted(expected_contrasts - seen_b)
                if missing_contrasts:
                    problems.append(f"{listener_id}: missing phase-B codes {missing_contrasts}")

    if problems:
        raise ValidationError("listening responses failed validation: " + "; ".join(problems[:12]))

    scores = {
        "schema_version": PACK_SCHEMA_VERSION,
        "pack_id": manifest["pack_id"],
        "pack_manifest_sha256": sha256_file(manifest_path),
        "transcriptions": transcriptions,
        "naturalness": naturalness,
        "contrasts": contrasts,
        "catastrophic_outputs": empty_outputs,
        "condition_roles": list(manifest["conditions"]),
        "notes": (
            "Phase-A transcription is blind and phase-B contrast follows separately. "
            "Naturalness records retain their condition id; the gate uses the candidate "
            "condition. Adjudication "
            "of critical errors is a separate native step and is not inferred here."
        ),
    }
    return scores

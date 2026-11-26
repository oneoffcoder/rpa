"""Host-specific configuration is data, never a hard-coded assumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import InputError
from .paths import find_repo_root


def _require(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in mapping:
        raise InputError(f"config is missing required key {where}.{key}")
    value = mapping[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        names = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise InputError(f"config key {where}.{key} must be {names}, got {type(value).__name__}")
    return value


def _optional(mapping: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in mapping:
        return None
    return _require(mapping, key, kind, where)


def _check_fraction(value: float, where: str) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise InputError(f"config key {where} must be between 0 and 1, got {value}")
    return float(value)


@dataclass(frozen=True)
class TrainingLane:
    name: str
    enabled: bool
    priority: int
    status: str
    upstream: dict[str, Any]
    artifacts: dict[str, Any]

    @property
    def ready_for_gate(self) -> bool:
        return self.status in {"in_progress", "gate_passed"}


@dataclass(frozen=True)
class Config:
    """Validated view of ``configs/tts/default.yaml``.

    The raw mapping is retained so new keys can be read without schema churn;
    the properties below are the fields code depends on.
    """

    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Config:
        resolved = Path(path).expanduser().resolve()
        try:
            data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise InputError(f"config file does not exist: {resolved}") from None
        except yaml.YAMLError as exc:
            raise InputError(f"config is not valid YAML: {resolved}: {exc}") from None
        if not isinstance(data, dict):
            raise InputError(f"config root must be a mapping: {resolved}")
        _validate(data, resolved)
        return cls(path=resolved, data=data)

    @property
    def schema_version(self) -> int:
        return int(self.data["schema_version"])

    @property
    def seed(self) -> int:
        return int(self.data["project"]["seed"])

    @property
    def device(self) -> str:
        return str(self.data["host"]["device"])

    @property
    def precision(self) -> str:
        return str(self.data["host"]["precision"])

    @property
    def workspace_config(self) -> str:
        return str(self.data["host"]["roots"]["workspace"])

    @property
    def model_sample_rate(self) -> int:
        return int(self.data["audio"]["model_sample_rate"])

    @property
    def dev_fraction(self) -> float:
        return float(self.data["splits"]["dev_fraction"])

    @property
    def test_fraction(self) -> float:
        return float(self.data["splits"]["test_fraction"])

    @property
    def bucket_field(self) -> str:
        return str(self.data["splits"]["bucket_field"])

    @property
    def group_field(self) -> str:
        return str(self.data["splits"]["group_field"])

    @property
    def lanes(self) -> dict[str, TrainingLane]:
        result: dict[str, TrainingLane] = {}
        for name, lane in self.data["model_lanes"].items():
            result[name] = TrainingLane(
                name=name,
                enabled=bool(lane["enabled"]),
                priority=int(lane["priority"]),
                status=str(lane["status"]),
                upstream=dict(lane.get("upstream", {})),
                artifacts=dict(lane.get("artifacts", {})),
            )
        return result

    @property
    def gate_thresholds(self) -> dict[str, float]:
        return dict(self.data["gates"]["f07"])

    def to_dict(self) -> dict[str, Any]:
        return self.data


def _validate(data: dict[str, Any], path: Path) -> None:
    allowed_top = {
        "schema_version",
        "project",
        "host",
        "audio",
        "splits",
        "coverage",
        "model_lanes",
        "gates",
    }
    unknown = set(data) - allowed_top
    if unknown:
        raise InputError(f"config has unknown top-level keys: {sorted(unknown)}")
    if _require(data, "schema_version", int, "config") != 1:
        raise InputError(f"config schema_version must be 1 in {path}")

    project = _require(data, "project", dict, "config")
    _require(project, "name", str, "project")
    seed = _require(project, "seed", int, "project")
    if seed < 0:
        raise InputError("project.seed must be non-negative")

    host = _require(data, "host", dict, "config")
    _require(host, "profile", str, "host")
    device = _require(host, "device", str, "host")
    if not (device == "cpu" or device.startswith("cuda")):
        raise InputError("host.device must be 'cpu' or a cuda device such as 'cuda:0'")
    precision = _require(host, "precision", str, "host")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise InputError("host.precision must be one of fp32, bf16, fp16")
    roots = _require(host, "roots", dict, "host")
    _require(roots, "workspace", str, "host.roots")
    _require(roots, "artifacts", str, "host.roots")

    audio = _require(data, "audio", dict, "config")
    for key in ("master_sample_rate", "model_sample_rate"):
        if _require(audio, key, int, "audio") <= 0:
            raise InputError(f"audio.{key} must be positive")
    if _require(audio, "channels", int, "audio") not in (1, 2):
        raise InputError("audio.channels must be 1 or 2")
    _require(audio, "master_bit_depth", int, "audio")

    splits = _require(data, "splits", dict, "config")
    dev = _check_fraction(
        _require(splits, "dev_fraction", (int, float), "splits"), "splits.dev_fraction"
    )
    test = _check_fraction(
        _require(splits, "test_fraction", (int, float), "splits"), "splits.test_fraction"
    )
    if dev + test >= 1.0:
        raise InputError("splits.dev_fraction + splits.test_fraction must be below 1")
    if _require(splits, "bucket_field", str, "splits") != "bucket":
        raise InputError("splits.bucket_field must be 'bucket'")
    if _require(splits, "group_field", str, "splits") != "group_id":
        raise InputError("splits.group_field must be 'group_id'")
    _require(splits, "buckets", list, "splits")

    coverage = _require(data, "coverage", dict, "config")
    if _require(coverage, "min_examples_per_attested_cell", int, "coverage") < 1:
        raise InputError("coverage.min_examples_per_attested_cell must be at least 1")

    lanes = _require(data, "model_lanes", dict, "config")
    if not lanes:
        raise InputError("model_lanes must declare at least one lane")
    for name, lane in lanes.items():
        where = f"model_lanes.{name}"
        if not isinstance(lane, dict):
            raise InputError(f"{where} must be a mapping")
        _require(lane, "enabled", bool, where)
        _require(lane, "priority", int, where)
        status = _require(lane, "status", str, where)
        if status not in {"not_started", "standby", "in_progress", "gate_passed", "failed"}:
            raise InputError(f"{where}.status is not a recognized lane status: {status!r}")
        _require(lane, "upstream", dict, where)
        _require(lane, "artifacts", dict, where)

    gates = _require(data, "gates", dict, "config")
    for gate in ("f06", "f07"):
        _require(gates, gate, dict, "gates")
    f07 = gates["f07"]
    for key in (
        "pooled_syllable_error_max",
        "bucket_syllable_error_max",
        "contrast_accuracy_min",
        "naturalness_mean_min",
        "naturalness_lower_bound_min",
        "naturalness_gap_max",
    ):
        if not 0 <= float(_require(f07, key, (int, float), "gates.f07")) <= 1e9:
            raise InputError(f"gates.f07.{key} must be a non-negative number")
    required_buckets = _require(f07, "required_buckets", list, "gates.f07")
    if not required_buckets or not all(isinstance(bucket, str) for bucket in required_buckets):
        raise InputError("gates.f07.required_buckets must be a non-empty list of bucket names")
    declared_buckets = set(splits["buckets"])
    unknown_buckets = sorted(set(required_buckets) - declared_buckets)
    if unknown_buckets:
        raise InputError(
            f"gates.f07.required_buckets contains undeclared buckets: {unknown_buckets}"
        )
    f06 = gates["f06"]
    _require(f06, "bootstrap_resamples", int, "gates.f06")
    _require(f06, "min_relative_syllable_error_reduction", (int, float), "gates.f06")
    _require(f06, "min_naturalness_gain", (int, float), "gates.f06")


def load_config(path: str | Path | None = None) -> Config:
    from .paths import default_config_path

    return Config.load(path if path is not None else default_config_path(find_repo_root()))


__all__ = ["Config", "TrainingLane", "load_config"]

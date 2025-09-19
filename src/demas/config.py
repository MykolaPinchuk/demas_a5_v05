from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml


DATASET_LOCK_PATH = Path("config/dataset.lock")
INSTANCES_I0_PATH = Path("config/instances_i0.yaml")
DEFAULT_DATASET_DIR = Path("data/swe_bench_lite/dev")
RUNS_ROOT = Path("runs")


@dataclass
class DatasetLock:
    dataset: str
    revision: str
    split: str
    retrieved_at: datetime

    @classmethod
    def from_dict(cls, raw: dict) -> "DatasetLock":
        try:
            timestamp = datetime.fromisoformat(raw["retrieved_at"])
        except (KeyError, ValueError) as exc:
            raise ValueError("dataset lock requires valid 'retrieved_at'") from exc
        return cls(
            dataset=str(raw["dataset"]),
            revision=str(raw["revision"]),
            split=str(raw.get("split", "dev")),
            retrieved_at=timestamp,
        )

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "split": self.split,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass
class InstancesConfig:
    instances: List[str]

    @classmethod
    def load(cls, path: Path = INSTANCES_I0_PATH) -> "InstancesConfig":
        if not path.exists():
            raise FileNotFoundError(f"instances file missing: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict) or "instances" not in data:
            raise ValueError(f"instances file malformed: {path}")
        instances_raw = data.get("instances")
        if not isinstance(instances_raw, list):
            raise ValueError("instances entry must be a list")
        instances = [str(item) for item in instances_raw if str(item).strip()]
        if not instances:
            raise ValueError("instances list cannot be empty")
        return cls(instances=instances)


def read_dataset_lock(path: Path = DATASET_LOCK_PATH) -> Optional[DatasetLock]:
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid dataset lock YAML at {path}") from exc
    if not data:
        return None
    return DatasetLock.from_dict(data)


def write_dataset_lock(lock: DatasetLock, path: Path = DATASET_LOCK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(lock.to_dict(), sort_keys=True))


__all__ = [
    "DATASET_LOCK_PATH",
    "DEFAULT_DATASET_DIR",
    "DatasetLock",
    "InstancesConfig",
    "RUNS_ROOT",
    "INSTANCES_I0_PATH",
    "read_dataset_lock",
    "write_dataset_lock",
]

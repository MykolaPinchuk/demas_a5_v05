from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset, load_from_disk

from .agent import AgentConfig, InstanceResult, load_dataset as load_cached_dataset, run_instance
from .config import DEFAULT_DATASET_DIR

LOGGER = logging.getLogger(__name__)


def ensure_dataset_cached(
    dataset_dir: Path,
    *,
    hf_dataset: str,
    split: str,
) -> Optional[str]:
    """Ensure the SWE-bench Lite dataset split is available locally.

    Returns the dataset fingerprint (a stable hash provided by `datasets`).
    """

    dataset_path = dataset_dir if dataset_dir.is_absolute() else dataset_dir.resolve()
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
        fingerprint = getattr(dataset, "_fingerprint", None)
        LOGGER.info("Dataset already cached at %s (fingerprint=%s)", dataset_path, fingerprint)
        return fingerprint

    LOGGER.info("Downloading dataset %s split=%s", hf_dataset, split)
    dataset = load_dataset(hf_dataset, split=split)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(dataset_path))
    fingerprint = getattr(dataset, "_fingerprint", None)
    LOGGER.info("Saved dataset to %s (fingerprint=%s)", dataset_path, fingerprint)
    return fingerprint


def run_batch(
    instances: Iterable[str],
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    provider: str,
    model: str,
    runs_root: Path,
    timeout_seconds: float,
    attempt_limit: int,
    dry_run: bool = False,
) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = f"i0_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = runs_root / run_id
    batch_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_cached_dataset(dataset_dir)
    fingerprint = getattr(dataset, "_fingerprint", None)

    instance_ids = list(instances)
    LOGGER.info("Starting i0 batch %s with %d instances", run_id, len(instance_ids))

    config = AgentConfig(
        dataset_dir=dataset_dir,
        provider=provider,
        model=model,
        runs_root=batch_dir,
        timeout_seconds=timeout_seconds,
        attempt_limit=attempt_limit,
        dry_run=dry_run,
    )

    results: list[InstanceResult] = []
    for instance_id in instance_ids:
        LOGGER.info("Running instance %s", instance_id)
        try:
            result = run_instance(config=config, dataset=dataset, instance_id=instance_id)
        except Exception as exc:  # noqa: BLE001 we want to capture unexpected errors
            LOGGER.exception("Instance %s crashed", instance_id)
            now = datetime.utcnow()
            result = InstanceResult(
                instance_id=instance_id,
                status="crash",
                attempts=0,
                success=False,
                run_dir=batch_dir / instance_id,
                started_at=now,
                finished_at=now,
                error=str(exc),
            )
        results.append(result)

    manifest = {
        "run_id": run_id,
        "iteration": "i0",
        "provider": provider,
        "model": model,
        "dataset_dir": str(dataset_dir),
        "dataset_fingerprint": fingerprint,
        "started_at": results[0].started_at.isoformat() if results else datetime.utcnow().isoformat(),
        "finished_at": datetime.utcnow().isoformat(),
        "attempt_limit": attempt_limit,
        "timeout_seconds": timeout_seconds,
        "dry_run": dry_run,
        "instances": [item.to_manifest() for item in results],
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    predictions_path = batch_dir / "predictions.jsonl"
    with predictions_path.open("w") as handle:
        for result in results:
            record = {
                "instance_id": result.instance_id,
                "status": result.status,
                "success": result.success,
            }
            handle.write(json.dumps(record) + "\n")

    LOGGER.info("Completed i0 batch %s", run_id)

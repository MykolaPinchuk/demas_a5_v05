from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import (
    DEFAULT_DATASET_DIR,
    INSTANCES_I0_PATH,
    RUNS_ROOT,
    DatasetLock,
    InstancesConfig,
    read_dataset_lock,
    write_dataset_lock,
)
from .i0 import ensure_dataset_cached, run_batch


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEMAS iteration i0 runner")
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier to evaluate (provider-specific)",
    )
    parser.add_argument(
        "--provider",
        default="chutes",
        help="LLM provider name (e.g., chutes, openrouter)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory where the SWE-bench Lite dataset is cached",
    )
    parser.add_argument(
        "--hf-dataset",
        default="SWE-bench/SWE-bench_Lite",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Maximum attempts per instance",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each agent attempt (seconds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare run directories without contacting the LLM",
    )
    parser.add_argument(
        "--runs-root",
        default=str(RUNS_ROOT),
        help="Directory where run artifacts will be stored",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level",
    )
    return parser.parse_args(argv)


def command_i0(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    dataset_dir = Path(args.dataset_dir)
    runs_root = Path(args.runs_root)

    instances_cfg = InstancesConfig.load(INSTANCES_I0_PATH)
    dataset_revision = ensure_dataset_cached(
        dataset_dir,
        hf_dataset=args.hf_dataset,
        split=args.split,
    )

    if dataset_revision:
        existing_lock = read_dataset_lock()
        if existing_lock is None or existing_lock.revision != dataset_revision:
            new_lock = DatasetLock(
                dataset=args.hf_dataset,
                revision=dataset_revision,
                split=args.split,
                retrieved_at=datetime.utcnow(),
            )
            write_dataset_lock(new_lock)

    run_batch(
        instances_cfg.instances,
        dataset_dir=dataset_dir,
        provider=args.provider,
        model=args.model,
        runs_root=runs_root,
        timeout_seconds=args.timeout_seconds,
        attempt_limit=args.attempts,
        dry_run=args.dry_run,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    command_i0(args)


def i0_entrypoint() -> None:
    main()


if __name__ == "__main__":
    main()

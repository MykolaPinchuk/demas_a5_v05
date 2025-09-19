from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    def __init__(self, name: str, result: CommandResult):
        super().__init__(f"command '{name}' failed with code {result.returncode}")
        self.name = name
        self.result = result


def run_command(
    name: str,
    argv: List[str],
    *,
    cwd: Path,
    timeout: float,
    expect_ok: bool = True,
    log_path: Path,
) -> CommandResult:
    cwd_path = cwd if cwd.is_absolute() else cwd.resolve()
    completed = subprocess.run(
        argv,
        cwd=str(cwd_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(argv)}\n\n{completed.stdout}\n--- STDERR ---\n{completed.stderr}"
    )
    result = CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if expect_ok and completed.returncode != 0:
        raise CommandError(name, result)
    return result


__all__ = ["CommandError", "CommandResult", "run_command"]

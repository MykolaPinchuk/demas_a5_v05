from __future__ import annotations

from pathlib import Path
from typing import Dict

CREDENTIALS_PATH = Path("secrets/credentials.txt")


def load_credentials(path: Path = CREDENTIALS_PATH) -> Dict[str, str]:
    if not path.exists():
        return {}
    creds: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        creds[key.strip()] = value.strip()
    return creds


__all__ = ["load_credentials", "CREDENTIALS_PATH"]

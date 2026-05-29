"""Configuration and path helpers.

Loads ``config/config.yaml`` and any environment variables from a local
``.env`` file (which is gitignored). All other modules import paths and
settings from here so there are no hard-coded paths scattered around the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo root = two levels up from this file (src/power_fv/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML config and populate environment variables from .env.

    Parameters
    ----------
    path:
        Path to the YAML config, relative to the repo root or absolute.

    Returns
    -------
    dict
        Parsed configuration with absolute ``raw_dir`` / ``processed_dir``.
    """
    load_dotenv(ROOT / ".env")  # no-op if the file is absent

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path

    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Resolve data directories to absolute paths and ensure they exist.
    data = cfg.setdefault("data", {})
    for key in ("raw_dir", "processed_dir"):
        rel = data.get(key, f"data/{key.split('_')[0]}")
        abs_dir = (ROOT / rel).resolve()
        abs_dir.mkdir(parents=True, exist_ok=True)
        data[key] = str(abs_dir)

    return cfg

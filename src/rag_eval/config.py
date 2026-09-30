"""Config loading with dotted-key overrides and hashing.

Design:
- One YAML file holds every knob. An experiment is base config + overrides,
  so each run differs from the baseline in exactly the variables you name.
- Unknown override keys raise an error: a typo like "chunking.chunksize"
  must never silently fall back to the baseline value.
- config_hash() fingerprints a subset of sections so expensive artifacts
  (e.g. a FAISS index depends only on chunking + embedding) can be cached
  and reused across experiments that share those settings.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"

# Loads .env if present. Existing environment variables (e.g. Codespaces
# secrets) take precedence because override=False.
load_dotenv(REPO_ROOT / ".env", override=False)


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section '{k}' in override '{dotted_key}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[keys[-1]] = value


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """Load YAML config and apply {'section.key': value} overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    return cfg


def config_hash(cfg: dict, sections: list[str] | None = None, length: int = 10) -> str:
    """Stable short hash of the whole config or of selected sections."""
    subset = {s: cfg[s] for s in sections} if sections else cfg
    blob = json.dumps(subset, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def resolve_path(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key]; creates the directory if missing."""
    p = REPO_ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p

"""Paths and environment config.

Override via environment variables:
  SABER_DATA  -- root for all data and artefact directories
                 (default: ``./data`` relative to the repo root)
  HF_HOME     -- Hugging Face cache root (standard transformers var)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("SABER_DATA", REPO_ROOT / "data"))
DATASETS_DIR = DATA_ROOT / "datasets" if DATA_ROOT.name != "datasets" else DATA_ROOT

LOGS_DIR = REPO_ROOT / "logs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def ensure_dirs() -> None:
    for p in (DATASETS_DIR, LOGS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR):
        p.mkdir(parents=True, exist_ok=True)

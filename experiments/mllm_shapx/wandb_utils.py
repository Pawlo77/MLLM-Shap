"""Thin wrappers around Weights & Biases to keep imports isolated."""
from __future__ import annotations

import os
import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from .config import WandBConfig
from .constants import WandbMode


def wandb_init_if_enabled(cfg: WandBConfig, run_name: str, run_config: Dict[str, Any]) -> Optional[Any]:
    """Initialize a W&B run if enabled; return the run object or None."""
    if not cfg.enabled or cfg.mode == WandbMode.DISABLED.value:
        return None
    if cfg.mode:
        os.environ["WANDB_MODE"] = cfg.mode

    wb = importlib.import_module("wandb")
    init = getattr(wb, "init")
    return init(
        project=cfg.project,
        entity=cfg.entity,
        group=cfg.group,
        name=run_name,
        tags=cfg.tags,
        config=run_config,
    )


def wandb_log_artifact(
    run: Optional[Any],
    local_path: Path,
    artifact_name: str,
    artifact_type: str,
    metadata: Dict[str, Any],
) -> None:
    """Upload a file as an artifact to the given run, if any."""
    if run is None:
        return

    wb = importlib.import_module("wandb")
    artifact = getattr(wb, "Artifact")
    art = artifact(name=artifact_name, type=artifact_type, metadata=metadata)
    art.add_file(str(local_path))
    run.log_artifact(art)


def log_metrics(run: Optional[Any], metrics: Dict[str, Any]) -> None:
    """Log a metrics dict to W&B if a run object is present."""
    if run is None:
        return
    wb = importlib.import_module("wandb")
    log = getattr(wb, "log")
    log(metrics)

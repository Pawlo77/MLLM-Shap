"""Filesystem helpers for runs, checkpoints, and JSON I/O."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch

from .constants import CHECKPOINT_VERSION

LOGGER = logging.getLogger(__name__)


def make_run_dir(output_root: str, experiment_set_id: str, run_slug: str) -> Path:
    """Create output directories for a run and return the run directory path."""
    d = Path(output_root) / experiment_set_id / run_slug
    (d / "samples").mkdir(parents=True, exist_ok=True)
    (d / "summary").mkdir(parents=True, exist_ok=True)
    return d


def _json_default(o: Any) -> Any:
    """JSON serializer for numpy/torch and raw bytes."""
    if isinstance(o, (bytes, bytearray, memoryview)):
        return {"_binary": True, "num_bytes": len(o)}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return repr(o)


def save_json(path: Path, obj: Any) -> None:
    """Save an object as pretty-printed JSON to the given path (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp_path, path)


def _default_checkpoint() -> Dict[str, Any]:
    """Create a new default checkpoint dict."""
    return {
        "version": CHECKPOINT_VERSION,
        "completed_indices": [],
        "next_index": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a run checkpoint from disk, or return a default if none exists."""
    if not path.exists():
        return _default_checkpoint()
    try:
        with open(path, "r", encoding="utf-8") as f:
            ckpt = cast(Dict[str, Any], json.load(f))

        # Version migration
        stored_version = ckpt.get("version", 1)
        if stored_version < CHECKPOINT_VERSION:
            LOGGER.info(
                "Migrating checkpoint from version %d to %d.",
                stored_version,
                CHECKPOINT_VERSION,
            )
            ckpt["version"] = CHECKPOINT_VERSION
            # Future migrations can be added here

        return ckpt
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring malformed checkpoint file: %s", path)
        return _default_checkpoint()


def update_checkpoint(
    path: Path,
    ckpt: Dict[str, Any],
    just_completed: int | None = None,
    next_index: int | None = None,
) -> None:
    """Update and save the checkpoint with new progress information."""
    if just_completed is not None and just_completed not in ckpt["completed_indices"]:
        ckpt["completed_indices"].append(just_completed)
    if next_index is not None:
        ckpt["next_index"] = next_index
    ckpt["updated_at"] = time.time()
    ckpt["version"] = CHECKPOINT_VERSION
    save_json(path, ckpt)


def existing_completed_from_disk(run_dir: Path) -> set[int]:
    """Infer completed rows by scanning samples/ files."""
    done: set[int] = set()
    for p in (run_dir / "samples").glob("sample_*_result.json"):
        try:
            num = int(p.stem.split("_")[1])
            done.add(num)
        except (ValueError, IndexError):
            LOGGER.debug("Ignoring filename that does not match pattern: %s", p.name)
    return done

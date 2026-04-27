"""Filesystem helpers for runs, checkpoints, and JSON I/O."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch


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
    """Save an object as pretty-printed JSON to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a run checkpoint from disk, or return a default if none exists."""
    if not path.exists():
        return {
            "completed_indices": [],
            "next_index": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    with open(path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


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
    save_json(path, ckpt)


def existing_completed_from_disk(run_dir: Path) -> set[int]:
    """Infer completed rows by scanning samples/ files. Filenames follow: sample_00012_result.json"""
    done: set[int] = set()
    for p in (run_dir / "samples").glob("sample_*_result.json"):
        try:
            num = int(p.stem.split("_")[1])
            done.add(num)
        except (ValueError, IndexError):
            logging.getLogger(__name__).debug(
                "Ignoring filename that does not match pattern: %s", p.name
            )
    return done

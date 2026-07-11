"""Incremental analysis-style snapshot over completed sample JSON files."""

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def build_snapshot(samples_dir: Path) -> Dict[str, Any]:
    """Aggregate lightweight stats from ``sample_*_result.json`` files."""
    paths = sorted(samples_dir.glob("sample_*_result.json"))
    if not paths:
        return {"n_samples": 0}
    runtimes: List[float] = []
    frac_text: List[float] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runtimes.append(float(data.get("runtime_sec", 0.0)))
        attr = data.get("attr_summary") or {}
        frac_text.append(float(attr.get("frac_text", 0.0)))
    return {
        "n_samples": len(paths),
        "avg_runtime_sec": mean(runtimes) if runtimes else 0.0,
        "avg_frac_text": mean(frac_text) if frac_text else 0.0,
    }


def flatten_snapshot_metrics(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert snapshot stats to a flat dict of numeric metrics for logging (e.g., to MLflow).
    Non-numeric values are ignored, and keys are prefixed with 'analysis/'.
    """
    out: Dict[str, float] = {}
    for k, v in snapshot.items():
        if isinstance(v, (int, float)) and k != "n_samples":
            out[f"analysis/{k}"] = float(v)
        elif k == "n_samples":
            out["analysis/n_samples"] = float(v)
    return out

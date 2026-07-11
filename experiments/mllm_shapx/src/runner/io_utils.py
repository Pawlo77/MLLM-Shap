"""Runner serialization and telemetry utility functions."""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from mllm_shap.shap.core.telemetry import TelemetryData

LOGGER = logging.getLogger(__name__)


def _safe_primitive(value: Any) -> Any:
    """Convert values into JSON-safe primitives for run artifacts."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"_binary": True, "num_bytes": len(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def serialize_conversation(conv: Any) -> List[List[Dict[str, Any]]]:
    """Serialize model conversation turns into JSON-safe nested structures."""
    result: List[List[Dict[str, Any]]] = []
    for turn in conv:
        out_turn: List[Dict[str, Any]] = []
        for entry in turn:
            content = _safe_primitive(getattr(entry, "content", None))
            shap_vals = getattr(entry, "shap_values", None)
            if shap_vals is not None:
                shap_vals = [
                    None
                    if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                    else float(v)
                    for v in shap_vals
                ]
            out_turn.append({
                "content_type": _safe_primitive(getattr(entry, "content_type", None)),
                "roles": _safe_primitive(getattr(entry, "roles", None)),
                "content": content,
                "shap_values": shap_vals,
            })
        result.append(out_turn)
    return result


def compute_modality_summary(conv: Any) -> Dict[str, Any]:
    """Compute modality attribution aggregates from serialized SHAP values."""
    modality_abs_sum = {"text": 0.0, "audio": 0.0}
    modality_counts = {"text": 0, "audio": 0}

    for turn in conv:
        for entry in turn:
            shap_vals = getattr(entry, "shap_values", None)
            if shap_vals is None:
                continue
            modality = "text" if getattr(entry, "content_type", None) == 0 else "audio"
            arr = np.array(
                [v for v in shap_vals if not (isinstance(v, float) and np.isnan(v))],
                dtype=float,
            )
            modality_abs_sum[modality] += float(np.abs(arr).sum())
            modality_counts[modality] += int(arr.size)

    total = modality_abs_sum["text"] + modality_abs_sum["audio"]
    return {
        "abs_sum_text": modality_abs_sum["text"],
        "abs_sum_audio": modality_abs_sum["audio"],
        "frac_text": (modality_abs_sum["text"] / total) if total > 0 else 0.0,
        "frac_audio": (modality_abs_sum["audio"] / total) if total > 0 else 0.0,
        "count_text_tokens": modality_counts["text"],
        "count_audio_segments": modality_counts["audio"],
    }


def flatten_telemetry_metrics(data: TelemetryData | None) -> Dict[str, float]:
    """Flatten structured telemetry into metric keys suitable for MLflow."""
    if data is None:
        return {}
    raw = data.to_dict()
    out: Dict[str, float] = {}
    for section in ("cache", "masks", "timing"):
        block = raw.get(section) or {}
        for key, value in block.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"shap/{section}/{key}"] = float(value)
    for key, value in (raw.get("custom") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"shap/custom/{key}"] = float(value)
    return out

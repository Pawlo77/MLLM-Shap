"""Conversation serialization and summarization helpers."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def _safe_primitive(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return {"_binary": True, "num_bytes": len(x)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def serialize_conversation(conv: Any) -> List[List[Dict[str, Any]]]:
    """
    Convert model conversation to JSON-safe nested lists with sanitized SHAP values.
    """
    result: List[List[Dict[str, Any]]] = []
    for turn in conv:
        out_turn: List[Dict[str, Any]] = []
        for entry in turn:
            content = _safe_primitive(getattr(entry, "content", None))
            shap_vals = getattr(entry, "shap_values", None)
            if shap_vals is not None:
                shap_vals = [
                    None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else float(v)
                    for v in shap_vals
                ]
            out_turn.append(
                {
                    "content_type": _safe_primitive(getattr(entry, "content_type", None)),
                    "roles": _safe_primitive(getattr(entry, "roles", None)),
                    "content": content,
                    "shap_values": shap_vals,
                }
            )
        result.append(out_turn)
    return result


def compute_modality_summary(conv: Any) -> Dict[str, Any]:
    """
    Aggregate absolute contribution per modality + element counts.
    """
    modality_abs_sum = {"text": 0.0, "audio": 0.0}
    modality_counts = {"text": 0, "audio": 0}

    for turn in conv:
        for entry in turn:
            shap_vals = getattr(entry, "shap_values", None)
            if shap_vals is None:
                continue
            modality = "text" if getattr(entry, "content_type", None) == 0 else "audio"
            arr = np.array([v for v in shap_vals if not (isinstance(v, float) and np.isnan(v))], dtype=float)
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

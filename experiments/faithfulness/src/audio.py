"""Audio segment manipulation for faithfulness evaluation."""

import math
from typing import Any

import numpy as np
import torch
from mllm_shap.connectors.base.audio import AudioSegment


def extract_audio_sv(sample_json: dict[str, Any]) -> list[float]:
    """Extract finite audio SHAP values from a saved sample JSON payload."""
    for turn in sample_json.get("conversation", []):
        for entry in turn:
            if entry.get("content_type") != 1:
                continue
            values: list[float] = []
            for value in entry.get("shap_values") or []:
                if value is None:
                    continue
                value_f = float(value)
                if math.isfinite(value_f):
                    values.append(value_f)
            if values:
                return values
    raise ValueError("No audio SHAP values found in sample JSON.")


def aggregate_sv_to_segments(
    sv_values: list[float],
    segments: list[AudioSegment],
    total_samples: int,
) -> tuple[list[float], list[tuple[int, int]]]:
    """Map per-codec-token SVs to word segments using actual temporal alignment.

    Each codec token covers a fixed duration (total_samples / n_tokens samples).
    For each word segment [start_sample, end_sample], we compute which token
    indices fall within that range and average their SVs.

    Uses mean aggregation (not sum) to avoid biasing toward longer segments.
    """
    segment_count = len(segments)
    if segment_count <= 0:
        raise ValueError("segment_count must be positive.")
    if not sv_values:
        raise ValueError("No audio SHAP values found.")

    n_tokens = len(sv_values)
    if n_tokens == segment_count:
        return list(sv_values), [(i, i + 1) for i in range(segment_count)]

    values = np.asarray(sv_values, dtype=float)
    hop_size = total_samples / n_tokens

    aggregated: list[float] = []
    bins: list[tuple[int, int]] = []
    for seg in segments:
        start_sample = seg.start_sample if seg.start_sample is not None else 0
        end_sample = seg.end_sample if seg.end_sample is not None else total_samples
        start_token = int(start_sample / hop_size)
        end_token = int(np.ceil(end_sample / hop_size))
        start_token = max(0, min(start_token, n_tokens))
        end_token = max(start_token + 1, min(end_token, n_tokens))
        token_slice = values[start_token:end_token]
        aggregated.append(float(token_slice.mean()))
        bins.append((start_token, end_token))
    return aggregated, bins


def remove_interval(waveform: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Remove an interval from the waveform by concatenating the parts before and after.

    This matches the SHAP computation's masking paradigm (segment removal via
    concatenation) rather than silence insertion.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    start = max(0, min(int(start), waveform.size(-1)))
    end = max(start, min(int(end), waveform.size(-1)))
    return torch.cat([waveform[:, :start], waveform[:, end:]], dim=1)


def segment_interval(seg: AudioSegment) -> tuple[int, int]:
    """Return segment [start, end] sample indices, validating availability."""
    if seg.start_sample is None or seg.end_sample is None:
        raise ValueError("Segment is missing sample indices.")
    return int(seg.start_sample), int(seg.end_sample)

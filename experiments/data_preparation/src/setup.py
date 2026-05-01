"""Shared notebook setup: device detection, model initialization."""

from __future__ import annotations

import torch

from mllm_shap.connectors import LiquidAudio
from mllm_shap.connectors.enums import ModelHistoryTrackingMode


def get_device(include_mps: bool = True) -> torch.device:
    """Detect the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if include_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_token_model(device: torch.device) -> LiquidAudio:
    """Create a LiquidAudio model configured for token counting."""
    return LiquidAudio(
        device=device, history_tracking_mode=ModelHistoryTrackingMode.TEXT
    )

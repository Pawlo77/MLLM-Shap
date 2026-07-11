"""Mask-space utilities for explainable feature indexing."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskSpace:
    """Describes explainable feature subset inside full token mask."""

    shap_values_mask: Tensor
    """Boolean mask selecting explainable feature positions from full chat."""
    target_length: int
    """Total length of the full token mask (including non-explainable tokens)."""

    @property
    def n_features(self) -> int:
        """Number of explainable features."""
        return int(self.shap_values_mask.sum().item())

    def materialize(self, split: Tensor, device: torch.device) -> Tensor:
        """Project split over explainable subset back to full chat mask."""
        prepared = torch.zeros((self.target_length,), dtype=torch.bool, device=device)
        prepared[self.shap_values_mask] = split
        prepared[~self.shap_values_mask] = True
        return prepared

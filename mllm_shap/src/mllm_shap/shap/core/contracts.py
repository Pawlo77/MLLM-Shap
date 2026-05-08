"""Core sampling contracts for composition-based SHAP internals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class SamplingStrategy(ABC):
    """Contract for split-sampling strategies."""

    @abstractmethod
    def get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        """Return next split or None when strategy is exhausted."""

    @abstractmethod
    def get_num_splits(self, n: int) -> int | None:
        """Return expected number of generated masks, if known."""


@dataclass(frozen=True)
class EstimationResult:
    """Container for estimator outputs.

    Attributes:
        values: Estimated attribution values.
        uncertainty: Optional uncertainty proxy (for example CI width or variance).
            ``None`` means uncertainty is not provided by this estimator.
    """

    values: Tensor
    """Estimated attribution values returned by the estimator."""
    uncertainty: Tensor | None = None
    """Optional uncertainty proxy aligned with ``values`` when available."""


class Estimator(ABC):
    """Contract for composable attribution estimators."""

    @abstractmethod
    def estimate(self, masks: Tensor, payoffs: Tensor) -> EstimationResult:
        """Estimate attributions from sampled masks and corresponding payoffs."""


@dataclass(frozen=True)
class StopDecision:
    """Decision returned by stopping policies."""

    should_stop: bool
    """Whether the iterative estimation loop should terminate now."""
    reason: str
    """Human-readable explanation for the stop/continue decision."""


class StoppingPolicy(ABC):
    """Contract for composable stopping policies."""

    @abstractmethod
    def should_stop(self, iteration: int, estimation: EstimationResult) -> StopDecision:
        """Return whether pipeline should stop and why."""

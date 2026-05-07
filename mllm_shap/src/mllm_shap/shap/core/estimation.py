"""Composable estimation and stopping adapters for SHAP pipelines."""

from collections.abc import Callable

from torch import Tensor

from .contracts import EstimationResult, Estimator, StopDecision, StoppingPolicy


class CallableEstimator(Estimator):
    """Adapter that bridges callables to the Estimator contract."""

    def __init__(
        self,
        estimate_fn: Callable[[Tensor, Tensor], Tensor | tuple[Tensor, Tensor]],
    ) -> None:
        """Initialize callable-backed estimator.

        Args:
            estimate_fn: Callable receiving ``(masks, payoffs)`` and returning either:
                - attribution tensor, or
                - tuple ``(attributions, uncertainty)``.
        """
        self._estimate_fn = estimate_fn

    def estimate(self, masks: Tensor, payoffs: Tensor) -> EstimationResult:
        """Delegate estimation to wrapped callable."""
        result = self._estimate_fn(masks, payoffs)
        if isinstance(result, tuple):
            values, uncertainty = result
            return EstimationResult(values=values, uncertainty=uncertainty)
        return EstimationResult(values=result, uncertainty=None)


class CallableStoppingPolicy(StoppingPolicy):
    """Adapter that bridges callables to the StoppingPolicy contract."""

    def __init__(
        self,
        should_stop_fn: Callable[[int, EstimationResult], bool | StopDecision],
        default_reason: str = "callable-policy",
    ) -> None:
        """Initialize callable-backed stopping policy."""
        self._should_stop_fn = should_stop_fn
        self._default_reason = default_reason

    def should_stop(self, iteration: int, estimation: EstimationResult) -> StopDecision:
        """Delegate stopping decision to wrapped callable."""
        decision = self._should_stop_fn(iteration, estimation)
        if isinstance(decision, StopDecision):
            return decision
        return StopDecision(
            should_stop=bool(decision),
            reason=self._default_reason,
        )


class FixedThresholdStoppingPolicy(StoppingPolicy):
    """Stop when scalar uncertainty falls below a configured threshold."""

    def __init__(self, threshold: float) -> None:
        """Initialize fixed-threshold policy.

        Args:
            threshold: Uncertainty threshold at or below which stopping is triggered.
        """
        self._threshold = float(threshold)

    def should_stop(self, iteration: int, estimation: EstimationResult) -> StopDecision:
        """Apply fixed uncertainty threshold rule."""
        if estimation.uncertainty is None:
            return StopDecision(
                should_stop=False,
                reason="uncertainty-missing",
            )

        uncertainty_value = float(estimation.uncertainty.detach().mean().item())
        should_stop = uncertainty_value <= self._threshold
        reason = (
            f"uncertainty<={self._threshold}"
            if should_stop
            else "uncertainty-above-threshold"
        )
        return StopDecision(should_stop=should_stop, reason=reason)

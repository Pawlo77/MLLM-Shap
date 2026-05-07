"""Unit tests for composable estimation and stopping adapters."""

import torch

from mllm_shap.shap.core import (
    CallableEstimator,
    CallableStoppingPolicy,
    EstimationResult,
    FixedThresholdStoppingPolicy,
    StopDecision,
)


def test_callable_estimator_accepts_values_only() -> None:
    """Callable estimator should wrap plain attribution tensor output."""

    def estimate_fn(masks: torch.Tensor, payoffs: torch.Tensor) -> torch.Tensor:
        return masks.float().sum(dim=0) + payoffs.mean()

    estimator = CallableEstimator(estimate_fn=estimate_fn)
    masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    payoffs = torch.tensor([0.1, 0.3], dtype=torch.float32)

    result = estimator.estimate(masks=masks, payoffs=payoffs)

    assert torch.allclose(result.values, torch.tensor([1.2, 1.2]))
    assert result.uncertainty is None


def test_callable_estimator_accepts_values_and_uncertainty() -> None:
    """Callable estimator should wrap tuple output with uncertainty."""

    def estimate_fn(
        masks: torch.Tensor, payoffs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = masks.float().mean(dim=0)
        uncertainty = torch.tensor([0.25], dtype=torch.float32)
        return values, uncertainty

    estimator = CallableEstimator(estimate_fn=estimate_fn)
    masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    payoffs = torch.tensor([0.2, 0.4], dtype=torch.float32)

    result = estimator.estimate(masks=masks, payoffs=payoffs)

    assert torch.allclose(result.values, torch.tensor([0.5, 0.5]))
    assert torch.allclose(result.uncertainty, torch.tensor([0.25]))


def test_callable_stopping_policy_accepts_bool_or_decision() -> None:
    """Stopping adapter should normalize bool and StopDecision responses."""

    bool_policy = CallableStoppingPolicy(
        should_stop_fn=lambda iteration, estimation: iteration >= 3,
        default_reason="iter-threshold",
    )
    decision = bool_policy.should_stop(
        iteration=3,
        estimation=EstimationResult(values=torch.tensor([0.0])),
    )
    assert decision == StopDecision(should_stop=True, reason="iter-threshold")

    explicit_policy = CallableStoppingPolicy(
        should_stop_fn=lambda iteration, estimation: StopDecision(
            should_stop=False,
            reason="custom",
        )
    )
    explicit = explicit_policy.should_stop(
        iteration=1,
        estimation=EstimationResult(values=torch.tensor([0.0])),
    )
    assert explicit == StopDecision(should_stop=False, reason="custom")


def test_fixed_threshold_stopping_policy_uses_uncertainty_mean() -> None:
    """Threshold stopper should stop when uncertainty mean is below threshold."""
    policy = FixedThresholdStoppingPolicy(threshold=0.1)

    no_uncertainty = policy.should_stop(
        iteration=1,
        estimation=EstimationResult(values=torch.tensor([1.0]), uncertainty=None),
    )
    assert no_uncertainty == StopDecision(
        should_stop=False,
        reason="uncertainty-missing",
    )

    not_stopped = policy.should_stop(
        iteration=2,
        estimation=EstimationResult(
            values=torch.tensor([1.0]),
            uncertainty=torch.tensor([0.2, 0.3]),
        ),
    )
    assert not_stopped == StopDecision(
        should_stop=False,
        reason="uncertainty-above-threshold",
    )

    stopped = policy.should_stop(
        iteration=3,
        estimation=EstimationResult(
            values=torch.tensor([1.0]),
            uncertainty=torch.tensor([0.01, 0.05]),
        ),
    )
    assert stopped == StopDecision(
        should_stop=True,
        reason="uncertainty<=0.1",
    )

"""Unit tests for lightweight runner helpers."""

from types import SimpleNamespace

from . import runner


def test_linear_sample_scaler_keeps_even_values() -> None:
    """Even scaled values should be returned unchanged."""
    scaler = runner._LinearSampleScaler(factor=0.5)
    # 0.5 * 4^2 = 8 (already even)
    assert scaler.scale(n_pre=4) == 8


def test_linear_sample_scaler_rounds_up_odd_values() -> None:
    """Odd scaled values should be rounded up by one."""
    scaler = runner._LinearSampleScaler(factor=0.12)
    # 0.12 * 5^2 = 3 -> odd => 4
    assert scaler.scale(n_pre=5) == 4


def test_try_set_num_samples_updates_supported_explainer(monkeypatch) -> None:
    """Helper should update in place when shap_explainer matches expected base."""

    class FakeApprox:
        """Approximation-like test double."""

        def __init__(self) -> None:
            self.num_samples = 1

    monkeypatch.setattr(runner, "BaseShapApproximation", FakeApprox)
    explainer = SimpleNamespace(shap_explainer=FakeApprox())

    result = runner._try_set_num_samples(explainer=explainer, num_samples=42)

    assert result is True
    assert explainer.shap_explainer.num_samples == 42


def test_try_set_num_samples_rejects_unsupported_explainer() -> None:
    """Helper should signal fallback path for unsupported explainers."""
    explainer = SimpleNamespace(shap_explainer=object())
    assert runner._try_set_num_samples(explainer=explainer, num_samples=10) is False

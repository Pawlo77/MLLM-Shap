"""Example tests demonstrating lightweight benchmarking integration."""

import pytest
import torch

from mllm_shap.shap.base._masks_manager import MasksManager
from .perf_meter import PerfMeter


@pytest.mark.bench
def test_mask_hash_lightweight_bench() -> None:
    """Lightweight benchmark: mask hash performance.

    This test runs under normal test execution and tracks performance
    when --benchmark-light flag is enabled or in CI environment.
    """
    mask_len = 256
    num_hashes = 100

    meter = PerfMeter("mask_hash_generation")
    for i in range(num_hashes):
        mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool)
        meter.measure(lambda m=mask: MasksManager.get_hash(m))

    result = meter.result()
    # Ensure hash is fast enough (typically < 1ms)
    assert result.median_ms < 1.0, (
        f"Mask hash is slow: {result.median_ms:.2f}ms (expected < 1ms). "
        "Performance regression detected."
    )


@pytest.mark.bench
def test_tensor_creation_lightweight_bench() -> None:
    """Lightweight benchmark: tensor creation and operations.

    Demonstrates multi-sample tracking for regression detection.
    """
    meter = PerfMeter("tensor_ops")

    # Simulate repeated tensor operations
    for _ in range(10):
        meter.measure(lambda: torch.ones(100, 100) @ torch.ones(100, 100))

    result = meter.result()
    # Check that median is within reasonable bounds
    assert result.median_ms < 100, (
        f"Tensor operation is slow: {result.median_ms:.2f}ms. "
        "Performance regression detected."
    )


@pytest.mark.bench
def test_benchmark_meter_formats_results() -> None:
    """Test PerfMeter result formatting."""
    meter = PerfMeter("test_operation")

    # Simulate some measurements
    meter.times = [0.01, 0.02, 0.03, 0.04, 0.05]

    result = meter.result()
    assert result.name == "test_operation"
    assert result.min_s == pytest.approx(0.01)
    assert result.median_s == pytest.approx(0.03)
    assert result.max_s == pytest.approx(0.05)
    assert result.samples == 5
    assert result.median_ms == pytest.approx(30.0)


@pytest.mark.bench
def test_benchmark_meter_context_manager() -> None:
    """Test PerfMeter context manager interface."""
    meter = PerfMeter("context_test")

    # Use as context manager multiple times
    for _ in range(3):
        with meter:
            # Simulate some work
            sum(range(1000))

    result = meter.result()
    assert len(meter.times) == 3
    assert result.samples == 3
    assert result.median_s > 0

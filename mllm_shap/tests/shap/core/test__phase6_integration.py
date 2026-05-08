"""Phase 6 integration tests: Optional telemetry end-to-end."""

import unittest
from unittest.mock import MagicMock

import torch

from mllm_shap.shap.core import (
    CallableAdapterStrategy,
    SamplingEngine,
    TelemetryProbe,
)
from mllm_shap.shap.complementary._engine import ComplementarySamplingEngine
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.base._cache_manager import CacheManager


class TestPhase6TelemetryIntegration(unittest.TestCase):
    """Integration tests for Phase 6 telemetry infrastructure."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.mock_chat = MagicMock()
        self.mock_chat.shap_values_mask = torch.ones(5, dtype=torch.bool)
        self.mock_chat.input_tokens_num = 5
        self.mock_response_chat = MagicMock()
        self.mock_response_chat.shap_values_mask = torch.ones(5, dtype=torch.bool)
        self.mock_response_chat.input_tokens_num = 5
        self.mock_response_chat.cache = None

    def test_telemetry_probe_optional_in_sampling_engine(self) -> None:
        """Verify SamplingEngine accepts optional probe parameter."""
        strategy = MagicMock()
        strategy.get_num_splits.return_value = 10
        strategy.get_next_split.return_value = None

        # Without probe (zero overhead)
        engine_no_probe = SamplingEngine(
            strategy=strategy,
            probe=None,
        )
        self.assertIsNone(engine_no_probe._probe)

        # With probe
        probe = TelemetryProbe.with_log_sink()
        engine_with_probe = SamplingEngine(
            strategy=strategy,
            probe=probe,
        )
        self.assertIsNotNone(engine_with_probe._probe)

    def test_telemetry_probe_optional_in_complementary_engine(self) -> None:
        """Verify ComplementarySamplingEngine accepts optional probe parameter."""
        strategy = MagicMock()
        strategy.get_num_splits.return_value = 10
        strategy.get_next_split.return_value = None

        # Without probe (zero overhead)
        engine_no_probe = ComplementarySamplingEngine(
            strategy=strategy,
            probe=None,
        )
        self.assertIsNone(engine_no_probe._probe)

        # With probe
        probe = TelemetryProbe.with_log_sink()
        engine_with_probe = ComplementarySamplingEngine(
            strategy=strategy,
            probe=probe,
        )
        self.assertIsNotNone(engine_with_probe._probe)

    def test_masks_manager_accepts_optional_probe(self) -> None:
        """Verify MasksManager accepts optional probe parameter."""
        # Without probe (zero overhead)
        manager_no_probe = MasksManager(chat=self.mock_chat, probe=None)
        self.assertIsNone(manager_no_probe._probe)

        # With probe
        probe = TelemetryProbe.with_log_sink()
        manager_with_probe = MasksManager(
            chat=self.mock_chat,
            probe=probe,
        )
        self.assertIs(manager_with_probe._probe, probe)

    def test_cache_manager_accepts_optional_probe(self) -> None:
        """Verify CacheManager accepts optional probe parameter."""
        # Without probe (zero overhead)
        cache_no_probe = CacheManager(
            chat=self.mock_response_chat,
            explainer_hash=42,
            probe=None,
        )
        self.assertIsNone(cache_no_probe._probe)

        # With probe
        probe = TelemetryProbe.with_log_sink()
        cache_with_probe = CacheManager(
            chat=self.mock_response_chat,
            explainer_hash=42,
            probe=probe,
        )
        self.assertIs(cache_with_probe._probe, probe)

    def test_callable_adapter_strategy_has_canonical_name(self) -> None:
        """Verify CallableAdapterStrategy uses canonical naming."""

        def get_next_split(**kwargs):
            return None

        def get_num_splits(n):
            return 10

        strategy = CallableAdapterStrategy(
            get_next_split=get_next_split,
            get_num_splits=get_num_splits,
        )

        # Should support the SamplingStrategy contract
        self.assertIsNone(
            strategy.get_next_split(
                n=5,
                device=self.device,
                generated_masks_num=0,
                existing_masks=None,
            )
        )
        self.assertEqual(strategy.get_num_splits(5), 10)

    def test_telemetry_probe_noop_zero_overhead(self) -> None:
        """Verify TelemetryProbe.noop() has zero overhead."""
        probe = TelemetryProbe.noop()

        # noop probe should have None sink
        self.assertIsNone(probe.sink)

        # Calling methods should be no-ops
        probe.cache_operation(is_hit=True)
        probe.mask_generated(is_unique=True, is_invalid=False)
        probe.record_timing("sampling", 1.5)
        probe.custom_metric("key", 42)

        # get_metrics should work but return empty
        self.assertIsNone(probe.get_metrics())

    def test_telemetry_probe_with_log_sink_records_metrics(self) -> None:
        """Verify TelemetryProbe with LogProbeSink collects metrics."""
        probe = TelemetryProbe.with_log_sink()
        self.assertIsNotNone(probe.sink)

        # Record operations
        probe.cache_operation(is_hit=True)
        probe.cache_operation(is_hit=False)
        probe.mask_generated(is_unique=True, is_invalid=False)
        probe.mask_generated(is_unique=False, is_invalid=False)
        probe.mask_generated(is_unique=False, is_invalid=True)
        probe.record_timing("sampling", 2.5)

        # Get metrics
        metrics = probe.get_metrics()

        # Verify cache metrics
        self.assertEqual(metrics.cache_metrics.hits, 1)
        self.assertEqual(metrics.cache_metrics.misses, 1)

        # Verify mask metrics
        self.assertEqual(metrics.mask_metrics.generated, 3)
        self.assertEqual(metrics.mask_metrics.unique, 1)
        self.assertEqual(metrics.mask_metrics.invalid, 1)

        # Verify timing
        self.assertGreater(metrics.timing_metrics.sampling_ms, 0)

    def test_telemetry_optional_in_pipeline(self) -> None:
        """Verify telemetry can be toggled on/off in the pipeline."""
        mock_chat = MagicMock()
        mock_chat.shap_values_mask = torch.ones(10, dtype=torch.bool)
        mock_chat.input_tokens_num = 10

        # Scenario 1: No telemetry (production mode)
        manager_prod = MasksManager(chat=mock_chat, probe=None)
        initial_mask_prod = manager_prod.get_initial_mask(device=self.device)
        self.assertIsNotNone(initial_mask_prod)

        # Scenario 2: With telemetry (debug mode)
        probe = TelemetryProbe.with_log_sink()
        manager_debug = MasksManager(chat=mock_chat, probe=probe)
        initial_mask_debug = manager_debug.get_initial_mask(device=self.device)
        self.assertIsNotNone(initial_mask_debug)

        # Both should produce identical masks
        self.assertTrue(torch.equal(initial_mask_prod, initial_mask_debug))

        # But only debug mode should have metrics
        metrics = probe.get_metrics()
        self.assertIsNotNone(metrics)


class TestCallableAdapterStrategyNaming(unittest.TestCase):
    """Verify CallableAdapterStrategy naming remains canonical."""

    def test_class_name_is_canonical(self) -> None:
        """Verify class name does not use transitional prefixes."""
        self.assertNotIn(
            "Legacy",
            CallableAdapterStrategy.__name__,
            "Strategy class should not use transitional 'Legacy*' naming",
        )

    def test_callable_adapter_is_descriptive(self) -> None:
        """Verify CallableAdapterStrategy name is descriptive."""
        self.assertIn(
            "Callable",
            CallableAdapterStrategy.__name__,
            "Class name should indicate it wraps callables",
        )
        self.assertIn(
            "Adapter",
            CallableAdapterStrategy.__name__,
            "Class name should indicate it's an adapter",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for telemetry and metrics collection."""

import pytest
import time

from mllm_shap.shap.core.telemetry import (
    CacheMetrics,
    JSONProbeSink,
    LogProbeSink,
    MaskMetrics,
    StageTimer,
    TelemetryData,
    TelemetryProbe,
    TimingMetrics,
)


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_cache_metrics_init(self) -> None:
        """Test CacheMetrics initialization."""
        metrics = CacheMetrics(hits=10, misses=5)
        assert metrics.hits == 10
        assert metrics.misses == 5
        assert metrics.total == 15

    def test_cache_metrics_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        metrics = CacheMetrics(hits=10, misses=5)
        assert metrics.hit_rate == pytest.approx(10 / 15)

    def test_cache_metrics_hit_rate_zero_total(self) -> None:
        """Test cache hit rate when total is zero."""
        metrics = CacheMetrics(hits=0, misses=0)
        assert metrics.hit_rate == 0.0

    def test_cache_metrics_to_dict(self) -> None:
        """Test CacheMetrics.to_dict()."""
        metrics = CacheMetrics(hits=10, misses=5)
        result = metrics.to_dict()
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["total"] == 15
        assert result["hit_rate"] == pytest.approx(10 / 15)

    def test_cache_metrics_frozen(self) -> None:
        """Test that CacheMetrics is immutable."""
        metrics = CacheMetrics(hits=10, misses=5)
        with pytest.raises(AttributeError):
            metrics.hits = 20


class TestMaskMetrics:
    """Tests for MaskMetrics dataclass."""

    def test_mask_metrics_init(self) -> None:
        """Test MaskMetrics initialization."""
        metrics = MaskMetrics(generated=100, unique=80, invalid=5)
        assert metrics.generated == 100
        assert metrics.unique == 80
        assert metrics.invalid == 5

    def test_mask_metrics_dedup_rate(self) -> None:
        """Test mask deduplication rate calculation."""
        metrics = MaskMetrics(generated=100, unique=80, invalid=5)
        # (100 - 80) / 100 = 0.2
        assert metrics.dedup_rate == pytest.approx(0.2)

    def test_mask_metrics_dedup_rate_zero_generated(self) -> None:
        """Test dedup rate when no masks generated."""
        metrics = MaskMetrics(generated=0, unique=0, invalid=0)
        assert metrics.dedup_rate == 0.0

    def test_mask_metrics_to_dict(self) -> None:
        """Test MaskMetrics.to_dict()."""
        metrics = MaskMetrics(generated=100, unique=80, invalid=5)
        result = metrics.to_dict()
        assert result["generated"] == 100
        assert result["unique"] == 80
        assert result["invalid"] == 5
        assert result["dedup_rate"] == pytest.approx(0.2)

    def test_mask_metrics_frozen(self) -> None:
        """Test that MaskMetrics is immutable."""
        metrics = MaskMetrics(generated=100, unique=80, invalid=5)
        with pytest.raises(AttributeError):
            metrics.generated = 50


class TestLogProbeSink:
    """Tests for LogProbeSink."""

    def test_log_sink_cache_operation_hit(self) -> None:
        """Test recording cache hits."""
        sink = LogProbeSink(verbose=False)
        sink.record_cache_operation(is_hit=True)
        sink.record_cache_operation(is_hit=True)
        sink.record_cache_operation(is_hit=False)

        metrics = sink.get_metrics()
        assert metrics.cache_metrics.hits == 2
        assert metrics.cache_metrics.misses == 1

    def test_log_sink_mask_generation_unique(self) -> None:
        """Test recording unique masks."""
        sink = LogProbeSink(verbose=False)
        sink.record_mask_generated(is_unique=True, is_invalid=False)
        sink.record_mask_generated(is_unique=True, is_invalid=False)
        sink.record_mask_generated(is_unique=False, is_invalid=False)

        metrics = sink.get_metrics()
        assert metrics.mask_metrics.generated == 3
        assert metrics.mask_metrics.unique == 2
        assert metrics.mask_metrics.invalid == 0

    def test_log_sink_mask_generation_invalid(self) -> None:
        """Test recording invalid masks."""
        sink = LogProbeSink(verbose=False)
        sink.record_mask_generated(is_unique=False, is_invalid=True)
        sink.record_mask_generated(is_unique=True, is_invalid=False)

        metrics = sink.get_metrics()
        assert metrics.mask_metrics.generated == 2
        assert metrics.mask_metrics.unique == 1
        assert metrics.mask_metrics.invalid == 1

    def test_log_sink_custom_metrics(self) -> None:
        """Test recording custom metrics."""
        sink = LogProbeSink(verbose=False)
        sink.record_custom_metric("latency_ms", 123.45)
        sink.record_custom_metric("batch_size", 32)

        metrics = sink.get_metrics()
        assert metrics.custom_metrics["latency_ms"] == 123.45
        assert metrics.custom_metrics["batch_size"] == 32

    def test_log_sink_reset(self) -> None:
        """Test resetting sink metrics."""
        sink = LogProbeSink(verbose=False)
        sink.record_cache_operation(is_hit=True)
        sink.record_mask_generated(is_unique=True)
        sink.record_custom_metric("key", "value")

        sink.reset()

        metrics = sink.get_metrics()
        assert metrics.cache_metrics.hits == 0
        assert metrics.cache_metrics.misses == 0
        assert metrics.mask_metrics.generated == 0
        assert metrics.mask_metrics.unique == 0
        assert len(metrics.custom_metrics) == 0


class TestJSONProbeSink:
    """Tests for JSONProbeSink."""

    def test_json_sink_cache_operation(self) -> None:
        """Test JSON sink recording cache operations."""
        sink = JSONProbeSink()
        sink.record_cache_operation(is_hit=True)
        sink.record_cache_operation(is_hit=False)

        metrics = sink.get_metrics()
        assert metrics.cache_metrics.hits == 1
        assert metrics.cache_metrics.misses == 1

    def test_json_sink_mask_generation(self) -> None:
        """Test JSON sink recording mask generation."""
        sink = JSONProbeSink()
        sink.record_mask_generated(is_unique=True, is_invalid=False)
        sink.record_mask_generated(is_unique=False, is_invalid=True)

        metrics = sink.get_metrics()
        assert metrics.mask_metrics.generated == 2
        assert metrics.mask_metrics.unique == 1
        assert metrics.mask_metrics.invalid == 1

    def test_json_sink_to_json(self) -> None:
        """Test JSON serialization."""
        sink = JSONProbeSink()
        sink.record_cache_operation(is_hit=True)
        sink.record_mask_generated(is_unique=True)
        sink.record_custom_metric("test", "value")

        json_str = sink.to_json()
        assert isinstance(json_str, str)
        assert "cache" in json_str
        assert "masks" in json_str
        assert "custom" in json_str
        assert "hits" in json_str
        assert "generated" in json_str

    def test_json_sink_reset(self) -> None:
        """Test JSON sink reset."""
        sink = JSONProbeSink()
        sink.record_cache_operation(is_hit=True)
        sink.reset()

        metrics = sink.get_metrics()
        assert metrics.cache_metrics.hits == 0


class TestTelemetryProbe:
    """Tests for TelemetryProbe."""

    def test_probe_noop(self) -> None:
        """Test no-op probe."""
        probe = TelemetryProbe.noop()
        probe.cache_operation(is_hit=True)
        probe.mask_generated(is_unique=True)
        probe.custom_metric("key", "value")

        metrics = probe.get_metrics()
        assert metrics is None

    def test_probe_with_log_sink(self) -> None:
        """Test probe with log sink."""
        probe = TelemetryProbe.with_log_sink(verbose=False)
        probe.cache_operation(is_hit=True)
        probe.mask_generated(is_unique=True)
        probe.custom_metric("key", "value")

        metrics = probe.get_metrics()
        assert metrics is not None
        assert metrics.cache_metrics.hits == 1
        assert metrics.mask_metrics.generated == 1
        assert metrics.custom_metrics["key"] == "value"

    def test_probe_with_json_sink(self) -> None:
        """Test probe with JSON sink."""
        probe = TelemetryProbe.with_json_sink()
        probe.cache_operation(is_hit=False)
        probe.mask_generated(is_unique=False, is_invalid=True)

        metrics = probe.get_metrics()
        assert metrics is not None
        assert metrics.cache_metrics.misses == 1
        assert metrics.mask_metrics.invalid == 1

    def test_probe_reset(self) -> None:
        """Test probe reset."""
        probe = TelemetryProbe.with_log_sink()
        probe.cache_operation(is_hit=True)
        probe.reset()

        metrics = probe.get_metrics()
        assert metrics.cache_metrics.hits == 0

    def test_probe_with_custom_sink(self) -> None:
        """Test probe with custom sink."""
        sink = LogProbeSink(verbose=False)
        probe = TelemetryProbe(sink=sink)
        probe.cache_operation(is_hit=True)

        metrics = probe.get_metrics()
        assert metrics.cache_metrics.hits == 1


class TestTelemetryData:
    """Tests for TelemetryData."""

    def test_telemetry_data_init(self) -> None:
        """Test TelemetryData initialization."""
        cache_metrics = CacheMetrics(hits=10, misses=5)
        mask_metrics = MaskMetrics(generated=100, unique=80)
        data = TelemetryData(
            cache_metrics=cache_metrics,
            mask_metrics=mask_metrics,
            custom_metrics={"key": "value"},
        )

        assert data.cache_metrics.hits == 10
        assert data.mask_metrics.generated == 100
        assert data.custom_metrics["key"] == "value"

    def test_telemetry_data_to_dict(self) -> None:
        """Test TelemetryData.to_dict()."""
        cache_metrics = CacheMetrics(hits=10, misses=5)
        mask_metrics = MaskMetrics(generated=100, unique=80)
        data = TelemetryData(cache_metrics=cache_metrics, mask_metrics=mask_metrics)

        result = data.to_dict()
        assert "cache" in result
        assert "masks" in result
        assert "custom" in result
        assert result["cache"]["hits"] == 10
        assert result["masks"]["generated"] == 100


class TestTimingMetrics:
    """Tests for TimingMetrics dataclass."""

    def test_timing_metrics_init(self) -> None:
        """Test TimingMetrics initialization."""
        metrics = TimingMetrics(
            sampling_ms=100.0,
            dedup_ms=50.0,
            masking_ms=25.0,
            model_ms=500.0,
            scoring_ms=75.0,
        )
        assert metrics.sampling_ms == 100.0
        assert metrics.dedup_ms == 50.0
        assert metrics.masking_ms == 25.0
        assert metrics.model_ms == 500.0
        assert metrics.scoring_ms == 75.0

    def test_timing_metrics_total_ms(self) -> None:
        """Test total_ms calculation."""
        metrics = TimingMetrics(
            sampling_ms=100.0,
            dedup_ms=50.0,
            masking_ms=25.0,
            model_ms=500.0,
            scoring_ms=75.0,
        )
        assert metrics.total_ms == pytest.approx(750.0)

    def test_timing_metrics_total_ms_zero(self) -> None:
        """Test total_ms with zero values."""
        metrics = TimingMetrics()
        assert metrics.total_ms == 0.0

    def test_timing_metrics_to_dict(self) -> None:
        """Test TimingMetrics.to_dict()."""
        metrics = TimingMetrics(
            sampling_ms=100.0,
            dedup_ms=50.0,
            masking_ms=25.0,
            model_ms=500.0,
            scoring_ms=75.0,
        )
        result = metrics.to_dict()
        assert result["sampling_ms"] == 100.0
        assert result["dedup_ms"] == 50.0
        assert result["masking_ms"] == 25.0
        assert result["model_ms"] == 500.0
        assert result["scoring_ms"] == 75.0
        assert result["total_ms"] == pytest.approx(750.0)

    def test_timing_metrics_frozen(self) -> None:
        """Test that TimingMetrics is immutable."""
        metrics = TimingMetrics(sampling_ms=100.0)
        with pytest.raises(AttributeError):
            metrics.sampling_ms = 200.0


class TestStageTimer:
    """Tests for StageTimer context manager."""

    def test_stage_timer_records_timing(self) -> None:
        """Test that StageTimer records timing correctly."""
        probe = TelemetryProbe.with_log_sink(verbose=False)

        with StageTimer(probe, "sampling"):
            time.sleep(0.01)  # Sleep for ~10ms

        metrics = probe.get_metrics()
        assert metrics.timing_metrics.sampling_ms >= 10.0

    def test_stage_timer_multiple_stages(self) -> None:
        """Test timing multiple stages."""
        probe = TelemetryProbe.with_log_sink(verbose=False)

        with StageTimer(probe, "sampling"):
            time.sleep(0.01)
        with StageTimer(probe, "dedup"):
            time.sleep(0.01)
        with StageTimer(probe, "masking"):
            time.sleep(0.01)

        metrics = probe.get_metrics()
        assert metrics.timing_metrics.sampling_ms >= 10.0
        assert metrics.timing_metrics.dedup_ms >= 10.0
        assert metrics.timing_metrics.masking_ms >= 10.0

    def test_stage_timer_all_stages(self) -> None:
        """Test timing all stages."""
        probe = TelemetryProbe.with_log_sink(verbose=False)

        for stage in ["sampling", "dedup", "masking", "model", "scoring"]:
            with StageTimer(probe, stage):
                time.sleep(0.005)

        metrics = probe.get_metrics()
        assert metrics.timing_metrics.total_ms >= 25.0

    def test_probe_timing_method(self) -> None:
        """Test TelemetryProbe.timing() context manager."""
        probe = TelemetryProbe.with_log_sink(verbose=False)

        with probe.timing("sampling"):
            time.sleep(0.01)

        metrics = probe.get_metrics()
        assert metrics.timing_metrics.sampling_ms >= 10.0

    def test_probe_record_timing_direct(self) -> None:
        """Test TelemetryProbe.record_timing() direct method."""
        probe = TelemetryProbe.with_log_sink(verbose=False)

        probe.record_timing("sampling", 100.0)
        probe.record_timing("model", 500.0)

        metrics = probe.get_metrics()
        assert metrics.timing_metrics.sampling_ms == 100.0
        assert metrics.timing_metrics.model_ms == 500.0

    def test_timing_metrics_in_telemetry_data(self) -> None:
        """Test that TimingMetrics is included in TelemetryData."""
        probe = TelemetryProbe.with_json_sink()
        probe.record_timing("sampling", 100.0)
        probe.record_timing("model", 500.0)

        metrics = probe.get_metrics()
        data_dict = metrics.to_dict()

        assert "timing" in data_dict
        assert data_dict["timing"]["sampling_ms"] == 100.0
        assert data_dict["timing"]["model_ms"] == 500.0
        assert data_dict["timing"]["total_ms"] == 600.0

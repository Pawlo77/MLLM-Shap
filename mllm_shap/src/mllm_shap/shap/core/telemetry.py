"""Telemetry and metrics collection for SHAP explainability."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from json import dumps
from logging import Logger
from time import perf_counter
from typing import Any, Generator

from ...utils.logger import get_logger

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheMetrics:
    """Metrics for cache operations."""

    hits: int = 0
    """Number of cache hits."""

    misses: int = 0
    """Number of cache misses."""

    @property
    def total(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": self.hit_rate,
        }


@dataclass(frozen=True)
class MaskMetrics:
    """Metrics for mask generation and deduplication."""

    generated: int = 0
    """Total masks generated."""

    unique: int = 0
    """Total unique masks (after dedup)."""

    invalid: int = 0
    """Total invalid/empty masks."""

    @property
    def dedup_rate(self) -> float:
        """Rate of duplicates caught (0-1)."""
        if self.generated == 0:
            return 0.0
        duplicates = self.generated - self.unique
        return duplicates / self.generated

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "generated": self.generated,
            "unique": self.unique,
            "invalid": self.invalid,
            "dedup_rate": self.dedup_rate,
        }


@dataclass(frozen=True)
class TimingMetrics:
    """Metrics for per-stage timing."""

    sampling_ms: float = 0.0
    """Time spent in mask sampling (milliseconds)."""

    dedup_ms: float = 0.0
    """Time spent in deduplication checks (milliseconds)."""

    masking_ms: float = 0.0
    """Time spent in mask preparation (milliseconds)."""

    model_ms: float = 0.0
    """Time spent in model inference (milliseconds)."""

    scoring_ms: float = 0.0
    """Time spent in SHAP value computation (milliseconds)."""

    @property
    def total_ms(self) -> float:
        """Total time across all stages."""
        return (
            self.sampling_ms
            + self.dedup_ms
            + self.masking_ms
            + self.model_ms
            + self.scoring_ms
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "sampling_ms": self.sampling_ms,
            "dedup_ms": self.dedup_ms,
            "masking_ms": self.masking_ms,
            "model_ms": self.model_ms,
            "scoring_ms": self.scoring_ms,
            "total_ms": self.total_ms,
        }


class StageTimer:
    """Context manager for timing code blocks with probe integration."""

    def __init__(self, probe: "TelemetryProbe", stage: str) -> None:
        """
        Initialize StageTimer.

        Args:
            probe: TelemetryProbe to record timing to.
            stage: Name of the stage being timed (sampling, dedup, masking, model, scoring).
        """
        self.probe = probe
        self.stage = stage
        self.start_time: float | None = None

    def __enter__(self) -> "StageTimer":
        """Start timing."""
        self.start_time = perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record to probe."""
        if self.start_time is not None:
            elapsed_ms = (perf_counter() - self.start_time) * 1000
            self.probe.record_timing(self.stage, elapsed_ms)

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Alternate context manager interface."""
        with self:
            yield


@dataclass
class TelemetryData:
    """Container for all telemetry data collected during a run."""

    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)
    """Cache operation metrics."""

    mask_metrics: MaskMetrics = field(default_factory=MaskMetrics)
    """Mask generation metrics."""

    timing_metrics: TimingMetrics = field(default_factory=TimingMetrics)
    """Per-stage timing metrics."""

    custom_metrics: dict[str, Any] = field(default_factory=dict)
    """Custom metrics dictionary."""

    def to_dict(self) -> dict[str, Any]:
        """Convert all telemetry to dictionary."""
        return {
            "cache": self.cache_metrics.to_dict(),
            "masks": self.mask_metrics.to_dict(),
            "timing": self.timing_metrics.to_dict(),
            "custom": self.custom_metrics,
        }


class ProbeSink(ABC):
    """Abstract base class for telemetry sinks."""

    @abstractmethod
    def record_cache_operation(self, is_hit: bool) -> None:
        """Record a cache operation (hit or miss)."""

    @abstractmethod
    def record_mask_generated(self, is_unique: bool, is_invalid: bool = False) -> None:
        """Record a generated mask and whether it was unique/invalid."""

    @abstractmethod
    def record_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a stage (sampling, dedup, masking, model, scoring)."""

    @abstractmethod
    def record_custom_metric(self, key: str, value: Any) -> None:
        """Record a custom metric."""

    @abstractmethod
    def get_metrics(self) -> TelemetryData:
        """Get collected telemetry data."""

    def reset(self) -> None:
        """Reset collected metrics. Optional override."""


class LogProbeSink(ProbeSink):
    """Telemetry sink that logs metrics to logger."""

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize LogProbeSink.

        Args:
            verbose: If True, log every operation. If False, only log summary.
        """
        self.verbose = verbose
        self._cache_hits = 0
        self._cache_misses = 0
        self._masks_generated = 0
        self._masks_unique = 0
        self._masks_invalid = 0
        self._sampling_ms = 0.0
        self._dedup_ms = 0.0
        self._masking_ms = 0.0
        self._model_ms = 0.0
        self._scoring_ms = 0.0
        self._custom_metrics: dict[str, Any] = {}

    def record_cache_operation(self, is_hit: bool) -> None:
        """Record a cache operation."""
        if is_hit:
            self._cache_hits += 1
            if self.verbose:
                logger.debug(
                    "Mask cache hit: reusing previously computed SHAP values "
                    "(cumulative hits: %d)",
                    self._cache_hits,
                )
        else:
            self._cache_misses += 1
            if self.verbose:
                logger.debug(
                    "Mask cache miss: mask not in cache, will compute SHAP values "
                    "(cumulative misses: %d)",
                    self._cache_misses,
                )

    def record_mask_generated(self, is_unique: bool, is_invalid: bool = False) -> None:
        """Record a generated mask."""
        self._masks_generated += 1
        if is_invalid:
            self._masks_invalid += 1
            if self.verbose:
                logger.debug(
                    "Invalid mask generated: failed validation (e.g., no tokens to explain). "
                    "Cumulative invalid masks: %d/%d",
                    self._masks_invalid,
                    self._masks_generated,
                )
        elif is_unique:
            self._masks_unique += 1
            if self.verbose:
                logger.debug(
                    "Unique mask generated: new mask pattern added to cache. "
                    "Cumulative unique masks: %d/%d",
                    self._masks_unique,
                    self._masks_generated,
                )
        else:
            if self.verbose:
                logger.debug(
                    "Duplicate mask skipped: mask pattern already cached, "
                    "will use existing results (duplicates save computation). "
                    "Cumulative duplicates: %d/%d",
                    self._masks_generated - self._masks_unique - self._masks_invalid,
                    self._masks_generated,
                )

    def record_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a stage."""
        if stage == "sampling":
            self._sampling_ms += elapsed_ms
        elif stage == "dedup":
            self._dedup_ms += elapsed_ms
        elif stage == "masking":
            self._masking_ms += elapsed_ms
        elif stage == "model":
            self._model_ms += elapsed_ms
        elif stage == "scoring":
            self._scoring_ms += elapsed_ms
        if self.verbose:
            stage_descriptions = {
                "sampling": "sampling mask generation strategy",
                "dedup": "mask deduplication and caching",
                "masking": "applying mask to chat",
                "model": "model inference on masked input",
                "scoring": "scoring computation",
            }
            description = stage_descriptions.get(stage, stage)
            logger.debug(
                "Stage completed: %s took %.2f ms",
                description,
                elapsed_ms,
            )

    def record_custom_metric(self, key: str, value: Any) -> None:
        """Record a custom metric."""
        self._custom_metrics[key] = value
        if self.verbose:
            logger.debug(
                "Custom telemetry metric recorded: %s = %s (type: %s)",
                key,
                value,
                type(value).__name__,
            )

    def get_metrics(self) -> TelemetryData:
        """Get collected telemetry data."""
        cache_metrics = CacheMetrics(hits=self._cache_hits, misses=self._cache_misses)
        mask_metrics = MaskMetrics(
            generated=self._masks_generated,
            unique=self._masks_unique,
            invalid=self._masks_invalid,
        )
        timing_metrics = TimingMetrics(
            sampling_ms=self._sampling_ms,
            dedup_ms=self._dedup_ms,
            masking_ms=self._masking_ms,
            model_ms=self._model_ms,
            scoring_ms=self._scoring_ms,
        )
        return TelemetryData(
            cache_metrics=cache_metrics,
            mask_metrics=mask_metrics,
            timing_metrics=timing_metrics,
            custom_metrics=self._custom_metrics.copy(),
        )

    def reset(self) -> None:
        """Reset collected metrics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._masks_generated = 0
        self._masks_unique = 0
        self._masks_invalid = 0
        self._sampling_ms = 0.0
        self._dedup_ms = 0.0
        self._masking_ms = 0.0
        self._model_ms = 0.0
        self._scoring_ms = 0.0
        self._custom_metrics.clear()


class JSONProbeSink(ProbeSink):
    """Telemetry sink that collects metrics for JSON serialization."""

    def __init__(self) -> None:
        """Initialize JSONProbeSink."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._masks_generated = 0
        self._masks_unique = 0
        self._masks_invalid = 0
        self._sampling_ms = 0.0
        self._dedup_ms = 0.0
        self._masking_ms = 0.0
        self._model_ms = 0.0
        self._scoring_ms = 0.0
        self._custom_metrics: dict[str, Any] = {}

    def record_cache_operation(self, is_hit: bool) -> None:
        """Record a cache operation."""
        if is_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def record_mask_generated(self, is_unique: bool, is_invalid: bool = False) -> None:
        """Record a generated mask."""
        self._masks_generated += 1
        if is_invalid:
            self._masks_invalid += 1
        elif is_unique:
            self._masks_unique += 1

    def record_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a stage."""
        if stage == "sampling":
            self._sampling_ms += elapsed_ms
        elif stage == "dedup":
            self._dedup_ms += elapsed_ms
        elif stage == "masking":
            self._masking_ms += elapsed_ms
        elif stage == "model":
            self._model_ms += elapsed_ms
        elif stage == "scoring":
            self._scoring_ms += elapsed_ms

    def record_custom_metric(self, key: str, value: Any) -> None:
        """Record a custom metric."""
        self._custom_metrics[key] = value

    def get_metrics(self) -> TelemetryData:
        """Get collected telemetry data."""
        cache_metrics = CacheMetrics(hits=self._cache_hits, misses=self._cache_misses)
        mask_metrics = MaskMetrics(
            generated=self._masks_generated,
            unique=self._masks_unique,
            invalid=self._masks_invalid,
        )
        timing_metrics = TimingMetrics(
            sampling_ms=self._sampling_ms,
            dedup_ms=self._dedup_ms,
            masking_ms=self._masking_ms,
            model_ms=self._model_ms,
            scoring_ms=self._scoring_ms,
        )
        return TelemetryData(
            cache_metrics=cache_metrics,
            mask_metrics=mask_metrics,
            timing_metrics=timing_metrics,
            custom_metrics=self._custom_metrics.copy(),
        )

    def to_json(self) -> str:
        """Serialize collected metrics to JSON string."""
        data = self.get_metrics()
        return dumps(data.to_dict(), indent=2)

    def reset(self) -> None:
        """Reset collected metrics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._masks_generated = 0
        self._masks_unique = 0
        self._masks_invalid = 0
        self._sampling_ms = 0.0
        self._dedup_ms = 0.0
        self._masking_ms = 0.0
        self._model_ms = 0.0
        self._scoring_ms = 0.0
        self._custom_metrics.clear()


class TelemetryProbe:
    """Main interface for collecting telemetry during SHAP computation."""

    def __init__(self, sink: ProbeSink | None = None) -> None:
        """
        Initialize TelemetryProbe.

        Args:
            sink: Optional ProbeSink to collect metrics. If None, no metrics collected.
        """
        self.sink = sink

    def cache_operation(self, is_hit: bool) -> None:
        """Record a cache operation (hit or miss)."""
        if self.sink:
            self.sink.record_cache_operation(is_hit)

    def mask_generated(self, is_unique: bool, is_invalid: bool = False) -> None:
        """Record a generated mask and whether it was unique/invalid."""
        if self.sink:
            self.sink.record_mask_generated(is_unique, is_invalid)

    def record_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a stage (sampling, dedup, masking, model, scoring)."""
        if self.sink:
            self.sink.record_timing(stage, elapsed_ms)

    def timing(self, stage: str) -> StageTimer:
        """Create a context manager for timing a stage."""
        return StageTimer(self, stage)

    def custom_metric(self, key: str, value: Any) -> None:
        """Record a custom metric."""
        if self.sink:
            self.sink.record_custom_metric(key, value)

    def get_metrics(self) -> TelemetryData | None:
        """Get collected telemetry data, or None if no sink."""
        if self.sink:
            return self.sink.get_metrics()
        return None

    def reset(self) -> None:
        """Reset collected metrics."""
        if self.sink:
            self.sink.reset()

    @staticmethod
    def noop() -> "TelemetryProbe":
        """Create a no-op probe (no sink, no collection)."""
        return TelemetryProbe(sink=None)

    @staticmethod
    def with_log_sink(verbose: bool = False) -> "TelemetryProbe":
        """Create a probe with log sink."""
        return TelemetryProbe(sink=LogProbeSink(verbose=verbose))

    @staticmethod
    def with_json_sink() -> "TelemetryProbe":
        """Create a probe with JSON sink."""
        return TelemetryProbe(sink=JSONProbeSink())

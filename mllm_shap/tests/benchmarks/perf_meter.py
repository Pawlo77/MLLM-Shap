"""Lightweight performance profiling utilities for tests."""

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PerfResult:
    """Performance measurement result."""

    name: str
    min_s: float
    median_s: float
    max_s: float
    mean_s: float
    samples: int

    @property
    def min_ms(self) -> float:
        """Return minimum time in milliseconds."""
        return self.min_s * 1000

    @property
    def median_ms(self) -> float:
        """Return median time in milliseconds."""
        return self.median_s * 1000

    @property
    def max_ms(self) -> float:
        """Return maximum time in milliseconds."""
        return self.max_s * 1000

    @property
    def mean_ms(self) -> float:
        """Return mean time in milliseconds."""
        return self.mean_s * 1000


class PerfMeter:
    """Lightweight performance meter for specific code sections."""

    def __init__(self, name: str) -> None:
        """Initialize performance meter.

        Args:
            name: Name of the operation being measured.
        """
        self.name = name
        self.times: list[float] = []
        self._start_time: float | None = None

    def __enter__(self) -> "PerfMeter":
        """Enter context manager, start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit context manager, record elapsed time."""
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.times.append(elapsed)

    def measure(self, fn: Callable[..., Any]) -> Any:
        """Measure execution time of a function.

        Args:
            fn: Callable to measure.

        Returns:
            Result of the function call.
        """
        with self:
            return fn()

    def result(self) -> PerfResult:
        """Get performance result summary.

        Returns:
            Performance metrics if samples exist, raises ValueError otherwise.

        Raises:
            ValueError: If no measurements have been taken.
        """
        if not self.times:
            raise ValueError(f"No measurements recorded for {self.name}")

        return PerfResult(
            name=self.name,
            min_s=min(self.times),
            median_s=statistics.median(self.times),
            max_s=max(self.times),
            mean_s=statistics.mean(self.times),
            samples=len(self.times),
        )

    def reset(self) -> None:
        """Reset all recorded times."""
        self.times.clear()

    def __str__(self) -> str:
        """Return formatted result."""
        if not self.times:
            return f"{self.name}: no measurements"
        res = self.result()
        return (
            f"{self.name}: min={res.min_ms:.2f}ms, median={res.median_ms:.2f}ms, "
            f"max={res.max_ms:.2f}ms ({res.samples} samples)"
        )

"""Pytest configuration and fixtures for mllm_shap tests."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)


class LightweightBenchmark:
    """Lightweight performance tracker for tests without slowing down execution."""

    def __init__(self) -> None:
        """Initialize benchmark tracker."""
        self.timings: dict[str, list[float]] = {}
        self.enabled = False

    def start_timing(self, test_name: str) -> float:
        """Record start time for a test."""
        return time.perf_counter()

    def end_timing(self, test_name: str, start_time: float) -> None:
        """Record end time and store duration."""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - start_time
        if test_name not in self.timings:
            self.timings[test_name] = []
        self.timings[test_name].append(elapsed)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all timed tests."""
        if not self.timings:
            return {}

        summary = {}
        for test_name, times in self.timings.items():
            if times:
                summary[test_name] = {
                    "min_s": min(times),
                    "median_s": sorted(times)[len(times) // 2],
                    "max_s": max(times),
                    "mean_s": sum(times) / len(times),
                    "runs": len(times),
                }
        return summary

    def save_report(self, output_path: Path | None = None) -> None:
        """Save benchmark report to JSON file."""
        if not self.enabled or not self.timings:
            return

        if output_path is None:
            output_path = Path(".pytest_benchmark/lightweight.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "benchmarks": self.get_summary(),
            "total_tests_tracked": len(self.timings),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Benchmark report saved to %s", output_path)


# Global benchmark instance
_benchmark = LightweightBenchmark()


@pytest.fixture
def benchmark() -> LightweightBenchmark:
    """Fixture to access benchmark tracker in tests."""
    return _benchmark


def pytest_configure(config: Any) -> None:
    """Configure pytest plugins and enable lightweight benchmarking if requested."""
    config.addinivalue_line(
        "markers", "bench: mark test for lightweight benchmarking (timed separately)"
    )

    # Enable benchmarking if --benchmark-light flag is set or in CI
    if config.option.benchmark_light or ("CI" in __import__("os").environ):  # noqa: F821
        _benchmark.enabled = True
        logger.info("Lightweight benchmarking enabled")


def pytest_addoption(parser: Any) -> None:
    """Add custom command-line options."""
    parser.addoption(
        "--benchmark-light",
        action="store_true",
        default=False,
        help="Enable lightweight benchmarking of all tests (tracks execution time)",
    )
    parser.addoption(
        "--benchmark-light-report",
        default=".pytest_benchmark/lightweight.json",
        help="Path to save lightweight benchmark report (JSON format)",
    )


def pytest_runtest_setup(item: Any) -> None:
    """Setup timing for each test if benchmarking enabled."""
    if _benchmark.enabled:
        item._start_time = _benchmark.start_timing(item.nodeid)


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    """Record timing after each test."""
    if _benchmark.enabled and call.when == "call":
        if hasattr(item, "_start_time"):
            _benchmark.end_timing(item.nodeid, item._start_time)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Save benchmark report at end of test session."""
    if _benchmark.enabled:
        report_path = Path(session.config.getoption("benchmark_light_report"))
        _benchmark.save_report(report_path)
        summary = _benchmark.get_summary()
        if summary:
            logger.info(
                "Benchmarked %d tests. Median times range: %.2f-%.2f ms",
                len(summary),
                min(s["median_s"] * 1000 for s in summary.values()),
                max(s["median_s"] * 1000 for s in summary.values()),
            )

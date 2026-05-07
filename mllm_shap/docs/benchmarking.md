# Lightweight Benchmarking

This guide describes the lightweight benchmark tooling used in the test workflow.

## Overview

The benchmark setup is designed to:
- track performance with minimal overhead,
- run automatically in CI,
- catch regressions early,
- require no changes to existing tests.

## Run Locally

```bash
# Run tests with lightweight benchmark tracking
make bench-light

# Equivalent pytest command
pytest mllm_shap/tests/ --benchmark-light

# Save report to custom location
pytest mllm_shap/tests/ --benchmark-light --benchmark-light-report ./perf_report.json
```

Results are stored in `.pytest_benchmark/lightweight.json`.

## Detailed Profiling with PerfMeter

Use `PerfMeter` when you need focused measurements for a specific operation.

```python
import pytest
from mllm_shap.tests.benchmarks.perf_meter import PerfMeter

@pytest.mark.bench
def test_my_performance():
    meter = PerfMeter("operation_name")

    for _ in range(100):
        meter.measure(lambda: my_expensive_function())

    result = meter.result()
    assert result.median_ms < 1.0
```

## CI Integration

The benchmark workflow in `.github/workflows/benchmark.yaml`:
- runs on pushes and PRs targeting `main`,
- uploads artifacts,
- compares PR performance against baseline,
- flags regressions by threshold.

Default thresholds:
- slowdown > 5%: warning,
- slowdown > 10%: failure.

## Implementation Files

- `mllm_shap/tests/conftest.py`: pytest benchmark hooks
- `mllm_shap/tests/benchmarks/perf_meter.py`: PerfMeter and PerfResult
- `mllm_shap/tests/benchmarks/test__lightweight_bench.py`: example benchmark tests

## See Also

- [Benchmark Quickstart](benchmarking_quickstart.md)
- `.github/workflows/BENCHMARK_GUIDE.md`

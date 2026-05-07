# Benchmark Quickstart

## 1) Enable Lightweight Benchmarking

```bash
make bench-light
# or
pytest mllm_shap/tests/ --benchmark-light
```

Report output:
- `.pytest_benchmark/lightweight.json`

## 2) Add Explicit Performance Checks

```python
import pytest
from mllm_shap.tests.benchmarks.perf_meter import PerfMeter

@pytest.mark.bench
def test_my_operation_performance():
    meter = PerfMeter("my_operation")

    for _ in range(100):
        meter.measure(lambda: expensive_function())

    result = meter.result()
    assert result.median_ms < 5.0
```

## 3) Context Manager Style

```python
def test_with_context_manager():
    meter = PerfMeter("operation")

    for _ in range(10):
        with meter:
            expensive_function()
```

## 4) Read Results Programmatically

```python
import json
from pathlib import Path

report_path = Path(".pytest_benchmark/lightweight.json")
if report_path.exists():
    report = json.loads(report_path.read_text())
    for test_name, stats in report["benchmarks"].items():
        print(f"{test_name}: {stats['median_s']*1000:.2f}ms (median)")
```

## Tips

- Use `@pytest.mark.bench` for discoverability.
- Prefer `median_ms` over `max_ms` in assertions.
- Run multiple samples before setting strict thresholds.

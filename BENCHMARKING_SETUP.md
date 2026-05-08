# Lightweight Benchmarking Setup

Goal: add low-overhead performance tracking to tests and CI.

## What Was Added

Core files:
- mllm_shap/tests/conftest.py
   - Pytest hooks for automatic per-test timing.
   - JSON report output.
- mllm_shap/tests/benchmarks/perf_meter.py
   - PerfMeter and PerfResult utilities.
   - min, median, max, and mean metrics.
- mllm_shap/tests/benchmarks/test__lightweight_bench.py
   - Example benchmark tests and assertions.
- mllm_shap/tests/benchmarks/__init__.py

Documentation:
- mllm_shap/docs/benchmarking.md
- mllm_shap/docs/benchmarking_quickstart.md

Build and CI:
- Makefile
   - New target: make bench-light
- .github/workflows/benchmark.yaml
   - Benchmark workflow for PRs and pushes to main.

## Key Behavior

- Existing tests run without modification.
- Benchmark data is generated in a lightweight JSON report.
- PRs are compared against the latest baseline from main.
- Regressions are highlighted by threshold.

Default regression thresholds:
- >5% slowdown: warning
- >10% slowdown: workflow failure

## Local Usage

Run all tests with lightweight benchmarking:

```bash
make bench-light
```

or

```bash
pytest mllm_shap/tests/ --benchmark-light
```

Example detailed measurement:

```python
@pytest.mark.bench
def test_critical_path():
      meter = PerfMeter("operation")
      for _ in range(100):
            meter.measure(lambda: function_to_test())
      result = meter.result()
      assert result.median_ms < 5.0
```

## CI Workflow

On pull requests to main:
- Fetch latest benchmark artifact from main.
- Run benchmarks.
- Compare PR vs baseline.
- Mark regressions according to thresholds.

On pushes to main:
- Run benchmarks.
- Archive benchmark artifacts for historical tracking.

Find results in GitHub Actions:
- Workflow: Benchmark
- Artifact path: benchmark-report/lightweight.json

## Performance Notes

Typical overhead per test is very low (timestamp calls and small JSON writes).
Report generation is fast, and artifact size scales roughly with test count.

## Next Steps

1. Push the branch and confirm benchmark workflow output.
2. Add strict performance assertions to critical tests.
3. Adjust regression thresholds in benchmark.yaml if needed.

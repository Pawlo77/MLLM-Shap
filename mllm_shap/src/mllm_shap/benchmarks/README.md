# Benchmarks

Micro-benchmarks for `mllm-shap` API/performance hotspots.

## Run

From repo root:

```bash
uv run python -m mllm_shap.benchmarks.bench_api_perf --bench all
```

Or:

```bash
make benchmarks
```

## Included benches

- `mask-hash`: measures `MasksManager.get_hash()` throughput.
- `responses`: measures orchestration cost in `generate_responses()` with lightweight stubs.
- `sampling-adapter`: measures split-callback sampling path in `run_sampling_generation()`.
- `pipeline-observability`: compares `ExplainPipeline` runtime with and without sink tracing.
- `linear-update`: measures in-place `num_samples` update path used for linear runs.

## Useful flags

```bash
uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench responses \
  --n-masks 512 \
  --mask-len 1024 \
  --jobs 4 \
  --repeats 7

uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench sampling-adapter \
  --n-masks 512 \
  --mask-len 1024 \
  --jobs 4 \
  --repeats 7

uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench pipeline-observability \
  --iters 2000 \
  --n-stages 6 \
  --repeats 7
```

## CI-friendly Output

You can export results and optionally fail a run when regressions exceed thresholds.

```bash
uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench all \
  --output-json .artifacts/bench/results.json \
  --output-csv .artifacts/bench/results.csv

uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench pipeline-observability \
  --iters 2000 \
  --n-stages 6 \
  --max-overhead-pct 25

uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench responses \
  --n-masks 512 \
  --mask-len 1024 \
  --max-p50-ms 20
```

Threshold flags:

- `--max-p50-ms`: fails with exit code `1` if any selected benchmark has p50 above the limit.
- `--max-overhead-pct`: fails with exit code `1` if observed pipeline observability overhead exceeds the limit.

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
- `linear-update`: measures in-place `num_samples` update path used for linear runs.

## Useful flags

```bash
uv run python -m mllm_shap.benchmarks.bench_api_perf \
  --bench responses \
  --n-masks 512 \
  --mask-len 1024 \
  --jobs 4 \
  --repeats 7
```

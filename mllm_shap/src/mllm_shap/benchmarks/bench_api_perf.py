"""Micro-benchmarks for API/performance hotspots."""

import argparse
import csv
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

import torch
from torch import Tensor

from ..observability.sink import InMemoryObservabilitySink
from ..connectors.base.model_response import ModelResponse
from ..shap.base._generate_responses import generate_responses
from ..shap.base._masks_manager import MasksManager
from ..shap.base.approx import BaseShapApproximation
from ..shap.pipeline import ExplainContext, ExplainPipeline, ExplainState
from ..shap.pipeline.stages.sampling_adapter import run_sampling_generation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchResult:
    """Structured benchmark result for optional machine-readable exports."""

    bench: str
    """Identifier for the benchmarked operation or code path."""
    label: str
    """Human-readable label describing the benchmark parameters or scenario."""
    min_s: float
    """Minimum observed runtime across repetitions, in seconds."""
    p50_s: float
    """Median observed runtime across repetitions, in seconds."""
    max_s: float
    """Maximum observed runtime across repetitions, in seconds."""
    overhead_p50_pct: float | None = None
    """Optional overhead percentage for the median runtime."""

    @property
    def min_ms(self) -> float:
        """Minimum observed runtime across repetitions, in milliseconds."""
        return self.min_s * 1000.0

    @property
    def p50_ms(self) -> float:
        """Median observed runtime across repetitions, in milliseconds."""
        return self.p50_s * 1000.0

    @property
    def max_ms(self) -> float:
        """Maximum observed runtime across repetitions, in milliseconds."""
        return self.max_s * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark result to a dictionary for serialization."""
        payload = asdict(self)
        payload["min_ms"] = self.min_ms
        payload["p50_ms"] = self.p50_ms
        payload["max_ms"] = self.max_ms
        return payload


def _time_many(fn: Any, repeats: int) -> tuple[float, float, float]:
    """Run the given function multiple times and return min, median, and max runtimes."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times), max(times)


def _fmt(label: str, values: tuple[float, float, float], unit: str = "ms") -> str:
    """Format benchmark results for human-readable console output."""
    mul = 1000.0 if unit == "ms" else 1.0
    lo, med, hi = values
    return f"{label:<36} min={lo * mul:8.3f} {unit}  p50={med * mul:8.3f} {unit}  max={hi * mul:8.3f} {unit}"


def bench_mask_hash(iters: int, mask_len: int, repeats: int) -> list[BenchResult]:
    """Benchmark current mask hash implementation."""
    mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool)

    def run() -> None:
        for _ in range(iters):
            _ = MasksManager.get_hash(mask)

    values = _time_many(run, repeats=repeats)
    print("== mask hash ==")
    label = f"{iters}x hash(len={mask_len})"
    print(_fmt(label, values))
    return [
        BenchResult(
            bench="mask-hash",
            label=label,
            min_s=values[0],
            p50_s=values[1],
            max_s=values[2],
        )
    ]


@dataclass
class _FakeChat:
    """Minimal chat class for testing mask management and response generation."""

    cache: Any = None
    """Optional cache attribute to satisfy potential cache manager interactions during generation."""
    input_tokens_num: int = 0
    """Number of input tokens, used to determine mask lengths for generation."""
    shap_values_mask: Tensor = torch.ones(1, dtype=torch.bool)
    """SHAP values mask tensor, used to determine which tokens are considered for attribution during generation."""

    @classmethod
    def from_chat(cls, mask: Tensor, chat: "_FakeChat") -> "_FakeChat":
        """Factory method to create a _FakeChat instance based on an existing chat and mask."""
        del chat
        return cls(
            input_tokens_num=int(mask.numel()),
            shap_values_mask=torch.ones(mask.numel(), dtype=torch.bool),
        )


class _FakeModel:
    """Minimal model class for testing response generation orchestration."""

    def generate(
        self,
        chat: _FakeChat,
        keep_history: bool = False,
        **_: dict[str, Any],
    ) -> ModelResponse:
        """Simulate model response generation, optionally keeping chat history."""
        del chat
        response_chat = _FakeChat() if keep_history else None
        return ModelResponse(
            chat=response_chat,
            generated_text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            generated_audio_tokens=torch.empty((0, 0), dtype=torch.long),
            generated_modality_flag=torch.ones((3,), dtype=torch.long),
        )


class _FakeCacheManager:
    """Minimal cache manager class for testing response generation orchestration."""

    def __init__(self) -> None:
        self._cache: dict[int, ModelResponse] = {}
        self._probe = None

    def contains(self, mask_hash: int) -> bool:
        """Check if the cache contains a response for the given mask hash."""
        return mask_hash in self._cache

    def extract(self, mask_hash: int) -> ModelResponse:
        """Extract a cached response for the given mask hash, or raise an error if not found."""
        return self._cache[mask_hash]


def _mask_gen(n_masks: int, mask_len: int) -> Iterable[tuple[Tensor, int]]:
    """Generate random boolean masks and their hashes for testing."""
    for i in range(n_masks):
        mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool)
        yield mask, hash((i, mask_len))


def bench_response_generation(
    n_masks: int, mask_len: int, repeats: int, n_jobs: int, verbose: bool
) -> list[BenchResult]:
    """Benchmark masked response generation orchestration."""

    def run() -> None:
        """Set up fake chat, mask manager, and cache manager,
        then run response generation with random masks."""
        masks: list[Tensor] = []
        responses: list[ModelResponse] = []
        _ = generate_responses(
            masks=masks,
            responses=responses,
            gen=_mask_gen(n_masks=n_masks, mask_len=mask_len),
            source_chat=_FakeChat(),
            model=_FakeModel(),
            cache_manager=_FakeCacheManager(),
            n_generator_jobs=n_jobs,
            progress_bar=False,
            verbose=verbose,
        )

    values = _time_many(run, repeats=repeats)
    print("== response generation ==")
    label = f"{n_masks} masks len={mask_len} jobs={n_jobs} verbose={verbose}"
    print(
        _fmt(
            label,
            values,
        )
    )
    return [
        BenchResult(
            bench="responses",
            label=label,
            min_s=values[0],
            p50_s=values[1],
            max_s=values[2],
        )
    ]


def bench_sampling_adapter(
    n_masks: int, mask_len: int, repeats: int, n_jobs: int, verbose: bool
) -> list[BenchResult]:
    """Benchmark split-callback sampling path via shared sampling adapter."""

    def run() -> None:
        """Set up fake chat, mask manager, and cache manager,
        then run sampling generation with random splits."""

        chat = _FakeChat(
            input_tokens_num=mask_len,
            shap_values_mask=torch.ones(mask_len, dtype=torch.bool),
        )
        mask_manager = MasksManager(chat=chat)
        masks = [mask_manager.get_initial_mask(device=torch.device("cpu"))]
        responses: list[ModelResponse] = []

        def get_num_splits(_: int) -> int:
            return n_masks

        def get_next_split(
            n: int,
            device: torch.device,
            generated_masks_num: int,
            existing_masks: list[Tensor] | None = None,
        ) -> Tensor | None:
            del existing_masks
            if generated_masks_num >= n_masks:
                return None
            return torch.randint(0, 2, (1, n), dtype=torch.bool, device=device)

        _ = run_sampling_generation(
            get_next_split=get_next_split,
            get_num_splits=get_num_splits,
            mask_manager=mask_manager,
            device=torch.device("cpu"),
            masks=masks,
            allow_mask_duplicates=False,
            allow_full_or_empty=False,
            logger=logger,
            responses=responses,
            source_chat=chat,
            model=_FakeModel(),
            cache_manager=_FakeCacheManager(),
            n_generator_jobs=n_jobs,
            progress_bar=False,
            verbose=verbose,
            tqdm_desc="bench",
        )

    values = _time_many(run, repeats=repeats)
    print("== sampling adapter ==")
    label = f"{n_masks} masks len={mask_len} jobs={n_jobs} verbose={verbose}"
    print(
        _fmt(
            label,
            values,
        )
    )
    return [
        BenchResult(
            bench="sampling-adapter",
            label=label,
            min_s=values[0],
            p50_s=values[1],
            max_s=values[2],
        )
    ]


@dataclass
class _NoopStage:
    """Pipeline stage that performs no operations, used for measuring
    baseline overhead of pipeline execution and observability integration."""

    def run(self, context: ExplainContext, state: ExplainState, probe=None) -> None:
        del context, state, probe


def bench_pipeline_observability_overhead(
    iters: int,
    repeats: int,
    n_stages: int,
) -> list[BenchResult]:
    """Benchmark pipeline execution overhead with and without observability sink."""

    pipeline = ExplainPipeline(stages=tuple(_NoopStage() for _ in range(n_stages)))
    base_context = ExplainContext(
        model=None,  # type: ignore[arg-type]
        source_chat=None,  # type: ignore[arg-type]
        response_chat=None,  # type: ignore[arg-type]
        base_response=None,  # type: ignore[arg-type]
        device=None,  # type: ignore[arg-type]
        params=MappingProxyType({}),
    )

    sink = InMemoryObservabilitySink()
    observed_context = ExplainContext(
        model=None,  # type: ignore[arg-type]
        source_chat=None,  # type: ignore[arg-type]
        response_chat=None,  # type: ignore[arg-type]
        base_response=None,  # type: ignore[arg-type]
        device=None,  # type: ignore[arg-type]
        params=MappingProxyType({"observability_sink": sink, "run_id": "bench-run"}),
    )

    def baseline() -> None:
        for _ in range(iters):
            _ = pipeline.run(context=base_context, state=ExplainState())

    def observed() -> None:
        for _ in range(iters):
            sink.events.clear()
            sink.spans.clear()
            _ = pipeline.run(context=observed_context, state=ExplainState())

    base_values = _time_many(baseline, repeats=repeats)
    observed_values = _time_many(observed, repeats=repeats)
    print("== pipeline observability overhead ==")
    base_label = f"{iters}x run stages={n_stages} no-sink"
    observed_label = f"{iters}x run stages={n_stages} with-sink"
    print(_fmt(base_label, base_values))
    print(_fmt(observed_label, observed_values))
    overhead_pct = (
        100.0 * (observed_values[1] - base_values[1]) / max(base_values[1], 1e-9)
    )
    print(f"observability overhead (p50): {overhead_pct:6.2f}%")
    return [
        BenchResult(
            bench="pipeline-observability",
            label=base_label,
            min_s=base_values[0],
            p50_s=base_values[1],
            max_s=base_values[2],
        ),
        BenchResult(
            bench="pipeline-observability",
            label=observed_label,
            min_s=observed_values[0],
            p50_s=observed_values[1],
            max_s=observed_values[2],
            overhead_p50_pct=overhead_pct,
        ),
    ]


class _DummyApprox(BaseShapApproximation):
    """Minimal approximation explainer for in-place budget update benchmark."""

    def _get_num_splits(self, n: int) -> int:
        """Calculate the number of splits based on the current configuration.
        For this dummy implementation, we ignore the input and return a fixed number."""
        del n
        return self.num_samples if self.num_samples is not None else 1

    def _calculate_shap_values(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> Tensor:
        """Calculate SHAP values based on the provided masks and similarities.
        For this dummy implementation, we ignore the inputs and return a zero tensor."""
        del similarities, device
        return torch.zeros(masks.shape[-1], dtype=torch.float32)

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        """Generate the next mask split based on the current state of generation.
        For this dummy implementation, we ignore the inputs and return None to indicate no more splits."""
        del n, device, generated_masks_num, existing_masks
        return None


def bench_linear_num_samples_update(iters: int, repeats: int) -> list[BenchResult]:
    """Benchmark in-place sampling budget update path used by experiments runner."""
    approx = _DummyApprox(num_samples=100, fraction=0.6)

    def run() -> None:
        """Update the num_samples attribute of the approximation explainer
        in-place multiple times to simulate the experiments runner adjusting the sampling budget."""
        for i in range(iters):
            approx.num_samples = 100 + i

    values = _time_many(run, repeats=repeats)
    print("== linear num_samples update ==")
    label = f"{iters}x in-place num_samples set"
    print(_fmt(label, values))
    return [
        BenchResult(
            bench="linear-update",
            label=label,
            min_s=values[0],
            p50_s=values[1],
            max_s=values[2],
        )
    ]


def _write_results_json(path: str, results: list[BenchResult]) -> None:
    """Write benchmark results to JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in results]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_results_csv(path: str, results: list[BenchResult]) -> None:
    """Write benchmark results to CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in results]
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _enforce_thresholds(
    results: list[BenchResult],
    max_p50_ms: float | None,
    max_overhead_pct: float | None,
) -> int:
    """Return non-zero when configured benchmark thresholds are violated."""
    violations = 0
    if max_p50_ms is not None:
        for record in results:
            if record.p50_ms > max_p50_ms:
                print(
                    f"THRESHOLD VIOLATION: {record.bench} ({record.label}) p50={record.p50_ms:.3f}ms > {max_p50_ms:.3f}ms"
                )
                violations += 1
    if max_overhead_pct is not None:
        for record in results:
            if (
                record.overhead_p50_pct is not None
                and record.overhead_p50_pct > max_overhead_pct
            ):
                print(
                    "THRESHOLD VIOLATION: "
                    f"{record.bench} ({record.label}) overhead_p50={record.overhead_p50_pct:.2f}% > {max_overhead_pct:.2f}%"
                )
                violations += 1
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mask-len", type=int, default=512)
    parser.add_argument("--n-masks", type=int, default=256)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--max-p50-ms", type=float, default=None)
    parser.add_argument("--max-overhead-pct", type=float, default=None)
    parser.add_argument(
        "--bench",
        choices=[
            "all",
            "mask-hash",
            "responses",
            "sampling-adapter",
            "pipeline-observability",
            "linear-update",
        ],
        default="all",
    )
    args = parser.parse_args()
    results: list[BenchResult] = []

    if args.bench in ("all", "mask-hash"):
        results.extend(
            bench_mask_hash(
                iters=args.iters,
                mask_len=args.mask_len,
                repeats=args.repeats,
            )
        )
        print()
    if args.bench in ("all", "responses"):
        results.extend(
            bench_response_generation(
                n_masks=args.n_masks,
                mask_len=args.mask_len,
                repeats=args.repeats,
                n_jobs=args.jobs,
                verbose=False,
            )
        )
        print()
    if args.bench in ("all", "sampling-adapter"):
        results.extend(
            bench_sampling_adapter(
                n_masks=args.n_masks,
                mask_len=args.mask_len,
                repeats=args.repeats,
                n_jobs=args.jobs,
                verbose=False,
            )
        )
        print()
    if args.bench in ("all", "pipeline-observability"):
        results.extend(
            bench_pipeline_observability_overhead(
                iters=args.iters,
                repeats=args.repeats,
                n_stages=args.n_stages,
            )
        )
        print()
    if args.bench in ("all", "linear-update"):
        results.extend(
            bench_linear_num_samples_update(
                iters=args.iters,
                repeats=args.repeats,
            )
        )

    if args.output_json:
        _write_results_json(args.output_json, results)
        print(f"wrote JSON results: {args.output_json}")
    if args.output_csv:
        _write_results_csv(args.output_csv, results)
        print(f"wrote CSV results: {args.output_csv}")

    violations = _enforce_thresholds(
        results=results,
        max_p50_ms=args.max_p50_ms,
        max_overhead_pct=args.max_overhead_pct,
    )
    if violations > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

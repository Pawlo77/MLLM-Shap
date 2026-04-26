"""Micro-benchmarks for API/performance hotspots."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

from ..connectors.base.model_response import ModelResponse
from ..shap.base._generate_responses import generate_responses
from ..shap.base._masks_manager import MasksManager
from ..shap.base.approx import BaseShapApproximation


def _time_many(fn: Any, repeats: int) -> tuple[float, float, float]:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times), max(times)


def _fmt(label: str, values: tuple[float, float, float], unit: str = "ms") -> str:
    mul = 1000.0 if unit == "ms" else 1.0
    lo, med, hi = values
    return f"{label:<36} min={lo * mul:8.3f} {unit}  p50={med * mul:8.3f} {unit}  max={hi * mul:8.3f} {unit}"


def bench_mask_hash(iters: int, mask_len: int, repeats: int) -> None:
    """Benchmark current mask hash implementation."""
    mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool)

    def run() -> None:
        for _ in range(iters):
            _ = MasksManager.get_hash(mask)

    values = _time_many(run, repeats=repeats)
    print("== mask hash ==")
    print(_fmt(f"{iters}x hash(len={mask_len})", values))


@dataclass
class _FakeChat:
    cache: Any = None

    @classmethod
    def from_chat(cls, mask: Tensor, chat: "_FakeChat") -> "_FakeChat":
        del mask, chat
        return cls()


class _FakeModel:
    def generate(
        self,
        *,
        chat: _FakeChat,
        keep_history: bool = False,
        **_: dict[str, Any],
    ) -> ModelResponse:
        del chat
        response_chat = _FakeChat() if keep_history else None
        return ModelResponse(
            chat=response_chat,
            generated_text_tokens=torch.tensor([1, 2, 3], dtype=torch.long),
            generated_audio_tokens=torch.empty((0, 0), dtype=torch.long),
            generated_modality_flag=torch.ones((3,), dtype=torch.long),
        )


class _FakeCacheManager:
    def __init__(self) -> None:
        self._cache: dict[int, ModelResponse] = {}

    def contains(self, mask_hash: int) -> bool:
        return mask_hash in self._cache

    def extract(self, mask_hash: int) -> ModelResponse:
        return self._cache[mask_hash]


def _mask_gen(n_masks: int, mask_len: int) -> Iterable[tuple[Tensor, int]]:
    for i in range(n_masks):
        mask = torch.randint(0, 2, (mask_len,), dtype=torch.bool)
        yield mask, hash((i, mask_len))


def bench_response_generation(
    n_masks: int, mask_len: int, repeats: int, n_jobs: int, verbose: bool
) -> None:
    """Benchmark masked response generation orchestration."""

    def run() -> None:
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
    print(
        _fmt(
            f"{n_masks} masks len={mask_len} jobs={n_jobs} verbose={verbose}",
            values,
        )
    )


class _DummyApprox(BaseShapApproximation):
    """Minimal approximation explainer for in-place budget update benchmark."""

    def _get_num_splits(self, n: int) -> int:
        del n
        return self.num_samples if self.num_samples is not None else 1

    def _calculate_shap_values(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> Tensor:
        del similarities, device
        return torch.zeros(masks.shape[-1], dtype=torch.float32)

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        del n, device, generated_masks_num, existing_masks
        return None


def bench_linear_num_samples_update(iters: int, repeats: int) -> None:
    """Benchmark in-place sampling budget update path used by experiments runner."""
    approx = _DummyApprox(num_samples=100, fraction=0.6)

    def run() -> None:
        for i in range(iters):
            approx.num_samples = 100 + i

    values = _time_many(run, repeats=repeats)
    print("== linear num_samples update ==")
    print(_fmt(f"{iters}x in-place num_samples set", values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mask-len", type=int, default=512)
    parser.add_argument("--n-masks", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--bench",
        choices=["all", "mask-hash", "responses", "linear-update"],
        default="all",
    )
    args = parser.parse_args()

    if args.bench in ("all", "mask-hash"):
        bench_mask_hash(iters=args.iters, mask_len=args.mask_len, repeats=args.repeats)
        print()
    if args.bench in ("all", "responses"):
        bench_response_generation(
            n_masks=args.n_masks,
            mask_len=args.mask_len,
            repeats=args.repeats,
            n_jobs=args.jobs,
            verbose=False,
        )
        print()
    if args.bench in ("all", "linear-update"):
        bench_linear_num_samples_update(iters=args.iters, repeats=args.repeats)


if __name__ == "__main__":
    main()

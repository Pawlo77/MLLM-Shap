"""Unit tests for phase-1 sampling engine."""

import torch

from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.core.contracts import SamplingStrategy
from mllm_shap.shap.core.engine import SamplingEngine

from ...dummy import DummyChat


class _PlannedStrategy(SamplingStrategy):
    def __init__(self, splits: list[torch.Tensor]) -> None:
        self._splits = splits

    def get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        del n, device, generated_masks_num, existing_masks
        if not self._splits:
            return None
        return self._splits.pop(0)

    def get_num_splits(self, n: int) -> int | None:
        del n
        return None


def test_sampling_engine_skips_duplicates_when_disabled() -> None:
    """Duplicate masks should be skipped when allow_mask_duplicates=False."""
    split = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    strategy = _PlannedStrategy(splits=[split.clone(), split.clone()])
    engine = SamplingEngine(strategy=strategy, allow_mask_duplicates=False)

    manager = MasksManager(chat=DummyChat(num_tokens=4))
    masks = [manager.get_initial_mask(device=torch.device("cpu"))]
    gen = engine.create_generator(
        mask_manager=manager,
        device=torch.device("cpu"),
        masks=masks,
    )

    generated = list(gen)
    assert len(generated) == 1
    assert gen.stats.candidate_splits == 2
    assert gen.stats.yielded_masks == 1
    assert gen.stats.skipped_duplicates == 1
    assert gen.stats.elapsed_ms >= 0.0


def test_sampling_engine_allows_full_masks_when_enabled() -> None:
    """Full/empty masks should be accepted only when explicitly enabled."""
    full = torch.ones((1, 4), dtype=torch.bool)
    strategy = _PlannedStrategy(splits=[full])
    engine = SamplingEngine(
        strategy=strategy,
        allow_mask_duplicates=True,
        allow_full_or_empty=True,
    )

    manager = MasksManager(chat=DummyChat(num_tokens=4))
    masks = [manager.get_initial_mask(device=torch.device("cpu"))]
    gen = engine.create_generator(
        mask_manager=manager,
        device=torch.device("cpu"),
        masks=masks,
    )

    generated = list(gen)
    assert len(generated) == 1
    assert gen.stats.candidate_splits == 1
    assert gen.stats.yielded_masks == 1

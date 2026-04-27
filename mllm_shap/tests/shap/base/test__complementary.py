"""Unit tests for BaseComplementaryShapApproximation."""

from typing import Iterable

import pytest
import torch
from torch import Tensor

from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.base.complementary import BaseComplementaryShapApproximation

from ...dummy import DummyChat


class DummyComplementaryExplainer(BaseComplementaryShapApproximation):
    """Concrete subclass exposing BaseComplementaryShapApproximation internals for testing."""

    def __init__(self, splits: Iterable[Tensor] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._scheduled_splits = (
            [s.clone() for s in splits] if splits is not None else []
        )
        self._split_index = 0

    def _initialize_state(self) -> None:
        super()._initialize_state()
        self._split_index = 0

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        if self._split_index >= len(self._scheduled_splits):
            return None
        split = self._scheduled_splits[self._split_index]
        self._split_index += 1
        return split.to(device=device)

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        return torch.zeros(masks.shape[1], device=device, dtype=similarities.dtype)


class TestInitializeState:
    """Tests covering state initialization for complementary explainer."""

    def test_initialize_state_resets_cached_values(self) -> None:
        explainer = DummyComplementaryExplainer()
        _ = explainer._get_num_splits(3)
        explainer._M = torch.ones((3, 4), dtype=torch.int16)
        explainer._C = torch.ones((3, 4), dtype=torch.float32)
        explainer._zero_mask_skipped = False

        explainer._initialize_state()

        assert explainer._M is None
        assert explainer._C is None
        assert explainer._zero_mask_skipped is True
        assert explainer._get_num_splits.cache_info().currsize == 0

    def test_get_num_splits_uses_lru_cache(self) -> None:
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()

        first = explainer._get_num_splits(4)
        second = explainer._get_num_splits(4)

        assert first == second
        info = explainer._get_num_splits.cache_info()
        assert info.hits >= 1


class TestNumSplitsStatic:
    """Tests for static num-splits calculation utility."""

    def test_requires_minimal_even_samples(self) -> None:
        with pytest.raises(
            ValueError, match="at least equal to the number of features times two"
        ):
            BaseComplementaryShapApproximation._get_num_splits_static(
                n=3, num_samples=4
            )

    def test_rejects_odd_num_samples(self) -> None:
        with pytest.raises(ValueError, match="must not be odd"):
            BaseComplementaryShapApproximation._get_num_splits_static(
                n=4, num_samples=9
            )

    def test_caps_num_samples_to_maximum(self) -> None:
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=3, num_samples=100
        )
        assert result == 6  # 2**3 - 2 = 6

    def test_fraction_result_is_even(self) -> None:
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=10, fraction=0.5
        )
        assert result == 510  # (2**10 - 2) * 0.5

    def test_fraction_small_value(self) -> None:
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=3, fraction=0.1
        )
        assert result % 2 == 0


class TestIncrementCoalitionVal:
    """Tests for coalition increment helper."""

    def test_updates_first_column_for_zero_size(self) -> None:
        tensor = torch.zeros((3, 4), dtype=torch.float32)
        BaseComplementaryShapApproximation._increment_coalition_val(
            tensor=tensor,
            indices=torch.tensor([True, False, False]),
            coalition_size=0,
            value=2.5,
        )
        assert torch.all(tensor[:, 0] == 2.5)

    def test_updates_selected_indices_for_positive_size(self) -> None:
        tensor = torch.zeros((3, 4), dtype=torch.float32)
        BaseComplementaryShapApproximation._increment_coalition_val(
            tensor=tensor,
            indices=torch.tensor([False, True, True]),
            coalition_size=2,
            value=-1.0,
        )
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        assert torch.equal(tensor, expected)


class TestCalculateCMatrix:
    """Tests for complementary C matrix construction."""

    def test_raises_when_m_not_initialized(self) -> None:
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        masks = torch.ones((2, 3), dtype=torch.bool)
        sims = torch.zeros(2)
        with pytest.raises(RuntimeError, match="M matrix must be initialized"):
            explainer._calculate_C_matrix(
                masks=masks, similarities=sims, device=torch.device("cpu")
            )

    def test_raises_for_non_complementary_pairs(self) -> None:
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        explainer._M = torch.zeros((3, 4), dtype=torch.int16)
        masks = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
            ],
            dtype=torch.bool,
        )
        sims = torch.tensor([0.5, 0.1])
        with pytest.raises(ValueError, match="not complementary pairs"):
            explainer._calculate_C_matrix(
                masks=masks, similarities=sims, device=torch.device("cpu")
            )

    def test_raises_for_odd_number_of_masks(self) -> None:
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        explainer._M = torch.zeros((3, 4), dtype=torch.int16)
        masks = torch.tensor(
            [
                [True, False, False],
                [False, True, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        )
        sims = torch.tensor([0.5, 0.1, 0.2])
        with pytest.raises(ValueError, match="Masks should be in complementary pairs"):
            explainer._calculate_C_matrix(
                masks=masks, similarities=sims, device=torch.device("cpu")
            )

    def test_populates_c_matrix_for_valid_pairs(self) -> None:
        explainer = DummyComplementaryExplainer()
        explainer._initialize_state()
        explainer._M = torch.zeros((3, 4), dtype=torch.int16)
        masks = torch.tensor(
            [
                [True, False, False],
                [False, True, True],
            ],
            dtype=torch.bool,
        )
        sims = torch.tensor([0.7, 0.2])
        explainer._calculate_C_matrix(
            masks=masks, similarities=sims, device=torch.device("cpu")
        )

        assert explainer._C is not None
        expected = torch.zeros((3, 4), dtype=sims.dtype)
        expected[0, 1] = 0.5
        expected[1, 2] = -0.5
        expected[2, 2] = -0.5
        torch.testing.assert_close(explainer._C, expected)


class TestMasksGenerator:
    """Tests for complementary mask generator behavior."""

    def test_generates_complementary_pairs_and_updates_m(self) -> None:
        device = torch.device("cpu")
        splits = [
            torch.tensor([[True, False, False]], dtype=torch.bool),
            torch.tensor([[True, True, False]], dtype=torch.bool),
        ]
        explainer = DummyComplementaryExplainer(splits=splits)
        explainer._initialize_state()
        chat = DummyChat(num_tokens=3)
        manager = MasksManager(chat=chat)

        gen = explainer._get_masks_generator(
            mask_manager=manager,
            device=device,
            masks=[],
        )

        produced: list[tuple[Tensor, int]] = []
        try:
            while True:
                produced.append(next(gen))
        except StopIteration:
            pass

        assert len(produced) == len(splits) * 2
        assert gen.generated_masks == len(splits) * 2
        for mask, _ in produced:
            assert mask.dtype == torch.bool
            assert mask.shape == (chat.input_tokens_num,)
            assert manager.seen(mask=mask)

        expected_M = torch.zeros((manager.n, manager.n + 1), dtype=torch.int16)
        for split in splits:
            positive = split.squeeze(0)
            complement = ~positive
            size_pos = int(positive.sum().item())
            size_neg = int(complement.sum().item())
            BaseComplementaryShapApproximation._increment_coalition_val(
                expected_M, positive, size_pos, 1
            )
            BaseComplementaryShapApproximation._increment_coalition_val(
                expected_M, complement, size_neg, 1
            )

        torch.testing.assert_close(explainer._M, expected_M)

    def test_skips_duplicates_when_only_unique(self) -> None:
        device = torch.device("cpu")
        split = torch.tensor([[True, False, False]], dtype=torch.bool)
        explainer = DummyComplementaryExplainer(splits=[split, split])
        explainer._initialize_state()
        chat = DummyChat(num_tokens=3)
        manager = MasksManager(chat=chat)

        gen = explainer._get_masks_generator(
            mask_manager=manager,
            device=device,
            masks=[],
        )

        produced = list(gen)

        assert len(produced) == 2
        assert gen.generated_masks == 2

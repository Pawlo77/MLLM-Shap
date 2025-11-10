"""Unit tests for ComplementaryNeymanShapExplainer class."""

import pytest
import torch
from mllm_shap.shap.neyman import ComplementaryNeymanShapExplainer, _Step
from mllm_shap.shap.base._masks_manager import MasksManager
from torch import Tensor


class DummyModel:
    """A minimal dummy model that simulates MLLM model behavior."""

    def __call__(self, *args, **kwargs) -> None:
        return None


class DummyChat:
    """A minimal dummy chat instance used for testing."""

    torch_device: torch.device = torch.device("cpu")

    def __init__(self) -> None:
        self.tokens = ["token1", "token2"]


class DummyResponse:
    """A minimal dummy response object containing a DummyChat."""

    def __init__(self) -> None:
        self.chat = DummyChat()


class DummyComplementaryNeymanExplainer(ComplementaryNeymanShapExplainer):
    """
    Concrete subclass of ComplementaryNeymanShapExplainer used for testing.
    Implements abstract methods with deterministic, simplified behavior.
    """

    def __init__(
        self,
        initial_num_samples: int | None = None,
        initial_fraction: float | None = 0.2,
    ) -> None:
        super().__init__(initial_num_samples=initial_num_samples, initial_fraction=initial_fraction)
        # Initialize deterministic state
        self._first_call: bool = True
        self._zero_mask_skipped: bool = True
        self._M: Tensor = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        self._C: Tensor = torch.tensor([[4, 6], [2, 8]], dtype=torch.float32)
        self._M_hat: Tensor = torch.tensor([1, 1], dtype=torch.float32)
        self._initial_num_splits: int = 1
        self._step: _Step = _Step.INITIAL_SAMPLING
        self._i: int = 0
        self._j: int = 0
        self._next_mask: Tensor | None = None
        self._last_iter_first_row_sum: float | None = 10

    def _get_similarities(self, responses: list, model: DummyModel) -> Tensor:
        """
        Return a constant similarity tensor of ones (for testing purposes).
        """
        return torch.ones(len(responses), dtype=torch.float32)

    def _calculate_C_matrix(self, masks: Tensor, similarities: Tensor, device: torch.device) -> None:
        """
        Set the C-matrix to a constant tensor of twos for deterministic behavior.
        """
        self._C = torch.ones_like(self._M) * 2


class TestComplementaryNeymanShapExplainerNumSplits:
    """Tests for the _get_num_splits() method."""

    @pytest.fixture
    def explainer(self) -> DummyComplementaryNeymanExplainer:
        """Provide a default explainer fixture."""
        return DummyComplementaryNeymanExplainer(initial_num_samples=2, initial_fraction=0.5)

    def test_num_splits_returns_integer(self, explainer: DummyComplementaryNeymanExplainer) -> None:
        """Ensure _get_num_splits() returns a valid integer greater than or equal to the initial splits."""
        result = explainer._get_num_splits(n=5)
        assert isinstance(result, int)
        assert result >= explainer._initial_num_splits

    def test_initial_num_splits_too_large_raises(self) -> None:
        """Verify that a ValueError is raised when initial_num_samples > total possible splits."""
        explainer: DummyComplementaryNeymanExplainer = DummyComplementaryNeymanExplainer(initial_num_samples=10)
        with pytest.raises(
            ValueError,
            match="Initial number of splits .* is larger than total number of splits .*",
        ):
            explainer._get_num_splits(n=2)


class TestComplementaryNeymanShapExplainerNextSplit:
    """Tests for the _get_next_split() method."""

    @pytest.fixture
    def explainer(self) -> DummyComplementaryNeymanExplainer:
        """Provide a default explainer fixture."""
        return DummyComplementaryNeymanExplainer(initial_num_samples=2)

    def test_complementary_mask_pair_generation(self, explainer: DummyComplementaryNeymanExplainer) -> None:
        """Tests that complementary mask pairs are generated correctly."""
        chat = DummyChat()
        device = torch.device("cpu")
        explainer._M = None
        chat.shap_values_mask = torch.tensor([True, True, True, True], dtype=torch.bool, device=device)
        chat.input_tokens_num = 4
        mask_manager = MasksManager(chat=chat)
        explainer._first_call = False

        gen = explainer._get_masks_generator(
            mask_manager=mask_manager,
            device=device,
            masks=[],
        )
        mask_1, _ = next(gen)
        mask_2, _ = next(gen)

        # mask2 should be complement of mask1
        assert torch.equal(mask_2, ~mask_1)

    def test_raises_without_update_M(self, explainer: DummyComplementaryNeymanExplainer) -> None:
        """Verify that None is returned when all splits are already generated."""
        n = 5
        with pytest.raises(RuntimeError, match="_update_M_position did not update position correctly"):
            explainer._get_next_split(n, device=torch.device("cpu"), generated_masks_num=10)


class TestComplementaryNeymanShapExplainerCalculateShapValues:
    """Tests for the _calculate_shap_values() method."""

    def test_shap_values_computation(self) -> None:
        """Check that SHAP value computation produces a tensor of the correct shape."""
        explainer: DummyComplementaryNeymanExplainer = DummyComplementaryNeymanExplainer()
        device: torch.device = torch.device("cpu")
        masks: Tensor = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities: Tensor = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result: Tensor = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)
        assert isinstance(result, Tensor)
        assert result.shape[0] == explainer._M.shape[0]

    def test_raises_if_zero_mask_not_skipped(self) -> None:
        """Ensure a RuntimeError is raised when zero mask was not skipped."""
        explainer: DummyComplementaryNeymanExplainer = DummyComplementaryNeymanExplainer()
        explainer._zero_mask_skipped = False
        device: torch.device = torch.device("cpu")
        masks: Tensor = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities: Tensor = torch.tensor([1.0, 1.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Zero mask was not skipped"):
            explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)

"""Unit tests for ComplementaryNeymanShapExplainer class."""

import pytest
import torch
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.neyman import ComplementaryNeymanShapExplainer
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


class DummyComplementaryNeymanShapExplainer(ComplementaryNeymanShapExplainer):
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
        self._first_call = True
        self._zero_mask_skipped = True
        self._M = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        self._C = torch.tensor([[4, 6], [2, 8]], dtype=torch.float32)
        self._M_hat = torch.tensor([1, 1], dtype=torch.float32)
        self._initial_num_splits = 1

        self._initialize_state()

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
    def explainer(self) -> DummyComplementaryNeymanShapExplainer:
        """Provide a default explainer fixture."""
        return DummyComplementaryNeymanShapExplainer(initial_num_samples=2, initial_fraction=0.5)

    def test_num_splits_returns_integer(self, explainer: DummyComplementaryNeymanShapExplainer) -> None:
        """Ensure _get_num_splits() returns a valid integer greater than or equal to the initial splits."""
        result = explainer._get_num_splits(n=5)
        assert isinstance(result, int)
        assert result >= explainer._initial_num_splits

    def test_initial_num_splits_too_large_raises(self) -> None:
        """Verify that a ValueError is raised when initial_num_samples > total possible splits."""
        explainer = DummyComplementaryNeymanShapExplainer(initial_num_samples=10)
        with pytest.raises(
            ValueError,
            match="Initial number of splits .* is larger than total number of splits .*",
        ):
            explainer._get_num_splits(n=2)


class TestComplementaryNeymanShapExplainerNextSplit:
    """Tests for the _get_next_split() method."""

    @pytest.fixture
    def explainer(self) -> DummyComplementaryNeymanShapExplainer:
        """Provide a default explainer fixture."""
        return DummyComplementaryNeymanShapExplainer(initial_num_samples=2)

    def test_complementary_mask_pair_generation(self, explainer: DummyComplementaryNeymanShapExplainer) -> None:
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


class TestComplementaryNeymanShapExplainerCalculateShapValues:
    """Tests for the _calculate_shap_values() method."""

    def test_shap_values_computation(self) -> None:
        """Check that SHAP value computation produces a tensor of the correct shape."""
        explainer = DummyComplementaryNeymanShapExplainer()
        explainer._initialize_state()
        explainer._M = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        explainer._C = torch.tensor([[4, 6], [2, 8]], dtype=torch.float32)
        device = torch.device("cpu")
        masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)
        assert isinstance(result, Tensor)
        assert result.shape[0] == explainer._M.shape[0]

    def test_raises_if_zero_mask_not_skipped(self) -> None:
        """Ensure a RuntimeError is raised when zero mask was not skipped."""
        explainer = DummyComplementaryNeymanShapExplainer()
        explainer._zero_mask_skipped = False
        device = torch.device("cpu")
        masks = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Zero mask was not skipped"):
            explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)

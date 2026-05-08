"""Unit tests for BaseShapApproximation class."""

from typing import Any

import pytest
import torch
from mllm_shap.shap.base.approx import BaseShapApproximation
from torch import Tensor


class DummyExplainer(BaseShapApproximation):
    """Concrete subclass for testing abstract BaseShapApproximation."""

    def _get_next_split(
        self,
        n: int,
        device: torch.device,
        generated_masks_num: int,
        existing_masks: list[Tensor] | None = None,
    ) -> Tensor | None:
        return super()._get_next_split(
            n=n,
            device=device,
            generated_masks_num=generated_masks_num,
            existing_masks=existing_masks,
        )

    def _get_num_splits(self, n: int) -> int:
        # Return a large enough value to avoid budget errors
        return 100

    def _calculate_shap_values(
        self,
        masks: Tensor,
        similarities: Tensor,
        device: torch.device,
    ) -> Tensor:
        return torch.zeros((masks.size(0),), device=device)


class TestBaseShapApproximationInit:
    """Unit tests for BaseShapApproximation initialization and validation."""

    def test_init_with_valid_fraction(self) -> None:
        """Should correctly initialize with a valid fraction and no num_samples."""
        explainer = DummyExplainer(num_samples=None, fraction=0.8)
        assert explainer.num_samples is None
        assert explainer.fraction == 0.8

    def test_init_with_valid_num_samples(self) -> None:
        """Should correctly initialize with a valid num_samples and default fraction."""
        explainer = DummyExplainer(num_samples=10)
        assert explainer.num_samples == 10
        assert explainer.fraction == 0.6  # default

    def test_init_with_minimal_num_samples(self) -> None:
        """Should accept num_samples = -1 as valid (minimal sampling mode)."""
        explainer = DummyExplainer(num_samples=-1)
        assert explainer.num_samples == -1

    def test_init_raises_if_both_none(self) -> None:
        """Should raise ValueError if both num_samples and fraction are None."""
        with pytest.raises(
            ValueError, match="Either num_samples or fraction must be provided"
        ):
            DummyExplainer(num_samples=None, fraction=None)

    def test_init_raises_for_invalid_fraction_type(self) -> None:
        """Should raise ValueError if fraction is not a float."""
        with pytest.raises(ValueError, match="fraction must be a float"):
            DummyExplainer(fraction="invalid")

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_init_raises_for_invalid_fraction_value(self, fraction: float) -> None:
        """Should raise ValueError for fractions outside (0,1]."""
        with pytest.raises(ValueError, match="fraction must be a float in the range"):
            DummyExplainer(fraction=fraction)

    @pytest.mark.parametrize("num_samples", [0, -5, 2.5, "abc"])
    def test_init_raises_for_invalid_num_samples(self, num_samples: Any) -> None:
        """Should raise ValueError for invalid num_samples values."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            DummyExplainer(num_samples=num_samples)


class TestGenerateMinimalSplits:
    """Tests for static minimal mask generation."""

    def test_output_shape(self) -> None:
        """Test that generate_minimal_splits produces correct shape."""
        device = torch.device("cpu")
        n = 4
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert masks.shape == (n + 1, n)

    def test_first_row_is_all_false(self) -> None:
        """Test that the first row of generated minimal splits is all False."""
        device = torch.device("cpu")
        n = 3
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert torch.equal(masks[0], torch.zeros(n, dtype=torch.bool, device=device))

    def test_each_subsequent_row_has_single_false(self) -> None:
        """Each subsequent row should have exactly one False at the correct position."""
        device = torch.device("cpu")
        n = 5
        masks = DummyExplainer._generate_minimal_splits(n, device)
        for i in range(1, n + 1):
            row = masks[i]
            assert torch.sum(~row) == 1  # exactly one False
            false_index = torch.where(~row)[0].item()
            assert false_index == i - 1

    def test_dtype_and_device(self) -> None:
        """Generated minimal splits have correct dtype and device."""
        device = torch.device("cpu")
        n = 2
        masks = DummyExplainer._generate_minimal_splits(n, device)
        assert masks.dtype == torch.bool
        assert masks.device == device


class TestGetNextSplitBase:
    """Tests for _get_next_split_base behavior and internal logic."""

    def setup_method(self) -> None:
        self.device = torch.device("cpu")
        self.n = 3
        self.explainer = DummyExplainer(num_samples=5)
        # initialize internal state
        self.explainer._first_call = True
        self.explainer._zero_mask_skipped = False
        self.explainer._base_masks = None
        self.explainer._base_calls_num = 0

    def test_first_call_generates_base_masks(self) -> None:
        """Should generate base masks on first call."""
        mask = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=0,
        )
        assert isinstance(mask, Tensor)
        assert self.explainer._base_masks is not None
        assert mask.shape == (self.n,)

    def test_returns_none_when_generated_masks_exceed_base(self) -> None:
        """Should return None when generated masks exceed base mask count."""
        self.explainer._get_next_split_base(self.n, self.device, 0)
        num_base = self.explainer._base_masks.shape[0]
        result = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=num_base,
        )
        assert result is None

    def test_runtime_error_if_base_masks_missing(self) -> None:
        """Should raise RuntimeError if base masks are unexpectedly missing."""
        self.explainer.include_minimal_masks = True
        self.explainer._base_masks = None
        self.explainer._first_call = False
        with pytest.raises(RuntimeError, match="Base masks are not present"):
            self.explainer._get_next_split_base(
                n=self.n,
                device=self.device,
                generated_masks_num=1,
            )

    def test_runtime_error_multiple_base_rejected(self) -> None:
        """Should raise RuntimeError when multiple base masks are rejected."""
        self.explainer._first_call = False
        self.explainer._zero_mask_skipped = True
        self.explainer._base_masks = torch.zeros((2, 3), dtype=torch.bool)
        self.explainer._base_calls_num = 0
        with pytest.raises(RuntimeError, match="Multiple base masks were rejected"):
            self.explainer._get_next_split_base(
                n=self.n,
                device=self.device,
                generated_masks_num=0,
            )

    def test_returns_none_when_minimal_masks_disabled(self) -> None:
        """Should return None immediately when minimal masks are disabled."""
        self.explainer.include_minimal_masks = False
        mask = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=0,
        )
        assert mask is None

    def test_returns_none_after_consuming_all_base_masks(self) -> None:
        """Should yield None once all minimal masks have been emitted."""
        first_mask = self.explainer._get_next_split_base(self.n, self.device, 0)
        assert first_mask is not None
        total = self.explainer._base_masks.shape[0]
        for idx in range(1, total):
            mask = self.explainer._get_next_split_base(self.n, self.device, idx)
            assert mask is not None
        result = self.explainer._get_next_split_base(self.n, self.device, total)
        assert result is None

    def test_raises_when_budget_too_small_for_minimal_masks(self) -> None:
        """Should raise if reported sampling budget is smaller than minimal mask count."""
        from types import MethodType

        total = self.explainer._generate_minimal_splits(self.n, self.device).shape[0]

        def limited_budget(_self: DummyExplainer, _n: int) -> int:
            return total - 1

        self.explainer._get_num_splits = MethodType(limited_budget, self.explainer)
        with pytest.raises(RuntimeError, match="Not enough sampling budget"):
            self.explainer._get_next_split_base(self.n, self.device, 0)

    def test_returns_none_when_minimal_generator_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive branch: if minimal split generation fails, no base split is returned."""

        monkeypatch.setattr(
            BaseShapApproximation,
            "_generate_minimal_splits",
            staticmethod(lambda n, device: None),
        )

        result = self.explainer._get_next_split_base(
            n=self.n,
            device=self.device,
            generated_masks_num=0,
        )
        assert result is None


class TestGetRandomSplit:
    """Tests for the random split helper."""

    def test_returns_binary_tensor_with_expected_shape(self) -> None:
        """Default call should produce boolean mask of shape (1, n)."""
        torch.manual_seed(0)
        mask = DummyExplainer._get_random_split(n=4, device=torch.device("cpu"))
        assert mask.shape == (1, 4)
        assert mask.dtype == torch.bool

    def test_honors_true_values_constraint(self) -> None:
        """When true_values_num is provided, mask should contain that many True values."""
        torch.manual_seed(1)
        mask = DummyExplainer._get_random_split(
            n=6,
            device=torch.device("cpu"),
            true_values_num=2,
        )
        assert mask.sum().item() == 2

    def test_include_token_keeps_position_true(self) -> None:
        """include_token should guarantee the specified index is True."""
        torch.manual_seed(2)
        mask = DummyExplainer._get_random_split(
            n=5,
            device=torch.device("cpu"),
            true_values_num=3,
            include_token=2,
        )
        assert mask.shape == (1, 5)
        assert mask[0, 2]
        assert mask.sum().item() == 3


class TestGetNextSplit:
    """Tests for _get_next_split orchestrating base and random masks."""

    def test_prefers_base_masks_before_random(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should emit base mask before falling back to random splits."""
        explainer = DummyExplainer(num_samples=5)
        device = torch.device("cpu")
        captured: list[str] = []
        explainer._initialize_state()

        def fake_random_split(n: int, device: torch.device, **_: Any) -> Tensor:
            captured.append("random")
            return torch.ones((1, n), dtype=torch.bool, device=device)

        monkeypatch.setattr(
            DummyExplainer,
            "_get_random_split",
            staticmethod(fake_random_split),
        )

        mask = explainer._get_next_split(
            n=3,
            device=device,
            generated_masks_num=0,
            existing_masks=[],
        )
        assert mask is not None
        assert captured == []

    def test_calls_random_after_minimal_masks_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should fall back to random mask generation after emitting minimal masks."""
        explainer = DummyExplainer(num_samples=10)
        device = torch.device("cpu")
        random_masks: list[Tensor] = []
        explainer._initialize_state()

        def fake_random_split(n: int, device: torch.device, **_: Any) -> Tensor:
            mask = torch.ones((1, n), dtype=torch.bool, device=device)
            random_masks.append(mask)
            return mask

        monkeypatch.setattr(
            DummyExplainer,
            "_get_random_split",
            staticmethod(fake_random_split),
        )

        total = explainer._generate_minimal_splits(3, device).shape[0]
        for generated in range(total + 1):
            explainer._get_next_split(
                n=3,
                device=device,
                generated_masks_num=generated,
                existing_masks=[],
            )

        assert random_masks  # random masks were produced

    def test_respects_sampling_budget_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return None when sampling budget has been reached."""
        explainer = DummyExplainer(num_samples=1)
        device = torch.device("cpu")
        explainer._initialize_state()

        def limited_budget(_self: DummyExplainer, n: int) -> int:
            return explainer._generate_minimal_splits(n, device).shape[0]

        monkeypatch.setattr(
            explainer,
            "_get_num_splits",
            limited_budget.__get__(explainer, DummyExplainer),
        )

        total = explainer._generate_minimal_splits(3, device).shape[0]
        for generated in range(total):
            explainer._get_next_split(
                n=3,
                device=device,
                generated_masks_num=generated,
                existing_masks=[],
            )
        mask = explainer._get_next_split(
            n=3,
            device=device,
            generated_masks_num=total,
            existing_masks=[],
        )
        assert mask is None


class TestValidateSamplingParams:
    """Additional tests for sampling parameter validation."""

    def test_accepts_valid_fraction_only(self) -> None:
        """Providing only fraction within (0,1] should be valid."""
        BaseShapApproximation._validate_sampling_params(num_samples=None, fraction=0.5)

    def test_accepts_minimal_num_samples(self) -> None:
        """num_samples == -1 should be allowed."""
        BaseShapApproximation._validate_sampling_params(num_samples=-1, fraction=None)

    @pytest.mark.parametrize("fraction", [None, 0.5])
    def test_rejects_invalid_num_samples_type(self, fraction: float | None) -> None:
        """Non-integer num_samples should be rejected regardless of fraction."""
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            BaseShapApproximation._validate_sampling_params(
                num_samples=1.2, fraction=fraction
            )

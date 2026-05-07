"""Edge-case and coverage tests for shap modules."""

import math

import pytest
import torch
from torch import Tensor

from mllm_shap.shap.base.approx import BaseShapApproximation
from mllm_shap.shap.complementary import BaseComplementaryShapApproximation
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.monte_carlo.utils import approximate_budget
from mllm_shap.shap.monte_carlo._base import BaseMcShapExplainer
from mllm_shap.shap.base._validators import BaseShapCallConfig
from mllm_shap.shap.base._mask_generator import MaskGenerator as MaskGeneratorABC
from mllm_shap.shap.hierarchical.graph import GraphNode


# ──────────────── Helpers / Stubs ────────────────


class StubApprox(BaseShapApproximation):
    """Minimal concrete BaseShapApproximation for testing."""

    def __init__(self, **kwargs):
        self.include_minimal_masks = kwargs.pop("include_minimal_masks", True)
        self.num_samples = kwargs.pop("num_samples", None)
        self.fraction = kwargs.pop("fraction", 1.0)
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0

    def _get_num_splits(self, n: int) -> int:
        if self.num_samples is not None:
            return self.num_samples
        return int((2**n - 1) * self.fraction)

    def _calculate_shap_values(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> Tensor:
        return torch.ones(masks.shape[1], device=device)


class StubMcExplainer(BaseMcShapExplainer):
    """Minimal concrete MC explainer."""

    def __init__(self, **kwargs):
        super().__init__()
        self.num_samples = kwargs.get("num_samples")
        self.fraction = kwargs.get("fraction", 0.5)
        self.include_minimal_masks = kwargs.get("include_minimal_masks", True)
        self._first_call = True
        self._zero_mask_skipped = False
        self._base_masks = None
        self._base_calls_num = 0


# ──────────────── Monte Carlo utils ────────────────


class TestApproximateBudget:
    """Tests for approximate_budget Hoeffding bound."""

    def test_basic_computation(self) -> None:
        """Checks that basic computation."""
        result = approximate_budget(error_bound=0.1, confidence=0.95)
        expected = math.ceil(2 * math.log(2 / (1 - 0.95)) / (0.1**2))
        assert result == expected

    def test_tighter_bound_needs_more_samples(self) -> None:
        """Checks that tighter bound needs more samples."""
        loose = approximate_budget(error_bound=0.5, confidence=0.9)
        tight = approximate_budget(error_bound=0.1, confidence=0.9)
        assert tight > loose

    def test_higher_confidence_needs_more_samples(self) -> None:
        """Checks that higher confidence needs more samples."""
        low = approximate_budget(error_bound=0.1, confidence=0.8)
        high = approximate_budget(error_bound=0.1, confidence=0.99)
        assert high > low

    def test_returns_integer(self) -> None:
        """Checks that it returns integer."""
        result = approximate_budget(error_bound=0.3, confidence=0.95)
        assert isinstance(result, int)


# ──────────────── BaseShapApproximation._get_next_split_base ────────────────


class TestApproxGetNextSplitBase:
    """Edge cases for _get_next_split_base."""

    def test_no_minimal_masks_returns_none(self) -> None:
        """When include_minimal_masks=False, _get_next_split_base returns None."""
        e = StubApprox(include_minimal_masks=False)
        result = e._get_next_split_base(
            n=3, device=torch.device("cpu"), generated_masks_num=0
        )
        assert result is None

    def test_first_call_generates_base_masks(self) -> None:
        """First call creates minimal splits tensor."""
        e = StubApprox(num_samples=10)
        mask = e._get_next_split_base(
            n=3, device=torch.device("cpu"), generated_masks_num=0
        )
        assert mask is not None
        assert e._base_masks is not None
        assert e._base_masks.shape == (4, 3)  # n+1, n

    def test_zero_mask_rejected_shifts_base_masks(self) -> None:
        """When zero mask is rejected (generated_masks_num still 0 after call),
        base_masks advances past the zero mask."""
        e = StubApprox(num_samples=10)
        # Simulate first call already happened (returned all-False mask, incremented counter)
        e._first_call = False
        e._base_masks = BaseShapApproximation._generate_minimal_splits(
            n=3, device=torch.device("cpu")
        )
        e._base_calls_num = 1  # first call already returned mask[0]
        e._zero_mask_skipped = False
        # The mask generator rejected the zero mask, so generated_masks_num stays 0
        mask = e._get_next_split_base(
            n=3, device=torch.device("cpu"), generated_masks_num=0
        )
        assert e._zero_mask_skipped is True
        assert mask is not None
        # After slicing, first mask is the identity mask with 2 True values
        assert mask.sum().item() == 2

    def test_multiple_rejections_raise_runtime_error(self) -> None:
        """If base masks are rejected multiple times, RuntimeError is raised."""
        e = StubApprox(num_samples=10)
        e._first_call = False
        e._zero_mask_skipped = True
        e._base_masks = BaseShapApproximation._generate_minimal_splits(
            n=3, device=torch.device("cpu")
        )
        with pytest.raises(RuntimeError, match="Multiple base masks were rejected"):
            e._get_next_split_base(
                n=3, device=torch.device("cpu"), generated_masks_num=0
            )

    def test_budget_too_small_raises(self) -> None:
        """If num_splits < base_masks count, RuntimeError."""
        e = StubApprox(num_samples=2)  # budget=2, need 4 base masks for n=3
        with pytest.raises(RuntimeError, match="Not enough sampling budget"):
            e._get_next_split_base(
                n=3, device=torch.device("cpu"), generated_masks_num=0
            )

    def test_base_calls_num_mismatch_raises(self) -> None:
        """If _base_calls_num doesn't match generated_masks_num, raise."""
        e = StubApprox(num_samples=10)
        e._first_call = False
        e._zero_mask_skipped = False
        e._base_masks = BaseShapApproximation._generate_minimal_splits(
            n=3, device=torch.device("cpu")
        )
        e._base_calls_num = 5  # mismatch
        with pytest.raises(RuntimeError, match="Multiple base masks were rejected"):
            e._get_next_split_base(
                n=3, device=torch.device("cpu"), generated_masks_num=2
            )


# ──────────────── BaseShapApproximation._get_random_split ────────────────


class TestGetRandomSplit:
    """Tests for _get_random_split static method."""

    def test_no_constraints_random(self) -> None:
        """Without true_values_num, generates random boolean."""
        torch.manual_seed(0)
        mask = BaseShapApproximation._get_random_split(n=10, device=torch.device("cpu"))
        assert mask.shape == (1, 10)
        assert mask.dtype == torch.bool

    def test_exact_true_count(self) -> None:
        """With true_values_num=k, exactly k bits are True."""
        mask = BaseShapApproximation._get_random_split(
            n=8, device=torch.device("cpu"), true_values_num=3
        )
        assert mask.sum().item() == 3

    def test_include_token_forces_position(self) -> None:
        """include_token=idx ensures that index is True."""
        for idx in range(5):
            mask = BaseShapApproximation._get_random_split(
                n=5, device=torch.device("cpu"), true_values_num=2, include_token=idx
            )
            assert mask[0, idx].item() is True
            assert mask.sum().item() == 2

    def test_include_token_with_all_true(self) -> None:
        """include_token with true_values_num=n fills all True."""
        mask = BaseShapApproximation._get_random_split(
            n=4, device=torch.device("cpu"), true_values_num=4, include_token=2
        )
        assert mask.all()


# ──────────────── MC _get_num_splits edge cases ────────────────


class TestMcGetNumSplits:
    """Additional MC _get_num_splits edge cases."""

    def test_num_samples_negative_one_without_minimal_raises(self) -> None:
        """num_samples=-1 with include_minimal_masks=False → ValueError."""
        e = StubMcExplainer(num_samples=-1)
        e.include_minimal_masks = False
        with pytest.raises(ValueError, match="cannot be -1"):
            e._get_num_splits(n=4)

    def test_num_samples_less_than_features_warns_returns_n_plus_1(self) -> None:
        """Budget < n+1 → clamp to n+1."""
        e = StubMcExplainer(num_samples=2)
        result = e._get_num_splits(n=5)
        assert result == 6  # n+1

    def test_fraction_too_small_clamps_to_minimal(self) -> None:
        """Very small fraction → clamped to n+1."""
        e = StubMcExplainer(num_samples=None, fraction=0.001)
        result = e._get_num_splits(n=5)
        assert result == 6  # n+1


# ──────────────── MaskGenerator ABC ────────────────


class TestMaskGeneratorABC:
    """Tests for the abstract MaskGenerator base class."""

    def test_concrete_generator_works(self) -> None:
        """Concrete subclass yields masks correctly."""
        from typing import Generator

        class TwoMaskGen(MaskGeneratorABC):
            def _mask_iter(self) -> Generator[tuple[Tensor | None, int], None, None]:
                yield torch.ones(3, dtype=torch.bool), 1
                yield torch.zeros(3, dtype=torch.bool), 2

        gen = TwoMaskGen()
        results = list(gen)
        assert len(results) == 2
        assert results[0][1] == 1
        assert results[1][1] == 2


# ──────────────── MasksManager edge cases ────────────────


class TestMasksManagerEdgeCases:
    """Additional MasksManager edge-case tests."""

    def _make_chat(self, mask: Tensor) -> object:
        """Helper to create a minimal chat-like object."""

        class C:
            shap_values_mask = mask
            input_tokens_num = mask.shape[0]
            torch_device = torch.device("cpu")

        return C()

    def test_hash_deterministic(self) -> None:
        """Same mask → same hash."""
        chat = self._make_chat(torch.tensor([True, False, True, True]))
        mgr = MasksManager(chat=chat)
        m = torch.tensor([True, False, True])
        assert mgr.get_hash(m) == mgr.get_hash(m.clone())

    def test_hash_different_masks_differ(self) -> None:
        """Different masks → different hashes (with high probability)."""
        chat = self._make_chat(torch.tensor([True, True, True]))
        mgr = MasksManager(chat=chat)
        h1 = mgr.get_hash(torch.tensor([True, False, True]))
        h2 = mgr.get_hash(torch.tensor([False, True, True]))
        assert h1 != h2

    def test_hash_2d_mask_with_wrong_shape_raises(self) -> None:
        """2D mask with >1 row raises MaskError."""
        from mllm_shap.errors import MaskError

        chat = self._make_chat(torch.tensor([True, True]))
        mgr = MasksManager(chat=chat)
        bad_mask = torch.ones(2, 3, dtype=torch.bool)
        with pytest.raises(MaskError):
            mgr.get_hash(bad_mask)


# ──────────────── Complementary _get_num_splits_static ────────────────


class TestComplementaryNumSplitsStatic:
    """Edge cases for _get_num_splits_static."""

    def test_num_samples_minus_one_with_minimal(self) -> None:
        """num_samples=-1 + include_minimal_masks → 2*n."""
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=4, num_samples=-1, include_minimal_masks=True
        )
        assert result == 8

    def test_num_samples_minus_one_without_minimal_raises(self) -> None:
        """num_samples=-1 without minimal → ValueError."""
        with pytest.raises(ValueError):
            BaseComplementaryShapApproximation._get_num_splits_static(
                n=4, num_samples=-1, include_minimal_masks=False
            )

    def test_num_samples_too_small_with_force_minimal_raises(self) -> None:
        """num_samples < 2*n with force_minimal → ValueError."""
        with pytest.raises(ValueError):
            BaseComplementaryShapApproximation._get_num_splits_static(
                n=5, num_samples=3, force_minimal=True
            )

    def test_num_samples_larger_than_max_clamps(self) -> None:
        """num_samples > 2^n-2 → clamp to 2^n-2."""
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=3, num_samples=999, force_minimal=False
        )
        assert result == 6  # 2^3 - 2

    def test_fraction_basic(self) -> None:
        """Fraction-based computation works."""
        result = BaseComplementaryShapApproximation._get_num_splits_static(
            n=4, fraction=0.5, force_minimal=True
        )
        expected = max(2 * 4, int((2**4 - 2) * 0.5))
        assert result == expected


# ──────────────── GraphNode ────────────────


class TestGraphNode:
    """Tests for hierarchical GraphNode."""

    def test_default_fields_none(self) -> None:
        """Checks that default fields none."""
        node = GraphNode()
        assert node.shap_values is None
        assert node.children is None
        assert node.group_ids is None
        assert node.group_mask is None

    def test_display_no_crash(self) -> None:
        """display() should not crash even for empty node."""
        node = GraphNode(shap_values=torch.tensor([1.0, 2.0]))
        node.display()  # just ensure no exception

    def test_nested_children(self) -> None:
        """GraphNode with children."""
        child = GraphNode(shap_values=torch.tensor([1.0]))
        parent = GraphNode(
            shap_values=torch.tensor([2.0]),
            children=[child, GraphNode()],
        )
        assert len(parent.children) == 2
        assert parent.children[0].shap_values is not None


# ──────────────── BaseShapCallConfig validation ────────────────


class TestBaseShapCallConfig:
    """Tests for BaseShapCallConfig validators."""

    def _make_chat(self, device="cpu"):
        class C:
            pass

        c = C()
        c.device = device
        c.torch_device = torch.device(device)
        return c

    def _make_response(self, chat):
        class R:
            pass

        r = R()
        r.chat = chat
        return r

    def _make_model(self):
        class M:
            pass

        return M()

    def test_mismatched_devices_raises(self) -> None:
        """Different devices on source_chat and response.chat → ValueError."""
        src = self._make_chat("cpu")
        resp_chat = self._make_chat("meta")
        resp = self._make_response(resp_chat)
        with pytest.raises(Exception):
            BaseShapCallConfig(
                model=self._make_model(),
                source_chat=src,
                response=resp,
                progress_bar=False,
                verbose=False,
            )

    def test_response_without_chat_raises(self) -> None:
        """Response with chat=None → ValueError."""
        src = self._make_chat("cpu")
        resp = self._make_response(None)
        with pytest.raises(Exception):
            BaseShapCallConfig(
                model=self._make_model(),
                source_chat=src,
                response=resp,
                progress_bar=False,
                verbose=False,
            )


# ──────────────── _generate_minimal_splits ────────────────


class TestGenerateMinimalSplits:
    """Tests for BaseShapApproximation._generate_minimal_splits."""

    def test_shape_and_content(self) -> None:
        """Checks that shape and content."""
        masks = BaseShapApproximation._generate_minimal_splits(
            n=4, device=torch.device("cpu")
        )
        assert masks.shape == (5, 4)
        # First row: all False
        assert not masks[0].any()
        # Rows 1-4: n-1 True values each (one False per row)
        for i in range(1, 5):
            assert masks[i].sum().item() == 3
            assert masks[i, i - 1].item() is False

    def test_single_feature(self) -> None:
        """n=1 produces shape (2, 1): all-False and single-True."""
        masks = BaseShapApproximation._generate_minimal_splits(
            n=1, device=torch.device("cpu")
        )
        assert masks.shape == (2, 1)
        assert masks[0, 0].item() is False
        assert masks[1, 0].item() is False  # all True except feature 0 → False

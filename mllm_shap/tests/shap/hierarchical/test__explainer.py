"""Tests for the HierarchicalExplainer module."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.errors import ValidationError
from mllm_shap.shap.normalizers import IdentityNormalizer, MinMaxNormalizer
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.hierarchical.enums import Mode
from mllm_shap.shap.precise import PreciseShapExplainer
from mllm_shap.shap.base.approx import BaseShapApproximation
from ...dummy import DummyChat as BaseDummyChat, DummyModel


class DummyApproxExplainer(BaseShapApproximation):
    """Minimal approximation explainer to satisfy importance sampling checks."""

    def _get_num_splits(self, n: int) -> int:
        if self.num_samples is not None:
            return self.num_samples
        fraction = self.fraction or 1.0
        return max(1, math.ceil((2**n - 1) * fraction))

    def _calculate_shap_values(
        self, masks: torch.Tensor, similarities: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        del similarities
        return torch.ones((masks.shape[-1],), dtype=torch.float, device=device)


class DummyChat:
    """Simple mock of BaseMllmChat for testing."""

    def __init__(self) -> None:
        # True = explainable token
        self.shap_values_mask = torch.tensor([
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ])
        # Modalities: 0=text, 1=image, etc.
        self.tokens_modality_flag = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
        # Roles: e.g., 0=system/user, 1=model
        self.token_roles = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.torch_device = torch.device("cpu")
        self.input_tokens_num = len(self.shap_values_mask)
        self.cache = None
        self.external_shap_values_mask = None
        self.external_group_ids = None

    def translate_groups_ids_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Return the mask unchanged for test purposes."""
        return mask

    def clone(self) -> "DummyChat":
        """Return a shallow copy used to mimic BaseMllmChat behavior."""
        new_chat = DummyChat()
        new_chat.shap_values_mask = self.shap_values_mask.clone()
        new_chat.tokens_modality_flag = self.tokens_modality_flag.clone()
        new_chat.token_roles = self.token_roles.clone()
        return new_chat


class TestHierarchicalExplainerStatic:
    """Unit tests for private static utility methods."""

    def test_repeated_buckets_regular_case(self) -> None:
        """Should repeat values and trim correctly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=10, k=3
        )
        expected = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
        assert torch.equal(result, expected)

    def test_repeated_buckets_exact_division(self) -> None:
        """If n is multiple of k, sequence ends cleanly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=6, k=2
        )
        expected = torch.tensor([1, 1, 2, 2, 3, 3])
        assert torch.equal(result, expected)

    def test_repeated_buckets_single_value(self) -> None:
        """Handle smallest possible n correctly."""
        result = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=1, k=5
        )
        assert torch.equal(result, torch.tensor([1]))

    def test_get_group_props_contiguous_block(self) -> None:
        """Return correct start, end, and size for contiguous True block."""
        mask = torch.tensor([False, True, True, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(
            mask
        )
        assert (start, end, n) == (1, 3, 3)

    def test_get_group_props_single_true(self) -> None:
        """Handle single True element correctly."""
        mask = torch.tensor([False, True, False])
        start, end, n = HierarchicalExplainer._HierarchicalExplainer__get_group_props(
            mask
        )
        assert (start, end, n) == (1, 1, 1)

    def test_get_group_props_raises_for_empty(self) -> None:
        """Empty masks should raise IndexError as there is no group."""
        mask = torch.tensor([False, False])
        with pytest.raises(IndexError):
            HierarchicalExplainer._HierarchicalExplainer__get_group_props(mask)


class TestHierarchicalExplainerGrouping:
    """Tests for group creation and segmentation logic."""

    def test_group_ids_with_roles(self) -> None:
        """Different modalities or roles should start new groups."""
        chat = DummyChat()
        result = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=True
        )
        expected = torch.tensor([1, 1, 0, 2, 2, 3, 0, 0, 4, 4])
        assert torch.equal(result, expected)

    def test_group_ids_without_roles(self) -> None:
        """When include_role=False, role changes should not split groups."""
        chat = DummyChat()
        result = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=False
        )
        expected = torch.tensor([1, 1, 0, 2, 2, 2, 0, 0, 3, 3])
        assert torch.equal(result, expected)

    def test_empty_mask_returns_all_zero_groups(self) -> None:
        """When no tokens are explainable, result should be all zeros."""
        chat = DummyChat()
        chat.shap_values_mask = torch.zeros_like(
            chat.shap_values_mask, dtype=torch.bool
        )
        group_ids = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(chat)
        assert torch.equal(
            group_ids, torch.zeros_like(chat.shap_values_mask, dtype=torch.long)
        )

    def test_group_ids_resets_to_zero_between_calls(self) -> None:
        """Reusing the same chat instance should not leak previous group ids."""
        chat = DummyChat()
        result_first = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=True
        )
        chat.shap_values_mask[0] = False
        result_second = HierarchicalExplainer._HierarchicalExplainer__get_group_ids(
            chat, include_role=True
        )
        assert result_first[0] == 1
        assert result_second[0] == 0


class TestHierarchicalExplainerCore:
    """Tests for initialization and key numeric properties."""

    def test_get_subgroups_num(self) -> None:
        """Subgroup count should equal ceil(log(n, k))."""
        explainer = HierarchicalExplainer(
            k=4, shap_explainer=DummyApproxExplainer(fraction=0.5), model=DummyModel()
        )
        assert explainer._HierarchicalExplainer__get_subgroups_num(10) == 2
        assert explainer._HierarchicalExplainer__get_subgroups_num(8) == 2
        assert explainer._HierarchicalExplainer__get_subgroups_num(4) == 1

    def test_invalid_k_raises_valueerror(self) -> None:
        """Non-positive or non-integer k should raise ValueError."""
        with pytest.raises(ValueError):
            HierarchicalExplainer(
                k=0, shap_explainer=PreciseShapExplainer(), model=DummyModel()
            )
        with pytest.raises(ValueError):
            HierarchicalExplainer(
                k=2.5, shap_explainer=PreciseShapExplainer(), model=DummyModel()
            )

    def test_default_mode_is_text(self) -> None:
        """Ensure default mode is Mode.TEXT."""
        explainer = HierarchicalExplainer(
            k=5, shap_explainer=DummyApproxExplainer(fraction=0.5), model=DummyModel()
        )
        assert explainer.mode == Mode.TEXT

    def test_mode_explicit_setting(self) -> None:
        """Ensure custom mode is properly set."""
        explainer = HierarchicalExplainer(
            k=5,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
            mode=Mode.MULTI_MODAL_MULTI_USER,
        )
        assert explainer.mode == Mode.MULTI_MODAL_MULTI_USER

    def test_importance_sampling_validation(self) -> None:
        """Enabling importance sampling without proper explainer should raise."""
        with pytest.raises(ValueError, match="importance_sampling"):
            HierarchicalExplainer(
                k=4,
                shap_explainer=PreciseShapExplainer(),
                model=DummyModel(),
                use_importance_sampling=True,
            )

    def test_get_subgroups_num_always_positive(self) -> None:
        """For any n >= k, subgroup count should be at least 1."""
        explainer = HierarchicalExplainer(
            k=4, shap_explainer=DummyApproxExplainer(fraction=0.5), model=DummyModel()
        )
        for n in (4, 5, 100):
            assert explainer._HierarchicalExplainer__get_subgroups_num(n) >= 1

    def test_repeated_buckets_respects_k(self) -> None:
        """Bucket output should never exceed the expected maximum value."""
        n, k = 11, 4
        buckets = HierarchicalExplainer._HierarchicalExplainer__repeated_buckets(
            n=n, k=k
        )
        assert buckets.max().item() == math.ceil(n / k)

    def test_invalid_importance_sampling_min_fraction_raises(self) -> None:
        """Out-of-range importance fraction should raise validation error."""
        with pytest.raises(ValidationError, match="importance_sampling_min_fraction"):
            HierarchicalExplainer(
                k=4,
                shap_explainer=DummyApproxExplainer(fraction=0.5),
                model=DummyModel(),
                importance_sampling_min_fraction=0.0,
            )

    def test_first_layer_explainer_type_validation(self) -> None:
        """Non-explainer first layer object should raise validation error."""
        with pytest.raises(ValidationError, match="first_layer_explainer"):
            HierarchicalExplainer(
                k=4,
                shap_explainer=DummyApproxExplainer(fraction=0.5),
                model=DummyModel(),
                first_layer_explainer="bad",
            )

    def test_warns_when_normalizer_is_not_minmax(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Constructor should log warning when main explainer normalizer is not MinMax."""
        _ = HierarchicalExplainer(
            k=4,
            shap_explainer=DummyApproxExplainer(
                fraction=0.5, normalizer=IdentityNormalizer()
            ),
            model=DummyModel(),
        )
        captured = capsys.readouterr()
        assert "recommended to use MinMaxNormalizer" in captured.err

    def test_warns_when_first_layer_normalizer_differs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Constructor should log warning for mismatched normalizers."""
        main_explainer = DummyApproxExplainer(
            fraction=0.5, normalizer=MinMaxNormalizer()
        )
        first_layer = PreciseShapExplainer(normalizer=IdentityNormalizer())
        _ = HierarchicalExplainer(
            k=4,
            shap_explainer=main_explainer,
            first_layer_explainer=first_layer,
            model=DummyModel(),
        )
        captured = capsys.readouterr()
        assert "uses the same normalizer" in captured.err

    def test_calculate_group_shap_values_single_token_short_circuit(self) -> None:
        """Single-token mask should return one-hot tensor without model call."""
        explainer = HierarchicalExplainer(
            k=4,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)
        mask = torch.tensor([False, True, False], dtype=torch.bool)
        out = explainer._HierarchicalExplainer__calculate_group_normalized_shap_values(
            chat=chat,
            response=response,
            shap_values_mask=mask,
        )
        assert torch.equal(out, torch.tensor([0.0, 1.0, 0.0]))

    def test_calculate_group_shap_values_single_group_short_circuit(self) -> None:
        """Single group id should return nan-filled vector with group set to 1."""
        explainer = HierarchicalExplainer(
            k=4,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)
        group_ids = torch.tensor([0, 1, 1], dtype=torch.long)
        out = explainer._HierarchicalExplainer__calculate_group_normalized_shap_values(
            chat=chat,
            response=response,
            group_ids=group_ids,
        )
        assert torch.isnan(out[0])
        assert out[1].item() == 1.0
        assert out[2].item() == 1.0

    def test_update_progress_updates_tqdm_bar(self) -> None:
        """Progress updater should forward explainer calls to active progress bar."""
        explainer = HierarchicalExplainer(
            k=4,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
        )
        bar = MagicMock()
        explainer._progress_bar = bar
        explainer.n_calls = 0
        explainer.total_n_calls = 0
        called_explainer = SimpleNamespace(total_n_calls=7)

        explainer._HierarchicalExplainer__update_progress(explainer=called_explainer)

        assert explainer.n_calls == 1
        assert explainer.total_n_calls == 7
        bar.update.assert_called_once_with(7)

    def test_calculate_group_shap_values_requires_mask_or_group_ids(self) -> None:
        """Missing both group_ids and shap_values_mask should raise validation error."""
        explainer = HierarchicalExplainer(
            k=4,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)

        with pytest.raises(
            ValidationError, match="Either shap_values_mask or group_ids"
        ):
            explainer._HierarchicalExplainer__calculate_group_normalized_shap_values(
                chat=chat,
                response=response,
                shap_values_mask=None,
                group_ids=None,
            )

    def test_importance_sampling_raises_when_fraction_missing(self) -> None:
        """Importance sampling path should fail fast when base fraction is None."""
        shap_explainer = DummyApproxExplainer(num_samples=4, fraction=None)
        explainer = HierarchicalExplainer(
            k=4,
            shap_explainer=shap_explainer,
            model=DummyModel(),
            use_importance_sampling=False,
        )
        explainer.use_importance_sampling = True
        chat = DummyChat()
        response = SimpleNamespace(chat=DummyChat())
        mask = torch.tensor([True, True, False, False], dtype=torch.bool)

        with pytest.raises(RuntimeError, match="fraction is None"):
            explainer._HierarchicalExplainer__calculate_group_normalized_shap_values(
                chat=chat,
                response=response,
                shap_values_mask=mask,
            )

    def test_compute_verbose_records_skipped_and_recursive_children(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verbose compute should add placeholder node for zero subgroup and child for non-zero subgroup."""
        explainer = HierarchicalExplainer(
            k=2,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)
        group_mask = torch.tensor([True, True, True, True], dtype=torch.bool)

        def _fake_calc(*args, **kwargs):
            group_ids = kwargs.get("group_ids")
            shap_values_mask = kwargs.get("shap_values_mask")
            if group_ids is not None:
                return torch.tensor([0.0, 0.0, 0.5, 0.5], dtype=torch.float)
            r = torch.zeros_like(shap_values_mask, dtype=torch.float)
            r[shap_values_mask] = 1.0
            return r

        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__calculate_group_normalized_shap_values",
            _fake_calc,
        )

        values, graph = explainer._HierarchicalExplainer__compute(
            chat=chat,
            response=response,
            group_mask=group_mask,
            _verbose=True,
        )

        assert graph is not None
        assert len(graph.children) == 2
        assert graph.children[0] is not None
        assert torch.isfinite(values[2:]).all()

    def test_handle_with_first_level_verbose_tracks_children(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First-level verbose path should save graph and include skipped/recursive children."""
        explainer = HierarchicalExplainer(
            k=2,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
            mode=Mode.MULTI_MODAL,
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)

        monkeypatch.setattr(
            HierarchicalExplainer,
            "_HierarchicalExplainer__get_group_ids",
            staticmethod(lambda chat, include_role=True: torch.tensor([1, 1, 2, 2])),
        )

        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__calculate_group_normalized_shap_values",
            lambda **kwargs: torch.tensor([0.0, 0.0, 0.6, 0.6]),
        )

        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__compute",
            lambda **kwargs: (torch.ones(4), SimpleNamespace(tag="child")),
        )

        result = explainer._HierarchicalExplainer__handle_with_first_level(
            chat=chat,
            response=response,
            _verbose=True,
        )

        assert result.shape == (4,)
        assert explainer.computation_graph is not None
        assert len(explainer.computation_graph.children) == 2

    def test_handle_with_first_level_skips_zero_group_without_verbose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero-SHAP subgroup should skip recursion cleanly when verbose=False."""
        explainer = HierarchicalExplainer(
            k=2,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
            mode=Mode.MULTI_MODAL,
        )
        chat = DummyChat()
        response = SimpleNamespace(chat=chat)

        monkeypatch.setattr(
            HierarchicalExplainer,
            "_HierarchicalExplainer__get_group_ids",
            staticmethod(lambda chat, include_role=True: torch.tensor([1, 1, 2, 2])),
        )

        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__calculate_group_normalized_shap_values",
            lambda **kwargs: torch.tensor([0.0, 0.0, 0.5, 0.5]),
        )

        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__compute",
            lambda **kwargs: (torch.ones(4), None),
        )

        out = explainer._HierarchicalExplainer__handle_with_first_level(
            chat=chat,
            response=response,
            _verbose=False,
        )
        assert out.shape == (4,)

    @patch("mllm_shap.shap.hierarchical.explainer.tqdm")
    @patch("mllm_shap.shap.hierarchical.explainer.BaseExplainer.__call__")
    def test_call_creates_and_closes_progress_bar(
        self,
        _mock_base_call: MagicMock,
        mock_tqdm: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When progress_bar=True, __call__ should create tqdm and close it at end."""
        explainer = HierarchicalExplainer(
            k=2,
            shap_explainer=DummyApproxExplainer(fraction=0.5),
            model=DummyModel(),
            mode=Mode.TEXT,
        )
        bar = MagicMock()
        mock_tqdm.return_value = bar

        fake_chat = BaseDummyChat(num_tokens=3)
        fake_response = SimpleNamespace(chat=fake_chat)

        monkeypatch.setattr(explainer.model, "generate", lambda **kwargs: fake_response)
        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__compute",
            lambda **kwargs: (torch.ones(3), None),
        )
        monkeypatch.setattr(
            explainer,
            "_HierarchicalExplainer__save_to_cache",
            lambda **kwargs: None,
        )

        _ = explainer(chat=fake_chat, progress_bar=True)

        mock_tqdm.assert_called_once()
        bar.close.assert_called_once()
        assert explainer._progress_bar is None

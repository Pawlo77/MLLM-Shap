"""Unit tests for MasksManager class."""

from unittest.mock import patch

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.errors import MaskError
from mllm_shap.shap.base._masks_manager import (
    MaskGenerator,
    MasksManager,
    NoTokensToExplainError,
)

from ...dummy import DummyChat


class _ForwardingMaskGenerator(MaskGenerator):
    """Concrete generator used to validate MaskGenerator forwarding methods."""

    def _mask_iter(self):
        received = yield torch.tensor([True], dtype=torch.bool), 1
        if received == "boom":
            raise RuntimeError("boom")
        yield torch.tensor([False], dtype=torch.bool), 2


class TestMaskGenerator:
    """Coverage-focused tests for MaskGenerator wrapper behavior."""

    def test_send_and_iter_and_next_forward_to_internal_iterator(self) -> None:
        """Checks that send and iter and next forward to internal iterator."""
        generator = _ForwardingMaskGenerator()
        assert iter(generator) is generator
        assert next(generator)[1] == 1
        assert generator.send("ok")[1] == 2

    def test_throw_is_forwarded_to_internal_iterator(self) -> None:
        """Checks that throw is forwarded to internal iterator."""
        generator = _ForwardingMaskGenerator()
        _ = next(generator)
        with pytest.raises(RuntimeError, match="boom"):
            _ = generator.throw(RuntimeError("boom"))


class TestMasksManager:
    """Unit tests for MasksManager methods and validation."""

    @pytest.fixture
    def chat(self) -> BaseMllmChat:
        """Fixture for a dummy chat with valid mask."""
        return DummyChat(
            shap_values_mask=torch.tensor([True, False, True, False, False]),
            num_tokens=5,
        )

    @pytest.fixture
    def manager(self, chat: BaseMllmChat) -> MasksManager:
        """Fixture for initialized MasksManager."""
        return MasksManager(chat=chat)

    def test_init_with_no_tokens_raises(self) -> None:
        """Should raise NoTokensToExplainError if no tokens to explain."""
        chat = DummyChat(
            shap_values_mask=torch.zeros(5, dtype=torch.bool), num_tokens=5
        )
        with pytest.raises(NoTokensToExplainError, match="no tokens to explain"):
            _ = MasksManager(chat)

    def test_init_raises_when_mask_sum_is_zero_despite_any_true(self) -> None:
        """Defensive branch: if sum()==0 while any()==True, manager should still reject."""

        class _MaskInconsistent:
            def any(self) -> bool:
                return True

            def sum(self) -> torch.Tensor:
                return torch.tensor(0)

        class _Chat:
            shap_values_mask = _MaskInconsistent()
            input_tokens_num = 5

        with pytest.raises(
            NoTokensToExplainError, match="Mask must have at least one True value"
        ):
            _ = MasksManager(chat=_Chat())

    def test_init_sets_correct_attributes(
        self, manager: MasksManager, chat: BaseMllmChat
    ) -> None:
        """Should correctly initialize attributes."""
        assert torch.equal(manager.shap_values_mask, chat.shap_values_mask)
        assert manager.target_length == chat.input_tokens_num
        assert manager.n == int(chat.shap_values_mask.sum().item())
        assert isinstance(manager._seen_masks, set)
        assert manager._seen_masks == set()

    def test_max_masks_number_computation(self, manager: MasksManager) -> None:
        """Should correctly compute max_masks_number."""
        expected = int(2**manager.n - 1)
        assert manager.max_masks_number == expected

    def test_mark_seen_and_seen_methods(self, manager: MasksManager) -> None:
        """Should correctly mark and detect seen masks."""
        mask = torch.tensor([True, False, True, False, False])
        mask_hash = manager.get_hash(mask)
        assert not manager.seen(mask_hash=mask_hash)
        manager.mark_seen(mask_hash=mask_hash)
        assert manager.seen(mask_hash=mask_hash)

    def test_get_initial_mask_creates_all_true_mask(
        self, manager: MasksManager
    ) -> None:
        """get_initial_mask() should return a full mask and mark it seen."""
        device = torch.device("cpu")
        mask = manager.get_initial_mask(device=device)
        assert mask.dtype == torch.bool
        assert mask.shape == (manager.target_length,)
        assert mask.any()
        # must be registered as seen
        h = manager.get_hash(mask)
        assert h in manager._seen_masks

    def test_prepare_mask_returns_correct_tensor(self, manager: MasksManager) -> None:
        """prepare_mask() should correctly expand split to full-length mask."""
        device = torch.device("cpu")
        split = torch.tensor([[True, False]], dtype=torch.bool)
        result = manager.prepare_mask(split=split, device=device)
        assert result.shape == (manager.target_length,)
        # original masked positions updated
        assert result[manager.shap_values_mask].tolist() == [True, False]
        # unmasked positions remain True
        assert result[~manager.shap_values_mask].all()

    def test_prepare_mask_returns_none_if_all_false(
        self, manager: MasksManager
    ) -> None:
        """Should return None if resulting mask has no True values."""
        device = torch.device("cpu")
        manager.shap_values_mask = torch.ones(manager.target_length, dtype=torch.bool)
        split = torch.zeros(manager.target_length, dtype=torch.bool)
        result = manager.prepare_mask(split=split, device=device)
        assert result is None

    def test_get_hash_with_1d_and_2d_input(self, manager: MasksManager) -> None:
        """get_hash() should accept both 1D and 2D single-row tensors."""
        mask_1d = torch.tensor([True, False, True])
        mask_2d = mask_1d.unsqueeze(0)
        h1 = manager.get_hash(mask_1d)
        h2 = manager.get_hash(mask_2d)
        assert isinstance(h1, int)
        assert h1 == h2

    def test_get_hash_raises_if_multiple_rows(self, manager: MasksManager) -> None:
        """get_hash() should raise if 2D mask has more than one row."""
        mask = torch.tensor([[True, False], [False, True]])
        with pytest.raises(
            MaskError, match="1D tensor or a 2D tensor with a single row"
        ):
            _ = manager.get_hash(mask)

    def test_get_mask_hash_from_mask_or_hash(self, manager: MasksManager) -> None:
        """__get_mask_hash() should compute from either mask or hash."""
        mask = torch.tensor([True, False, True, False, False])
        h1 = manager._MasksManager__get_mask_hash(mask=mask)
        h2 = manager._MasksManager__get_mask_hash(mask_hash=h1)
        assert h1 == h2

    def test_get_mask_hash_raises_if_missing_both(self, manager: MasksManager) -> None:
        """Should raise ValueError if both mask and mask_hash are None."""
        with pytest.raises(
            MaskError, match="Either mask or mask_hash must be provided"
        ):
            _ = manager._MasksManager__get_mask_hash()

    def test_mask_hash_strategy_normalize_returns_1d(self) -> None:
        """Hash strategy should normalize single-row masks."""
        mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        normalized = MasksManager._MaskHashStrategy.normalize(mask)
        assert normalized.ndim == 1
        assert normalized.tolist() == [True, False, True]

    def test_mask_hash_strategy_hash_consistent(self) -> None:
        """Hash strategy should match get_hash API output."""
        mask = torch.tensor([True, True, False], dtype=torch.bool)
        assert MasksManager._MaskHashStrategy.hash(mask) == MasksManager.get_hash(mask)

    def test_mark_seen_with_mask_argument(self, manager: MasksManager) -> None:
        """Marking by mask tensor should register hash."""
        mask = torch.tensor([False, True, True, True, False], dtype=torch.bool)
        manager.mark_seen(mask=mask)
        assert manager.seen(mask=mask)

    def test_seen_returns_false_for_untracked_mask(self, manager: MasksManager) -> None:
        """Unseen mask should be reported as not present."""
        mask = torch.tensor([True, True, False, False, True], dtype=torch.bool)
        assert not manager.seen(mask=mask)

    def test_mark_seen_is_idempotent(self, manager: MasksManager) -> None:
        """Repeated registrations should not duplicate hashes."""
        mask = torch.tensor([True, False, False, True, True], dtype=torch.bool)
        manager.mark_seen(mask=mask)
        size_after_first = len(manager._seen_masks)
        manager.mark_seen(mask=mask)
        assert len(manager._seen_masks) == size_after_first

    def test_prepare_mask_accepts_1d_split(self, manager: MasksManager) -> None:
        """1D split tensors should be expanded correctly."""
        device = torch.device("cpu")
        split = torch.tensor([True, False], dtype=torch.bool)
        result = manager.prepare_mask(split=split, device=device)
        assert result is not None
        assert result[manager.shap_values_mask].tolist() == [True, False]

    def test_prepare_mask_preserves_dtype_and_device(
        self, manager: MasksManager
    ) -> None:
        """Returned mask keeps boolean dtype on the provided device."""
        device = torch.device("cpu")
        split = torch.tensor([[False, True]], dtype=torch.bool)
        result = manager.prepare_mask(split=split, device=device)
        assert result is not None
        assert result.dtype is torch.bool
        assert result.device == device

    def test_get_initial_mask_marks_single_entry(self, chat: BaseMllmChat) -> None:
        """Initial mask should align with shap mask positions."""
        manager = MasksManager(chat=chat)
        mask = manager.get_initial_mask(device=torch.device("cpu"))
        assert torch.equal(
            mask[manager.shap_values_mask], torch.ones(manager.n, dtype=torch.bool)
        )
        assert torch.equal(
            mask[~manager.shap_values_mask],
            torch.ones(manager.target_length - manager.n, dtype=torch.bool),
        )

    def test_get_initial_mask_raises_when_prepare_returns_none(
        self, manager: MasksManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If prepare_mask fails, get_initial_mask should raise."""

        def _fail_prepare(
            split: torch.Tensor, device: torch.device
        ) -> torch.Tensor | None:
            del split, device
            return None

        monkeypatch.setattr(manager, "prepare_mask", _fail_prepare)
        with pytest.raises(ValueError, match="Starting mask cannot be None"):
            _ = manager.get_initial_mask(device=torch.device("cpu"))

    def test_get_mask_hash_prefers_explicit_hash(self, manager: MasksManager) -> None:
        """Providing mask_hash should short-circuit mask computation."""
        mask = torch.tensor([True, False, False, True, False], dtype=torch.bool)
        forced_hash = 123456
        result = manager._MasksManager__get_mask_hash(mask=mask, mask_hash=forced_hash)
        assert result == forced_hash

    def test_get_hash_returns_stable_value(self, manager: MasksManager) -> None:
        """Repeated calls to get_hash should be deterministic."""
        mask = torch.tensor([False, True, False, True, False], dtype=torch.bool)
        assert manager.get_hash(mask) == manager.get_hash(mask)

    def test_init_with_log_stats_logs_max_masks(self, chat: BaseMllmChat) -> None:
        """When log_stats=True, init should emit max-mask statistics."""
        with patch("mllm_shap.shap.base._masks_manager.logger.info") as mock_info:
            _ = MasksManager(chat=chat, log_stats=True)
        mock_info.assert_called_once()
        assert "Number of tokens for explainability" in mock_info.call_args.args[0]

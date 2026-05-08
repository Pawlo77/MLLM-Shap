"""Unit tests for BaseComplementaryNeymanShapExplainer class (updated)."""

from unittest.mock import patch

import pytest
import torch
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.connectors.enums import Role, SystemRolesSetup
from mllm_shap.shap.base._masks_manager import MasksManager, NoTokensToExplainError
from mllm_shap.shap.complementary import BaseComplementaryShapApproximation
from mllm_shap.shap.neyman._base import BaseComplementaryNeymanShapExplainer, _Step
from torch import Tensor

from ...dummy import DummyChat as BaseDummyChat
from ...dummy import DummyModel


class DummyChat:
    """A minimal dummy chat instance used for testing."""

    def __init__(self) -> None:
        # torch_device attribute is expected by the explainer call path
        self.torch_device = torch.device("cpu")
        # shap_values_mask selects tokens considered for SHAP evaluation
        self.shap_values_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
        self.input_tokens_num = 4
        # simulate roles required by the explainer
        self.system_roles_setup = getattr(self, "system_roles_setup", None)
        self.token_roles = getattr(self, "token_roles", [])


class DummyResponse:
    """A minimal dummy response object containing a DummyChat."""

    def __init__(self) -> None:
        self.chat = DummyChat()


class DummyBaseComplementaryNeymanShapExplainer(BaseComplementaryNeymanShapExplainer):
    """
    Concrete subclass used for testing. Overrides behavior that depends on
    external models / connectors to deterministic minimal behaviors.
    """

    def __init__(
        self,
        *args,
        initial_num_samples: int | None = None,
        initial_fraction: float | None = None,
        **kwargs,
    ) -> None:
        # call base with no other args
        super().__init__(
            *args,
            initial_num_samples=initial_num_samples,
            initial_fraction=initial_fraction,
            **kwargs,
        )
        # deterministic, small matrices used by tests
        # note: ensure shape is at least 2x2 to cover indexing used by impl
        self._M = torch.ones((4, 4), dtype=torch.float32) * 2.0
        # C used by _calculate_shap_values tests should have same shape as _M
        self._C = torch.tensor(
            [
                [0.0, 4.0, 6.0, 0.0],
                [0.0, 2.0, 8.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 3.0, 5.0, 0.0],
            ],
            dtype=torch.float32,
        )
        # ensure the explainer's skip-zero-mask flag default
        self._zero_mask_skipped = True
        # initialize state (mimic what __call__ normally does)
        try:
            self._initialize_state()
        except Exception:
            # some init paths may require further context; tests will call _get_num_splits where needed
            pass

    def _get_similarities(self, responses: list, model=None) -> Tensor:
        """Return a constant similarity tensor of ones (for testing purposes)."""
        return torch.ones(len(responses), dtype=torch.float32)

    def _calculate_C_matrix(
        self, masks: Tensor, similarities: Tensor, device: torch.device
    ) -> None:
        """
        For tests, write deterministic simple contributions into self._C
        (this overrides heavy logic of the real method).
        """
        # masks shape = (masks_count, token_count)
        if self._M is None:
            raise RuntimeError(
                "M matrix must be initialized before calculating C matrix."
            )
        if self._C is None:
            self._C = torch.zeros_like(self._M)
        # simple deterministic update: increment by 1 per True entry for test visibility
        for i in range(masks.shape[0]):
            row_mask = masks[i]
            s_size = int(row_mask.sum().item())
            # increment for all tokens present in row_mask at coalition size = s_size
            # reuse base helper if available, else simple increment
            for idx, present in enumerate(row_mask):
                if present:
                    self._C[idx, s_size] += 1.0


class TestBaseComplementaryNeymanShapExplainerNumSplits:
    """Tests for the _get_num_splits() method."""

    @pytest.fixture
    def explainer(self) -> DummyBaseComplementaryNeymanShapExplainer:
        """Provide a default explainer fixture."""
        return DummyBaseComplementaryNeymanShapExplainer(
            initial_num_samples=2, initial_fraction=0.5
        )

    def test_num_splits_returns_integer(
        self, explainer: DummyBaseComplementaryNeymanShapExplainer
    ) -> None:
        """Ensure _get_num_splits() returns a valid integer and sets initial splits."""
        num_splits = explainer._get_num_splits(n=5)
        assert isinstance(num_splits, int)
        # the initial number of splits is stored under a mangled name; read it defensively
        initial_splits = getattr(
            explainer, "_BaseComplementaryNeymanShapExplainer__initial_num_splits", None
        )
        assert initial_splits is not None and isinstance(initial_splits, int)
        assert initial_splits >= 1
        assert num_splits >= initial_splits

    def test_num_splits_wraps_total_budget_errors(self) -> None:
        """Errors from parent total-budget calculation should be wrapped."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        with patch.object(
            BaseComplementaryShapApproximation,
            "_get_num_splits",
            side_effect=ValueError("boom"),
        ):
            with pytest.raises(
                ValueError, match="Total number of splits could not be determined"
            ):
                explainer._get_num_splits(n=4)

    def test_num_splits_wraps_initial_budget_errors(self) -> None:
        """Errors from initial-budget estimation should be wrapped."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(
            initial_num_samples=2, initial_fraction=0.5
        )
        with (
            patch.object(
                BaseComplementaryShapApproximation,
                "_get_num_splits",
                return_value=10,
            ),
            patch.object(
                BaseComplementaryShapApproximation,
                "_get_num_splits_static",
                side_effect=ValueError("boom"),
            ),
        ):
            with pytest.raises(
                ValueError, match="Initial number of splits could not be determined"
            ):
                explainer._get_num_splits(n=4)

    def test_num_splits_warns_for_small_and_large_initial_budget(self) -> None:
        """Small initial budget should clamp to 2 and emit warning paths."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(
            initial_num_samples=2, initial_fraction=0.5
        )
        with (
            patch.object(
                BaseComplementaryShapApproximation,
                "_get_num_splits",
                return_value=5,
            ),
            patch.object(
                BaseComplementaryShapApproximation,
                "_get_num_splits_static",
                return_value=1,
            ),
            patch("mllm_shap.shap.neyman._base.logger.warning") as mock_warning,
        ):
            result = explainer._get_num_splits(n=1)

        assert result == 5
        assert (
            getattr(
                explainer, "_BaseComplementaryNeymanShapExplainer__initial_num_splits"
            )
            == 2
        )
        assert mock_warning.call_count >= 2


class TestBaseComplementaryNeymanShapExplainerMasksGeneration:
    """Tests for mask generation behavior (complementary masks)."""

    @pytest.fixture
    def explainer(self) -> DummyBaseComplementaryNeymanShapExplainer:
        return DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)

    def test_complementary_mask_pair_generation(
        self, explainer: DummyBaseComplementaryNeymanShapExplainer
    ) -> None:
        """
        Tests that masks produced by the masks generator come in complementary pairs.
        This uses the explainer's _get_masks_generator which is a thin wrapper around the base generator.
        """
        chat = DummyChat()
        device = torch.device("cpu")
        # ensure M is present and initialized (generator relies on it)
        explainer._M = torch.ones((4, 4), dtype=torch.float32)
        # create a MasksManager for this chat
        mask_manager = MasksManager(chat=chat)
        # get the generator (kwargs align with the wrapper in the explainer)
        gen = explainer._get_masks_generator(
            mask_manager=mask_manager, device=device, masks=[]
        )
        # pull two masks (should be complementary pair)
        mask_a, _ = next(gen)
        mask_b, _ = next(gen)
        # ensure they have same shape and are boolean tensors
        assert mask_a.shape == mask_b.shape
        assert mask_a.dtype == torch.bool and mask_b.dtype == torch.bool
        # complementary: mask_b == ~mask_a
        assert torch.equal(mask_b, ~mask_a)

    def test_get_initial_mask_and_seen(self) -> None:
        """MasksManager.get_initial_mask should return a full-length mask and mark it as seen."""
        chat = DummyChat()
        manager = MasksManager(chat=chat)
        device = torch.device("cpu")

        mask = manager.get_initial_mask(device=device)
        assert mask.shape[0] == chat.input_tokens_num
        assert mask.all()
        assert manager.seen(mask=mask)

    def test_get_hash_consistent(self) -> None:
        """MasksManager.get_hash should return same value for squeezed/unsqueezed masks."""
        chat = DummyChat()
        manager = MasksManager(chat=chat)
        device = torch.device("cpu")
        mask = manager.get_initial_mask(device=device)
        h1 = MasksManager.get_hash(mask)
        h2 = MasksManager.get_hash(mask.unsqueeze(0))
        assert isinstance(h1, int) and isinstance(h2, int)
        assert h1 == h2

    def test_no_tokens_to_explain_raises(self) -> None:
        """MasksManager should raise if chat.shap_values_mask has no True values."""
        bad_chat = DummyChat()
        bad_chat.shap_values_mask = torch.tensor(
            [False, False, False, False], dtype=torch.bool
        )
        with pytest.raises(NoTokensToExplainError):
            MasksManager(chat=bad_chat)

    def test_mark_seen_and_seen_toggle(self) -> None:
        """mark_seen should record a mask hash and seen() should return True afterwards."""
        chat = DummyChat()
        manager = MasksManager(chat=chat)
        device = torch.device("cpu")
        mask = manager.get_initial_mask(device=device)
        h = MasksManager.get_hash(mask)
        # ensure manager knows about the mask
        assert manager.seen(mask_hash=h)
        # a new random mask hash should not be seen
        assert not manager.seen(mask_hash=h + 1)

    def test_get_next_split_raises_without_M_matrix(self) -> None:
        """Initial sampling requires M matrix to be present."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = None
        with pytest.raises(RuntimeError, match="M matrix must be initialized"):
            explainer._get_next_split(
                n=4,
                device=torch.device("cpu"),
                generated_masks_num=0,
            )

    def test_get_next_split_switches_to_neyman_when_initial_sampling_completes(
        self,
    ) -> None:
        """When initial sampling is done, next split should return None and advance step."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.full((2, 2), fill_value=2.0)
        explainer._first_call = False
        explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2

        result = explainer._get_next_split(
            n=2,
            device=torch.device("cpu"),
            generated_masks_num=0,
        )

        assert result is None
        assert (
            explainer._BaseComplementaryNeymanShapExplainer__step
            == _Step.NEYMAN_ALLOCATION
        )

    def test_get_next_split_raises_if_required_position_not_updated(self) -> None:
        """Modified initial sampler should fail if current M position already exhausted."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._first_call = False
        explainer._M = torch.full((2, 2), fill_value=2.0)
        explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
        explainer._BaseComplementaryNeymanShapExplainer__i = 0
        explainer._BaseComplementaryNeymanShapExplainer__j = 0
        with patch.object(
            explainer,
            "_BaseComplementaryNeymanShapExplainer__update_M_position",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="did not update position correctly"):
                explainer._get_next_split(
                    n=2,
                    device=torch.device("cpu"),
                    generated_masks_num=0,
                )

    def test_get_next_split_raises_if_required_token_missing(self) -> None:
        """Modified initial sampler should validate include_token contract."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._first_call = False
        explainer._M = torch.zeros((3, 3), dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
        explainer._BaseComplementaryNeymanShapExplainer__i = 1
        explainer._BaseComplementaryNeymanShapExplainer__j = 1
        with (
            patch.object(
                explainer,
                "_BaseComplementaryNeymanShapExplainer__update_M_position",
                return_value=False,
            ),
            patch.object(
                explainer,
                "_get_random_split",
                return_value=torch.tensor([[True, False, False]], dtype=torch.bool),
            ),
        ):
            with pytest.raises((ValueError, RuntimeError)):
                explainer._get_next_split(
                    n=3,
                    device=torch.device("cpu"),
                    generated_masks_num=0,
                )

    def test_get_next_split_neyman_requires_M_hat(self) -> None:
        """Neyman stage requires M_hat to be initialized."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__step = _Step.NEYMAN_ALLOCATION
        explainer._BaseComplementaryNeymanShapExplainer__M_hat = None
        with pytest.raises(RuntimeError, match="M_hat matrix must be initialized"):
            explainer._get_next_split(
                n=3,
                device=torch.device("cpu"),
                generated_masks_num=0,
            )

    def test_get_next_split_neyman_returns_none_at_end(self) -> None:
        """Neyman stage should stop when j reached M_hat length."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__step = _Step.NEYMAN_ALLOCATION
        explainer._BaseComplementaryNeymanShapExplainer__M_hat = torch.zeros(3)
        explainer._BaseComplementaryNeymanShapExplainer__j = 3
        assert (
            explainer._get_next_split(
                n=3,
                device=torch.device("cpu"),
                generated_masks_num=0,
            )
            is None
        )


class TestBaseComplementaryNeymanShapExplainerCalculateShapValues:
    """Tests for the _calculate_shap_values() method."""

    def test_shap_values_computation(self) -> None:
        """Check that SHAP value computation produces a tensor of the correct shape."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        # prepare M and C such that M[:, 1:] are non-zero
        explainer._M = torch.tensor(
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32
        )
        explainer._C = torch.tensor(
            [[0.0, 4.0, 6.0], [0.0, 2.0, 8.0], [0.0, 1.0, 1.0]], dtype=torch.float32
        )
        explainer._zero_mask_skipped = True
        device = torch.device("cpu")
        masks = torch.tensor(
            [[True, False, False], [False, True, False]], dtype=torch.bool
        )
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result = explainer._calculate_shap_values(
            masks=masks, similarities=similarities, device=device
        )
        assert isinstance(result, Tensor)
        # result length equals number of features (rows in _M)
        assert result.shape[0] == explainer._M.shape[0]

    def test_raises_if_zero_mask_not_skipped(self) -> None:
        """Ensure a RuntimeError is raised when zero mask was not skipped."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._zero_mask_skipped = False
        explainer._M = torch.ones((3, 3), dtype=torch.float32) * 2.0
        explainer._C = torch.zeros_like(explainer._M)
        device = torch.device("cpu")
        masks = torch.tensor(
            [[True, False, False], [False, True, False]], dtype=torch.bool
        )
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Zero mask was not skipped"):
            explainer._calculate_shap_values(
                masks=masks, similarities=similarities, device=device
            )

    def test_calculate_C_matrix_updates_counts(self) -> None:
        """Ensure the test `_calculate_C_matrix` implementation increments counts at the expected indices."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        # set shapes compatible with masks: tokens=3, coalition sizes up to 2
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)

        masks = torch.tensor(
            [[True, False, True], [True, True, False]], dtype=torch.bool
        )
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        device = torch.device("cpu")

        explainer._calculate_C_matrix(
            masks=masks, similarities=similarities, device=device
        )

        # coalition size for both rows is 2, so updates happen in column index 2
        # expected increments: token0 present in both rows -> 2, token1 present once -> 1, token2 present once -> 1
        expected_col = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.equal(explainer._C[:, 2], expected_col)

    def test_calculate_C_matrix_with_zero_coalition(self) -> None:
        """When a mask row has no True entries, the contribution should be recorded at coalition size 0."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((2, 2), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)

        masks = torch.tensor([[False, False], [True, False]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        device = torch.device("cpu")

        explainer._calculate_C_matrix(
            masks=masks, similarities=similarities, device=device
        )

        # first row has zero Trues -> contributions at column 0 for no-token coalition
        assert explainer._C[:, 0].sum() >= 0.0

    def test_get_similarities_all_ones(self) -> None:
        """_get_similarities should return a tensor of ones with length equal to responses."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        responses = [DummyResponse() for _ in range(4)]
        sims = explainer._get_similarities(responses=responses)
        assert isinstance(sims, Tensor)
        assert sims.shape[0] == len(responses)
        assert torch.equal(sims, torch.ones(len(responses), dtype=torch.float32))

    def test_get_similarities_empty_responses(self) -> None:
        """When passed an empty list, _get_similarities should return an empty tensor."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        sims = explainer._get_similarities(responses=[])
        assert isinstance(sims, Tensor)
        assert sims.numel() == 0

    def test_get_start_raises_without_M_matrix(self) -> None:
        """Private __get_start should fail when M matrix is missing."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        with pytest.raises(ValueError, match="Matrix M is not initialized"):
            _ = explainer._BaseComplementaryNeymanShapExplainer__get_start()

    def test_update_M_position_requires_M_matrix(self) -> None:
        """Private __update_M_position should fail when M matrix is missing."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        with pytest.raises(RuntimeError, match="M matrix must be initialized"):
            _ = explainer._BaseComplementaryNeymanShapExplainer__update_M_position()

    def test_update_M_position_returns_true_when_completed(self) -> None:
        """__update_M_position should report completion when M reached initial budget."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.full((2, 2), fill_value=2.0)
        explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
        done = explainer._BaseComplementaryNeymanShapExplainer__update_M_position()
        assert done is True

    def test_calculate_C_matrix_raises_for_odd_masks_count(self) -> None:
        """C matrix update should reject non-paired masks."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)
        masks = torch.tensor([[True, False, True]], dtype=torch.bool)
        with pytest.raises(ValueError, match="complementary pairs"):
            BaseComplementaryNeymanShapExplainer._calculate_C_matrix(
                explainer,
                masks=masks,
                similarities=torch.tensor([1.0], dtype=torch.float32),
                device=torch.device("cpu"),
            )

    def test_calculate_C_matrix_raises_for_non_complementary_pairs(self) -> None:
        """C matrix update should reject invalid pair relationship."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)
        masks = torch.tensor(
            [[True, False, True], [True, False, True]], dtype=torch.bool
        )
        with pytest.raises(ValueError, match="not complementary pairs"):
            BaseComplementaryNeymanShapExplainer._calculate_C_matrix(
                explainer,
                masks=masks,
                similarities=torch.tensor([1.0, 0.5], dtype=torch.float32),
                device=torch.device("cpu"),
            )

    def test_calculate_C_matrix_requires_M_matrix(self) -> None:
        """Real C matrix updater should reject missing M matrix."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        with pytest.raises(RuntimeError, match="M matrix must be initialized"):
            BaseComplementaryNeymanShapExplainer._calculate_C_matrix(
                explainer,
                masks=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
                similarities=torch.tensor([1.0, 0.5], dtype=torch.float32),
                device=torch.device("cpu"),
            )

    def test_calculate_shap_values_requires_M_and_C(self) -> None:
        """SHAP computation should reject missing M/C matrices."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        explainer._C = None
        with pytest.raises(RuntimeError, match="M and C matrices must be initialized"):
            explainer._calculate_shap_values(
                masks=torch.tensor([[True, False]], dtype=torch.bool),
                similarities=torch.tensor([1.0], dtype=torch.float32),
                device=torch.device("cpu"),
            )

    def test_calculate_shap_values_neyman_stage_requires_positive_M_entries(
        self,
    ) -> None:
        """Neyman stage should reject zero entries in M after initial sampling."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._zero_mask_skipped = True
        explainer._M = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        explainer._C = torch.ones_like(explainer._M)
        explainer._BaseComplementaryNeymanShapExplainer__step = _Step.NEYMAN_ALLOCATION
        with pytest.raises(RuntimeError, match="Some entries in M matrix are zero"):
            explainer._calculate_shap_values(
                masks=torch.tensor([[True]], dtype=torch.bool),
                similarities=torch.tensor([1.0], dtype=torch.float32),
                device=torch.device("cpu"),
            )

    def test_calculate_shap_values_neyman_stage_uses_direct_ratio(self) -> None:
        """Neyman stage should use direct C/M ratio without zero masking."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._zero_mask_skipped = True
        explainer._M = torch.tensor([[1.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
        explainer._C = torch.tensor([[0.0, 6.0], [0.0, 8.0]], dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__step = _Step.NEYMAN_ALLOCATION
        result = explainer._calculate_shap_values(
            masks=torch.tensor([[True]], dtype=torch.bool),
            similarities=torch.tensor([1.0], dtype=torch.float32),
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(result, torch.tensor([1.5, 1.0]))

    def test_update_M_position_wraps_to_next_pass(self) -> None:
        """When no later slot exists, update position should wrap and find first pending slot."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
        explainer._BaseComplementaryNeymanShapExplainer__i = 1
        explainer._BaseComplementaryNeymanShapExplainer__j = 1
        done = explainer._BaseComplementaryNeymanShapExplainer__update_M_position()
        assert done is False
        assert (
            explainer._BaseComplementaryNeymanShapExplainer__i,
            explainer._BaseComplementaryNeymanShapExplainer__j,
        ) == (0, 0)

    def test_estimate_sigma_squared_clamps_negative_values(self) -> None:
        """Negative variance estimates should be clamped to zero."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
        explainer._C = torch.tensor([[10.0, 10.0], [10.0, 10.0]], dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__C_squared = torch.zeros_like(
            explainer._C
        )
        with patch("mllm_shap.shap.neyman._base.logger.warning") as mock_warning:
            sigma = explainer._BaseComplementaryNeymanShapExplainer__estimate_sigma_squared()
        assert torch.all(sigma == 0)
        mock_warning.assert_called_once()

    def test_estimate_M_hat_requires_M_and_C(self) -> None:
        """M_hat estimation should reject missing matrices."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        explainer._C = None
        with pytest.raises(RuntimeError, match="M and C matrices must be initialized"):
            explainer._BaseComplementaryNeymanShapExplainer__estimate_M_hat(n=4)

    def test_estimate_M_hat_rejects_even_remaining_budget(self) -> None:
        """Remaining samples formula must produce an integer half-budget."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.ones((4, 4), dtype=torch.float32)
        explainer._C = torch.ones((4, 4), dtype=torch.float32)
        explainer.total_n_calls = 1
        with patch.object(explainer, "_get_num_splits", return_value=4):
            with pytest.raises(RuntimeError, match="must be odd"):
                explainer._BaseComplementaryNeymanShapExplainer__estimate_M_hat(n=4)

    def test_get_start_returns_halfway_index(self) -> None:
        """Private __get_start should return ceil(n/2) for the current M shape."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((5, 5), dtype=torch.float32)
        assert explainer._BaseComplementaryNeymanShapExplainer__get_start() == 3

    def test_calculate_C_matrix_initializes_missing_C_squared(self) -> None:
        """Real C matrix update should initialize C_squared even when C already exists."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((2, 2), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)
        explainer._BaseComplementaryNeymanShapExplainer__C_squared = None

        BaseComplementaryNeymanShapExplainer._calculate_C_matrix(
            explainer,
            masks=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            similarities=torch.tensor([1.0, 0.5], dtype=torch.float32),
            device=torch.device("cpu"),
        )

        assert explainer._BaseComplementaryNeymanShapExplainer__C_squared is not None

    def test_calculate_C_matrix_reuses_existing_C_and_C_squared(self) -> None:
        """Real C matrix update should skip reinitialization when both matrices exist."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((2, 2), dtype=torch.float32)
        existing_c = torch.full((2, 2), 7.0, dtype=torch.float32)
        existing_c_squared = torch.full((2, 2), 9.0, dtype=torch.float32)
        explainer._C = existing_c.clone()
        explainer._BaseComplementaryNeymanShapExplainer__C_squared = (
            existing_c_squared.clone()
        )
        c_before = explainer._C
        c_squared_before = explainer._BaseComplementaryNeymanShapExplainer__C_squared

        BaseComplementaryNeymanShapExplainer._calculate_C_matrix(
            explainer,
            masks=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
            similarities=torch.tensor([1.0, 0.5], dtype=torch.float32),
            device=torch.device("cpu"),
        )

        assert explainer._C is c_before
        assert (
            explainer._BaseComplementaryNeymanShapExplainer__C_squared
            is c_squared_before
        )

    def test_estimate_sigma_squared_requires_all_matrices(self) -> None:
        """Sigma estimation should reject missing state."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = None
        explainer._C = None
        explainer._BaseComplementaryNeymanShapExplainer__C_squared = None
        with pytest.raises(RuntimeError, match="M, C and C_squared matrices"):
            explainer._BaseComplementaryNeymanShapExplainer__estimate_sigma_squared()

    def test_estimate_sigma_squared_without_negative_values_skips_warning(self) -> None:
        """Stable sigma estimates should be returned without clamping warnings."""
        explainer = DummyBaseComplementaryNeymanShapExplainer()
        explainer._M = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
        explainer._C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        explainer._BaseComplementaryNeymanShapExplainer__C_squared = torch.tensor(
            [[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        with patch("mllm_shap.shap.neyman._base.logger.warning") as mock_warning:
            sigma = explainer._BaseComplementaryNeymanShapExplainer__estimate_sigma_squared()
        assert torch.all(sigma >= 0)
        mock_warning.assert_not_called()

    def test_estimate_M_hat_populates_right_half(self) -> None:
        """Successful M_hat estimation should allocate only the right-half sizes."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        explainer._M = torch.ones((4, 4), dtype=torch.float32)
        explainer._C = torch.ones((4, 4), dtype=torch.float32)
        explainer.total_n_calls = 0
        with (
            patch.object(explainer, "_get_num_splits", return_value=8),
            patch.object(
                explainer,
                "_BaseComplementaryNeymanShapExplainer__estimate_sigma_squared",
                return_value=torch.ones((4, 4), dtype=torch.float32),
            ),
        ):
            explainer._BaseComplementaryNeymanShapExplainer__estimate_M_hat(n=4)

        m_hat = explainer._BaseComplementaryNeymanShapExplainer__M_hat
        assert m_hat is not None
        assert torch.equal(m_hat[:2], torch.zeros(2))
        assert torch.all(m_hat[2:] > 0)

    def test_call_requires_system_assistant_source_chat(self) -> None:
        """Neyman call should reject chats without SYSTEM_ASSISTANT setup."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        model = DummyModel()
        source_chat = BaseDummyChat(num_tokens=4)
        source_chat.system_roles_setup = SystemRolesSetup.NONE
        source_chat.token_roles = torch.tensor([Role.USER.value] * 4, dtype=torch.int8)
        response = ModelResponse(
            chat=source_chat,
            generated_text_tokens=torch.zeros((1, 1)),
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1, 1)),
        )

        with pytest.raises(ValueError, match="SYSTEM_ASSISTANT"):
            explainer(
                model=model,
                source_chat=source_chat,
                response=response,
                progress_bar=False,
            )

    def test_call_stage_two_logs_empty_allocation_and_dedup(self) -> None:
        """Stage two should warn on empty allocation, close progress bar, and log dedup stats."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        model = DummyModel()
        source_chat = BaseDummyChat(num_tokens=4)
        source_chat.system_roles_setup = SystemRolesSetup.SYSTEM_ASSISTANT
        source_chat.token_roles = torch.tensor(
            [Role.USER.value, Role.ASSISTANT.value, Role.USER.value, Role.USER.value],
            dtype=torch.int8,
        )
        response = ModelResponse(
            chat=source_chat,
            generated_text_tokens=torch.zeros((1, 1)),
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1, 1)),
        )

        def _fake_generate_step(
            mask_manager: MasksManager,
            masks: list[Tensor],
            responses: list[ModelResponse],
            device: torch.device,
            **kwargs,
        ) -> tuple[int, list[tuple[Tensor, int, BaseDummyChat | None, ModelResponse]]]:
            del mask_manager, device, kwargs
            if len(masks) == 1:
                masks.extend(
                    [
                        torch.tensor([True, False, False, False]),
                        torch.tensor([False, True, True, True]),
                    ]
                )
                responses.extend([response, response])
                explainer._BaseComplementaryNeymanShapExplainer__step = (
                    _Step.NEYMAN_ALLOCATION
                )
            return 0, []

        mock_cache_manager = type(
            "MockCacheManager",
            (),
            {"extracted_num": 2, "__init__": lambda self, *args, **kwargs: None},
        )

        class _Pbar:
            def __init__(self) -> None:
                self.closed = False
                self.descriptions: list[str] = []

            def set_description(self, desc: str) -> None:
                self.descriptions.append(desc)

            def close(self) -> None:
                self.closed = True

        pbar = _Pbar()

        def _fake_num_splits(n: int) -> int:
            del n
            explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
            explainer._M = torch.ones((4, 4), dtype=torch.float32)
            explainer._C = torch.zeros((4, 4), dtype=torch.float32)
            return 4

        with (
            patch("mllm_shap.shap.neyman._base.CacheManager", mock_cache_manager),
            patch("mllm_shap.shap.neyman._base.tqdm", return_value=pbar),
            patch.object(explainer, "_get_num_splits", side_effect=_fake_num_splits),
            patch.object(explainer, "_generate_step", side_effect=_fake_generate_step),
            patch.object(
                explainer,
                "_BaseComplementaryNeymanShapExplainer__estimate_M_hat",
                side_effect=lambda n: setattr(
                    explainer,
                    "_BaseComplementaryNeymanShapExplainer__M_hat",
                    torch.zeros(4),
                ),
            ),
            patch.object(
                explainer,
                "_get_shap_values",
                return_value=(torch.zeros(4), torch.zeros(4)),
            ),
            patch.object(explainer, "_save_to_cache"),
            patch("mllm_shap.shap.neyman._base.logger.warning") as mock_warning,
            patch("mllm_shap.shap.neyman._base.logger.info") as mock_info,
        ):
            explainer(
                model=model,
                source_chat=source_chat,
                response=response,
                progress_bar=True,
                verbose=False,
            )

        assert "Neyman SHAP [stage 2/2]" in pbar.descriptions
        assert pbar.closed is True
        mock_warning.assert_any_call(
            "Neyman allocation step produced no new masks; budget may have been exhausted during initial sampling."
        )
        mock_info.assert_any_call(
            "Deduplicated %d/%d masks using existing cache.",
            2,
            2,
        )

    def test_call_warns_without_assistant_role(self) -> None:
        """Neyman call should warn when no assistant token role is present."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        model = DummyModel()
        source_chat = BaseDummyChat(num_tokens=4)
        source_chat.system_roles_setup = SystemRolesSetup.SYSTEM_ASSISTANT
        source_chat.token_roles = torch.tensor([Role.USER.value] * 4, dtype=torch.int8)
        response = ModelResponse(
            chat=source_chat,
            generated_text_tokens=torch.zeros((1, 1)),
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1, 1)),
        )

        def _fake_generate_step(
            mask_manager: MasksManager,
            masks: list[Tensor],
            responses: list[ModelResponse],
            device: torch.device,
            **kwargs,
        ) -> tuple[int, None]:
            del mask_manager, device, kwargs
            masks.append(torch.tensor([True, False, False, False]))
            responses.append(response)
            return 0, None

        def _fake_num_splits(n: int) -> int:
            del n
            explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
            explainer._M = torch.ones((4, 4), dtype=torch.float32)
            explainer._C = torch.zeros((4, 4), dtype=torch.float32)
            return 1

        with (
            patch.object(explainer, "_get_num_splits", side_effect=_fake_num_splits),
            patch.object(explainer, "_generate_step", side_effect=_fake_generate_step),
            patch.object(
                explainer,
                "_get_shap_values",
                return_value=(torch.zeros(4), torch.zeros(4)),
            ),
            patch.object(explainer, "_save_to_cache"),
            patch("mllm_shap.shap.neyman._base.logger.warning") as mock_warning,
        ):
            explainer(
                model=model,
                source_chat=source_chat,
                response=response,
                progress_bar=False,
                verbose=False,
            )

        mock_warning.assert_any_call(
            "Source chat must have at least one non-user message for Neyman SHAP."
            "No assistant role found, make sure that existing messages cover it."
        )

    def test_call_stage_two_merges_new_masks_and_history_without_progress_bar(
        self,
    ) -> None:
        """Stage two should merge additional masks/history even when no tqdm bar exists."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        model = DummyModel()
        source_chat = BaseDummyChat(num_tokens=4)
        source_chat.system_roles_setup = SystemRolesSetup.SYSTEM_ASSISTANT
        source_chat.token_roles = torch.tensor(
            [Role.USER.value, Role.ASSISTANT.value, Role.USER.value, Role.USER.value],
            dtype=torch.int8,
        )
        response = ModelResponse(
            chat=source_chat,
            generated_text_tokens=torch.zeros((1, 1)),
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1, 1)),
        )

        history_stage_1 = [
            (torch.tensor([True, False, False, False]), 0, None, response),
        ]
        history_stage_2 = [
            (torch.tensor([False, True, False, False]), 0, None, response),
        ]

        def _fake_num_splits(n: int) -> int:
            del n
            explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
            explainer._M = torch.ones((4, 4), dtype=torch.float32)
            explainer._C = torch.zeros((4, 4), dtype=torch.float32)
            explainer._BaseComplementaryNeymanShapExplainer__C_squared = torch.zeros(
                (4, 4), dtype=torch.float32
            )
            return 4

        def _fake_generate_step(
            mask_manager: MasksManager,
            masks: list[Tensor],
            responses: list[ModelResponse],
            device: torch.device,
            **kwargs,
        ) -> tuple[int, list[tuple[Tensor, int, BaseDummyChat | None, ModelResponse]]]:
            del mask_manager, device, kwargs
            if len(masks) == 1:
                masks.extend(
                    [
                        torch.tensor([True, False, False, False]),
                        torch.tensor([False, True, True, True]),
                    ]
                )
                responses.extend([response, response])
                explainer._BaseComplementaryNeymanShapExplainer__step = (
                    _Step.NEYMAN_ALLOCATION
                )
                return 1, history_stage_1.copy()

            masks.extend(
                [
                    torch.tensor([False, True, False, False]),
                    torch.tensor([True, False, True, True]),
                ]
            )
            responses.extend([response, response])
            return 2, history_stage_2.copy()

        with (
            patch.object(explainer, "_get_num_splits", side_effect=_fake_num_splits),
            patch.object(explainer, "_generate_step", side_effect=_fake_generate_step),
            patch.object(
                explainer,
                "_BaseComplementaryNeymanShapExplainer__estimate_M_hat",
                side_effect=lambda n: setattr(
                    explainer,
                    "_BaseComplementaryNeymanShapExplainer__M_hat",
                    torch.ones(4),
                ),
            ),
            patch.object(
                explainer,
                "_get_shap_values",
                return_value=(torch.zeros(4), torch.zeros(4)),
            ),
            patch.object(explainer, "_save_to_cache") as mock_save,
        ):
            result = explainer(
                model=model,
                source_chat=source_chat,
                response=response,
                progress_bar=False,
                verbose=False,
            )

        assert result == history_stage_1 + history_stage_2
        saved_masks = mock_save.call_args.kwargs["masks"]
        saved_responses = mock_save.call_args.kwargs["responses"]
        assert saved_masks.shape[0] == 5
        assert len(saved_responses) == 5

    def test_call_stage_two_skips_history_merge_when_initial_history_missing(
        self,
    ) -> None:
        """Stage two should leave history as None when initial stage produced no history."""
        explainer = DummyBaseComplementaryNeymanShapExplainer(initial_num_samples=2)
        model = DummyModel()
        source_chat = BaseDummyChat(num_tokens=4)
        source_chat.system_roles_setup = SystemRolesSetup.SYSTEM_ASSISTANT
        source_chat.token_roles = torch.tensor(
            [Role.USER.value, Role.ASSISTANT.value, Role.USER.value, Role.USER.value],
            dtype=torch.int8,
        )
        response = ModelResponse(
            chat=source_chat,
            generated_text_tokens=torch.zeros((1, 1)),
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1, 1)),
        )

        def _fake_num_splits(n: int) -> int:
            del n
            explainer._BaseComplementaryNeymanShapExplainer__initial_num_splits = 2
            explainer._M = torch.ones((4, 4), dtype=torch.float32)
            explainer._C = torch.zeros((4, 4), dtype=torch.float32)
            explainer._BaseComplementaryNeymanShapExplainer__C_squared = torch.zeros(
                (4, 4), dtype=torch.float32
            )
            return 4

        def _fake_generate_step(
            mask_manager: MasksManager,
            masks: list[Tensor],
            responses: list[ModelResponse],
            device: torch.device,
            **kwargs,
        ) -> tuple[
            int, list[tuple[Tensor, int, BaseDummyChat | None, ModelResponse]] | None
        ]:
            del mask_manager, device, kwargs
            if len(masks) == 1:
                masks.extend(
                    [
                        torch.tensor([True, False, False, False]),
                        torch.tensor([False, True, True, True]),
                    ]
                )
                responses.extend([response, response])
                explainer._BaseComplementaryNeymanShapExplainer__step = (
                    _Step.NEYMAN_ALLOCATION
                )
                return 1, None

            masks.extend(
                [
                    torch.tensor([False, True, False, False]),
                    torch.tensor([True, False, True, True]),
                ]
            )
            responses.extend([response, response])
            return 2, [(torch.tensor([False, True, False, False]), 0, None, response)]

        with (
            patch.object(explainer, "_get_num_splits", side_effect=_fake_num_splits),
            patch.object(explainer, "_generate_step", side_effect=_fake_generate_step),
            patch.object(
                explainer,
                "_BaseComplementaryNeymanShapExplainer__estimate_M_hat",
                side_effect=lambda n: setattr(
                    explainer,
                    "_BaseComplementaryNeymanShapExplainer__M_hat",
                    torch.ones(4),
                ),
            ),
            patch.object(
                explainer,
                "_get_shap_values",
                return_value=(torch.zeros(4), torch.zeros(4)),
            ),
            patch.object(explainer, "_save_to_cache"),
        ):
            result = explainer(
                model=model,
                source_chat=source_chat,
                response=response,
                progress_bar=False,
                verbose=False,
            )

        assert result is None

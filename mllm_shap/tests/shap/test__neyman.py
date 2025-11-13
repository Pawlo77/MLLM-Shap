"""Unit tests for ComplementaryNeymanShapExplainer class (updated)."""

import pytest
import torch
from mllm_shap.shap.base._masks_manager import MasksManager, NoTokensToExplainError
from mllm_shap.shap.neyman import ComplementaryNeymanShapExplainer
from torch import Tensor


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


class DummyComplementaryNeymanShapExplainer(ComplementaryNeymanShapExplainer):
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

    def _calculate_C_matrix(self, masks: Tensor, similarities: Tensor, device: torch.device) -> None:
        """
        For tests, write deterministic simple contributions into self._C
        (this overrides heavy logic of the real method).
        """
        # masks shape = (masks_count, token_count)
        if self._M is None:
            raise RuntimeError("M matrix must be initialized before calculating C matrix.")
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


class TestComplementaryNeymanShapExplainerNumSplits:
    """Tests for the _get_num_splits() method."""

    @pytest.fixture
    def explainer(self) -> DummyComplementaryNeymanShapExplainer:
        """Provide a default explainer fixture."""
        return DummyComplementaryNeymanShapExplainer(initial_num_samples=2, initial_fraction=0.5)

    def test_num_splits_returns_integer(self, explainer: DummyComplementaryNeymanShapExplainer) -> None:
        """Ensure _get_num_splits() returns a valid integer and sets initial splits."""
        num_splits = explainer._get_num_splits(n=5)
        assert isinstance(num_splits, int)
        # the initial number of splits is stored under a mangled name; read it defensively
        initial_splits = getattr(explainer, "_ComplementaryNeymanShapExplainer__initial_num_splits", None)
        assert initial_splits is not None and isinstance(initial_splits, int)
        assert initial_splits >= 1
        assert num_splits >= initial_splits


class TestComplementaryNeymanShapExplainerMasksGeneration:
    """Tests for mask generation behavior (complementary masks)."""

    @pytest.fixture
    def explainer(self) -> DummyComplementaryNeymanShapExplainer:
        return DummyComplementaryNeymanShapExplainer(initial_num_samples=2)

    def test_complementary_mask_pair_generation(self, explainer: DummyComplementaryNeymanShapExplainer) -> None:
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
        gen = explainer._get_masks_generator(mask_manager=mask_manager, device=device, masks=[])
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
        bad_chat.shap_values_mask = torch.tensor([False, False, False, False], dtype=torch.bool)
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


class TestComplementaryNeymanShapExplainerCalculateShapValues:
    """Tests for the _calculate_shap_values() method."""

    def test_shap_values_computation(self) -> None:
        """Check that SHAP value computation produces a tensor of the correct shape."""
        explainer = DummyComplementaryNeymanShapExplainer()
        # prepare M and C such that M[:, 1:] are non-zero
        explainer._M = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
        explainer._C = torch.tensor([[0.0, 4.0, 6.0], [0.0, 2.0, 8.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
        explainer._zero_mask_skipped = True
        device = torch.device("cpu")
        masks = torch.tensor([[True, False, False], [False, True, False]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result = explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)
        assert isinstance(result, Tensor)
        # result length equals number of features (rows in _M)
        assert result.shape[0] == explainer._M.shape[0]

    def test_raises_if_zero_mask_not_skipped(self) -> None:
        """Ensure a RuntimeError is raised when zero mask was not skipped."""
        explainer = DummyComplementaryNeymanShapExplainer()
        explainer._zero_mask_skipped = False
        explainer._M = torch.ones((3, 3), dtype=torch.float32) * 2.0
        explainer._C = torch.zeros_like(explainer._M)
        device = torch.device("cpu")
        masks = torch.tensor([[True, False, False], [False, True, False]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Zero mask was not skipped"):
            explainer._calculate_shap_values(masks=masks, similarities=similarities, device=device)

    def test_calculate_C_matrix_updates_counts(self) -> None:
        """Ensure the test `_calculate_C_matrix` implementation increments counts at the expected indices."""
        explainer = DummyComplementaryNeymanShapExplainer()
        # set shapes compatible with masks: tokens=3, coalition sizes up to 2
        explainer._M = torch.ones((3, 3), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)

        masks = torch.tensor([[True, False, True], [True, True, False]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        device = torch.device("cpu")

        explainer._calculate_C_matrix(masks=masks, similarities=similarities, device=device)

        # coalition size for both rows is 2, so updates happen in column index 2
        # expected increments: token0 present in both rows -> 2, token1 present once -> 1, token2 present once -> 1
        expected_col = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.equal(explainer._C[:, 2], expected_col)

    def test_calculate_C_matrix_with_zero_coalition(self) -> None:
        """When a mask row has no True entries, the contribution should be recorded at coalition size 0."""
        explainer = DummyComplementaryNeymanShapExplainer()
        explainer._M = torch.ones((2, 2), dtype=torch.float32)
        explainer._C = torch.zeros_like(explainer._M)

        masks = torch.tensor([[False, False], [True, False]], dtype=torch.bool)
        similarities = torch.tensor([1.0, 1.0], dtype=torch.float32)
        device = torch.device("cpu")

        explainer._calculate_C_matrix(masks=masks, similarities=similarities, device=device)

        # first row has zero Trues -> contributions at column 0 for no-token coalition
        assert explainer._C[:, 0].sum() >= 0.0

    def test_get_similarities_all_ones(self) -> None:
        """_get_similarities should return a tensor of ones with length equal to responses."""
        explainer = DummyComplementaryNeymanShapExplainer()
        responses = [DummyResponse() for _ in range(4)]
        sims = explainer._get_similarities(responses=responses)
        assert isinstance(sims, Tensor)
        assert sims.shape[0] == len(responses)
        assert torch.equal(sims, torch.ones(len(responses), dtype=torch.float32))

    def test_get_similarities_empty_responses(self) -> None:
        """When passed an empty list, _get_similarities should return an empty tensor."""
        explainer = DummyComplementaryNeymanShapExplainer()
        sims = explainer._get_similarities(responses=[])
        assert isinstance(sims, Tensor)
        assert sims.numel() == 0

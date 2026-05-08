"""Integration tests for Hierarchical and Neyman explainers using Mock connector."""

import pytest
import torch

from mllm_shap.connectors.mock import Mock, MockChat
from mllm_shap.connectors.enums import Role, SystemRolesSetup
from mllm_shap.shap.hierarchical import HierarchicalExplainer
from mllm_shap.shap.hierarchical.enums import Mode
from mllm_shap.shap.precise import PreciseShapExplainer
from mllm_shap.shap.monte_carlo import StandardMcShapExplainer, LimitedMcShapExplainer
from mllm_shap.shap.complementary import StandardComplementaryShapExplainer
from mllm_shap.shap.neyman import StandardComplementaryNeymanShapExplainer
from mllm_shap.shap.normalizers import MinMaxNormalizer, IdentityNormalizer
from mllm_shap.shap.base.shap_explainer import NotEnoughTokensToExplainError
from mllm_shap.errors import ValidationError


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model(device: torch.device) -> Mock:
    return Mock(device=device)


def _build_chat(model: Mock, text: str = "A B C D") -> MockChat:
    """Helper: build a chat with user text + assistant response."""
    chat = model.get_new_chat(
        system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
    )
    chat.new_turn(Role.SYSTEM)
    chat.add_text("Be helpful.")
    chat.end_turn()
    chat.new_turn(Role.USER)
    chat.add_text(text)
    chat.end_turn()
    return chat


# ──────────────── HierarchicalExplainer Integration ────────────────


class TestHierarchicalExplainerIntegration:
    """Integration tests for HierarchicalExplainer with Mock model."""

    def test_basic_text_mode_explanation(
        self, model: Mock, device: torch.device
    ) -> None:
        """Full hierarchical explanation in TEXT mode produces valid result."""
        chat = _build_chat(model)
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            mode=Mode.TEXT,
            k=3,
        )

        result = explainer(chat=chat, progress_bar=False)
        assert result is not None
        assert result.source_chat is chat
        assert result.full_chat is not None
        assert explainer.n_calls > 0
        assert explainer.total_n_calls > 0

    def test_k_must_be_at_least_2(self, model: Mock) -> None:
        """k < 2 raises ValidationError."""
        with pytest.raises(ValidationError, match="k must be an integer"):
            HierarchicalExplainer(
                model=model,
                shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
                k=1,
            )

    def test_verbose_saves_computation_graph(
        self, model: Mock, device: torch.device
    ) -> None:
        """verbose=True populates computation_graph."""
        chat = _build_chat(model, text="A B C")
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            mode=Mode.TEXT,
            k=3,
        )

        result = explainer(chat=chat, progress_bar=False, verbose=True)
        assert result is not None
        assert explainer.computation_graph is not None

    def test_multi_modal_mode(self, model: Mock, device: torch.device) -> None:
        """Text-only input in MULTI_MODAL mode can create undersized groups."""
        chat = _build_chat(model)
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            mode=Mode.MULTI_MODAL,
            k=4,
        )

        with pytest.raises(NotEnoughTokensToExplainError):
            explainer(chat=chat, progress_bar=False)

    def test_multi_modal_multi_user_mode(
        self, model: Mock, device: torch.device
    ) -> None:
        """Text-only input in MULTI_MODAL_MULTI_USER mode can create undersized groups."""
        chat = _build_chat(model)
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            mode=Mode.MULTI_MODAL_MULTI_USER,
            k=4,
        )

        with pytest.raises(NotEnoughTokensToExplainError):
            explainer(chat=chat, progress_bar=False)

    def test_importance_sampling_with_fraction_explainer(
        self, model: Mock, device: torch.device
    ) -> None:
        """use_importance_sampling=True with fraction-based explainer works."""
        chat = _build_chat(model, text="A B C D E")
        mc = StandardMcShapExplainer(
            normalizer=IdentityNormalizer(),
            fraction=0.5,
            allow_mask_duplicates=True,
        )
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=mc,
            mode=Mode.TEXT,
            k=3,
            use_importance_sampling=True,
            importance_sampling_min_fraction=0.2,
        )

        result = explainer(chat=chat, progress_bar=False)
        assert result is not None

    def test_importance_sampling_without_fraction_raises(self, model: Mock) -> None:
        """use_importance_sampling=True without fraction → ValidationError."""
        with pytest.raises(ValidationError, match="fraction-based"):
            HierarchicalExplainer(
                model=model,
                shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
                mode=Mode.TEXT,
                k=3,
                use_importance_sampling=True,
            )

    def test_bad_importance_sampling_min_fraction_raises(self, model: Mock) -> None:
        """Invalid min_fraction raises ValidationError."""
        mc = StandardMcShapExplainer(normalizer=MinMaxNormalizer(), fraction=0.5)
        with pytest.raises(ValidationError, match="importance_sampling_min_fraction"):
            HierarchicalExplainer(
                model=model,
                shap_explainer=mc,
                mode=Mode.TEXT,
                k=3,
                use_importance_sampling=True,
                importance_sampling_min_fraction=0.0,
            )

    def test_first_layer_explainer(self, model: Mock, device: torch.device) -> None:
        """first_layer_explainer is used when provided."""
        chat = _build_chat(model, text="A B C D E")
        first_layer = PreciseShapExplainer(normalizer=MinMaxNormalizer())
        explainer = HierarchicalExplainer(
            model=model,
            shap_explainer=PreciseShapExplainer(normalizer=MinMaxNormalizer()),
            first_layer_explainer=first_layer,
            mode=Mode.MULTI_MODAL,
            k=3,
        )

        result = explainer(chat=chat, progress_bar=False)
        assert result is not None


# ──────────────── StandardComplementaryNeymanShapExplainer Integration ────────────────


class TestNeymanIntegration:
    """Integration tests for StandardComplementaryNeymanShapExplainer with Mock model."""

    def test_basic_neyman_explanation(self, model: Mock, device: torch.device) -> None:
        """Full Neyman two-phase explanation runs without error."""
        chat = _build_chat(model, text="A B C")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=8)

        explainer = StandardComplementaryNeymanShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=40,
            initial_num_samples=4,
            allow_mask_duplicates=True,
        )

        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        # Returns history or None
        assert result is None or isinstance(result, list)

    def test_neyman_with_default_formula(
        self, model: Mock, device: torch.device
    ) -> None:
        """Neyman with no initial params uses default formula."""
        chat = _build_chat(model, text="A B C")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryNeymanShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=30,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)
        assert explainer.initial_steps > 0

    def test_neyman_with_fraction(self, model: Mock, device: torch.device) -> None:
        """Neyman with fraction-based initial sampling."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryNeymanShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=20,
            initial_fraction=0.1,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)

    def test_neyman_standard_method(self, model: Mock, device: torch.device) -> None:
        """use_standard_method=True runs random initial sampling."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryNeymanShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=30,
            initial_num_samples=4,
            allow_mask_duplicates=True,
        )
        assert explainer.use_standard_method is True
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)

    def test_neyman_verbose_returns_history(
        self, model: Mock, device: torch.device
    ) -> None:
        """verbose=True returns history list."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryNeymanShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=20,
            initial_num_samples=2,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
            verbose=True,
        )
        assert isinstance(result, list)


# ──────────────── StandardComplementaryShapExplainer Integration ────────────────


class TestComplementaryIntegration:
    """Integration tests for StandardComplementaryShapExplainer."""

    def test_basic_complementary(self, model: Mock, device: torch.device) -> None:
        """Complementary explainer produces valid results."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=12,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)

    def test_complementary_with_fraction(
        self, model: Mock, device: torch.device
    ) -> None:
        """Fraction-based complementary."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardComplementaryShapExplainer(
            normalizer=MinMaxNormalizer(),
            fraction=0.8,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)


# ──────────────── StandardMcShapExplainer Integration ────────────────


class TestMonteCarloIntegration:
    """Integration tests for MC explainer with Mock."""

    def test_mc_minimal_budget(self, model: Mock, device: torch.device) -> None:
        """MC with num_samples=-1 uses minimal masks (LimitedMcShapExplainer)."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = LimitedMcShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=-1,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
        )
        assert result is None or isinstance(result, list)

    def test_mc_verbose_returns_history(
        self, model: Mock, device: torch.device
    ) -> None:
        """verbose=True produces history."""
        chat = _build_chat(model, text="A B C D")
        resp = model.generate(chat=chat, keep_history=True, max_new_tokens=4)

        explainer = StandardMcShapExplainer(
            normalizer=MinMaxNormalizer(),
            num_samples=8,
            allow_mask_duplicates=True,
        )
        result = explainer(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=False,
            verbose=True,
        )
        assert isinstance(result, list)

"""Unit tests for the refactored BaseShapExplainer class."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.connectors.base.explainer_cache import ExplainerCache
from mllm_shap.observability.sink import InMemoryObservabilitySink
from mllm_shap.shap.base.shap_explainer import (
    BaseShapExplainer,
    NotEnoughTokensToExplainError,
)
from mllm_shap.shap.base._masks_manager import MasksManager
from mllm_shap.shap.embeddings import MeanReducer
from mllm_shap.shap.enums import Mode
from mllm_shap.shap.normalizers import PowerShiftNormalizer
from mllm_shap.shap.similarity import CosineSimilarity

from ...dummy import DummyChat, DummyModel, DummyShapExplainer


class TestBaseShapExplainer:
    """Tests for the refactored BaseShapExplainer class."""

    @staticmethod
    @pytest.fixture
    def explainer_instance() -> BaseShapExplainer:
        """Fixture for DummyShapExplainer."""
        return DummyShapExplainer()

    @staticmethod
    @pytest.fixture
    def dummy_chat_instance() -> BaseMllmChat:
        """Fixture for BaseMllmChat."""
        return DummyChat(num_tokens=3)

    @staticmethod
    @pytest.fixture
    def dummy_model_instance() -> BaseMllmModel:
        """Fixture for DummyModel."""
        return DummyModel()

    @staticmethod
    @pytest.fixture
    def dummy_response_instance(dummy_chat_instance: BaseMllmChat) -> ModelResponse:
        """Fixture for ModelResponse."""
        return ModelResponse(
            chat=dummy_chat_instance,
            generated_audio_tokens=torch.tensor([]),
            generated_text_tokens=torch.tensor([1, 2, 3]),
            generated_modality_flag=torch.ones(3, dtype=torch.bool),
        )

    def test_initialization_defaults(
        self, explainer_instance: BaseShapExplainer
    ) -> None:
        """Test default initialization components."""
        expl = explainer_instance
        assert isinstance(expl.embedding_reducer, MeanReducer)
        assert isinstance(expl.similarity_measure, CosineSimilarity)
        assert isinstance(expl.normalizer, PowerShiftNormalizer)
        assert expl.mode in (Mode.STATIC, Mode.CONTEXTUAL)

    def test_hash_returns_int(self, explainer_instance: BaseShapExplainer) -> None:
        """Test that __hash__ returns an integer and is deterministic."""
        h1 = hash(explainer_instance)
        h2 = hash(explainer_instance)
        assert isinstance(h1, int)
        assert h1 == h2

    def test_get_shap_values_computation(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_response_instance: ModelResponse,
        dummy_chat_instance: BaseMllmChat,
    ) -> None:
        """Test that SHAP and normalized SHAP values are computed properly."""
        masks = torch.tensor([[True, False, True], [False, True, True]])
        responses = [dummy_response_instance, dummy_response_instance]

        explainer_instance.similarity_measure.operates_on_embeddings = True
        shap_values, normalized = explainer_instance._get_shap_values(
            source_chat=dummy_chat_instance,
            model=dummy_model_instance,
            masks=masks,
            responses=responses,
            device=torch.device("cpu"),
        )

        assert shap_values.shape == (3,)
        assert normalized.shape == (3,)
        assert torch.isfinite(normalized).all()

    def test_call_returns_history(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test __call__ returns history correctly when verbose=True."""
        history = explainer_instance(
            model=dummy_model_instance,
            source_chat=dummy_chat_instance,
            response=dummy_response_instance,
            progress_bar=False,
            verbose=True,
        )

        assert isinstance(history, list)
        assert len(history) > 0
        for el in history:
            assert len(el) == 4
            assert isinstance(el[0], Tensor)  # generated mask
            assert isinstance(el[1], int)  # mask hash
            assert isinstance(el[2], BaseMllmChat)  # chat
            assert isinstance(el[3], ModelResponse)  # model response

    @patch("mllm_shap.shap.base.shap_explainer.MasksManager")
    @patch("mllm_shap.shap.base.shap_explainer.CacheManager")
    @patch("mllm_shap.shap.base.shap_explainer.generate_responses")
    def test_call_raises_not_enough_tokens(
        self,
        mock_generate_responses: MagicMock,
        mock_cache_manager: MagicMock,
        mock_masks_manager: MagicMock,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: BaseMllmModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test that NotEnoughTokensToExplainError is raised when all chats are skipped."""
        mock_masks_manager.return_value.n = 3
        mock_masks_manager.return_value.max_masks_number = 2
        mock_masks_manager.return_value.get_initial_mask.return_value = torch.tensor(
            [True, True, True]
        )
        mock_cache_manager.return_value.extracted_num = 0
        # simulate all chats skipped
        mock_generate_responses.return_value = (2, [])

        with pytest.raises(NotEnoughTokensToExplainError):
            explainer_instance(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=True,
            )

    @patch("mllm_shap.connectors.base.explainer_cache.ExplainerCache.create")
    def test_save_to_cache_creates_new_cache(
        self,
        mock_create: MagicMock,
        explainer_instance: BaseShapExplainer,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Test _save_to_cache assigns a new ExplainerCache when no existing cache."""
        responses = [dummy_response_instance]
        masks = torch.ones((2, 3), dtype=torch.bool)
        shap_values = torch.zeros(3)
        norm_values = torch.ones(3)
        explainer_instance._save_to_cache(
            chat=dummy_chat_instance,
            source_chat=deepcopy(dummy_chat_instance),
            responses=responses,
            masks=masks,
            shap_values=shap_values,
            normalized_shap_values=norm_values,
        )
        mock_create.assert_called_once()

    def test_initialize_state_resets_tracking(
        self, explainer_instance: BaseShapExplainer
    ) -> None:
        """Ensure internal counters reset before explanation."""
        explainer_instance.total_n_calls = 5
        setattr(explainer_instance, "_first_call", False)

        explainer_instance._initialize_state()

        assert explainer_instance.total_n_calls == 0
        assert explainer_instance._first_call is True

    def test_masks_generator_filters_invalid_splits(
        self, dummy_chat_instance: BaseMllmChat
    ) -> None:
        """Generator should skip zero, all-ones, and duplicate splits."""

        class ControlledExplainer(DummyShapExplainer):
            def __init__(self) -> None:
                super().__init__()
                self._splits = [
                    torch.zeros((1, 3), dtype=torch.bool),
                    torch.ones((1, 3), dtype=torch.bool),
                    torch.tensor([[True, False, True]], dtype=torch.bool),
                    torch.tensor([[True, False, True]], dtype=torch.bool),
                    torch.tensor([[False, True, True]], dtype=torch.bool),
                    None,
                ]

            def _get_num_splits(self, n: int) -> int:
                return 5

            def _get_next_split(
                self,
                n: int,
                device: torch.device,
                generated_masks_num: int,
                existing_masks: list[Tensor] | None = None,
            ) -> Tensor | None:
                del n, device, generated_masks_num, existing_masks
                return self._splits.pop(0)

        explainer = ControlledExplainer()
        mask_manager = MasksManager(chat=dummy_chat_instance)
        device = dummy_chat_instance.torch_device
        masks = [mask_manager.get_initial_mask(device=device)]

        gen = explainer._get_masks_generator(
            mask_manager=mask_manager, device=device, masks=masks
        )
        produced = list(gen)

        assert len(produced) == 2
        assert gen.generated_masks == 2
        valid_masks = [prod[0] for prod in produced]
        expected_masks = [
            torch.tensor([True, False, True], dtype=torch.bool),
            torch.tensor([False, True, True], dtype=torch.bool),
        ]
        for obtained, expected in zip(valid_masks, expected_masks):
            assert torch.equal(obtained, expected)

    def test_masks_generator_skips_when_prepare_mask_returns_none(
        self, dummy_chat_instance: BaseMllmChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generator should skip splits when mask manager cannot prepare a valid mask."""

        class SingleSplitExplainer(DummyShapExplainer):
            def __init__(self) -> None:
                super().__init__()
                self._splits = [
                    torch.tensor([[True, False, True]], dtype=torch.bool),
                    None,
                ]

            def _get_num_splits(self, n: int) -> int:
                del n
                return 1

            def _get_next_split(
                self,
                n: int,
                device: torch.device,
                generated_masks_num: int,
                existing_masks: list[Tensor] | None = None,
            ) -> Tensor | None:
                del n, device, generated_masks_num, existing_masks
                return self._splits.pop(0)

        explainer = SingleSplitExplainer()
        mask_manager = MasksManager(chat=dummy_chat_instance)
        monkeypatch.setattr(mask_manager, "prepare_mask", lambda split, device: None)

        produced = list(
            explainer._get_masks_generator(
                mask_manager=mask_manager,
                device=dummy_chat_instance.torch_device,
                masks=[],
            )
        )

        assert produced == []

    def test_save_to_cache_raises_when_cache_already_exists(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """_save_to_cache should reject overwriting existing cache."""
        dummy_chat_instance.cache = MagicMock()
        with pytest.raises(ValueError, match="SHAP cache already exists"):
            explainer_instance._save_to_cache(
                chat=dummy_chat_instance,
                source_chat=deepcopy(dummy_chat_instance),
                responses=[dummy_response_instance],
                masks=torch.ones((2, 3), dtype=torch.bool),
                shap_values=torch.zeros(3),
                normalized_shap_values=torch.zeros(3),
            )

    def test_get_similarities_uses_embeddings(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: DummyModel,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """When measure operates on embeddings, model embeddings should be queried."""
        responses = [dummy_response_instance, deepcopy(dummy_response_instance)]
        similarity = MagicMock(return_value=torch.tensor([0.1, 0.2]))
        similarity.operates_on_embeddings = True
        explainer_instance.similarity_measure = similarity

        with patch.object(
            dummy_model_instance,
            "get_contextual_embeddings",
            wraps=dummy_model_instance.get_contextual_embeddings,
        ) as spy:
            result = explainer_instance._get_similarities(
                responses=responses, model=dummy_model_instance
            )

        assert isinstance(result, torch.Tensor)
        spy.assert_called_once()
        similarity.assert_called_once()
        kwargs = similarity.call_args.kwargs
        assert torch.equal(kwargs["base"], kwargs["other"][0])

    def test_get_similarities_with_raw_responses(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: DummyModel,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Raw similarity path should bypass embedding extraction."""
        responses = [dummy_response_instance, deepcopy(dummy_response_instance)]
        similarity = MagicMock(return_value=torch.tensor([0.3]))
        similarity.operates_on_embeddings = False
        explainer_instance.similarity_measure = similarity

        with patch.object(
            dummy_model_instance,
            "get_contextual_embeddings",
            wraps=dummy_model_instance.get_contextual_embeddings,
        ) as spy:
            result = explainer_instance._get_similarities(
                responses=responses, model=dummy_model_instance
            )

        assert isinstance(result, torch.Tensor)
        spy.assert_not_called()
        similarity.assert_called_once_with(base=responses[0], other=responses)

    def test_get_similarities_uses_embedding_model_when_provided(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: DummyModel,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Embedding model path should bypass model embedding methods."""
        responses = [dummy_response_instance, deepcopy(dummy_response_instance)]
        similarity = MagicMock(return_value=torch.tensor([0.2, 0.3]))
        similarity.operates_on_embeddings = True
        explainer_instance.similarity_measure = similarity
        explainer_instance.embedding_model = lambda responses: [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]
        explainer_instance.embedding_reducer = lambda embeddings: torch.stack(
            embeddings
        )

        with (
            patch.object(dummy_model_instance, "get_static_embeddings") as static_spy,
            patch.object(dummy_model_instance, "get_contextual_embeddings") as ctx_spy,
        ):
            _ = explainer_instance._get_similarities(
                responses=responses, model=dummy_model_instance
            )

        static_spy.assert_not_called()
        ctx_spy.assert_not_called()

    def test_get_similarities_static_mode_uses_static_embeddings(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: DummyModel,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """STATIC mode without embedding_model should call model.get_static_embeddings."""
        responses = [dummy_response_instance, deepcopy(dummy_response_instance)]
        similarity = MagicMock(return_value=torch.tensor([0.4, 0.5]))
        similarity.operates_on_embeddings = True
        explainer_instance.similarity_measure = similarity
        explainer_instance.embedding_model = None
        explainer_instance.mode = Mode.STATIC

        with (
            patch.object(
                dummy_model_instance,
                "get_static_embeddings",
                wraps=dummy_model_instance.get_static_embeddings,
            ) as static_spy,
            patch.object(
                dummy_model_instance,
                "get_contextual_embeddings",
                wraps=dummy_model_instance.get_contextual_embeddings,
            ) as ctx_spy,
        ):
            _ = explainer_instance._get_similarities(
                responses=responses, model=dummy_model_instance
            )

        static_spy.assert_called_once()
        ctx_spy.assert_not_called()

    def test_get_shap_values_with_external_groups(
        self,
        dummy_model_instance: DummyModel,
    ) -> None:
        """External group ids should broadcast group scores to all members."""

        class DeterministicExplainer(DummyShapExplainer):
            def _calculate_shap_values(
                self, masks: Tensor, similarities: Tensor, device: torch.device
            ) -> Tensor:
                del masks, similarities, device
                return torch.tensor([0.2, 0.9, 0.1, 0.7, 0.4], dtype=torch.float32)

        explainer = DeterministicExplainer()
        explainer.normalizer = lambda values: values
        source_chat = DummyChat(num_tokens=5)
        source_chat.external_group_ids = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)
        source_chat.shap_values_mask = torch.tensor(
            [True, True, True, True, True], dtype=torch.bool
        )

        masks = torch.tensor(
            [
                [True, True, True, True, True],
                [True, False, True, False, True],
            ],
            dtype=torch.bool,
        )
        base_response = ModelResponse(
            chat=source_chat,
            generated_audio_tokens=torch.zeros((1, 1)),
            generated_text_tokens=torch.zeros((1, 1)),
            generated_modality_flag=torch.zeros((1,), dtype=torch.bool),
        )
        responses = [base_response, deepcopy(base_response)]

        shap_values, normalized = explainer._get_shap_values(
            model=dummy_model_instance,
            masks=masks,
            responses=responses,
            source_chat=source_chat,
            device=torch.device("cpu"),
            similarities=torch.tensor([1.0, 0.5]),
        )

        assert torch.allclose(shap_values[1:3], torch.tensor([0.9, 0.9]))
        assert torch.allclose(shap_values[3:], torch.tensor([0.7, 0.7]))
        assert torch.equal(shap_values, normalized)

    @patch("mllm_shap.shap.base.shap_explainer.generate_responses")
    def test_generate_step_updates_total_calls(
        self,
        mock_generate: MagicMock,
        dummy_chat_instance: BaseMllmChat,
        dummy_model_instance: DummyModel,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """_generate_step should propagate masks, responses, and total call count."""

        class SingleSplitExplainer(DummyShapExplainer):
            def __init__(self) -> None:
                super().__init__()
                self._splits = [
                    torch.tensor([[True, False, True]], dtype=torch.bool),
                    None,
                ]

            def _get_next_split(
                self,
                n: int,
                device: torch.device,
                generated_masks_num: int,
                existing_masks: list[Tensor] | None = None,
            ) -> Tensor | None:
                del n, device, generated_masks_num, existing_masks
                return self._splits.pop(0)

        explainer = SingleSplitExplainer()
        mask_manager = MasksManager(chat=dummy_chat_instance)
        device = dummy_chat_instance.torch_device
        masks = [mask_manager.get_initial_mask(device=device)]
        responses = [dummy_response_instance]

        def fake_generate(**kwargs):
            gen = kwargs["gen"]
            verbose = kwargs.get("verbose", False)
            history = [] if verbose else None
            count = 0
            for mask, mask_hash in gen:
                count += 1
                masks.append(mask)
                new_response = deepcopy(dummy_response_instance)
                responses.append(new_response)
                if history is not None:
                    history.append((mask, mask_hash, None, new_response))
            gen.generated_masks = count
            return 0, history

        mock_generate.side_effect = fake_generate

        chats_skipped, history = explainer._generate_step(
            mask_manager=mask_manager,
            device=device,
            masks=masks,
            responses=responses,
            source_chat=dummy_chat_instance,
            model=dummy_model_instance,
            cache_manager=MagicMock(),
            progress_bar=False,
            verbose=True,
        )

        assert chats_skipped == 0
        assert explainer.total_n_calls == 1
        assert history is not None and len(history) == 1
        assert torch.equal(history[0][0], masks[-1])

    def test_call_returns_none_when_not_verbose(
        self,
        explainer_instance: BaseShapExplainer,
        dummy_model_instance: DummyModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """__call__ should populate cache and return None when verbose=False."""

        mask_sequence = [torch.tensor([[True, False, True]], dtype=torch.bool), None]

        with (
            patch.object(
                explainer_instance, "_get_next_split", side_effect=mask_sequence
            ),
            patch.object(explainer_instance, "_get_num_splits", return_value=1),
        ):
            result = explainer_instance(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=False,
            )

        assert result is None
        assert isinstance(dummy_response_instance.chat.cache, ExplainerCache)
        assert explainer_instance.total_n_calls >= 1

    @patch("mllm_shap.shap.base.shap_explainer.CacheManager")
    @patch("mllm_shap.shap.base.shap_explainer.logger.info")
    def test_call_logs_cache_deduplication_info(
        self,
        mock_log_info: MagicMock,
        mock_cache_manager_cls: MagicMock,
        dummy_model_instance: DummyModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """__call__ should log deduplication stats when cache hits are extracted."""
        explainer = DummyShapExplainer()
        mock_cache_manager_cls.return_value.extracted_num = 1

        def _fake_generate_step(
            mask_manager: MasksManager,
            device: torch.device,
            masks: list[Tensor],
            responses: list[ModelResponse],
            **kwargs,
        ) -> tuple[
            int, list[tuple[Tensor, int, BaseMllmChat | None, ModelResponse]] | None
        ]:
            del kwargs
            masks.append(mask_manager.get_initial_mask(device=device))
            responses.append(deepcopy(dummy_response_instance))
            return 0, None

        with (
            patch.object(explainer, "_generate_step", side_effect=_fake_generate_step),
            patch.object(
                explainer,
                "_get_shap_values",
                return_value=(torch.zeros(3), torch.zeros(3)),
            ),
            patch.object(explainer, "_save_to_cache"),
        ):
            _ = explainer(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=False,
            )

        mock_log_info.assert_any_call(
            "Deduplicated %d/%d masks using existing cache.",
            1,
            1,
        )

    def test_call_auto_injects_observability_sink_when_enabled(
        self,
        dummy_model_instance: DummyModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Enabling observability should auto-create an in-memory sink."""
        explainer = DummyShapExplainer()
        mask_sequence = [torch.tensor([[True, False, True]], dtype=torch.bool), None]

        with (
            patch.object(explainer, "_get_next_split", side_effect=mask_sequence),
            patch.object(explainer, "_get_num_splits", return_value=1),
        ):
            _ = explainer(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=False,
                observability_enabled=True,
            )

        assert isinstance(explainer.last_observability_sink, InMemoryObservabilitySink)
        sink = explainer.last_observability_sink
        assert sink is not None
        assert len(sink.events) > 0
        assert any(event.name == "stage_start" for event in sink.events)
        assert any(event.name == "stage_end" for event in sink.events)

    def test_call_uses_user_observability_sink_and_run_id(
        self,
        dummy_model_instance: DummyModel,
        dummy_chat_instance: BaseMllmChat,
        dummy_response_instance: ModelResponse,
    ) -> None:
        """Provided sink and run id should propagate to pipeline observability events."""
        explainer = DummyShapExplainer()
        sink = InMemoryObservabilitySink()
        mask_sequence = [torch.tensor([[True, False, True]], dtype=torch.bool), None]

        with (
            patch.object(explainer, "_get_next_split", side_effect=mask_sequence),
            patch.object(explainer, "_get_num_splits", return_value=1),
        ):
            _ = explainer(
                model=dummy_model_instance,
                source_chat=dummy_chat_instance,
                response=dummy_response_instance,
                progress_bar=False,
                verbose=False,
                observability_sink=sink,
                observability_run_id="test-run-id",
            )

        assert explainer.last_observability_sink is sink
        assert len(sink.events) > 0
        assert all(event.run_id == "test-run-id" for event in sink.events)
        assert all(span.run_id == "test-run-id" for span in sink.spans)

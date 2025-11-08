"""Dummy classes for testing SHAP explainers."""

import torch
from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.model_response import ModelResponse
from mllm_shap.connectors.enums import ModalityFlag, Role
from mllm_shap.shap.base.explainer import BaseShapExplainer
from torch import Tensor


class DummyChat(BaseMllmChat):
    """Minimal concrete implementation of BaseMllmChat for testing."""

    def __init__(self, num_tokens: int = 3, shap_values_mask: Tensor | None = None):
        super().__init__(
            device=torch.device("cpu"),
            empty_turn_sequences=set(),
        )

        self._num_tokens = num_tokens

        if shap_values_mask is None:
            shap_values_mask = torch.ones(num_tokens, dtype=torch.bool)
        else:
            assert num_tokens == shap_values_mask.numel()
        self.shap_values_mask = shap_values_mask

    @property
    def input_tokens(self) -> Tensor:
        """Return dummy token indices (0..N-1)."""
        return torch.arange(self._num_tokens, device=self.torch_device)

    @property
    def tokens_modality_flag(self) -> Tensor:
        """All tokens are treated as TEXT for simplicity."""
        return torch.full(
            (self._num_tokens,),
            ModalityFlag.TEXT,
            dtype=torch.int8,
            device=self.torch_device,
        )

    @property
    def text_tokens(self) -> Tensor:
        """Return identical dummy tokens for text."""
        return torch.arange(self._num_tokens, device=self.torch_device)

    @property
    def audio_tokens(self) -> Tensor:
        """Return identical dummy tokens for audio."""
        return torch.arange(self._num_tokens, device=self.torch_device)

    @classmethod
    def _set_new_instance(
        cls,
        full_mask: Tensor,
        text_mask_relative: Tensor,
        audio_mask_relative: Tensor,
        chat: "BaseMllmChat",
    ):
        """Create a new DummyChat with number of tokens equal to remaining mask."""
        num_tokens = int(full_mask.sum().item())
        return cls(num_tokens=num_tokens)

    def _decode_text(self, text_tokens: Tensor) -> str:
        """Return decoded text as space-separated string."""
        return " ".join(str(t.item()) for t in text_tokens)

    def _decode_audio(self, audio_tokens: Tensor) -> Tensor | None:
        """Return dummy zero waveform."""
        return torch.zeros_like(audio_tokens, device=self.torch_device)

    def _add_text(self, text: str) -> int:
        """Simulate adding a single text token."""
        self._num_tokens += 1
        return 1

    def _add_audio(self, waveform: Tensor, sample_rate: int) -> int:
        """Simulate adding a single audio token."""
        self._num_tokens += 1
        return 1

    def _append(
        self,
        text: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
        history_tracking_mode,
    ) -> tuple[int, int]:
        """Simulate appending tokens (increment counters)."""
        n_text = text.numel()
        n_audio = audio_out.numel()
        self._num_tokens += n_text + n_audio
        return n_text, n_audio

    def _new_turn(self, speaker: Role) -> None:
        """No-op for testing."""

    def _end_turn(self) -> None:
        """No-op for testing."""

    def _get_tokens_sequences_to_exclude(self, phrases_to_exclude: set[str]) -> list[Tensor]:
        """No exclusions for dummy."""
        return []


class DummyModel(BaseMllmModel):
    """Dummy model class for testing."""

    def __init__(self, device: torch.device = torch.device("cpu"), **kwargs):
        self.config = None
        self.device = device
        self.processor = None
        self.model = None

    def get_new_chat(self) -> BaseMllmChat:
        """Return a dummy chat instance."""
        return DummyChat()

    def generate(
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config=None,
        keep_history: bool = False,
    ) -> ModelResponse:
        """
        Return a dummy ModelResponse for testing.
        """
        dummy_tensor = torch.zeros((1, 2))
        return ModelResponse(
            chat=chat,
            generated_text_tokens=dummy_tensor,
            generated_audio_tokens=dummy_tensor,
            generated_modality_flag=torch.zeros((1, 2)),
        )

    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:
        """Return dummy static embeddings."""
        return [torch.zeros((1, 2)) for _ in responses]

    def _get_contextual_embeddings(self, static_embeddings: list[Tensor]) -> list[Tensor]:
        """Return dummy contextual embeddings."""
        return [torch.zeros_like(e) for e in static_embeddings]


class DummyShapExplainer(BaseShapExplainer):
    """Dummy SHAP explainer for testing purposes."""

    def _get_num_splits(self, target_length: int) -> int | None:
        return 3

    def _get_next_split(self, target_length: int, device: torch.device, generated_masks: int) -> Tensor | None:
        if generated_masks > self._get_num_splits(target_length):
            return None

        return torch.rand(target_length, device=device) > 0.5

    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        # return a simple 1D tensor of ones with length equal to number of tokens
        return torch.ones(masks.shape[1], device=device, dtype=torch.float32)

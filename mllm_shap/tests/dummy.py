"""Dummy classes for testing SHAP explainers."""

import torch
from torch import Tensor
from mllm_shap.shap.base.explainer import BaseShapExplainer
from mllm_shap.connectors.base.model import BaseMllmModel
from mllm_shap.connectors.base.chat import BaseMllmChat, Role
from mllm_shap.connectors.enums import ModalityFlag


class DummyChat(BaseMllmChat):
    """Dummy chat class for testing purposes."""

    def __init__(self, num_tokens: int = 3):
        super().__init__(
            added_vocab_tokens=set(),
            device=torch.device("cpu"),
            empty_turn_sequences=set(),
        )
        self._num_tokens = num_tokens
        # All tokens are initially explainable
        self.shap_values_mask = torch.ones(self._num_tokens, dtype=torch.bool, device=self.torch_device)

    @property
    def input_tokens(self) -> Tensor:
        # Return tokens as a range from 0 to num_tokens-1
        return torch.arange(self._num_tokens, device=self.torch_device)

    @property
    def tokens_modality_flag(self) -> Tensor:
        return torch.full((self._num_tokens,), ModalityFlag.TEXT, dtype=torch.int8, device=self.torch_device)

    @property
    def text_tokens(self) -> Tensor:
        return torch.arange(self._num_tokens, device=self.torch_device)

    @property
    def audio_tokens(self) -> Tensor:
        return torch.arange(self._num_tokens, device=self.torch_device)

    @classmethod
    def _set_new_instance(
        cls,
        full_mask: Tensor,
        text_mask_relative: Tensor,
        audio_mask_relative: Tensor,
        chat: "BaseMllmChat",
    ):
        # Return a new DummyChat with the same number of tokens as the mask
        return cls(num_tokens=full_mask.numel())

    def _decode_text(self, text_tokens: Tensor) -> str:
        return " ".join([str(t.item()) for t in text_tokens])

    def _decode_audio(self, audio_tokens: Tensor) -> Tensor | None:
        return torch.zeros_like(audio_tokens, device=self.torch_device)

    def _add_text(self, text: str) -> int:
        self._num_tokens += 1
        return self._num_tokens - 1

    def _add_audio(self, waveform: Tensor, sample_rate: int) -> int:
        self._num_tokens += 1
        return self._num_tokens - 1

    def _append(
        self,
        text: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
        history_tracking_mode,
    ) -> tuple[int, int]:
        # Append tokens (increment counters)
        old_num_tokens = self._num_tokens
        self._num_tokens += text.numel() + audio_out.numel()
        return old_num_tokens, self._num_tokens

    def _new_turn(self, speaker: Role) -> None:
        pass

    def _end_turn(self) -> None:
        pass

    def _get_tokens_sequences_to_exclude(self, phrases_to_exclude: set[str]) -> list[Tensor]:
        return []


class DummyModel(BaseMllmModel):
    """Dummy model class for testing"""

    def __init__(self):
        # provide dummy attributes without calling BaseMllmModel.__init__
        self.config = None
        self.device = torch.device("cpu")
        self.processor = None
        self.model = None
        self.history_tracking_mode = None

    def _get_contextual_embeddings(self, static_embeddings: Tensor) -> Tensor:
        return torch.zeros((1, 2))

    def get_new_chat(self) -> BaseMllmChat:
        return DummyChat()

    def get_static_embeddings(self, chat: BaseMllmChat) -> Tensor:
        return torch.zeros((1, 2))

    def generate(self, chat: BaseMllmChat, **kwargs):
        return chat, chat


class DummyShapExplainer(BaseShapExplainer):
    """Dummy SHAP explainer for testing purposes."""

    def _generate_masks(self, n: int, device: torch.device, existing_masks: Tensor | None = None) -> Tensor:
        # generate n simple masks: identity matrix of size n
        return torch.eye(n, dtype=torch.bool, device=device)

    def _calculate_shap_values(self, masks: Tensor, similarities: Tensor, device: torch.device) -> Tensor:
        # return a simple 1D tensor of ones with length equal to number of tokens
        return torch.ones(masks.shape[1], device=device, dtype=torch.float32)

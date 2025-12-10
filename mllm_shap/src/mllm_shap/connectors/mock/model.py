"""Mock model connector for debugging."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import logging
import os

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..base.chat import BaseMllmChat
from ..base.model import BaseMllmModel
from ..base.model_response import ModelResponse
from ..config import ModelConfig
from ..enums import ModelHistoryTrackingMode, Role, ModalityFlag
from .chat import MockChat
from .config import CONFIG


class Mock(BaseMllmModel):
    """
    Mock model connector for debugging.

    Returns placeholder tokens of size equal to max_new_tokens without performing
    actual model inference. Useful for testing and debugging the SHAP pipeline.
    """

    processor: PreTrainedTokenizerBase
    _KW_HISTORY_TRACKING_MODE = "history_tracking_mode"
    _TOKEN_EMB_RANK = 2
    _PLACEHOLDER_TOKEN_ID = 0  # Use token ID 0 as placeholder

    def __init__(self, device: torch.device, **kwargs: Any) -> None:
        # configure logger and debug toggle
        self._logger = logging.getLogger(__name__)
        self._debug_memory = bool(os.getenv("MLLM_SHAP_DEBUG_MEMORY"))

        # Load a simple tokenizer for text encoding/decoding
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            "gpt2",
            trust_remote_code=True,
        )  # nosec: B615

        # Create a dummy model object for interface compatibility
        dummy_model = None

        # Force text-only history tracking
        if self._KW_HISTORY_TRACKING_MODE in kwargs and \
           kwargs[self._KW_HISTORY_TRACKING_MODE] != ModelHistoryTrackingMode.TEXT:
            warnings.warn(
                "Non-TEXT history tracking requested but this connector is text-only. Forcing TEXT mode.",
                stacklevel=2,
            )
            kwargs[self._KW_HISTORY_TRACKING_MODE] = ModelHistoryTrackingMode.TEXT

        super().__init__(
            config=CONFIG,
            device=device,
            processor=tokenizer,
            model=dummy_model,
            history_tracking_mode=kwargs.pop(self._KW_HISTORY_TRACKING_MODE, ModelHistoryTrackingMode.TEXT),
        )

    def get_new_chat(self, **kwargs: Any) -> MockChat:
        kwargs = dict(kwargs or {})
        kwargs.pop("device", None)
        kwargs["tokenizer"] = self.processor
        return MockChat(device=self.device, **kwargs)

    # pylint: disable=too-many-locals
    def generate(
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        super().generate(chat=chat, max_new_tokens=max_new_tokens, model_config=model_config, keep_history=keep_history)

        chat = deepcopy(chat)

        chat.new_turn(Role.ASSISTANT)

        # Generate placeholder tokens with size equal to max_new_tokens
        # create generated tokens on CPU to avoid holding GPU memory in responses/cache
        generated = torch.full(
            (max_new_tokens,),
            self._PLACEHOLDER_TOKEN_ID,
            dtype=torch.long,
            device=torch.device("cpu"),
        )

        # All generated tokens are TEXT
        modality_flag = torch.full(
            (generated.shape[0],),
            ModalityFlag.TEXT,
            dtype=torch.long,
            device=torch.device("cpu"),
        )

        # History update
        if keep_history:
            # For API parity with other connectors: pass [1, T] tensors
            text_tokens_2d = generated.unsqueeze(0)  # [1, seq_len] (on CPU)
            empty_audio = torch.empty((0, 0), dtype=torch.long, device=torch.device("cpu"))  # [0, 0]
            self._set_chat_history(chat, text_tokens_2d, empty_audio, modality_flag)

        return ModelResponse(
            chat=chat if keep_history else None,
            generated_text_tokens=generated,  # [seq_len]
            generated_audio_tokens=torch.empty((0, 0), dtype=torch.long, device=torch.device("cpu")),  # [0, 0]
            generated_modality_flag=modality_flag,  # [seq_len]
        )

    # -- embeddings API --

    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:
        super().get_static_embeddings(responses=responses)

        # Return dummy embeddings (all zeros with shape [T, embedding_dim])
        # Using a small embedding dimension for testing
        embedding_dim = 768
        static_embeddings: list[Tensor] = []

        for response in responses:
            num_tokens = response.generated_text_tokens.shape[0]
            # Create zero embeddings for each token
            emb = torch.zeros(
                (num_tokens, embedding_dim),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            static_embeddings.append(emb)

        return static_embeddings

    def _get_contextual_embeddings(self, static_embeddings: list[Tensor]) -> list[Tensor]:
        # For mock, contextual embeddings are same as static (no model context)
        contextual = [emb.clone() for emb in static_embeddings]
        return contextual

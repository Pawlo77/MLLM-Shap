"""Base model connector class."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, cast

import torch
from torch import Tensor

from ...utils.logger import get_logger
from ...utils.other import raise_connector_error
from ..config import HuggingFaceModelConfig, ModelConfig
from ..enums import ModelHistoryTrackingMode
from ._validators import BaseModelConfig, BaseModelGenerateConfig
from .chat import BaseMllmChat
from .model_response import ModelResponse

logger: Logger = get_logger(__name__)


# pylint: disable=duplicate-code
class BaseMllmModel(ABC):
    """Base class for model connectors."""

    config: HuggingFaceModelConfig
    """The model configuration."""
    device: torch.device
    """The device to run the model on."""

    processor: Any
    """The model processor (tokenizer)."""
    model: Any
    """The model instance."""

    history_tracking_mode: ModelHistoryTrackingMode
    """The mode for tracking chat history."""

    # pylint: disable=too-many-positional-arguments,too-many-arguments
    def __init__(
        self,
        config: HuggingFaceModelConfig,
        device: torch.device,
        processor: Any,
        model: Any,
        history_tracking_mode: ModelHistoryTrackingMode = ModelHistoryTrackingMode.TEXT,
    ) -> None:
        """
        Initialize the model connector.

        Args:
            config: The model configuration.
            device: The device to run the model on.
            processor: The model processor.
            model: The model instance.
            history_tracking_mode: The mode for tracking chat history.
        """
        # validation
        __config = BaseModelConfig(
            config=config,
            device=device,
            processor=processor,
            model=model,
            history_tracking_mode=history_tracking_mode,
        )

        self.device = __config.device
        self.config = __config.config
        self.processor = __config.processor
        self.model = __config.model
        self.history_tracking_mode = __config.history_tracking_mode

    @abstractmethod
    def get_new_chat(self) -> BaseMllmChat:
        """Get a new chat state for the model."""

    @abstractmethod
    def generate(  # type: ignore[return]
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        """
        Generate audio based on the current chat state.

        Args:
            chat: The current chat state.
            max_new_tokens: The maximum number of new tokens to generate (default is 20).
            model_config: Additional model configuration parameters.
            keep_history: Whether to return chat state with full history or only generated content.
        Returns:
            ModelResponse: The updated chat state after generation.
        """
        logger.debug("Generating audio with max_new_tokens=%d, keep_history=%s", max_new_tokens, keep_history)
        # validation
        _ = BaseModelGenerateConfig(
            max_new_tokens=max_new_tokens,
            model_config_=model_config,
            keep_history=keep_history,
        )

    @abstractmethod
    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:  # type: ignore[return]
        """
        Get static embeddings for the current chat state.

        Args:
            responses: The model responses to get embeddings for.
        Returns:
            The static embeddings for the text and audio tokens.
        Raises:
            ValueError: If responses is not a list of ModelResponse.
        """
        logger.debug("Getting static embeddings.")
        if not isinstance(responses, list) or not all(isinstance(r, ModelResponse) for r in responses):
            raise ValueError(f"responses must be a list of ModelResponse, got {type(responses)}")

    def get_contextual_embeddings(
        self, *args: Any, static_embeddings: list[Tensor] | None = None, **kwargs: Any
    ) -> list[Tensor]:
        """
        Get contextual embeddings for the current chat state.

        Args:
            static_embeddings: Precomputed static embeddings (if any).
            *args: Additional positional arguments for :func:`get_static_embeddings`.
                Used if static_embeddings is None.
            **kwargs: Additional keyword arguments for :func:`get_static_embeddings`.
                Used if static_embeddings is None.
        Returns:
            The context embeddings for the text and audio tokens, same format as in
                :func:`get_static_embeddings`.
        Raises:
            ValueError: If static_embeddings is not an instance of Tensor.
        """
        logger.debug("Getting contextual embeddings.")
        if static_embeddings is None:
            static_embeddings = self.get_static_embeddings(*args, **kwargs)
        if not isinstance(static_embeddings, list):
            raise ValueError(f"static_embeddings must be an instance of list, got {type(static_embeddings)}")
        for emb in static_embeddings:
            if not isinstance(emb, Tensor):
                raise ValueError(f"Each item in static_embeddings must be an instance of Tensor, got {type(emb)}")
        with torch.no_grad():
            return cast(list[Tensor], raise_connector_error(self._get_contextual_embeddings, static_embeddings))

    @abstractmethod
    def _get_contextual_embeddings(self, static_embeddings: list[Tensor]) -> list[Tensor]:
        """
        Get contextual embeddings for the current chat state.

        Args:
            static_embeddings: Precomputed static embeddings.
        Returns:
            The contextual embeddings for the text and audio tokens.
        """

    def _set_chat_history(
        self,
        chat: BaseMllmChat,
        text_tokens: Tensor,
        audio_tokens: Tensor,
        modality_flag: Tensor,
    ) -> None:
        """
        Set the chat history with provided text and audio tokens.

        Args:
            chat: The chat instance to update.
            text_tokens: The text tokens to set.
            audio_tokens: The audio tokens to set.
            modality_flag: The modality flags corresponding to the tokens.
        """
        logger.debug(
            "Setting chat history with text tokens (%d), audio tokens (%d), modality flags (%d).",
            text_tokens.shape[0],
            audio_tokens.shape[0],
            modality_flag.shape[0],
        )
        chat.append(
            text=text_tokens,
            audio_out=audio_tokens,
            modality_flag=modality_flag,
            history_tracking_mode=self.history_tracking_mode,
        )
        chat.end_turn()

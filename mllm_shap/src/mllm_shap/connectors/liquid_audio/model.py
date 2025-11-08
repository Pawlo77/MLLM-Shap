"""LiquidAudio model connector."""

import warnings
from copy import deepcopy
from typing import Any, cast

import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality
from torch import Tensor

from ..base.chat import BaseMllmChat
from ..base.model import BaseMllmModel
from ..config import ModelConfig
from ..enums import ModelHistoryTrackingMode, Role
from .chat import LiquidAudioChat
from .config import CONFIG
from ..base.model_response import ModelResponse

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


class _PatchedLFM2AudioProcessor(LFM2AudioProcessor):
    """Patched LFM2AudioProcessor to handle device management."""

    __device: str | None

    @property  # type: ignore[override]
    def device(self) -> str:
        """
        Get the device.

        Returns:
            The device.
        Raises:
            ValueError: If device is not set.
        """
        if self.__device is None:
            raise ValueError("Device not set. Please set the device before using the processor.")
        return self.__device

    @device.setter
    def device(self, value: str) -> None:
        """Set the device."""
        self.__device = value


class LiquidAudio(BaseMllmModel):
    """
    Connector for LiquidAudio model.

    Fields:
        processor (LFM2AudioProcessor): The audio processor.
        model (LFM2AudioModel): The LiquidAudio model.
    """

    processor: LFM2AudioProcessor
    model: LFM2AudioModel

    def __init__(self, device: torch.device, *args: Any, **kwargs: Any) -> None:
        _kw: dict[str, Any] = {
            "repo_id": CONFIG.repo_id,
            "revision": CONFIG.revision,
            "device": device,
        }

        if "config" in kwargs or "model" in kwargs or "processor" in kwargs:  # pylint: disable=magic-value-comparison
            raise ValueError(
                "Please do not provide 'config', 'model' or 'processor' arguments. They are set automatically."
            )

        super().__init__(
            *args,
            config=CONFIG,
            device=device,
            processor=_PatchedLFM2AudioProcessor.from_pretrained(**_kw).eval(),
            model=LFM2AudioModel.from_pretrained(**_kw).eval(),
            **kwargs,
        )  # type: ignore[misc]

        # it is a patch to set device properly
        self.processor.device = str(device)  # type: ignore

    def get_new_chat(self, *args: Any, **kwargs: Any) -> LiquidAudioChat:
        kwargs = kwargs or {}
        kwargs["processor"] = self.processor

        return LiquidAudioChat(*args, device=self.device, **kwargs)  # type: ignore[misc]

    def generate(
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        super().generate(
            chat=chat,
            max_new_tokens=max_new_tokens,
            model_config=model_config,
            keep_history=keep_history,
        )

        # use copy of chat as it is immutable
        chat = deepcopy(chat)

        # Mark assistant reply
        chat.new_turn(Role.ASSISTANT)

        # Prepare chat containing only generated content
        new_chat = self.get_new_chat()
        new_chat.new_turn(Role.ASSISTANT)

        text_tokens: list[Tensor] = []
        audio_tokens: list[Tensor] = []
        modality_out: list[LFMModality] = []

        # if history tracking mode is text only, use generate_sequential to generate only text tokens
        gen_callable = (
            self.model.generate_interleaved
            if not self.history_tracking_mode == ModelHistoryTrackingMode.TEXT
            else self.model.generate_sequential
        )

        # generate audio and text interleaved
        for t in gen_callable(
            **cast(dict[str, Any], chat),
            max_new_tokens=max_new_tokens,
            text_temperature=model_config.text_temperature,
            text_top_k=model_config.text_top_k,
            audio_temperature=model_config.audio_temperature,
            audio_top_k=model_config.audio_top_k,
        ):
            # text tokens
            if t.numel() == 1:
                text_tokens.append(t)
                modality_out.append(LFMModality.TEXT)
            # audio tokens
            else:
                audio_tokens.append(t)
                modality_out.append(LFMModality.AUDIO_OUT)

        if len(text_tokens) > 0:
            text_tokens_tensor = torch.stack(text_tokens, 1)
        else:
            text_tokens_tensor = torch.empty((1, 0), dtype=torch.long, device=self.device)
        del text_tokens

        if len(audio_tokens) > 0:
            audio_tokens_tensor = torch.stack(audio_tokens, 1)
        else:
            audio_tokens_tensor = torch.empty(
                (cast(ChatState, chat).codebooks, 0),
                dtype=torch.long,
                device=self.device,
            )
        del audio_tokens

        modality_flag = torch.tensor(modality_out, device=self.device)

        if keep_history:
            self._set_chat_history(chat, text_tokens_tensor, audio_tokens_tensor, modality_flag)
        return ModelResponse(
            chat=chat if keep_history else None,
            generated_text_tokens=text_tokens_tensor.squeeze(0),  # shape: [seq_len]
            generated_audio_tokens=audio_tokens_tensor.T,  # shape: [seq_len, codebooks]
            generated_modality_flag=modality_flag,  # shape: [seq_len]
        )

    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:
        super().get_static_embeddings(responses=responses)

        static_embeddings: list[Tensor] = []
        for response in responses:
            chat = self.get_new_chat()
            chat.new_turn(Role.ASSISTANT)
            self._set_chat_history(
                chat,
                response.generated_text_tokens.unsqueeze(0),
                response.generated_audio_tokens.T,
                response.generated_modality_flag,
            )
            # pylint: disable=protected-access # type: ignore[arg-type]
            static_embeddings.append(self.model._prefill(**cast(dict[str, Any], chat)).squeeze(0))

        return static_embeddings

    def _get_contextual_embeddings(self, static_embeddings: list[Tensor]) -> list[Tensor]:
        contextual_embeddings = []

        for emb in static_embeddings:
            if len(emb.shape) == 2:  # pylint: disable=magic-value-comparison
                emb = emb.unsqueeze(0)
            # Last hidden states: [seq_len, hidden_dim]
            contextual_embeddings.append(
                cast(
                    Tensor,
                    self.model.lfm(
                        inputs_embeds=emb,
                        past_key_values=None,
                        use_cache=False,
                    ).last_hidden_state.squeeze(0),
                )
            )

        return contextual_embeddings

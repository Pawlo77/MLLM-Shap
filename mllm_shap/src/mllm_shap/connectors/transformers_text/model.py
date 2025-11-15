"""Transformers text-only model connector."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, cast

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from ..base.chat import BaseMllmChat
from ..base.model import BaseMllmModel
from ..base.model_response import ModelResponse
from ..config import ModelConfig
from ..enums import ModelHistoryTrackingMode, Role, ModalityFlag
from .chat import TransformersTextChat
from .config import CONFIG


class TransformersCausalText(BaseMllmModel):
    """
    Connector for classic Hugging Face causal LMs (text-only).

    Fields:
        processor (PreTrainedTokenizerBase): the tokenizer
        model (PreTrainedModel): the causal LM (AutoModelForCausalLM)
    """

    processor: PreTrainedTokenizerBase
    model: PreTrainedModel
    _KW_HISTORY_TRACKING_MODE = "history_tracking_mode"
    _TOKEN_EMB_RANK = 2

    def __init__(self, device: torch.device, **kwargs: Any) -> None:
        # Disallow overriding these to keep parity with LiquidAudio pattern.
        forbidden = {"config", "model", "processor"}
        if any(k in kwargs for k in forbidden):
            raise ValueError("Do not pass 'config', 'model', or 'processor'—they are set automatically.")

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            CONFIG.repo_id,
            revision=CONFIG.revision,
        )  # nosec: B615 - pinned to immutable commit
        _model = AutoModelForCausalLM.from_pretrained(
            CONFIG.repo_id,
            revision=CONFIG.revision,
            load_in_4bit=True,
        )  # nosec: B615
        model = cast(PreTrainedModel, _model)
        model.to(device)  # type: ignore[arg-type]
        model.eval()  # type: ignore[no-untyped-call]

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
            model=model,
            history_tracking_mode=kwargs.pop(self._KW_HISTORY_TRACKING_MODE, ModelHistoryTrackingMode.TEXT),
        )

        # Set reasonable defaults for EOS if missing
        if (
            getattr(self.processor, "pad_token_id", None) is None
            and getattr(self.processor, "eos_token_id", None) is not None
            and getattr(self.processor, "eos_token", None) is not None
        ):
            # guarded, no broad-except
            self.processor.pad_token = self.processor.eos_token

        gen_cfg = self.model.generation_config
        if not isinstance(gen_cfg, GenerationConfig):
            gen_cfg = GenerationConfig()  # type: ignore[no-untyped-call]
            setattr(self.model, "generation_config", gen_cfg)
        if getattr(gen_cfg, "pad_token_id", None) is None and self.processor.pad_token_id is not None:
            gen_cfg.pad_token_id = self.processor.pad_token_id
        if getattr(gen_cfg, "eos_token_id", None) is None and self.processor.eos_token_id is not None:
            gen_cfg.eos_token_id = self.processor.eos_token_id

    def get_new_chat(self, **kwargs: Any) -> TransformersTextChat:
        kwargs = dict(kwargs or {})
        kwargs.pop("device", None)
        kwargs["tokenizer"] = self.processor
        return TransformersTextChat(device=self.device, **kwargs)

    # pylint: disable=too-many-locals
    def generate(
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        super().generate(chat=chat, max_new_tokens=max_new_tokens, model_config=model_config, keep_history=keep_history)

        # Enforce text-only semantics and surface warnings for audio knobs
        if model_config.audio_temperature is not None or model_config.audio_top_k is not None:
            warnings.warn(
                "Audio generation parameters were provided but this connector is text-only; \
                    audio settings are ignored.",
                stacklevel=2,
            )

        # Copy chat (immutable input contract), mark assistant reply turn
        chat = deepcopy(chat)
        chat.new_turn(Role.ASSISTANT)

        # Build input ids from chat history (pure text)
        input_ids = chat.text_tokens.unsqueeze(0)  # [1, prompt_len]
        input_ids = input_ids.to(dtype=torch.long, device=self.device)
        prompt_len = int(input_ids.shape[1])

        # Explicit attention mask avoids warning about attention mask not set
        # and is correct for unpadded 1xT inputs
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        do_sample = model_config.text_temperature is not None and model_config.text_temperature > 0.0
        temperature: float | None = (
            float(model_config.text_temperature) if do_sample and model_config.text_temperature is not None else None
        )
        top_k: int | None = (
            int(model_config.text_top_k) if do_sample and model_config.text_top_k is not None else None
        )
        gen_out = self.model.generate(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            return_dict_in_generate=True,
            pad_token_id=self.processor.pad_token_id,
            eos_token_id=self.processor.eos_token_id,
        )

        sequences: Tensor = gen_out.sequences  # [1, prompt_len + seq_len]
        generated = sequences[0, prompt_len:] if sequences.shape[1] > prompt_len else sequences.new_empty((0,))
        generated = generated.to(dtype=torch.long, device=self.device)  # [seq_len]

        # All generated tokens are TEXT
        modality_flag = torch.full((generated.shape[0],), ModalityFlag.TEXT, dtype=torch.long, device=self.device)

        # History update
        if keep_history:
            # For API parity with LiquidAudio: pass [1, T] tensors
            text_tokens_2d = generated.unsqueeze(0)  # [1, seq_len]
            empty_audio = torch.empty((0, 0), dtype=torch.long, device=self.device)  # [0, 0]
            self._set_chat_history(chat, text_tokens_2d, empty_audio, modality_flag)

        return ModelResponse(
            chat=chat if keep_history else None,
            generated_text_tokens=generated,                   # [seq_len]
            generated_audio_tokens=torch.empty((0, 0), dtype=torch.long, device=self.device),  # [0, 0]
            generated_modality_flag=modality_flag,             # [seq_len]
        )

    # -- embeddings API --

    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:
        super().get_static_embeddings(responses=responses)

        emb_layer = self.model.get_input_embeddings()  # standard HF API
        static_embeddings: list[Tensor] = []

        for response in responses:
            ids = response.generated_text_tokens.to(device=self.device, dtype=torch.long).unsqueeze(0)  # [1, T]
            # Shape: [1, T, hidden]
            emb = emb_layer(ids)
            static_embeddings.append(emb.squeeze(0))  # [T, hidden]

        return static_embeddings

    def _get_contextual_embeddings(self, static_embeddings: list[Tensor]) -> list[Tensor]:
        contextual: list[Tensor] = []
        for emb in static_embeddings:
            if emb.dim() == self._TOKEN_EMB_RANK:
                emb = emb.unsqueeze(0)  # [1, T, hidden]
            # Call base model to obtain last_hidden_state (see HF outputs contract)
            base = getattr(self.model, "base_model", self.model)
            outputs = base(inputs_embeds=emb, use_cache=False)
            # Shape: [1, T, hidden]
            contextual.append(outputs.last_hidden_state.squeeze(0))
        return contextual

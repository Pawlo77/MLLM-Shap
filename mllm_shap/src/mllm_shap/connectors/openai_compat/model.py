"""Text-only causal LM via OpenAI-compatible HTTP API + local HF embeddings."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Callable, Mapping, cast

import httpx
import torch
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
)

from ..base.chat import BaseMllmChat
from ..base.model import BaseMllmModel
from ..base.model_response import ModelResponse
from ..config import HuggingFaceModelConfig, ModelConfig
from ..enums import ModalityFlag, ModelHistoryTrackingMode, Role
from ..transformers_text.chat import TransformersTextChat
from ..transformers_text.config import CONFIG as DEFAULT_HF_CONFIG


ChatCompletionsTransport = Callable[
    [str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]
]
"""POST body to parsed JSON dict (OpenAI-style chat.completions response)."""


def _default_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    with httpx.Client(timeout=600.0) as client:
        r = client.post(url, headers=dict(headers), json=dict(payload))
        r.raise_for_status()
        return cast(Mapping[str, Any], r.json())


class OpenAICompatCausalText(BaseMllmModel):
    """
    Generation uses an OpenAI-compatible ``/chat/completions`` endpoint (LM Studio).

    Static/contextual embeddings use a **local** Hugging Face causal LM (same tokenizer
    vocabulary as the served model) so SHAP similarity stages keep per-token vectors.
    """

    processor: Any
    model: PreTrainedModel

    def __init__(
        self,
        device: torch.device,
        
        base_url: str | None = None,
        chat_model: str | None = None,
        api_key: str | None = None,
        hf_repo_id: str | None = None,
        hf_revision: str | None = None,
        history_tracking_mode: ModelHistoryTrackingMode = ModelHistoryTrackingMode.TEXT,
        transport: ChatCompletionsTransport | None = None,
        **kwargs: Any,
    ) -> None:
        forbidden = {"config", "model", "processor"}
        overlap = forbidden.intersection(kwargs)
        if overlap:
            raise ValueError(f"Do not pass reserved keys: {sorted(overlap)}")

        hf_cfg = HuggingFaceModelConfig(
            repo_id=hf_repo_id or DEFAULT_HF_CONFIG.repo_id,
            revision=hf_revision or DEFAULT_HF_CONFIG.revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            hf_cfg.repo_id, revision=hf_cfg.revision
        )
        _lm = AutoModelForCausalLM.from_pretrained(
            hf_cfg.repo_id, revision=hf_cfg.revision
        )
        lm = cast(PreTrainedModel, _lm)
        cast(Any, lm).to(device)
        cast(Any, lm).eval()

        super().__init__(
            config=hf_cfg,
            device=device,
            processor=tokenizer,
            model=lm,
            history_tracking_mode=history_tracking_mode,
        )

        if (
            getattr(self.processor, "pad_token_id", None) is None
            and getattr(self.processor, "eos_token_id", None) is not None
        ):
            self.processor.pad_token = self.processor.eos_token

        gen_cfg = self.model.generation_config
        if not isinstance(gen_cfg, GenerationConfig):
            gen_cfg = cast(Any, GenerationConfig)()
            setattr(self.model, "generation_config", gen_cfg)
        if (
            getattr(gen_cfg, "pad_token_id", None) is None
            and self.processor.pad_token_id is not None
        ):
            gen_cfg.pad_token_id = self.processor.pad_token_id
        if (
            getattr(gen_cfg, "eos_token_id", None) is None
            and self.processor.eos_token_id is not None
        ):
            gen_cfg.eos_token_id = self.processor.eos_token_id

        root = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:1234/v1"
        ).rstrip("/")
        self._chat_url = f"{root}/chat/completions"
        self._chat_model = (
            chat_model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("LMSTUDIO_MODEL")
            or hf_cfg.repo_id
        )
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LMSTUDIO_API_KEY")
        )
        self._transport = transport or _default_transport

    def get_new_chat(self, **kwargs: Any) -> TransformersTextChat:
        kwargs = dict(kwargs or {})
        kwargs.pop("device", None)
        kwargs["tokenizer"] = self.processor
        return TransformersTextChat(device=self.device, **kwargs)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _prompt_from_chat(self, chat: BaseMllmChat) -> str:
        ids = chat.text_tokens.detach().to("cpu").reshape(-1).tolist()
        decoded = self.processor.decode(
            [int(x) for x in ids], skip_special_tokens=False
        )
        if isinstance(decoded, list):
            return "".join(decoded)
        return str(decoded)

    def generate(
        self,
        chat: BaseMllmChat,
        max_new_tokens: int = 128,
        model_config: ModelConfig = ModelConfig(),
        keep_history: bool = False,
    ) -> ModelResponse:
        model_config = model_config.model_copy(deep=True)
        super().generate(
            chat=chat,
            max_new_tokens=max_new_tokens,
            model_config=model_config,
            keep_history=keep_history,
        )

        chat = deepcopy(chat)
        chat.new_turn(Role.ASSISTANT)

        prompt = self._prompt_from_chat(chat)
        do_sample = (
            model_config.text_temperature is not None
            and model_config.text_temperature > 0.0
        )
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_new_tokens),
        }
        if do_sample:
            payload["temperature"] = float(model_config.text_temperature)
            if model_config.text_top_k is not None:
                # OpenAI uses top_p; approximate top_k via omitting if unset
                payload["top_p"] = 0.95
        else:
            payload["temperature"] = 0.0

        data = self._transport(self._chat_url, self._headers(), payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected chat/completions payload: {json.dumps(data)[:800]}"
            ) from e

        ids = self.processor.encode(str(content), add_special_tokens=False)
        if len(ids) == 0:
            generated = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            generated = torch.tensor(ids, dtype=torch.long, device=self.device)

        modality_flag = torch.full(
            (generated.shape[0],),
            ModalityFlag.TEXT,
            dtype=torch.long,
            device=self.device,
        )

        if keep_history:
            text_tokens_2d = generated.unsqueeze(0)
            empty_audio = torch.empty((0, 0), dtype=torch.long, device=self.device)
            self._set_chat_history(chat, text_tokens_2d, empty_audio, modality_flag)

        return ModelResponse(
            chat=chat if keep_history else None,
            generated_text_tokens=generated,
            generated_audio_tokens=torch.empty(
                (0, 0), dtype=torch.long, device=self.device
            ),
            generated_modality_flag=modality_flag,
        )

    def get_static_embeddings(self, responses: list[ModelResponse]) -> list[Tensor]:
        super().get_static_embeddings(responses=responses)
        emb_layer = self.model.get_input_embeddings()
        static_embeddings: list[Tensor] = []
        for response in responses:
            ids = response.generated_text_tokens.to(
                device=self.device, dtype=torch.long
            ).unsqueeze(0)
            emb = emb_layer(ids)
            static_embeddings.append(emb.squeeze(0))
        return static_embeddings

    def _get_contextual_embeddings(
        self, static_embeddings: list[Tensor]
    ) -> list[Tensor]:
        contextual: list[Tensor] = []
        for emb in static_embeddings:
            if emb.dim() == 2:
                emb = emb.unsqueeze(0)
            base = getattr(self.model, "base_model", self.model)
            outputs = base(inputs_embeds=emb, use_cache=False)
            contextual.append(outputs.last_hidden_state.squeeze(0))
        return contextual

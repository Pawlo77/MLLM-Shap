"""Tests for the LiquidAudio model connector."""

from enum import IntEnum
from functools import cached_property
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from mllm_shap.connectors.base.chat import BaseMllmChat
from mllm_shap.connectors.enums import ModelHistoryTrackingMode, Role
from mllm_shap.connectors.liquid import model as liquid_model
from mllm_shap.connectors.base.model_response import ModelResponse


class FakeLFMModality(IntEnum):
    """Minimal replacement for liquid_audio.LFMModality."""

    TEXT = 0
    AUDIO_OUT = 1
    AUDIO_IN = 2


@pytest.fixture
def stubbed_liquid_audio(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch liquid_audio dependencies with lightweight stubs for testing."""

    created_chats: list["ChatStub"] = []

    class ProcessorStub:
        last_from_pretrained_kwargs: dict[str, Any] | None = None

        def __init__(self) -> None:
            self.eval_called = False

        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> "ProcessorStub":
            instance = cls()
            cls.last_from_pretrained_kwargs = kwargs
            ProcessorStub.last_from_pretrained_kwargs = kwargs
            return instance

        def eval(self) -> "ProcessorStub":
            self.eval_called = True
            return self

    class ModelStub:
        last_from_pretrained_kwargs: dict[str, Any] | None = None

        def __init__(self) -> None:
            self.eval_called = False
            self.generate_sequential_calls: list[dict[str, Any]] = []
            self.generate_interleaved_calls: list[dict[str, Any]] = []
            self.prefill_calls: list[dict[str, Any]] = []
            self.lfm_calls: list[tuple[Tensor, Any, Any]] = []
            self.sequential_output: list[Tensor] = []
            self.interleaved_output: list[Tensor] = []
            self.prefill_output: Tensor = torch.zeros((1, 1, 1))

        @classmethod
        def from_pretrained(cls, **kwargs: Any) -> "ModelStub":
            instance = cls()
            cls.last_from_pretrained_kwargs = kwargs
            ModelStub.last_from_pretrained_kwargs = kwargs
            return instance

        def eval(self) -> "ModelStub":
            self.eval_called = True
            return self

        def generate_sequential(self, *args: Any, **kwargs: Any):
            self.generate_sequential_calls.append(kwargs)
            yield from self.sequential_output

        def generate_interleaved(self, *args: Any, **kwargs: Any):
            self.generate_interleaved_calls.append(kwargs)
            yield from self.interleaved_output

        def _prefill(self, **kwargs: Any) -> Tensor:
            self.prefill_calls.append(kwargs)
            return self.prefill_output

        def lfm(
            self, inputs_embeds: Tensor, past_key_values: Any, use_cache: bool
        ) -> SimpleNamespace:
            self.lfm_calls.append((inputs_embeds, past_key_values, use_cache))
            return SimpleNamespace(last_hidden_state=inputs_embeds + 1)

    class ChatStub(BaseMllmChat, dict):
        """Lightweight BaseMllmChat substitute for tests."""

        def __init__(
            self,
            device: torch.device,
            processor: Any | None = None,
            codebooks: int = 2,
            **chat_kwargs: Any,
        ) -> None:
            dict.__init__(self)
            BaseMllmChat.__init__(
                self,
                device=device,
                empty_turn_sequences=set(),
                get_new_chat_callable=lambda: ChatStub(
                    device=device,
                    processor=processor,
                    codebooks=codebooks,
                    **chat_kwargs,
                ),
            )
            self.device = device
            self.processor = processor
            self.codebooks = codebooks
            self.new_turn_calls: list[Role] = []
            self.append_calls: list[
                tuple[Tensor, Tensor, Tensor, ModelHistoryTrackingMode]
            ] = []
            self.end_turn_calls = 0
            self.update(chat_kwargs)
            created_chats.append(self)

        def new_turn(self, role: Role) -> None:
            self.new_turn_calls.append(role)

        def append(
            self,
            text: Tensor,
            audio_out: Tensor,
            modality_flag: Tensor,
            history_tracking_mode: ModelHistoryTrackingMode,
        ) -> None:
            self.append_calls.append(
                (
                    text.clone(),
                    audio_out.clone(),
                    modality_flag.clone(),
                    history_tracking_mode,
                )
            )

        def end_turn(self) -> None:
            self.end_turn_calls += 1

        def __deepcopy__(self, memo: dict[int, Any]) -> "ChatStub":
            copied = ChatStub(
                device=self.torch_device,
                processor=self.processor,
                codebooks=self.codebooks,
                **self,
            )
            copied.new_turn_calls = self.new_turn_calls.copy()
            copied.append_calls = self.append_calls.copy()
            copied.end_turn_calls = self.end_turn_calls
            memo[id(self)] = copied
            return copied

        @cached_property
        def input_tokens(self) -> list[Tensor]:
            return []

        @cached_property
        def tokens_modality_flag(self) -> Tensor:
            return torch.zeros(0, dtype=torch.int8, device=self.torch_device)

        @cached_property
        def text_tokens(self) -> Tensor:
            return torch.zeros(0, dtype=torch.long, device=self.torch_device)

        @cached_property
        def audio_tokens(self) -> Tensor:
            return torch.zeros(0, dtype=torch.long, device=self.torch_device)

    monkeypatch.setattr(liquid_model, "LFM2AudioProcessor", ProcessorStub)
    monkeypatch.setattr(liquid_model, "LFM2AudioModel", ModelStub)
    monkeypatch.setattr(liquid_model, "LiquidAudioChat", ChatStub)
    monkeypatch.setattr(liquid_model, "LFMModality", FakeLFMModality)
    # Ensure the patched processor base is used by the subclass
    liquid_model._PatchedLFM2AudioProcessor.__bases__ = (ProcessorStub,)

    def factory(
        device: torch.device = torch.device("cpu"),
    ) -> "liquid_model.LiquidAudio":
        return liquid_model.LiquidAudio(device=device)

    return SimpleNamespace(
        factory=factory,
        processor_cls=ProcessorStub,
        model_cls=ModelStub,
        chat_cls=ChatStub,
        created_chats=created_chats,
    )


def test_patched_processor_device_property() -> None:
    """_PatchedLFM2AudioProcessor should guard device access until set."""
    processor = liquid_model._PatchedLFM2AudioProcessor.__new__(
        liquid_model._PatchedLFM2AudioProcessor
    )
    processor._PatchedLFM2AudioProcessor__device = None
    with pytest.raises(ValueError, match="Device not set"):
        _ = processor.device

    processor.device = "cuda:0"
    assert processor.device == "cuda:0"


def test_init_sets_components_and_device(stubbed_liquid_audio: SimpleNamespace) -> None:
    """LiquidAudio initialization should load pretrained artifacts and set device string."""
    model = stubbed_liquid_audio.factory(device=torch.device("cpu"))

    proc_kwargs = stubbed_liquid_audio.processor_cls.last_from_pretrained_kwargs
    mdl_kwargs = stubbed_liquid_audio.model_cls.last_from_pretrained_kwargs
    assert proc_kwargs == mdl_kwargs
    assert proc_kwargs["repo_id"] == liquid_model.CONFIG.repo_id
    assert proc_kwargs["revision"] == liquid_model.CONFIG.revision

    assert model.processor.eval_called is True
    assert model.model.eval_called is True
    assert model.processor.device == "cpu"


def test_init_rejects_explicit_components(
    stubbed_liquid_audio: SimpleNamespace,
) -> None:
    """Providing custom processor/model/config should be rejected."""
    with pytest.raises(
        ValueError, match="Please do not provide 'config', 'model' or 'processor'"
    ):
        liquid_model.LiquidAudio(device=torch.device("cpu"), processor=object())


def test_get_new_chat_passes_processor(stubbed_liquid_audio: SimpleNamespace) -> None:
    """get_new_chat should inject the shared processor into chat instances."""
    model = stubbed_liquid_audio.factory()
    chat = model.get_new_chat(custom_flag=True)

    assert isinstance(chat, stubbed_liquid_audio.chat_cls)
    assert chat.processor is model.processor
    assert chat.device == model.device


def test_generate_uses_sequential_for_text_mode(
    stubbed_liquid_audio: SimpleNamespace,
) -> None:
    """History tracking TEXT mode should call generate_sequential and skip history saving."""
    model = stubbed_liquid_audio.factory()
    model.model.sequential_output = [torch.tensor([11]), torch.tensor([22])]

    chat = stubbed_liquid_audio.chat_cls(device=model.device, processor=model.processor)
    response = model.generate(chat, max_new_tokens=2, keep_history=False)

    assert len(model.model.generate_sequential_calls) == 1
    assert model.model.generate_interleaved_calls == []
    assert response.chat is None
    assert torch.equal(response.generated_text_tokens, torch.tensor([11, 22]))
    assert response.generated_audio_tokens.shape == (0, chat.codebooks)
    assert torch.equal(
        response.generated_modality_flag,
        torch.tensor([FakeLFMModality.TEXT, FakeLFMModality.TEXT]),
    )


def test_generate_interleaved_keeps_history(
    stubbed_liquid_audio: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-TEXT history tracking should use interleaved generation and persist chat history."""
    monkeypatch.setattr(liquid_model, "deepcopy", lambda obj: obj)

    model = stubbed_liquid_audio.factory()
    model.history_tracking_mode = ModelHistoryTrackingMode.AUDIO
    model.model.interleaved_output = [
        torch.tensor([5]),
        torch.tensor([1, 2]),
    ]

    chat = stubbed_liquid_audio.chat_cls(device=model.device, processor=model.processor)
    response = model.generate(chat, max_new_tokens=3, keep_history=True)

    assert len(model.model.generate_interleaved_calls) == 1
    assert response.chat is chat
    assert chat.new_turn_calls[0] == Role.ASSISTANT
    assert chat.append_calls  # history stored
    assert chat.end_turn_calls == 1
    assert torch.equal(
        response.generated_modality_flag,
        torch.tensor([FakeLFMModality.TEXT, FakeLFMModality.AUDIO_OUT]),
    )
    assert response.generated_audio_tokens.shape == (1, 2)


def test_get_static_embeddings_prefills_responses(
    stubbed_liquid_audio: SimpleNamespace,
) -> None:
    """get_static_embeddings should rebuild chats, append history, and call _prefill."""
    model = stubbed_liquid_audio.factory()
    model.model.prefill_output = torch.ones((1, 3, 2))

    responses = [
        ModelResponse(
            chat=None,
            generated_text_tokens=torch.tensor([1, 2, 3]),
            generated_audio_tokens=torch.zeros((2, 0)),
            generated_modality_flag=torch.tensor([FakeLFMModality.TEXT] * 3),
        )
    ]

    embeddings = model.get_static_embeddings(responses)

    assert len(model.model.prefill_calls) == 1
    assert len(embeddings) == 1
    assert embeddings[0].shape == (3, 2)

    created_chat = stubbed_liquid_audio.created_chats[-1]
    assert created_chat.new_turn_calls[0] == Role.ASSISTANT
    assert created_chat.append_calls  # history appended prior to prefill


def test_get_contextual_embeddings_invokes_lfm(
    stubbed_liquid_audio: SimpleNamespace,
) -> None:
    """_get_contextual_embeddings should normalise tensor ranks and call model.lfm."""
    model = stubbed_liquid_audio.factory()
    two_dim = torch.zeros((2, 3))
    three_dim = torch.zeros((1, 2, 4))

    contextual = model._get_contextual_embeddings([two_dim, three_dim])

    assert len(model.model.lfm_calls) == 2
    first_call_tensor, _, _ = model.model.lfm_calls[0]
    assert first_call_tensor.shape == (1, 2, 3)
    assert contextual[0].shape == (2, 3)
    assert contextual[1].shape == (2, 4)

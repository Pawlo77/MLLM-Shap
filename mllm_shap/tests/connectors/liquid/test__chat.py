"""Tests for the LiquidAudio chat connector."""

from enum import IntEnum
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from mllm_shap.connectors.enums import ModelHistoryTrackingMode, Role
from mllm_shap.connectors.liquid import chat as liquid_chat


class FakeLFMModality(IntEnum):
    """Minimal replacement for liquid_audio.LFMModality."""

    TEXT = 0
    AUDIO_OUT = 1
    AUDIO_IN = 2


@pytest.fixture
def patched_liquid_audio_chat(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch liquid_audio ChatState and LFMModality with lightweight stubs."""

    class TextProcessorStub:
        def encode(self, phrase: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(char) % 97 for char in phrase]

        def decode(self, tokens: Tensor) -> str:
            return ",".join(str(int(v)) for v in tokens.view(-1))

    class MimiProcessorStub:
        def __init__(self) -> None:
            self.calls: list[Tensor] = []

        def decode(self, codes: Tensor) -> Tensor:
            self.calls.append(codes.clone())
            return codes.to(dtype=torch.float32)

    class ProcessorStub:
        def __init__(self) -> None:
            self.text = TextProcessorStub()
            self.mimi = MimiProcessorStub()

    class ChatStateStub:
        """Stand-in for liquid_audio.ChatState with deterministic tensor updates."""

        def __init__(
            self,
            proc: ProcessorStub | None = None,
            text: Tensor | None = None,
            audio_in: Tensor | None = None,
            audio_out: Tensor | None = None,
            modality_flag: Tensor | None = None,
            audio_in_lens: Tensor | None = None,
            codebooks: int = liquid_chat.LiquidAudioChat.AUDIO_OUT_SHAPE,
            **_: Any,
        ) -> None:
            device = torch.device("cpu")
            self.proc = proc if proc is not None else ProcessorStub()
            self.codebooks = codebooks

            self.text = (
                text
                if text is not None
                else torch.zeros((1, 1), dtype=torch.long, device=device)
            )
            if audio_in is None:
                audio_in = torch.zeros(
                    (liquid_chat.LiquidAudioChat.AUDIO_IN_SHAPE, 0),
                    dtype=torch.long,
                    device=device,
                )
            self.audio_in = audio_in
            if audio_out is None:
                audio_out = torch.zeros((codebooks, 0), dtype=torch.long, device=device)
            self.audio_out = audio_out
            if modality_flag is None:
                modality_flag = torch.full(
                    (1, self.text.shape[1]),
                    FakeLFMModality.TEXT,
                    dtype=torch.long,
                    device=device,
                )
            self.modality_flag = modality_flag
            self.audio_in_lens = (
                audio_in_lens
                if audio_in_lens is not None
                else torch.zeros((0,), dtype=torch.long, device=device)
            )

            self.new_turn_log: list[str] = []
            self.end_turn_count = 0
            self.append_calls: list[tuple[Tensor, Tensor, Tensor]] = []

        # chat state API -------------------------------------------------
        def add_text(self, text: str) -> None:
            tokens = torch.tensor(
                self.proc.text.encode(text, add_special_tokens=False),
                dtype=torch.long,
                device=self.text.device,
            ).unsqueeze(0)
            self.text = torch.cat([self.text, tokens], dim=1)
            mod = torch.full(
                (1, tokens.shape[1]),
                FakeLFMModality.TEXT,
                dtype=torch.long,
                device=self.modality_flag.device,
            )
            self.modality_flag = torch.cat([self.modality_flag, mod], dim=1)

        def add_audio(self, waveform: Tensor, sample_rate: int) -> None:
            del sample_rate
            token_count = int(waveform.shape[-1])
            total_columns = liquid_chat.LiquidAudioChat.AUDIO_OUT_SHAPE * token_count
            audio_chunk = torch.arange(
                total_columns,
                dtype=torch.long,
                device=waveform.device,
            ).repeat(liquid_chat.LiquidAudioChat.AUDIO_IN_SHAPE, 1)
            self.audio_in = torch.cat([self.audio_in, audio_chunk], dim=1)
            self.audio_in_lens = torch.cat(
                [
                    self.audio_in_lens,
                    torch.tensor(
                        [total_columns], dtype=torch.long, device=waveform.device
                    ),
                ],
                dim=0,
            )
            mod = torch.full(
                (1, token_count),
                FakeLFMModality.AUDIO_IN,
                dtype=torch.long,
                device=self.modality_flag.device,
            )
            self.modality_flag = torch.cat([self.modality_flag, mod], dim=1)

        def append(
            self, text: Tensor, audio_out: Tensor, modality_flag: Tensor
        ) -> None:
            if text.dim() == 1:
                text = text.unsqueeze(0)
            if audio_out.dim() == 1:
                audio_out = audio_out.unsqueeze(0)
            if modality_flag.dim() == 1:
                modality_flag = modality_flag.unsqueeze(0)

            text_added = int(text.shape[1]) if text.numel() else 0
            audio_added = int(audio_out.shape[1]) if audio_out.numel() else 0

            if text_added:
                self.text = torch.cat([self.text, text], dim=1)
            if audio_added:
                self.audio_out = torch.cat([self.audio_out, audio_out], dim=1)
            self.modality_flag = torch.cat(
                [self.modality_flag, modality_flag.to(self.modality_flag.device)], dim=1
            )

            self.append_calls.append(
                (text.clone(), audio_out.clone(), modality_flag.clone())
            )

        def new_turn(self, role: str) -> None:
            self.new_turn_log.append(role)

        def end_turn(self) -> None:
            self.end_turn_count += 1

    monkeypatch.setattr(liquid_chat, "LFMModality", FakeLFMModality)
    monkeypatch.setattr(liquid_chat, "_ChatState", ChatStateStub)
    try:
        liquid_chat.LiquidAudioChat.__bases__ = (
            liquid_chat.BaseMllmChat,
            ChatStateStub,
        )
    except TypeError:
        pass

    return SimpleNamespace(Processor=ProcessorStub, ChatState=ChatStateStub)


def _build_chat_with_state(**kwargs: Any) -> liquid_chat.LiquidAudioChat:
    chat = liquid_chat.LiquidAudioChat(device=torch.device("cpu"), **kwargs)
    chat.refresh(full=True)
    return chat


def test_init_sets_system_token_and_empty_audio_map(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()

    assert chat._audio_map.numel() == 0
    assert chat.speaker is None
    assert chat.token_roles.tolist() == [Role.SYSTEM.value]
    assert chat.text_tokens_no_system_mask.tolist() == [False]


def test_get_relative_audio_masks_handles_in_and_out(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    modality = torch.tensor(
        [
            [
                FakeLFMModality.AUDIO_IN,
                FakeLFMModality.TEXT,
                FakeLFMModality.AUDIO_OUT,
                FakeLFMModality.AUDIO_IN,
            ]
        ],
        dtype=torch.long,
    )
    chat = _build_chat_with_state(modality_flag=modality)
    audio_mask = torch.tensor([-1, 1, -2], dtype=torch.long)
    chat._audio_map = audio_mask.to(chat.torch_device)
    chat.refresh(full=True)

    audio_in_mask, audio_out_mask = chat._get_relative_audio_masks()

    expected_in = torch.tensor(
        [True, False, True], dtype=torch.bool, device=chat.torch_device
    )
    expected_out = torch.tensor(
        [False, True, False], dtype=torch.bool, device=chat.torch_device
    )
    assert torch.equal(audio_in_mask, expected_in)
    assert torch.equal(audio_out_mask, expected_out)


def test_append_text_history_keeps_audio_map_empty(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    text = torch.tensor([[10, 11]], dtype=torch.long)
    audio = torch.arange(chat.codebooks * 2, dtype=torch.long).reshape(
        chat.codebooks, 2
    )
    modality = torch.tensor(
        [FakeLFMModality.TEXT, FakeLFMModality.AUDIO_OUT], dtype=torch.long
    )

    text_added, audio_added = chat._append(
        text=text,
        audio_out=audio,
        modality_flag=modality,
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )

    assert (text_added, audio_added) == (2, 0)
    assert chat._audio_map.tolist() == []
    appended_text, appended_audio, appended_modality = chat.append_calls[-1]
    assert appended_audio.numel() == 0
    assert appended_text.shape[1] == 2
    assert appended_modality.shape[1] == 1


def test_append_audio_history_updates_audio_tokens_and_map(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    text = torch.tensor([[10, 20]], dtype=torch.long)
    audio = torch.arange(chat.codebooks * 2, dtype=torch.long).reshape(
        chat.codebooks, 2
    )
    modality = torch.tensor(
        [FakeLFMModality.TEXT, FakeLFMModality.AUDIO_OUT], dtype=torch.long
    )

    text_added, audio_added = chat._append(
        text=text,
        audio_out=audio,
        modality_flag=modality,
        history_tracking_mode=ModelHistoryTrackingMode.AUDIO,
    )

    assert (text_added, audio_added) == (0, 2)
    assert chat._audio_map.tolist() == [1, 2]
    appended_text, appended_audio, _ = chat.append_calls[-1]
    assert appended_text.numel() == 0
    assert torch.equal(appended_audio, audio)


def test_add_audio_appends_tokens_and_negative_map(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    waveform = torch.zeros((chat.codebooks, 2), dtype=torch.float32)

    added = chat._add_audio(waveform, sample_rate=24_000)

    assert added == 2
    assert chat._audio_map.tolist() == [-1, -2]
    assert chat.audio_in_lens.tolist() == [16]


def test_decode_audio_returns_none_for_audio_in_shape(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    audio_tokens = torch.zeros(
        (liquid_chat.LiquidAudioChat.AUDIO_IN_SHAPE, 3), dtype=torch.long
    )

    decoded = chat._decode_audio(audio_tokens)
    assert decoded is None


def test_decode_audio_uses_mimi_for_audio_out(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    audio_tokens = torch.arange(chat.codebooks * 3, dtype=torch.long).reshape(
        chat.codebooks, 3
    )

    decoded = chat._decode_audio(audio_tokens)

    assert torch.equal(decoded, audio_tokens.float())
    assert len(chat.proc.mimi.calls) == 1


def test_decode_audio_mixed_sign_raises(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    mixed = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="audio_tokens should contain either only audio in or only audio out tokens",
    ):
        chat._decode_audio(mixed)


def test_get_tokens_sequences_to_exclude_encodes_phrases(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()
    sequences = chat._get_tokens_sequences_to_exclude({"ab"})

    assert len(sequences) == 1
    encoded = torch.tensor(
        chat.proc.text.encode("ab", add_special_tokens=False), device=chat.torch_device
    )
    assert torch.equal(sequences[0], encoded)


def test_new_turn_and_end_turn_forward_to_chat_state(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    chat = _build_chat_with_state()

    chat.new_turn(Role.USER)
    assert chat.turn_number == 1
    assert chat.speaker == Role.USER
    assert chat.new_turn_log[-1] == "user"

    chat.end_turn()
    assert chat.speaker is None
    assert chat.end_turn_count == 1


def test_input_tokens_mixes_text_and_audio_sources(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    codebooks = liquid_chat.LiquidAudioChat.AUDIO_OUT_SHAPE
    text = torch.tensor([[5, 6]], dtype=torch.long)
    audio_in = (
        torch.arange(codebooks, dtype=torch.long)
        .view(codebooks, 1)
        .repeat(1, codebooks)
    )
    audio_out = torch.arange(codebooks, dtype=torch.long).unsqueeze(1)
    modality = torch.tensor(
        [
            [
                FakeLFMModality.TEXT,
                FakeLFMModality.AUDIO_IN,
                FakeLFMModality.AUDIO_OUT,
                FakeLFMModality.TEXT,
            ]
        ],
        dtype=torch.long,
    )

    chat = _build_chat_with_state(
        text=text,
        audio_in=audio_in,
        audio_out=audio_out,
        modality_flag=modality,
        audio_in_lens=torch.tensor([codebooks], dtype=torch.long),
    )
    chat._audio_map = torch.tensor([-1, 1], dtype=torch.long)
    chat.text_tokens_no_system_mask = torch.tensor(
        [False, True], dtype=torch.bool, device=chat.torch_device
    )
    chat.audio_tokens_no_system_mask = torch.tensor(
        [True, True], dtype=torch.bool, device=chat.torch_device
    )
    chat.token_turns = torch.zeros(4, dtype=torch.int16)
    chat.token_roles = torch.tensor(
        [Role.SYSTEM.value, Role.SYSTEM.value, Role.ASSISTANT.value, Role.USER.value],
        dtype=torch.int8,
    )
    chat.refresh(full=True)

    tokens = chat.input_tokens
    assert len(tokens) == 4
    assert torch.equal(tokens[0], chat.text_tokens[0].unsqueeze(-1))
    assert torch.equal(tokens[1], chat.audio_in[..., 0].unsqueeze(-1))
    assert torch.equal(tokens[2], chat.audio_out[..., 0].unsqueeze(-1))
    assert torch.equal(tokens[3], chat.text_tokens[1].unsqueeze(-1))


def test_set_new_instance_filters_audio_components(
    patched_liquid_audio_chat: SimpleNamespace,
) -> None:
    codebooks = liquid_chat.LiquidAudioChat.AUDIO_OUT_SHAPE
    text = torch.tensor([[11, 22]], dtype=torch.long)
    audio_in = (
        torch.arange(codebooks, dtype=torch.long)
        .view(codebooks, 1)
        .repeat(1, codebooks)
    )
    audio_out = torch.arange(codebooks, dtype=torch.long).unsqueeze(1)
    modality = torch.tensor(
        [
            [
                FakeLFMModality.TEXT,
                FakeLFMModality.AUDIO_IN,
                FakeLFMModality.AUDIO_OUT,
                FakeLFMModality.TEXT,
            ]
        ],
        dtype=torch.long,
    )
    chat = _build_chat_with_state(
        text=text,
        audio_in=audio_in,
        audio_out=audio_out,
        modality_flag=modality,
        audio_in_lens=torch.tensor([codebooks], dtype=torch.long),
    )
    chat._audio_map = torch.tensor([-1, 1], dtype=torch.long)
    chat.validate_from_chat = True
    chat.text_tokens_no_system_mask = torch.tensor(
        [False, True], dtype=torch.bool, device=chat.torch_device
    )
    chat.audio_tokens_no_system_mask = torch.tensor(
        [True, True], dtype=torch.bool, device=chat.torch_device
    )
    chat.token_turns = torch.zeros(4, dtype=torch.int16)
    chat.token_roles = torch.tensor(
        [Role.SYSTEM.value, Role.SYSTEM.value, Role.ASSISTANT.value, Role.USER.value],
        dtype=torch.int8,
    )
    chat.refresh(full=True)

    full_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
    text_mask_relative = torch.tensor([True, True], dtype=torch.bool)
    audio_mask_relative = torch.tensor([False, True], dtype=torch.bool)

    new_chat = liquid_chat.LiquidAudioChat._set_new_instance(
        full_mask=full_mask,
        text_mask_relative=text_mask_relative,
        audio_mask_relative=audio_mask_relative,
        chat=chat,
    )

    assert torch.equal(
        new_chat.modality_flag,
        torch.tensor(
            [[FakeLFMModality.TEXT, FakeLFMModality.AUDIO_OUT, FakeLFMModality.TEXT]],
            dtype=torch.long,
        ),
    )
    assert new_chat.audio_in.shape[1] == 0
    assert new_chat.audio_in_lens.numel() == 0
    assert new_chat.audio_out.shape[1] == 1
    assert new_chat._audio_map.tolist() == [1]

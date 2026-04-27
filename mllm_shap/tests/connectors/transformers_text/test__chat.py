"""Tests for the Transformers text chat connector."""

import pytest
import torch

from mllm_shap.connectors.enums import ModelHistoryTrackingMode, ModalityFlag
from mllm_shap.connectors.transformers_text.chat import TransformersTextChat


class _TokenizerStub:
    """Minimal tokenizer stub for chat tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) % 31 for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "|".join(str(i) for i in ids)


@pytest.fixture
def chat() -> TransformersTextChat:
    """Create text-only chat with deterministic tokenizer."""
    return TransformersTextChat(
        device=torch.device("cpu"),
        tokenizer=_TokenizerStub(),
    )


def test_add_text_and_decode(chat: TransformersTextChat) -> None:
    """Adding text should append token ids and decode back as string."""
    added = chat._add_text("ab")
    assert added == 2
    assert chat.text_tokens.shape[0] == 2
    decoded = chat._decode_text(chat.text_tokens)
    assert decoded == "|".join(str(i) for i in chat.text_tokens.tolist())


def test_add_audio_warns_and_noops(chat: TransformersTextChat) -> None:
    """Audio add path should warn and not append tokens."""
    with pytest.warns(UserWarning, match="Audio input is not supported"):
        added = chat._add_audio(torch.zeros((1, 10)), sample_rate=16_000)
    assert added == 0
    assert chat.audio_tokens.numel() == 0


def test_append_audio_tracking_warns_and_skips(chat: TransformersTextChat) -> None:
    """AUDIO-only tracking should be rejected with warning."""
    with pytest.warns(UserWarning, match="AUDIO-only history tracking"):
        text_added, audio_added = chat._append(
            text=torch.tensor([1, 2, 3], dtype=torch.long),
            audio_out=torch.empty((0, 0), dtype=torch.long),
            modality_flag=torch.tensor([ModalityFlag.TEXT] * 3, dtype=torch.long),
            history_tracking_mode=ModelHistoryTrackingMode.AUDIO,
        )
    assert (text_added, audio_added) == (0, 0)


def test_append_text_tracking_appends_and_refreshes(chat: TransformersTextChat) -> None:
    """TEXT tracking should append ids and expose text-only modality flags."""
    text_added, audio_added = chat._append(
        text=torch.tensor([7, 8], dtype=torch.long),
        audio_out=torch.empty((0, 0), dtype=torch.long),
        modality_flag=torch.tensor([ModalityFlag.TEXT] * 2, dtype=torch.long),
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )
    assert (text_added, audio_added) == (2, 0)
    assert chat.text_tokens.tolist() == [7, 8]
    assert chat.tokens_modality_flag.tolist() == [ModalityFlag.TEXT, ModalityFlag.TEXT]


def test_set_new_instance_applies_relative_text_mask(
    chat: TransformersTextChat,
) -> None:
    """Masked clone should keep only selected relative text tokens."""
    _ = chat._append(
        text=torch.tensor([10, 11, 12], dtype=torch.long),
        audio_out=torch.empty((0, 0), dtype=torch.long),
        modality_flag=torch.tensor([ModalityFlag.TEXT] * 3, dtype=torch.long),
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )
    # Mirror runtime state where text-only masks align with full text length.
    chat.text_tokens_no_system_mask = torch.tensor([True, True, True], dtype=torch.bool)

    cloned = TransformersTextChat._set_new_instance(
        full_mask=torch.tensor([True, False, True], dtype=torch.bool),
        text_mask_relative=torch.tensor([True, False, True], dtype=torch.bool),
        audio_mask_relative=torch.tensor([], dtype=torch.bool),
        chat=chat,
    )
    assert cloned.text_tokens.tolist() == [10, 12]

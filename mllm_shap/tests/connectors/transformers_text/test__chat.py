"""Tests for the Transformers text chat connector."""

import pytest
import torch

from mllm_shap.connectors.enums import ModelHistoryTrackingMode, ModalityFlag
from mllm_shap.connectors.transformers_text.chat import TransformersTextChat
from .conftest import _TokenizerStub


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


def test_input_tokens_returns_singleton_tensors(chat: TransformersTextChat) -> None:
    """input_tokens should expose one [1]-shaped tensor per token id."""
    _ = chat._add_text("abc")
    tokens = chat.input_tokens
    assert len(tokens) == 3
    assert all(t.shape == (1,) for t in tokens)


def test_decode_text_joins_list_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """_decode_text should join list outputs from tokenizer.decode."""
    c = TransformersTextChat(device=torch.device("cpu"), tokenizer=_TokenizerStub())
    _ = c._add_text("ab")

    def _decode(ids: list[int], skip_special_tokens: bool = False) -> list[str]:
        del skip_special_tokens
        return [str(i) for i in ids]

    monkeypatch.setattr(c.tokenizer, "decode", _decode)
    decoded = c._decode_text(c.text_tokens)
    assert decoded == "".join(str(i) for i in c.text_tokens.tolist())


def test_add_text_empty_string_returns_zero(chat: TransformersTextChat) -> None:
    """Empty text should produce no tokens."""
    assert chat._add_text("") == 0


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


def test_append_handles_two_dimensional_input_branch(
    chat: TransformersTextChat,
) -> None:
    """The explicit 2D branch should execute and still append flattened tokens."""
    text_added, audio_added = chat._append(
        text=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        audio_out=torch.empty((0, 0), dtype=torch.long),
        modality_flag=torch.tensor([ModalityFlag.TEXT] * 4, dtype=torch.long),
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )
    assert (text_added, audio_added) == (4, 0)
    assert chat.text_tokens.tolist() == [1, 2, 3, 4]


def test_append_handles_scalar_input_branch(chat: TransformersTextChat) -> None:
    """Scalar inputs should be unsqueezed and appended as a single token."""
    text_added, audio_added = chat._append(
        text=torch.tensor(9, dtype=torch.long),
        audio_out=torch.empty((0, 0), dtype=torch.long),
        modality_flag=torch.tensor([ModalityFlag.TEXT], dtype=torch.long),
        history_tracking_mode=ModelHistoryTrackingMode.TEXT,
    )
    assert (text_added, audio_added) == (1, 0)
    assert chat.text_tokens.tolist() == [9]


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


def test_get_tokens_sequences_to_exclude_encodes_phrases(
    chat: TransformersTextChat,
) -> None:
    """Exclude phrases should be converted to long tensors on chat device."""
    seqs = chat._get_tokens_sequences_to_exclude({"hi", "ok"})
    assert len(seqs) == 2
    assert all(seq.dtype == torch.long for seq in seqs)

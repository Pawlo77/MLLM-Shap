"""Tests for Mock connector (chat + model)."""

import warnings

import pytest
import torch
from torch import Tensor
from transformers import AutoTokenizer

from mllm_shap.connectors.mock import Mock, MockChat
from mllm_shap.connectors.enums import (
    ModalityFlag,
    ModelHistoryTrackingMode,
    Role,
)
from mllm_shap.connectors.base.model_response import ModelResponse


# ──────────────────── Fixtures ────────────────────


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def mock_model(device: torch.device) -> Mock:
    return Mock(device=device)


@pytest.fixture
def chat(device: torch.device, tokenizer) -> MockChat:
    return MockChat(device=device, tokenizer=tokenizer)


# ──────────────────── MockChat Tests ────────────────────


class TestMockChat:
    """Unit tests for MockChat."""

    def test_init_empty(self, chat: MockChat) -> None:
        """Fresh chat has zero text tokens."""
        assert chat.text_tokens.shape[0] == 0
        assert chat.audio_tokens.shape[0] == 0

    def test_add_text_encodes_correctly(self, chat: MockChat) -> None:
        """_add_text encodes text and appends to _text_ids."""
        n = chat._add_text("hello")
        assert n > 0
        assert chat._text_ids.shape[0] == n

    def test_add_text_empty_string_returns_zero(self, chat: MockChat) -> None:
        """Empty string produces no tokens."""
        n = chat._add_text("")
        assert n == 0

    def test_add_audio_warns_and_returns_zero(self, chat: MockChat) -> None:
        """Audio is unsupported—should warn and return 0."""
        waveform = torch.randn(1, 16000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = chat._add_audio(waveform, sample_rate=16000)
        assert n == 0
        assert any("not supported" in str(warning.message).lower() for warning in w)

    def test_input_tokens_per_token(self, device: torch.device, tokenizer) -> None:
        """input_tokens returns one tensor per token."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("hello world")
        tokens = c.input_tokens
        assert len(tokens) == c._text_ids.shape[0]
        for t in tokens:
            assert t.shape == (1,)

    def test_tokens_modality_flag_all_text(
        self, device: torch.device, tokenizer
    ) -> None:
        """All tokens flagged as TEXT."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("foo bar")
        flags = c.tokens_modality_flag
        assert (flags == ModalityFlag.TEXT).all()

    def test_decode_text_round_trip(self, device: torch.device, tokenizer) -> None:
        """Encode→decode round-trip preserves content."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("test")
        decoded = c._decode_text(c._text_ids)
        assert "test" in decoded

    def test_decode_audio_returns_none(self, chat: MockChat) -> None:
        """_decode_audio always returns None."""
        result = chat._decode_audio(torch.tensor([1, 2, 3]))
        assert result is None

    def test_decode_text_joins_list_output(
        self, device: torch.device, tokenizer
    ) -> None:
        """_decode_text should join tokenizer outputs when decode returns a list."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("ab")

        original_decode = tokenizer.decode

        def _decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return [original_decode([token_id]) for token_id in ids]

        tokenizer.decode = _decode
        try:
            decoded = c._decode_text(c._text_ids)
        finally:
            tokenizer.decode = original_decode

        assert decoded
        assert isinstance(decoded, str)

    def test_apply_text_mask(self, device: torch.device, tokenizer) -> None:
        """apply_text_mask filters token IDs by boolean mask."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c.new_turn(Role.USER)
        c.add_text("a b c d")
        original_len = c._text_ids.shape[0]
        mask = torch.ones(original_len, dtype=torch.bool)
        mask[1] = False  # drop one token
        c.apply_text_mask(mask)
        assert c._text_ids.shape[0] == original_len - 1

    def test_set_new_instance_creates_copy(
        self, device: torch.device, tokenizer
    ) -> None:
        """_set_new_instance deepcopies and masks."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c.new_turn(Role.USER)
        c.add_text("hello world")
        n = c._text_ids.shape[0]
        full_mask = torch.ones(n, dtype=torch.bool)
        # keep all text tokens
        text_mask = torch.ones(n, dtype=torch.bool)
        text_mask[-1] = False  # drop last
        audio_mask = torch.empty(0, dtype=torch.bool)
        new = MockChat._set_new_instance(full_mask, text_mask, audio_mask, c)
        assert new._text_ids.shape[0] == int(text_mask.sum().item())
        # original unchanged
        assert c._text_ids.shape[0] == n

    def test_append_text_tokens(self, device: torch.device, tokenizer) -> None:
        """_append adds text tokens and refreshes state."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("initial")
        initial_count = c._text_ids.shape[0]
        new_tokens = torch.tensor([42, 43], dtype=torch.long)
        n_text, n_audio = c._append(
            new_tokens.unsqueeze(0),  # shape [1, 2]
            torch.empty(0),
            torch.empty(0),
            ModelHistoryTrackingMode.TEXT,
        )
        assert n_text == 2
        assert n_audio == 0
        assert c._text_ids.shape[0] == initial_count + 2

    def test_append_audio_mode_warns(self, device: torch.device, tokenizer) -> None:
        """AUDIO tracking mode warns and appends nothing."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("x")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n_text, n_audio = c._append(
                torch.tensor([[1]]),
                torch.empty(0),
                torch.empty(0),
                ModelHistoryTrackingMode.AUDIO,
            )
        assert n_text == 0 and n_audio == 0
        assert any("not supported" in str(warning.message).lower() for warning in w)

    def test_append_scalar_tensor(self, device: torch.device, tokenizer) -> None:
        """_append handles 0-dim tensor."""
        c = MockChat(device=device, tokenizer=tokenizer)
        c._add_text("x")
        initial = c._text_ids.shape[0]
        n_text, _ = c._append(
            torch.tensor(99),
            torch.empty(0),
            torch.empty(0),
            ModelHistoryTrackingMode.TEXT,
        )
        assert n_text == 1
        assert c._text_ids.shape[0] == initial + 1

    def test_append_higher_rank_tensor_flattens(
        self, device: torch.device, tokenizer
    ) -> None:
        """_append should flatten non-1D tensors outside the single-batch fast path."""
        c = MockChat(device=device, tokenizer=tokenizer)
        initial = c._text_ids.shape[0]
        n_text, n_audio = c._append(
            torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long),
            torch.empty(0),
            torch.empty(0),
            ModelHistoryTrackingMode.TEXT,
        )
        assert n_text == 4
        assert n_audio == 0
        assert c._text_ids.shape[0] == initial + 4

    def test_new_turn_and_end_turn_noop(self, chat: MockChat) -> None:
        """_new_turn and _end_turn do nothing (no error)."""
        chat._new_turn(Role.ASSISTANT)
        chat._end_turn()

    def test_get_tokens_sequences_to_exclude(
        self, device: torch.device, tokenizer
    ) -> None:
        """Returns encoded tensor sequences."""
        c = MockChat(device=device, tokenizer=tokenizer)
        seqs = c._get_tokens_sequences_to_exclude({"hello", "world"})
        assert len(seqs) == 2
        for s in seqs:
            assert isinstance(s, Tensor)
            assert s.dtype == torch.long

    def test_empty_turn_sequences_filter(self, device: torch.device, tokenizer) -> None:
        """Constructor accepts empty_turn_sequences."""
        c = MockChat(
            device=device,
            tokenizer=tokenizer,
            empty_turn_sequences={"<|endoftext|>"},
        )
        assert c is not None


# ──────────────────── Mock Model Tests ────────────────────


class TestMockModel:
    """Unit tests for Mock model connector."""

    def test_init_default(self, mock_model: Mock) -> None:
        """Model initializes with gpt2 tokenizer."""
        assert mock_model.processor is not None
        assert mock_model.device == torch.device("cpu")

    def test_init_forces_text_mode_with_warning(self, device: torch.device) -> None:
        """Non-TEXT history tracking warns and forces TEXT."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = Mock(
                device=device,
                history_tracking_mode=ModelHistoryTrackingMode.AUDIO,
            )
        assert m.history_tracking_mode == ModelHistoryTrackingMode.TEXT
        assert any("forcing text" in str(warning.message).lower() for warning in w)

    def test_get_new_chat_returns_mock_chat(self, mock_model: Mock) -> None:
        """get_new_chat returns a MockChat instance."""
        c = mock_model.get_new_chat()
        assert isinstance(c, MockChat)

    def test_generate_without_history(self, mock_model: Mock) -> None:
        """generate returns placeholder tokens, no chat in response."""
        chat = mock_model.get_new_chat()
        chat._add_text("test input")
        resp = mock_model.generate(chat=chat, max_new_tokens=10, keep_history=False)
        assert isinstance(resp, ModelResponse)
        assert resp.chat is None
        assert resp.generated_text_tokens.shape[0] == 10
        assert (resp.generated_text_tokens == 0).all()  # placeholder
        assert (resp.generated_modality_flag == ModalityFlag.TEXT).all()

    def test_generate_with_history(self, mock_model: Mock) -> None:
        """generate with keep_history=True returns chat in response."""
        chat = mock_model.get_new_chat()
        chat._add_text("hello")
        resp = mock_model.generate(chat=chat, max_new_tokens=5, keep_history=True)
        assert resp.chat is not None

    def test_generate_max_tokens_respected(self, mock_model: Mock) -> None:
        """Output length matches max_new_tokens."""
        chat = mock_model.get_new_chat()
        chat._add_text("abc")
        for n in [1, 16, 64]:
            resp = mock_model.generate(chat=chat, max_new_tokens=n)
            assert resp.generated_text_tokens.shape[0] == n

    def test_get_static_embeddings(self, mock_model: Mock) -> None:
        """Static embeddings have shape [T, 768]."""
        chat = mock_model.get_new_chat()
        chat._add_text("hi")
        resp = mock_model.generate(chat=chat, max_new_tokens=4, keep_history=False)
        embs = mock_model.get_static_embeddings([resp])
        assert len(embs) == 1
        assert embs[0].shape == (4, 768)

    def test_get_contextual_embeddings_clones_static(self, mock_model: Mock) -> None:
        """Contextual embeddings clone static ones."""
        static = [torch.randn(3, 768)]
        ctx = mock_model._get_contextual_embeddings(static)
        assert len(ctx) == 1
        assert torch.equal(ctx[0], static[0])
        # verify it's a clone not same object
        ctx[0][0, 0] = 999.0
        assert static[0][0, 0] != 999.0

    def test_generate_audio_tokens_empty(self, mock_model: Mock) -> None:
        """Generated audio tokens are empty."""
        chat = mock_model.get_new_chat()
        chat._add_text("x")
        resp = mock_model.generate(chat=chat, max_new_tokens=2)
        assert resp.generated_audio_tokens.numel() == 0

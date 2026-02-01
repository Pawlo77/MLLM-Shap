"""Unit tests for the BaseMllmChat connector and its methods."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.connectors.base.chat import BaseMllmChat, Role
from mllm_shap.connectors.base.chat_entry import ChatEntry
from mllm_shap.connectors.enums import ModalityFlag

from ...dummy import DummyChat


class TestDummyChat:
    """Tests for the DummyChat class apart from get_conversation."""

    @staticmethod
    @pytest.fixture
    def chat() -> BaseMllmChat:
        """Fixture for DummyChat instance."""
        return DummyChat(num_tokens=5)

    def test_initialization(self, chat: BaseMllmChat) -> None:
        """Test initialization of DummyChat."""
        assert isinstance(chat.torch_device, torch.device)
        assert chat.turn_number == 0
        assert chat.speaker is None
        assert chat.input_tokens_num == 5
        assert chat.text_tokens.shape[0] == 5
        assert chat.audio_tokens.shape[0] == 5

    def test_input_tokens_and_masks(self, chat: BaseMllmChat) -> None:
        """Test input_tokens and various masks."""
        assert torch.all(chat.input_tokens == torch.arange(5))
        assert torch.all(chat.tokens_modality_flag == ModalityFlag.TEXT)
        # text_tokens_mask
        assert torch.all(chat.text_tokens_mask)
        # audio_tokens_mask
        assert not chat.audio_tokens_mask.any()
        # shap_values_mask
        mask = chat.shap_values_mask
        assert mask.shape[0] == chat.input_tokens_num
        assert mask.dtype == torch.bool

    def test_shap_setter_getter_deleter(self, chat: BaseMllmChat) -> None:
        """Test shap property setter, getter, and deleter."""
        dummy_cache = MagicMock()
        dummy_cache.chat = chat
        # set
        chat.cache = dummy_cache
        assert chat.cache == dummy_cache
        # changing to different chat should raise
        dummy_cache2 = MagicMock()
        dummy_cache2.chat = DummyChat(1)
        with pytest.raises(ValueError):
            chat.cache = dummy_cache2
        # deleter
        del chat.cache
        assert chat.cache is None

    def test_extend_and_after_add(self, chat: BaseMllmChat) -> None:
        """Test _extend_token_roles and _after_add methods."""
        # speaker must be set for _extend_token_roles
        chat.speaker = Role.USER
        chat._after_add(num_tokens=2, text_added=True, refresh=False)
        # token_turns and roles extended
        assert chat.token_turns.shape[0] >= 2
        assert chat.token_roles.shape[0] >= 2
        assert chat.text_tokens_no_system_mask.shape[0] >= 2

    def test_new_turn_and_end_turn(self, chat: BaseMllmChat) -> None:
        """Test new_turn and end_turn methods."""
        chat.turn_number = 0
        chat.speaker = None
        with patch("mllm_shap.connectors.base.chat.raise_connector_error"):
            chat.new_turn(Role.USER)
            assert chat.turn_number == 1
            assert chat.speaker == Role.USER
            chat.end_turn()
            assert chat.speaker is None

    def test_add_text_and_add_audio(self, chat: BaseMllmChat) -> None:
        """Test add_text and add_audio methods."""
        chat.speaker = Role.USER
        # add text
        initial_tokens = chat.input_tokens_num
        chat.add_text("hello")
        assert chat.input_tokens_num > initial_tokens
        # add audio
        with patch("mllm_shap.utils.audio.TorchAudioHandler.from_bytes") as mock_audio:
            mock_audio.return_value = (torch.zeros(2), 16000)
            chat.add_audio(b"abcd", audio_format="mp3")
        assert chat.input_tokens_num > initial_tokens

    def test_add_text_rejects_invalid_input(self, chat: BaseMllmChat) -> None:
        """add_text should reject non-string or empty values before touching chat state."""
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            chat.add_text(123)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            chat.add_text("")

    def test_add_audio_rejects_invalid_input(self, chat: BaseMllmChat) -> None:
        """add_audio should reject non-bytes or empty payloads."""
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="audio_content must be non-empty bytes"):
            chat.add_audio("not-bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="audio_content must be non-empty bytes"):
            chat.add_audio(b"")

    def test_append_method(self, chat: BaseMllmChat) -> None:
        """Test append method."""
        chat.speaker = Role.USER
        text_tensor = torch.arange(2)
        audio_tensor = torch.arange(2)
        modality_flag = torch.tensor([ModalityFlag.TEXT, ModalityFlag.AUDIO], dtype=torch.int8)
        with patch("mllm_shap.connectors.base.chat.raise_connector_error", return_value=(2, 2)):
            chat.append(text_tensor, audio_tensor, modality_flag, history_tracking_mode=None)
        # after_add logic should extend masks
        assert chat.text_tokens_no_system_mask.shape[0] >= 2
        assert chat.audio_tokens_no_system_mask.shape[0] >= 2

    def test_detect_sequence(self, chat: BaseMllmChat) -> None:
        """Test _detect method for sequence detection."""
        tokens = torch.tensor([1, 2, 3, 1, 2, 3])
        seq_tensor = torch.tensor([1, 2])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        new_mask = chat._detect(tokens, seq_tensor, mask=mask, mark=True)
        # positions [0,3] should be marked False
        assert not new_mask[0]
        assert not new_mask[3]

    def test_detect_returns_indices_when_not_marking(self, chat: BaseMllmChat) -> None:
        """_detect should return match indices when mark=False."""
        tokens = torch.tensor([4, 5, 4, 5, 6])
        seq_tensor = torch.tensor([4, 5])
        matches = chat._detect(tokens, seq_tensor, mask=None, mark=False)
        assert matches.tolist() == [0, 2]

    def test_new_turn_raises_if_already_active(self, chat: BaseMllmChat) -> None:
        """new_turn should guard against nested turns."""
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="Cannot start a new turn"):
            chat.new_turn(Role.USER)

    def test_end_turn_raises_if_no_active(self, chat: BaseMllmChat) -> None:
        """end_turn should fail when there is no active speaker."""
        chat.speaker = None
        with pytest.raises(ValueError, match="No active turn"):
            chat.end_turn()

    def test_from_chat(self) -> None:
        """Test from_chat class method."""
        base_chat = DummyChat(num_tokens=3)
        mask = torch.ones(3, dtype=torch.bool)
        new_chat = DummyChat.from_chat(mask, base_chat)
        assert isinstance(new_chat, DummyChat)
        assert new_chat.input_tokens_num == 3
        # mask of wrong size raises
        with pytest.raises(ValueError):
            DummyChat.from_chat(torch.ones(2, dtype=torch.bool), base_chat)
        # all-False mask raises
        with pytest.raises(ValueError):
            DummyChat.from_chat(torch.zeros(3, dtype=torch.bool), base_chat)

    def test_from_chat_clears_external_masks(self) -> None:
        """External group ids and shap masks must not leak to new chat."""
        base_chat = DummyChat(num_tokens=3)
        base_chat.external_group_ids = torch.tensor([0, 1, 1], dtype=torch.int32)
        base_chat.external_shap_values_mask = torch.tensor([True, False, True])

        new_chat = DummyChat.from_chat(torch.tensor([True, True, True]), base_chat)

        assert new_chat.external_group_ids is None
        assert new_chat.external_shap_values_mask is None

    def test_before_add_blocks_when_external_masks_set(self, chat: BaseMllmChat) -> None:
        """Setting external masks should prevent further additions."""
        ids = torch.tensor([0, 1, 1, 0, 2], dtype=torch.int32)
        chat.external_group_ids = ids[: chat.input_tokens_num]
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="Cannot add tokens when external_group_ids is set"):
            chat.add_text("hello")

        del chat.external_group_ids
        chat.speaker = Role.USER
        chat.add_text("hello")
        assert chat.input_tokens_num == 6

        chat.external_shap_values_mask = torch.ones(chat.input_tokens_num, dtype=torch.bool)
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="Cannot add tokens when external_shap_values_mask is set"):
            chat.add_text("world")
        del chat.external_shap_values_mask

    def test_external_masks_size_validation(self, chat: BaseMllmChat) -> None:
        """External mask setters should validate tensor lengths."""
        with pytest.raises(ValueError, match="External SHAP values mask size"):
            chat.external_shap_values_mask = torch.ones(chat.input_tokens_num + 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="External group IDs size"):
            chat.external_group_ids = torch.ones(chat.input_tokens_num + 2, dtype=torch.int32)

    def test_translate_group_ids_mask_marks_full_groups(self, chat: BaseMllmChat) -> None:
        """translate_groups_ids_mask should expand selections to entire groups."""
        ids = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32)
        chat.external_group_ids = ids
        group_mask = torch.zeros(chat.input_tokens_num, dtype=torch.bool)
        positions = chat.external_group_ids_first_positions
        group_mask[positions] = torch.tensor([False, True])

        translated = chat.translate_groups_ids_mask(group_mask.clone())
        expected = torch.tensor([False, False, False, True, True])
        assert torch.equal(translated, expected)

        del chat.external_group_ids

    def test_is_system_turn_flag(self, chat: BaseMllmChat) -> None:
        """is_system_turn should reflect membership in system roles set."""
        chat.speaker = Role.SYSTEM
        assert chat.is_system_turn is True
        chat.speaker = Role.USER
        assert chat.is_system_turn is False


class TestGetConversation:
    """Tests for the get_conversation method of DummyChat."""

    @pytest.fixture
    def chat(self) -> DummyChat:
        """Fixture for DummyChat with multi-turn setup."""
        chat = DummyChat(num_tokens=4)
        chat.turn_number = 2
        chat.token_turns = torch.tensor([1, 1, 2, 2], dtype=torch.int16)
        chat.token_roles = torch.tensor(
            [
                Role.USER.value,
                Role.USER.value,
                Role.ASSISTANT.value,
                Role.ASSISTANT.value,
            ],
            dtype=torch.int8,
        )
        chat.text_tokens_mask = torch.ones(4, dtype=torch.bool)
        return chat

    def test_returns_empty_when_no_turns(self) -> None:
        """Chats without turns should yield an empty conversation."""
        chat = DummyChat(num_tokens=2)
        assert chat.get_conversation() == []

    def test_single_turn_text(self, chat: BaseMllmChat) -> None:
        """Test get_conversation for single turn with text modality."""
        chat.turn_number = 1
        chat.token_turns = torch.tensor([1, 1, 1], dtype=torch.int16)
        chat.token_roles = torch.tensor([Role.USER.value] * 3, dtype=torch.int8)
        chat.text_tokens_mask = torch.ones(3, dtype=torch.bool)

        with patch.object(
            chat,
            "decode_text",
            side_effect=lambda text_tokens, **kwargs: str(text_tokens.item()),
        ):
            conversation = chat.get_conversation()

        assert isinstance(conversation, list)
        assert len(conversation) == 1
        turn_entries = conversation[0]
        assert all(isinstance(entry, ChatEntry) for entry in turn_entries)
        content_flat = [token for entry in turn_entries for token in entry.content]
        assert content_flat == ["0", "1", "2"]
        for entry in turn_entries:
            assert entry.roles == [Role.USER.value] * 3

    def test_multi_turn_text(self, chat: BaseMllmChat) -> None:
        """Test get_conversation for multi-turn with text modality."""
        with patch.object(
            chat,
            "decode_text",
            side_effect=lambda text_tokens, **kwargs: str(text_tokens.item()),
        ):
            conversation = chat.get_conversation()

        assert len(conversation) == 2
        # First turn
        first_turn_tokens = [token for entry in conversation[0] for token in entry.content]
        assert first_turn_tokens == ["0", "1"]
        # Second turn
        second_turn_tokens = [token for entry in conversation[1] for token in entry.content]
        assert second_turn_tokens == ["2", "3"]

    def test_text_and_audio_modality(self) -> None:
        """Test get_conversation with mixed text and audio modalities."""
        chat = DummyChat(num_tokens=4)
        chat.turn_number = 1
        chat.token_turns = torch.tensor([1, 1, 1, 1], dtype=torch.int16)
        chat.token_roles = torch.tensor([Role.USER.value] * 4, dtype=torch.int8)
        chat.text_tokens_mask = torch.tensor([True, True, False, False], dtype=torch.bool)

        with (
            patch.object(
                chat,
                "decode_text",
                side_effect=lambda text_tokens, **kwargs: str(text_tokens.item()),
            ),
            patch.object(
                chat,
                "decode_audio",
                side_effect=lambda audio_tokens, **kwargs: b"audio",
            ),
        ):
            conversation = chat.get_conversation()

        assert len(conversation) == 1
        turn_entries = conversation[0]
        assert len(turn_entries) == 2
        # Check decoded content
        assert turn_entries[0].content == ["0", "1"]
        assert turn_entries[1].content == [b"audio", b"audio"]
        # Check roles
        for entry in turn_entries:
            assert entry.roles == [Role.USER.value] * len(entry.content)

    def test_shap_values_included(self) -> None:
        """Test that SHAP values are correctly included in ChatEntry."""
        chat = DummyChat(num_tokens=3)
        chat.turn_number = 1
        chat.token_turns = torch.tensor([1, 1, 1], dtype=torch.int16)
        chat.token_roles = torch.tensor([Role.USER.value] * 3, dtype=torch.int8)
        chat.text_tokens_mask = torch.ones(3, dtype=torch.bool)

        # Mock SHAP
        shap_mock = MagicMock()
        shap_mock.normalized_values = torch.tensor([0.1, 0.2, 0.3])
        chat.cache = shap_mock

        with patch.object(
            chat,
            "decode_text",
            side_effect=lambda text_tokens, **kwargs: str(text_tokens.item()),
        ):
            conversation = chat.get_conversation()

        turn_entry = conversation[0][0]
        assert turn_entry.shap_values == pytest.approx([0.1, 0.2, 0.3])

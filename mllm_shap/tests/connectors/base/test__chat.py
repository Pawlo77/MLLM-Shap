"""Unit tests for the BaseMllmChat connector and its methods."""

from copy import deepcopy
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.connectors.base.audio import AudioSegment
from mllm_shap.connectors.base.chat import BaseMllmChat, Role
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.connectors.base.chat_entry import ChatEntry
from mllm_shap.connectors.enums import ModalityFlag

from ...dummy import DummyChat


class _BranchCoverageChat(BaseMllmChat):
    """Minimal chat implementation that allows explicit text/audio token splits."""

    def __init__(self, input_tokens: torch.Tensor, text_mask: torch.Tensor) -> None:
        super().__init__(
            device=torch.device("cpu"),
            empty_turn_sequences=set(),
            get_new_chat_callable=lambda: _BranchCoverageChat(
                input_tokens=torch.tensor([], dtype=torch.long),
                text_mask=torch.tensor([], dtype=torch.bool),
            ),
        )
        self._input_tokens = input_tokens.clone().to(torch.long)
        self._text_mask = text_mask.clone().to(torch.bool)
        self.text_tokens_no_system_mask = torch.ones(
            int(self._text_mask.sum().item()), dtype=torch.bool
        )
        self.audio_tokens_no_system_mask = torch.ones(
            int((~self._text_mask).sum().item()), dtype=torch.bool
        )

    @property
    def input_tokens(self) -> torch.Tensor:
        return self._input_tokens

    @property
    def tokens_modality_flag(self) -> torch.Tensor:
        out = torch.full(
            (self._input_tokens.numel(),), ModalityFlag.AUDIO, dtype=torch.int8
        )
        out[self._text_mask] = ModalityFlag.TEXT
        return out

    @property
    def text_tokens(self) -> torch.Tensor:
        return self._input_tokens[self._text_mask]

    @property
    def audio_tokens(self) -> torch.Tensor:
        return self._input_tokens[~self._text_mask]

    @classmethod
    def _set_new_instance(
        cls,
        full_mask: torch.Tensor,
        text_mask_relative: torch.Tensor,
        audio_mask_relative: torch.Tensor,
        chat: BaseMllmChat,
    ) -> "_BranchCoverageChat":
        del text_mask_relative, audio_mask_relative, chat
        kept = torch.arange(int(full_mask.sum().item()), dtype=torch.long)
        return cls(kept, torch.ones_like(kept, dtype=torch.bool))

    def _decode_text(self, text_tokens: torch.Tensor) -> str:
        return " ".join(str(int(t.item())) for t in text_tokens)

    def _decode_audio(self, audio_tokens: torch.Tensor) -> torch.Tensor | None:
        del audio_tokens
        return torch.zeros(1, 8)

    def _add_text(self, text: str) -> int:
        del text
        new_token = torch.tensor([self._input_tokens.numel()], dtype=torch.long)
        self._input_tokens = torch.cat([self._input_tokens, new_token])
        self._text_mask = torch.cat([self._text_mask, torch.tensor([True])])
        return 1

    def _add_audio(self, waveform: torch.Tensor, sample_rate: int) -> int:
        del waveform, sample_rate
        new_token = torch.tensor([self._input_tokens.numel()], dtype=torch.long)
        self._input_tokens = torch.cat([self._input_tokens, new_token])
        self._text_mask = torch.cat([self._text_mask, torch.tensor([False])])
        return 1

    def _append(
        self,
        text: torch.Tensor,
        audio_out: torch.Tensor,
        modality_flag: torch.Tensor,
        history_tracking_mode,
    ) -> tuple[int, int]:
        del text, audio_out, modality_flag, history_tracking_mode
        return 0, 0

    def _new_turn(self, speaker: Role) -> None:
        del speaker

    def _end_turn(self) -> None:
        return

    def _get_tokens_sequences_to_exclude(
        self, phrases_to_exclude: set[str]
    ) -> list[torch.Tensor]:
        del phrases_to_exclude
        return []


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
            chat.add_text(cast(Any, 123))
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            chat.add_text("")

    def test_add_audio_rejects_invalid_input(self, chat: BaseMllmChat) -> None:
        """add_audio should reject non-bytes or empty payloads."""
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="audio_content must be non-empty bytes"):
            chat.add_audio(cast(Any, "not-bytes"))
        with pytest.raises(ValueError, match="audio_content must be non-empty bytes"):
            chat.add_audio(b"")

    def test_add_audio_rejects_when_segments_exist(self, chat: BaseMllmChat) -> None:
        """add_audio should require transcript path once segmented audio is in use."""
        chat.speaker = Role.USER
        chat._audio_segments = {1: []}
        with pytest.raises(ValueError, match="Cannot add audio without transcript"):
            chat.add_audio(b"abc")

    def test_add_audio_rejects_second_audio_in_same_turn(
        self, chat: BaseMllmChat
    ) -> None:
        """add_audio should block multiple audio additions in one turn."""
        chat.speaker = Role.USER
        chat._audio_added_in_last_turn = True
        with pytest.raises(ValueError, match="already been added in the current turn"):
            chat.add_audio(b"abc")

    def test_append_method(self, chat: BaseMllmChat) -> None:
        """Test append method."""
        chat.speaker = Role.USER
        text_tensor = torch.arange(2)
        audio_tensor = torch.arange(2)
        modality_flag = torch.tensor(
            [ModalityFlag.TEXT, ModalityFlag.AUDIO], dtype=torch.int8
        )
        with patch(
            "mllm_shap.connectors.base.chat.raise_connector_error", return_value=(2, 2)
        ):
            chat.append(
                text_tensor, audio_tensor, modality_flag, history_tracking_mode=None
            )
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

    def test_detect_requires_mask_when_marking(self, chat: BaseMllmChat) -> None:
        """_detect should require mask argument when mark=True."""
        with pytest.raises(ValueError, match="Mask must be provided"):
            _ = chat._detect(
                tokens=torch.tensor([1, 2, 3]),
                seq_tensor=torch.tensor([1]),
                mask=None,
                mark=True,
            )

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

    def test_before_add_blocks_when_external_masks_set(
        self, chat: BaseMllmChat
    ) -> None:
        """Setting external masks should prevent further additions."""
        ids = torch.tensor([0, 1, 1, 0, 2], dtype=torch.int32)
        chat.external_group_ids = ids[: chat.input_tokens_num]
        chat.speaker = Role.USER
        with pytest.raises(
            ValueError, match="Cannot add tokens when external_group_ids is set"
        ):
            chat.add_text("hello")

        del chat.external_group_ids
        chat.speaker = Role.USER
        chat.add_text("hello")
        assert chat.input_tokens_num == 6

        chat.external_shap_values_mask = torch.ones(
            chat.input_tokens_num, dtype=torch.bool
        )
        chat.speaker = Role.USER
        with pytest.raises(
            ValueError, match="Cannot add tokens when external_shap_values_mask is set"
        ):
            chat.add_text("world")
        del chat.external_shap_values_mask

    def test_external_masks_size_validation(self, chat: BaseMllmChat) -> None:
        """External mask setters should validate tensor lengths."""
        with pytest.raises(ValueError, match="External SHAP values mask size"):
            chat.external_shap_values_mask = torch.ones(
                chat.input_tokens_num + 1, dtype=torch.bool
            )

        with pytest.raises(ValueError, match="External group IDs size"):
            chat.external_group_ids = torch.ones(
                chat.input_tokens_num + 2, dtype=torch.int32
            )

    def test_translate_group_ids_mask_marks_full_groups(
        self, chat: BaseMllmChat
    ) -> None:
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

    def test_external_group_ids_properties_raise_when_missing(
        self, chat: BaseMllmChat
    ) -> None:
        """external_group_ids-derived properties should require configured ids."""
        del chat.external_group_ids
        with pytest.raises(ValueError, match="external_group_ids is not set"):
            _ = chat.external_group_ids_first_positions
        with pytest.raises(ValueError, match="external_group_ids is not set"):
            _ = chat.external_group_ids_positive_mask

    def test_shap_values_mask_uses_external_group_ids(self, chat: BaseMllmChat) -> None:
        """shap_values_mask should keep first token from each positive group only."""
        chat.text_tokens_no_system_mask = torch.ones(
            chat.input_tokens_num, dtype=torch.bool
        )
        chat.external_group_ids = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32)
        chat.refresh(shap=True)

        mask = chat.shap_values_mask
        assert torch.equal(mask, torch.tensor([False, True, False, True, False]))

    def test_shap_values_mask_uses_external_mask_when_no_groups(
        self, chat: BaseMllmChat
    ) -> None:
        """shap_values_mask should be intersected with external_shap_values_mask."""
        chat.text_tokens_no_system_mask = torch.ones(
            chat.input_tokens_num, dtype=torch.bool
        )
        chat.external_shap_values_mask = torch.tensor(
            [True, False, True, False, True], dtype=torch.bool
        )
        chat.refresh(shap=True)

        mask = chat.shap_values_mask
        assert torch.equal(mask, torch.tensor([True, False, True, False, True]))

    def test_decode_text_paths(self, chat: BaseMllmChat) -> None:
        """decode_text should support default and list-based token sources."""
        text_all = chat.decode_text()
        assert text_all == "0 1 2 3 4"

        text_list = chat.decode_text([torch.tensor([1]), torch.tensor([2])])
        assert text_list == "1 2"

    def test_decode_audio_returns_empty_when_undecodable(
        self, chat: BaseMllmChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """decode_audio should return empty bytes when connector cannot decode."""
        monkeypatch.setattr(chat, "_decode_audio", lambda _: None)
        assert chat.decode_audio(audio_tokens=torch.tensor([[1]])) == b""

    def test_decode_audio_list_calls_encoder(
        self, chat: BaseMllmChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """decode_audio should join lists and pass waveform to byte encoder."""
        monkeypatch.setattr(chat, "_decode_audio", lambda _: torch.zeros(1, 4))
        monkeypatch.setattr(
            "mllm_shap.connectors.base.chat.TorchAudioHandler.to_bytes",
            lambda *args, **kwargs: b"encoded",
        )
        result = chat.decode_audio(
            audio_tokens=[torch.tensor([[1]]), torch.tensor([[2]])]
        )
        assert result == b"encoded"

    def test_extend_token_roles_requires_active_speaker(
        self, chat: BaseMllmChat
    ) -> None:
        """_extend_token_roles should reject updates without current speaker."""
        chat.speaker = None
        with pytest.raises(ValueError, match="no active speaker"):
            chat._extend_token_roles(1)

    def test_after_add_noop_for_zero_tokens(self, chat: BaseMllmChat) -> None:
        """_after_add should no-op when zero tokens were added."""
        before_turns = chat.token_turns.clone()
        before_roles = chat.token_roles.clone()
        chat._after_add(num_tokens=0)
        assert torch.equal(chat.token_turns, before_turns)
        assert torch.equal(chat.token_roles, before_roles)

    def test_deepcopy_without_cache(self, chat: BaseMllmChat) -> None:
        """Deepcopy should preserve shared attrs and copy mutable state."""
        copied = deepcopy(chat)
        assert copied is not chat
        assert copied.token_sequences_to_exclude is chat.token_sequences_to_exclude
        assert copied.token_turns is not chat.token_turns

    def test_deepcopy_with_cache_rebinds_chat(self, chat: BaseMllmChat) -> None:
        """Deepcopy should clone cache and set its chat reference to the clone."""
        cache = MagicMock()
        cache.chat = chat
        chat.cache = cache

        copied = deepcopy(chat)
        assert copied.cache is not None
        assert copied.cache is not cache
        assert copied.cache.chat is copied

    def test_add_audio_with_transcript_validates_aligner_type(
        self, chat: BaseMllmChat
    ) -> None:
        """add_audio_with_transcript should reject non-aligner objects."""
        chat.speaker = Role.USER
        with pytest.raises(ValueError, match="SpectrogramGuidedAligner"):
            chat.add_audio_with_transcript(
                audio_content=b"abc",
                transcript="hello",
                aligner=cast(Any, object()),
            )

    def test_add_audio_with_transcript_rejects_empty_transcript(
        self, chat: BaseMllmChat
    ) -> None:
        """add_audio_with_transcript should reject empty transcript values."""
        chat.speaker = Role.USER
        aligner = object.__new__(SpectrogramGuidedAligner)
        with pytest.raises(ValueError, match="transcript must be a non-empty string"):
            chat.add_audio_with_transcript(
                audio_content=b"abc",
                transcript="",
                aligner=aligner,
            )

    def test_add_audio_with_transcript_rejects_mixed_mode(
        self, chat: BaseMllmChat
    ) -> None:
        """Cannot switch to transcript mode after regular audio was already added."""
        chat.speaker = Role.USER
        chat.audio_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        aligner = object.__new__(SpectrogramGuidedAligner)
        with pytest.raises(ValueError, match="Cannot add audio with transcript"):
            chat.add_audio_with_transcript(
                audio_content=b"abc",
                transcript="hello",
                aligner=aligner,
            )

    def test_add_audio_with_transcript_raises_when_no_segments(
        self, chat: BaseMllmChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """add_audio_with_transcript should fail if aligner returns no segments."""
        chat.speaker = Role.USER

        class _Aligner(SpectrogramGuidedAligner):
            pass

        aligner = object.__new__(_Aligner)
        monkeypatch.setattr(_Aligner, "__call__", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            "mllm_shap.connectors.base.chat.TorchAudioHandler.from_bytes",
            lambda *args, **kwargs: (torch.zeros(1, 16000), 16000),
        )
        with pytest.raises(ValueError, match="No audio segments were created"):
            chat.add_audio_with_transcript(
                audio_content=b"abc",
                transcript="hello",
                aligner=aligner,
            )

    def test_attach_audio_to_segments_validates_state(self, chat: BaseMllmChat) -> None:
        """attach_audio_to_segments should validate presence of segment metadata."""
        aligner = object.__new__(SpectrogramGuidedAligner)
        with pytest.raises(ValueError, match="No audio segments exist in this chat"):
            chat.attach_audio_to_segments(aligner=aligner, audio_content=b"abc")

        chat._audio_segments = {1: []}
        with pytest.raises(ValueError, match="No audio segments exist for turn 2"):
            chat.attach_audio_to_segments(
                aligner=aligner,
                audio_content=b"abc",
                turn_number=2,
            )

    def test_extend_tensor_runtime_error_path(
        self, chat: BaseMllmChat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Internal tensor extension failures should raise descriptive RuntimeError."""
        chat.speaker = Role.USER
        original_cat = torch.cat

        def _boom(
            values: list[torch.Tensor], *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            del values, args, kwargs
            raise RuntimeError("cat failed")

        monkeypatch.setattr(torch, "cat", _boom)
        with pytest.raises(RuntimeError, match="Failed to extend tensor 'token_roles'"):
            chat._extend_token_roles(1)
        monkeypatch.setattr(torch, "cat", original_cat)

    def test_from_chat_reconstructs_audio_turn_with_waveform_cache(self) -> None:
        """from_chat should rebuild segmented audio turns from cached source waveform."""
        base_chat = _BranchCoverageChat(
            input_tokens=torch.tensor([1, 2], dtype=torch.long),
            text_mask=torch.tensor([True, False], dtype=torch.bool),
        )
        base_chat.turn_number = 2
        base_chat.token_turns = torch.tensor([1, 2], dtype=torch.int16)
        base_chat.token_roles = torch.tensor(
            [Role.USER.value, Role.USER.value], dtype=torch.int8
        )
        base_chat.text_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        base_chat.audio_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        base_chat._audio_segments = {
            2: [
                AudioSegment(
                    token="A",
                    start_time=0.0,
                    end_time=0.01,
                    confidence=1.0,
                    audio_format="wav",
                    sample_rate=16000,
                    start_sample=0,
                    end_sample=4,
                )
            ]
        }
        base_chat._audio_waveforms = {2: (torch.zeros(1, 8), 16000, "wav")}

        new_chat = _BranchCoverageChat.from_chat(torch.tensor([True, True]), base_chat)
        assert isinstance(new_chat, _BranchCoverageChat)
        assert new_chat.turn_number == 2
        assert new_chat.input_tokens_num == 2

    def test_from_chat_raises_for_segment_sr_mismatch(self) -> None:
        """from_chat should fail if segment sample rate differs from cached waveform rate."""
        base_chat = _BranchCoverageChat(
            input_tokens=torch.tensor([1, 2], dtype=torch.long),
            text_mask=torch.tensor([True, False], dtype=torch.bool),
        )
        base_chat.turn_number = 2
        base_chat.token_turns = torch.tensor([1, 2], dtype=torch.int16)
        base_chat.token_roles = torch.tensor(
            [Role.USER.value, Role.USER.value], dtype=torch.int8
        )
        base_chat.text_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        base_chat.audio_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        base_chat._audio_segments = {
            2: [
                AudioSegment(
                    token="A",
                    start_time=0.0,
                    end_time=0.01,
                    confidence=1.0,
                    audio_format="wav",
                    sample_rate=22050,
                    start_sample=0,
                    end_sample=4,
                )
            ]
        }
        base_chat._audio_waveforms = {2: (torch.zeros(1, 8), 16000, "wav")}

        with pytest.raises(RuntimeError, match="sample rate mismatch"):
            _ = _BranchCoverageChat.from_chat(torch.tensor([True, True]), base_chat)

    def test_shap_values_mask_raises_when_segments_exceed_audio_tokens(self) -> None:
        """shap_values_mask should validate per-turn segment/token consistency."""
        chat = DummyChat(num_tokens=1)
        chat.turn_number = 1
        chat.token_turns = torch.tensor([1], dtype=torch.int16)
        chat.text_tokens_mask = torch.tensor([False], dtype=torch.bool)
        chat.audio_tokens_mask = torch.tensor([True], dtype=torch.bool)
        chat.text_tokens_no_system_mask = torch.tensor([], dtype=torch.bool)
        chat.audio_tokens_no_system_mask = torch.tensor([True], dtype=torch.bool)
        chat._audio_segments = {
            1: [
                AudioSegment(token="a", start_time=0.0, end_time=0.1, confidence=1.0),
                AudioSegment(token="b", start_time=0.1, end_time=0.2, confidence=1.0),
            ]
        }
        chat.refresh(shap=True)
        with pytest.raises(RuntimeError, match="larger then number of audio tokens"):
            _ = chat.shap_values_mask


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
        first_turn_tokens = [
            token for entry in conversation[0] for token in entry.content
        ]
        assert first_turn_tokens == ["0", "1"]
        # Second turn
        second_turn_tokens = [
            token for entry in conversation[1] for token in entry.content
        ]
        assert second_turn_tokens == ["2", "3"]

    def test_text_and_audio_modality(self) -> None:
        """Test get_conversation with mixed text and audio modalities."""
        chat = DummyChat(num_tokens=4)
        chat.turn_number = 1
        chat.token_turns = torch.tensor([1, 1, 1, 1], dtype=torch.int16)
        chat.token_roles = torch.tensor([Role.USER.value] * 4, dtype=torch.int8)
        chat.text_tokens_mask = torch.tensor(
            [True, True, False, False], dtype=torch.bool
        )

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

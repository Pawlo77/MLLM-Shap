"""Unit tests for the ChatEntry data model."""

from unittest.mock import MagicMock, patch

import pytest
from mllm_shap.connectors.base.chat_entry import ChatEntry
from mllm_shap.connectors.enums import ModalityFlag, Role


class DummyDisplayAudio:
    """Mock audio display function that returns a fake displayable object."""

    def __call__(self, audio_bytes: bytes):
        return f"<Audio len={len(audio_bytes)}>"


class TestChatEntry:
    """Unit tests for the ChatEntry data model."""

    @staticmethod
    @pytest.fixture
    def text_entry() -> ChatEntry:
        """Fixture for a text-based ChatEntry."""
        return ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=[Role.USER.value, Role.SYSTEM.value, Role.USER.value],
            content=["Who", "are", "you?"],
            shap_values=[0.1, 0.2, 0.3],
        )

    @staticmethod
    @pytest.fixture
    def audio_entry() -> ChatEntry:
        """Fixture for an audio-based ChatEntry."""
        return ChatEntry(
            content_type=ModalityFlag.AUDIO.value,
            roles=[Role.USER.value, Role.SYSTEM.value],
            content=[b"\x00\x01", b"\x02\x03"],
            shap_values=None,
        )

    def test_repr_for_text_entry(self, text_entry: ChatEntry) -> None:
        """Ensure __repr__ correctly represents text content."""
        result = repr(text_entry)
        assert "content_type=0" in result
        assert "Who, are, you?" in result
        assert "shap_values=[0.1, 0.2, 0.3]" in result
        assert "roles=[USER, SYSTEM, USER]" in result

    def test_repr_truncates_long_content(self) -> None:
        """Ensure very long text content is truncated in __repr__."""
        long_content = ["word" for _ in range(20)]
        entry = ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=[Role.USER.value] * 20,
            content=long_content,
            shap_values=None,
        )
        result = repr(entry)
        assert "..." in result
        assert "content='word, word" in result
        assert result.count("word") < len(long_content)

    def test_repr_for_audio_entry(self, audio_entry: ChatEntry) -> None:
        """Ensure __repr__ describes audio bytes length correctly."""
        result = repr(audio_entry)
        assert "Audio bytes of total length 4" in result
        assert "content_type" in result

    def test_repr_truncates_roles_when_many(self) -> None:
        """Roles list longer than five should be summarized."""
        roles = [
            Role.USER.value,
            Role.ASSISTANT.value,
            Role.SYSTEM.value,
            Role.USER.value,
            Role.ASSISTANT.value,
            Role.SYSTEM.value,
        ]
        entry = ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=roles,
            content=["hi"] * len(roles),
            shap_values=None,
        )
        result = repr(entry)
        assert "roles=[USER, ASSISTANT, ..., ASSISTANT, SYSTEM]" in result

    def test_repr_truncates_shap_values_when_many(self) -> None:
        """Long shap values list should be summarized."""
        shap_values = [float(i) for i in range(8)]
        entry = ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=[Role.USER.value] * len(shap_values),
            content=["token"] * len(shap_values),
            shap_values=shap_values,
        )
        result = repr(entry)
        assert "shap_values=[0.0, 1.0, ..., 6.0, 7.0]" in result

    def test_repr_escapes_newlines_in_text_content(self) -> None:
        """Newline characters should be escaped in repr output."""
        entry = ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=[Role.USER.value, Role.SYSTEM.value],
            content=["hello\nworld", "final"],
            shap_values=None,
        )
        result = repr(entry)
        assert "hello\\nworld" in result

    @patch("IPython.display.display")
    @patch("mllm_shap.connectors.base.chat_entry.display_audio")
    def test_display_audio_entry(
        self,
        mock_display_audio: MagicMock,
        mock_display: MagicMock,
        audio_entry: ChatEntry,
    ) -> None:
        """Ensure display() calls display_audio for audio entries."""
        mock_display_audio.return_value = "<Audio len=4>"

        audio_entry.display()

        mock_display_audio.assert_called_once()
        mock_display.assert_called_once_with("<Audio len=4>")
        byte_arg = mock_display_audio.call_args.args[0]
        assert byte_arg == b"\x00\x01\x02\x03"

    @patch("builtins.print")
    def test_display_text_entry(
        self, mock_print: MagicMock, text_entry: ChatEntry
    ) -> None:
        """Ensure display() prints textual content for text entries."""
        text_entry.display()
        printed = " ".join("".join(call.args[0]) for call in mock_print.call_args_list)
        assert "BY: USER, SYSTEM" in printed
        assert "TEXT CONTENT" in printed
        assert "Who are you?" in printed

    def test_str_delegates_to_repr(self, text_entry: ChatEntry) -> None:
        """Ensure __str__ delegates to __repr__."""
        assert str(text_entry) == repr(text_entry)

    @patch("IPython.display.display")
    @patch("mllm_shap.connectors.base.chat_entry.display_audio")
    @patch("builtins.print")
    def test_display_audio_entry_prints_and_joins_bytes(
        self,
        mock_print: MagicMock,
        mock_display_audio: MagicMock,
        mock_display: MagicMock,
    ) -> None:
        """Audio display should print headers and join all byte chunks."""
        entry = ChatEntry(
            content_type=ModalityFlag.AUDIO.value,
            roles=[Role.SYSTEM.value, Role.USER.value, Role.SYSTEM.value],
            content=[b"ab", b"cd", b"ef"],
            shap_values=None,
        )
        mock_display_audio.return_value = "<audio>"

        entry.display()

        calls = [call.args[0] for call in mock_print.call_args_list]
        assert calls[0] == "BY: USER, SYSTEM"
        assert calls[1] == "AUDIO CONTENT:"
        mock_display_audio.assert_called_once_with(b"abcdef")
        mock_display.assert_called_once_with("<audio>")

    def test_display_raises_when_lengths_mismatch(self) -> None:
        """display() should raise when roles and content lengths differ."""
        entry = ChatEntry(
            content_type=ModalityFlag.TEXT.value,
            roles=[Role.USER.value],
            content=["hello", "world"],
            shap_values=None,
        )
        with pytest.raises(ValueError, match="Number of roles must match"):
            entry.display()

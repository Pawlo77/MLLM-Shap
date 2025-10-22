"""Unit tests for the ChatEntry data model."""

import pytest
from unittest.mock import patch, MagicMock

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
        assert "content_type" in result
        assert "Who, are, you?" in result
        assert "shap_values=[0.1, 0.2, 0.3]" in result

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
        assert "roles=[USER, USER, ...," in result

    def test_repr_for_audio_entry(self, audio_entry: ChatEntry) -> None:
        """Ensure __repr__ describes audio bytes length correctly."""
        result = repr(audio_entry)
        assert "Audio bytes of total length 4" in result
        assert "content_type" in result

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

    @patch("builtins.print")
    def test_display_text_entry(self, mock_print: MagicMock, text_entry: ChatEntry) -> None:
        """Ensure display() prints textual content for text entries."""
        text_entry.display()
        printed = " ".join("".join(call.args[0]) for call in mock_print.call_args_list)
        assert "BY: USER, SYSTEM" in printed
        assert "TEXT CONTENT" in printed
        assert "Who are you?" in printed

    def test_str_delegates_to_repr(self, text_entry: ChatEntry) -> None:
        """Ensure __str__ delegates to __repr__."""
        assert str(text_entry) == repr(text_entry)

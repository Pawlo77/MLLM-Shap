"""Unit tests for mllm_shap.utils.audio module."""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.utils.audio import TorchAudioHandler, display_audio


class TestDisplayAudio:
    """Tests for display_audio utility function."""

    @patch("IPython.display.Audio")
    def test_display_audio_returns_ipython_audio(self, mock_audio: MagicMock) -> None:
        """Test that display_audio returns an IPython Audio object."""
        audio_bytes = b"fake_audio_data"
        mock_audio.return_value = "audio_object"

        result = display_audio(audio_bytes)

        mock_audio.assert_called_once_with(data=audio_bytes, autoplay=True)
        assert result == "audio_object"


class TestTorchAudioHandler:
    """Tests for TorchAudioHandler methods."""

    @staticmethod
    @pytest.fixture
    def dummy_waveform() -> torch.Tensor:
        """Fixture for sample mono waveform (1, 4)."""
        return torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    @staticmethod
    @pytest.fixture
    def stereo_waveform() -> torch.Tensor:
        """Fixture for sample stereo waveform (2, 4)."""
        return torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)

    # --- from_bytes ---

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_reads_and_returns_tensor(self, mock_read: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test that from_bytes loads bytes and returns waveform + sample rate."""
        waveform_np = dummy_waveform.squeeze(0).numpy()
        mock_read.return_value = (waveform_np, 24000)
        fake_bytes = b"audio_bytes"

        waveform, sample_rate = TorchAudioHandler.from_bytes(fake_bytes, audio_format="mp3")

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == dummy_waveform.shape
        assert sample_rate == 24000
        mock_read.assert_called_once()
        args, _ = mock_read.call_args
        assert isinstance(args[0], io.BytesIO)

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_converts_stereo_to_mono(self, mock_read: MagicMock, stereo_waveform: torch.Tensor) -> None:
        """Test that stereo waveform is averaged to mono."""
        mock_read.return_value = (stereo_waveform.T.numpy(), 16000)
        fake_bytes = b"audio_bytes"

        waveform, sr = TorchAudioHandler.from_bytes(fake_bytes, audio_format="wav")

        expected = stereo_waveform.mean(dim=0, keepdim=True)
        assert waveform.shape == (1, stereo_waveform.shape[1])
        torch.testing.assert_close(waveform, expected)
        assert sr == 16000

    @patch("mllm_shap.utils.audio.sf.read", side_effect=Exception("read error"))
    def test_from_bytes_raises_on_invalid_audio(self, mock_read: MagicMock) -> None:
        """Test that from_bytes raises if soundfile.read fails."""
        with pytest.raises(Exception, match="read error"):
            TorchAudioHandler.from_bytes(b"invalid", audio_format="mp3")

    @patch("mllm_shap.utils.audio.save")
    def test_to_bytes_writes_and_returns_bytes(self, mock_save: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test that to_bytes calls torchaudio.save and returns byte data."""

        def fake_save_side_effect(buffer, waveform, sr, format):
            buffer.write(b"encoded_audio")

        mock_save.side_effect = fake_save_side_effect

        result = TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=24000, audio_format="mp3")

        assert isinstance(result, bytes)
        assert b"encoded_audio" in result
        mock_save.assert_called_once()
        _, args, kwargs = mock_save.mock_calls[0]
        assert args[1] is dummy_waveform
        assert kwargs["format"] == "mp3"

    @patch("mllm_shap.utils.audio.save")
    def test_to_bytes_supports_custom_format(self, mock_save: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test that to_bytes works with non-default audio format."""
        mock_save.side_effect = lambda buffer, waveform, sr, format: buffer.write(b"flac_bytes")

        result = TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=44100, audio_format="flac")

        assert b"flac_bytes" in result
        mock_save.assert_called_once()
        _, _, kwargs = mock_save.mock_calls[0]
        assert kwargs["format"] == "flac"

    @patch("mllm_shap.utils.audio.save", side_effect=Exception("Save failed"))
    def test_to_bytes_raises_on_error(self, mock_save: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test that to_bytes raises exception if torchaudio.save fails."""
        with pytest.raises(Exception, match="Save failed"):
            TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=16000, audio_format="mp3")

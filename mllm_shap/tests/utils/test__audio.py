"""Tests for TorchAudioHandler utility class."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from mllm_shap.utils.audio import TorchAudioHandler


class TestTorchAudioHandler:
    """Tests for TorchAudioHandler methods."""

    @staticmethod
    @pytest.fixture
    def dummy_waveform() -> torch.Tensor:
        """Fixture for sample mono waveform - shape (1, 4)."""
        return torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    @staticmethod
    @pytest.fixture
    def stereo_waveform() -> torch.Tensor:
        """Fixture for sample stereo waveform - shape (2, 4)."""
        return torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)

    @patch("mllm_shap.utils.audio.load")
    def test_from_bytes_mono(self, mock_load: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test from_bytes returns waveform and sample rate correctly (mono)."""
        mock_load.return_value = (dummy_waveform, 24000)
        audio_bytes = b"fake_audio_bytes"

        waveform, sample_rate = TorchAudioHandler.from_bytes(audio_bytes, audio_format="mp3")

        mock_load.assert_called_once()
        assert torch.equal(waveform, dummy_waveform)
        assert sample_rate == 24000

    @patch("mllm_shap.utils.audio.load")
    def test_from_bytes_stereo_converts_to_mono(self, mock_load: MagicMock, stereo_waveform: torch.Tensor) -> None:
        """Test stereo waveform is converted to mono."""
        mock_load.return_value = (stereo_waveform, 16000)
        audio_bytes = b"fake_audio_bytes"

        waveform, sample_rate = TorchAudioHandler.from_bytes(audio_bytes, audio_format="wav")

        # Expected mono average of stereo channels
        expected_mono = stereo_waveform.mean(dim=0, keepdim=True)
        assert waveform.shape == (1, stereo_waveform.shape[1])
        assert torch.allclose(waveform, expected_mono)
        assert sample_rate == 16000

    @patch("mllm_shap.utils.audio.load")
    def test_from_bytes_handles_different_formats(self, mock_load: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test from_bytes works with custom format."""
        mock_load.return_value = (dummy_waveform, 44100)
        TorchAudioHandler.from_bytes(b"audio", audio_format="flac")
        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs["format"] == "flac"

    # --- to_bytes ---

    @patch("mllm_shap.utils.audio.save")
    def test_to_bytes_round_trip(self, mock_save, dummy_waveform):
        """Test to_bytes calls save and returns bytes."""

        # Mock write/read behavior
        def mock_save_side_effect(buffer, *_, **__):
            buffer.write(b"encoded_audio_data")

        mock_save.side_effect = mock_save_side_effect

        output = TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=24_000, audio_format="mp3")

        assert isinstance(output, bytes)
        assert b"encoded_audio_data" in output
        mock_save.assert_called_once()
        _, args, kwargs = mock_save.mock_calls[0]
        assert args[1] is dummy_waveform
        assert kwargs.get("format") == "mp3"

    @patch("mllm_shap.utils.audio.save")
    def test_to_bytes_handles_different_formats(self, mock_save: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test to_bytes works with non-default audio formats."""
        mock_save.side_effect = lambda buffer, waveform, sr, format: buffer.write(b"flac_data")
        out = TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=44_100, audio_format="flac")
        assert b"flac_data" in out
        mock_save.assert_called_once()
        _, _, kwargs = mock_save.mock_calls[0]
        assert kwargs["format"] == "flac"

    @patch("mllm_shap.utils.audio.save", side_effect=Exception("Save failed"))
    def test_to_bytes_raises_error(self, _: MagicMock, dummy_waveform: torch.Tensor) -> None:
        """Test to_bytes raises exception if save fails."""
        with pytest.raises(Exception, match="Save failed"):
            TorchAudioHandler.to_bytes(dummy_waveform, sample_rate=16_000, audio_format="mp3")

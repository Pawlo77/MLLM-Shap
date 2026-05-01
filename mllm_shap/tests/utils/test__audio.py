"""Unit tests for mllm_shap.utils.audio module."""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.utils.audio import TorchAudioHandler, display_audio, TARGET_SAMPLE_RATE


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
        return torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=torch.float32
        )

    # --- from_bytes ---

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_reads_and_returns_tensor(
        self, mock_read: MagicMock, dummy_waveform: torch.Tensor
    ) -> None:
        """Test that from_bytes loads bytes and returns waveform + sample rate."""
        waveform_np = dummy_waveform.squeeze(0).numpy()
        mock_read.return_value = (waveform_np, 24000)
        fake_bytes = b"audio_bytes"

        waveform, sample_rate = TorchAudioHandler.from_bytes(
            fake_bytes, audio_format="mp3"
        )

        assert isinstance(waveform, torch.Tensor)
        assert waveform.shape == dummy_waveform.shape
        assert sample_rate == 24000
        mock_read.assert_called_once()
        args, _ = mock_read.call_args
        assert isinstance(args[0], io.BytesIO)

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_converts_stereo_to_mono(
        self, mock_read: MagicMock, stereo_waveform: torch.Tensor
    ) -> None:
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

    @patch("mllm_shap.utils.audio.sf.write")
    def test_to_bytes_wav_writes_and_returns_bytes(
        self, mock_sf_write: MagicMock, dummy_waveform: torch.Tensor
    ) -> None:
        """WAV path uses soundfile.write to a buffer and returns its contents."""

        def fake_sf_write(buffer, data, sr, format, subtype):
            # Validate basic invariants
            assert format == "WAV"
            assert subtype == "PCM_16"
            assert sr == TARGET_SAMPLE_RATE
            # Write sentinel bytes so we can assert on the output
            assert hasattr(buffer, "write")
            buffer.write(b"encoded_wav_bytes")

        mock_sf_write.side_effect = fake_sf_write

        result = TorchAudioHandler.to_bytes(
            dummy_waveform, sample_rate=TARGET_SAMPLE_RATE, audio_format="wav"
        )

        assert isinstance(result, bytes)
        assert b"encoded_wav_bytes" in result
        # Ensure soundfile.write was called once
        mock_sf_write.assert_called_once()
        # Buffer was first positional arg
        args, kwargs = mock_sf_write.call_args
        assert isinstance(args[0], io.BytesIO)

    @patch("mllm_shap.utils.audio.AudioSegment")
    def test_to_bytes_mp3_exports_via_pydub_and_returns_bytes(
        self, mock_audio_segment_cls: MagicMock, dummy_waveform: torch.Tensor
    ) -> None:
        """MP3 path constructs an AudioSegment and exports with correct params."""
        # Prepare a mock segment instance with export writing known bytes
        mock_segment = MagicMock()

        def fake_export(buffer, format, bitrate):
            assert format == "mp3"
            assert bitrate == "192k"
            buffer.write(b"encoded_mp3_bytes")

        mock_segment.export.side_effect = fake_export
        mock_audio_segment_cls.return_value = mock_segment

        result = TorchAudioHandler.to_bytes(
            dummy_waveform, sample_rate=24_000, audio_format="mp3"
        )

        # Returned bytes come from export buffer
        assert isinstance(result, bytes)
        assert result == b"encoded_mp3_bytes"

        # Constructor was called with raw PCM16 bytes and correct metadata
        # We only check keyword args that are set in the util.
        _, kwargs = mock_audio_segment_cls.call_args
        assert kwargs["frame_rate"] == 24_000
        assert kwargs["sample_width"] == 2  # PCM16_SAMPLE_WIDTH_BYTES
        assert kwargs["channels"] == 1  # MONO_CHANNELS

        # Export was invoked once
        mock_segment.export.assert_called_once()

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

    def test_to_bytes_unsupported_format_raises(
        self, dummy_waveform: torch.Tensor
    ) -> None:
        """Should raise ValueError for unsupported audio format."""
        with pytest.raises(ValueError, match="Unsupported audio_format"):
            TorchAudioHandler.to_bytes(dummy_waveform, audio_format="ogg")

    def test_to_bytes_handles_nan_inf(self) -> None:
        """NaN and Inf values clamped to valid range (no crash)."""
        waveform = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.5]])
        result = TorchAudioHandler.to_bytes(waveform, audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_to_bytes_stereo_to_mono(self) -> None:
        """Stereo waveform (2, N) is averaged to mono in to_bytes."""
        stereo = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        result = TorchAudioHandler.to_bytes(stereo, audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_to_bytes_1d_waveform(self) -> None:
        """1D waveform passes through correctly."""
        waveform = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float32)
        result = TorchAudioHandler.to_bytes(waveform, audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_to_bytes_fallback_path_handles_higher_rank_waveform(self) -> None:
        """Higher-rank tensors should use fallback mean(dim=0) path without crashing."""
        waveform = torch.tensor(
            [[[0.1, -0.1, 0.2]], [[0.3, -0.3, 0.4]]], dtype=torch.float32
        )
        result = TorchAudioHandler.to_bytes(waveform, audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_2d_input_transposed(self, mock_read: MagicMock) -> None:
        """2D numpy array from sf.read gets transposed correctly."""
        # sf.read returns (samples, channels) format for multichannel
        import numpy as np

        stereo_np = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.4, 0.8]])
        mock_read.return_value = (stereo_np, 44100)
        waveform, sr = TorchAudioHandler.from_bytes(b"data", audio_format="wav")
        # Should be mono (1, 4) after T and mean
        assert waveform.shape == (1, 4)
        assert sr == 44100

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_single_channel_2d_keeps_channel_after_transpose(
        self, mock_read: MagicMock
    ) -> None:
        """(samples, 1) input should transpose to (1, samples) without averaging."""
        import numpy as np

        mono_np = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
        mock_read.return_value = (mono_np, 22050)

        waveform, sr = TorchAudioHandler.from_bytes(b"mono", audio_format="wav")

        assert waveform.shape == (1, 4)
        torch.testing.assert_close(
            waveform,
            torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        )
        assert sr == 22050

    @patch("mllm_shap.utils.audio.sf.read")
    def test_from_bytes_higher_rank_waveform_skips_transpose_branch(
        self, mock_read: MagicMock
    ) -> None:
        """Higher-rank waveform should bypass 1D/2D shape branches and still return tensor."""
        import numpy as np

        waveform_np = np.zeros((2, 3, 4), dtype=np.float32)
        mock_read.return_value = (waveform_np, 8000)

        waveform, sr = TorchAudioHandler.from_bytes(b"3d", audio_format="wav")

        assert waveform.dim() == 3
        assert waveform.shape[0] == 1
        assert sr == 8000

    def test_combine_empty_list_returns_empty(self) -> None:
        """Combine with no segments returns empty bytes."""
        result = TorchAudioHandler.combine([])
        assert result == b""

    def test_combine_single_segment(self) -> None:
        """Combine single segment returns valid audio bytes."""
        from mllm_shap.connectors.base.audio import AudioSegment

        # Create a real WAV segment
        waveform = torch.randn(1, 1000)
        wav_bytes = TorchAudioHandler.to_bytes(
            waveform, sample_rate=24000, audio_format="wav"
        )
        seg = AudioSegment(
            token="test",
            start_time=0.0,
            end_time=0.04,
            confidence=1.0,
            audio=wav_bytes,
            audio_format="wav",
        )
        result = TorchAudioHandler.combine([seg], target_audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_combine_resamples_different_rates(self) -> None:
        """Combine resamples to first segment's rate when rates differ."""
        from mllm_shap.connectors.base.audio import AudioSegment

        wf1 = torch.randn(1, 2400)
        wf2 = torch.randn(1, 1600)
        wav1 = TorchAudioHandler.to_bytes(wf1, sample_rate=24000, audio_format="wav")
        wav2 = TorchAudioHandler.to_bytes(wf2, sample_rate=16000, audio_format="wav")
        seg1 = AudioSegment(
            token="a",
            start_time=0.0,
            end_time=0.1,
            confidence=1.0,
            audio=wav1,
            audio_format="wav",
        )
        seg2 = AudioSegment(
            token="b",
            start_time=0.1,
            end_time=0.2,
            confidence=1.0,
            audio=wav2,
            audio_format="wav",
        )
        result = TorchAudioHandler.combine([seg1, seg2], target_audio_format="wav")
        assert isinstance(result, bytes)
        assert len(result) > len(wav1)  # combined is longer

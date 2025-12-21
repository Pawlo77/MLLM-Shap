"""Unit tests for SpectrogramGuidedAligner and AudioSegment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from mllm_shap.connectors.base.audio import AudioSegment, SpectrogramGuidedAligner


class TestAudioSegment:
    """Tests for the AudioSegment dataclass."""

    def test_duration_property(self) -> None:
        """Duration should be end_time - start_time."""
        seg = AudioSegment(token="hello", start_time=0.25, end_time=1.0, confidence=0.9)
        assert seg.duration == pytest.approx(0.75)

    def test_repr_includes_core_fields(self) -> None:
        """__repr__ should include token and rounded times."""
        seg = AudioSegment(token="world", start_time=0.0, end_time=0.5, confidence=0.5)
        rep = repr(seg)
        assert "AudioSegment(" in rep
        assert "token='world'" in rep
        assert "start=0.000" in rep
        assert "end=0.500" in rep


class TestSpectrogramGuidedAlignerInit:
    """Tests for SpectrogramGuidedAligner initialisation behaviour."""

    @patch("mllm_shap.connectors.base.audio.Wav2Vec2ForCTC.from_pretrained")
    @patch("mllm_shap.connectors.base.audio.Wav2Vec2Processor.from_pretrained")
    def test_init_loads_models_and_vocab(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
    ) -> None:
        """__init__ should load processor/model and set vocab and blank id."""
        device = torch.device("cpu")

        # Prepare a minimal tokenizer stub
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"A": 0, "B": 1}
        tokenizer.pad_token_id = 0

        processor = MagicMock()
        processor.tokenizer = tokenizer
        mock_processor_from_pretrained.return_value = processor

        model = MagicMock()
        model.to.return_value = model
        mock_model_from_pretrained.return_value = model

        aligner = SpectrogramGuidedAligner(device=device, model_name="dummy-model")

        # from_pretrained should be called once for processor and model
        assert mock_processor_from_pretrained.call_count == 1
        assert mock_model_from_pretrained.call_count == 1

        proc_args, proc_kwargs = mock_processor_from_pretrained.call_args
        model_args, model_kwargs = mock_model_from_pretrained.call_args
        # First positional argument is the model name we pass
        assert proc_args[0] == "dummy-model"
        assert model_args[0] == "dummy-model"
        # Revision is passed via keyword, defaulting to "main" in implementation
        assert proc_kwargs.get("revision") == "main"
        assert model_kwargs.get("revision") == "main"
        # Attributes are set from processor/model
        assert aligner.device is device
        assert aligner.vocab == {"A": 0, "B": 1}
        assert aligner.blank_id == 0

    @patch(
        "mllm_shap.connectors.base.audio.Wav2Vec2ForCTC.from_pretrained",
        side_effect=OSError("fail"),
    )
    @patch("mllm_shap.connectors.base.audio.Wav2Vec2Processor.from_pretrained")
    def test_init_raises_on_invalid_model(
        self,
        _mock_processor_from_pretrained: MagicMock,
        _mock_model_from_pretrained: MagicMock,
    ) -> None:
        """Invalid model name should surface as ValueError with a helpful message."""
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="Could not load"):
            SpectrogramGuidedAligner(device=device, model_name="non-existent-model")


class TestSpectrogramGuidedAlignerAlign:
    """Tests for the high-level align() pipeline."""

    @staticmethod
    def _make_aligner_with_mocks() -> SpectrogramGuidedAligner:
        """Create an aligner instance with model loading patched out."""

        with (
            patch(
                "mllm_shap.connectors.base.audio.Wav2Vec2Processor.from_pretrained"
            ) as proc_ft,
            patch(
                "mllm_shap.connectors.base.audio.Wav2Vec2ForCTC.from_pretrained"
            ) as model_ft,
        ):
            tokenizer = MagicMock()
            tokenizer.get_vocab.return_value = {"A": 0}
            tokenizer.pad_token_id = 0

            processor = MagicMock()
            processor.tokenizer = tokenizer
            proc_ft.return_value = processor

            model = MagicMock()
            model.to.return_value = model
            model_ft.return_value = model

            return SpectrogramGuidedAligner(
                device=torch.device("cpu"), model_name="dummy-model"
            )

    def test_align_runs_full_pipeline_and_returns_segments(self) -> None:
        """align should orchestrate helpers and return aggregated segments with audio attached."""
        aligner = self._make_aligner_with_mocks()

        dummy_waveform = torch.zeros(1, 16000)
        dummy_sr = 16_000
        dummy_tokens = [1, 2, 3]
        dummy_alignment = torch.tensor([1, 1, 2, 2, 3, 3])
        dummy_emissions = torch.zeros(6, 4)

        char_segments = [
            {"char": "A", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"char": "B", "start": 0.5, "end": 1.0, "confidence": 0.8},
        ]
        target_segments = ["AB"]
        final_segments = [
            AudioSegment(token="AB", start_time=0.0, end_time=1.0, confidence=0.85),
        ]

        with (
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__load_waveform_from_bytes",
                return_value=(dummy_waveform, dummy_sr),
            ) as mock_load,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__prepare_transcript",
                return_value=("AB", target_segments, "AB", dummy_tokens),
            ) as mock_prepare,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__perform_forced_alignment",
                return_value=(dummy_alignment, dummy_emissions),
            ) as mock_align,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__merge_tokens",
                return_value=[(1, 0, 2), (2, 2, 4), (3, 4, 6)],
            ) as mock_merge,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__refine_token_spans",
                return_value=char_segments,
            ) as mock_refine,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__aggregate_chars_to_segments",
                return_value=final_segments,
            ) as mock_aggregate,
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__attach_audio_to_segments",
            ) as mock_attach,
        ):
            out = aligner(b"fake-bytes", transcript="AB")

        # Pipeline wiring
        mock_load.assert_called_once_with(b"fake-bytes")
        mock_prepare.assert_called_once()
        mock_align.assert_called_once()
        mock_merge.assert_called_once()
        mock_refine.assert_called_once()
        mock_aggregate.assert_called_once()
        mock_attach.assert_called_once_with(final_segments, dummy_waveform, dummy_sr)

        # align should return the aggregated segments
        assert out is final_segments
        assert len(out) == 1
        assert isinstance(out[0], AudioSegment)

    def test_align_propagates_transcript_errors(self) -> None:
        """Errors from __prepare_transcript should bubble up as-is."""
        aligner = self._make_aligner_with_mocks()
        dummy_waveform = torch.zeros(1, 8000)
        dummy_sr = 8_000

        with (
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__load_waveform_from_bytes",
                return_value=(dummy_waveform, dummy_sr),
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__prepare_transcript",
                side_effect=ValueError("Transcript contains no valid characters"),
            ),
        ):
            with pytest.raises(ValueError, match="no valid characters"):
                aligner(b"invalid", transcript="@@@")

    def test_attach_audio_to_segments_writes_non_empty_audio(self) -> None:
        """__attach_audio_to_segments should populate non-empty audio bytes for each segment."""
        aligner = self._make_aligner_with_mocks()

        # Small mono waveform of 0.2s at 16kHz
        sr = 16_000
        num_samples = int(0.2 * sr)
        waveform = torch.zeros(1, num_samples)

        segments = [
            AudioSegment(token="a", start_time=0.0, end_time=0.01, confidence=1.0),
            AudioSegment(token="b", start_time=0.05, end_time=0.07, confidence=0.5),
        ]

        # Call the private helper via name-mangled attribute
        attach_fn = getattr(
            aligner,
            "_SpectrogramGuidedAligner__attach_audio_to_segments",
        )
        attach_fn(segments, waveform, sr)

        for seg in segments:
            assert isinstance(seg.audio, (bytes, bytearray))
            # At least some content should be written
            assert len(seg.audio) > 0

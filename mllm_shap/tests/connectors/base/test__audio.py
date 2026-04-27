"""Unit tests for SpectrogramGuidedAligner and AudioSegment."""

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

    def test_add_segment_raises_for_mismatched_tokens(self) -> None:
        """Adding segments with different tokens should fail."""
        a = AudioSegment(token="a", start_time=0.0, end_time=0.1, confidence=0.7)
        b = AudioSegment(token="b", start_time=0.1, end_time=0.2, confidence=0.8)
        with pytest.raises(ValueError, match="different tokens"):
            _ = a + b


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
        assert aligner.boundary_energy_weight == pytest.approx(0.8)
        assert aligner.boundary_flux_weight == pytest.approx(0.2)

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

    @patch("mllm_shap.connectors.base.audio.Wav2Vec2ForCTC.from_pretrained")
    @patch("mllm_shap.connectors.base.audio.Wav2Vec2Processor.from_pretrained")
    def test_init_raises_on_invalid_boundary_weights(
        self,
        mock_processor_from_pretrained: MagicMock,
        mock_model_from_pretrained: MagicMock,
    ) -> None:
        """Boundary refinement weights should be non-negative and not both zero."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"A": 0}
        tokenizer.pad_token_id = 0

        processor = MagicMock()
        processor.tokenizer = tokenizer
        mock_processor_from_pretrained.return_value = processor

        model = MagicMock()
        model.to.return_value = model
        mock_model_from_pretrained.return_value = model

        device = torch.device("cpu")
        with pytest.raises(ValueError, match="non-negative"):
            SpectrogramGuidedAligner(
                device=device,
                boundary_energy_weight=-0.1,
                boundary_flux_weight=1.0,
            )
        with pytest.raises(ValueError, match="greater than zero"):
            SpectrogramGuidedAligner(
                device=device,
                boundary_energy_weight=0.0,
                boundary_flux_weight=0.0,
            )


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
            # Test with attach_audio=True by passing waveform directly
            out = aligner(
                transcript="AB",
                waveform=dummy_waveform,
                original_sr=dummy_sr,
                attach_audio=True,
            )

        # Pipeline wiring
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

        with patch.object(
            SpectrogramGuidedAligner,
            "_SpectrogramGuidedAligner__prepare_transcript",
            side_effect=ValueError("Transcript contains no valid characters"),
        ):
            with pytest.raises(ValueError, match="no valid characters"):
                aligner(
                    transcript="@@@",
                    waveform=dummy_waveform,
                    original_sr=dummy_sr,
                )

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
        attach_fn(segments, waveform, sr, attach_audio=True)

        for seg in segments:
            assert isinstance(seg.audio, (bytes, bytearray))
            # At least some content should be written
            assert len(seg.audio) > 0

    def test_align_without_attach_audio_returns_empty_audio_bytes(self) -> None:
        """align with attach_audio=False (default) should return segments with empty audio bytes."""
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
                "_SpectrogramGuidedAligner__prepare_transcript",
                return_value=("AB", target_segments, "AB", dummy_tokens),
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__perform_forced_alignment",
                return_value=(dummy_alignment, dummy_emissions),
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__merge_tokens",
                return_value=[(1, 0, 2), (2, 2, 4), (3, 4, 6)],
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__refine_token_spans",
                return_value=char_segments,
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__aggregate_chars_to_segments",
                return_value=final_segments,
            ),
            patch.object(
                SpectrogramGuidedAligner,
                "_SpectrogramGuidedAligner__attach_audio_to_segments",
            ) as mock_attach,
        ):
            # Test with default attach_audio=False by passing waveform directly
            out = aligner(
                transcript="AB",
                waveform=dummy_waveform,
                original_sr=dummy_sr,
            )

        # __attach_audio_to_segments should NOT be called when attach_audio=False
        mock_attach.assert_not_called()

        # Segments should still be returned
        assert out is final_segments
        assert len(out) == 1
        assert isinstance(out[0], AudioSegment)
        # Audio bytes should be empty (default field value)
        assert out[0].audio == b""

    def test_attach_audio_to_segments_public_method(self) -> None:
        """attach_audio_to_segments public method should attach audio to existing segments."""
        aligner = self._make_aligner_with_mocks()

        # Create segments without audio
        segments = [
            AudioSegment(token="hello", start_time=0.0, end_time=0.5, confidence=0.9),
            AudioSegment(token="world", start_time=0.5, end_time=1.0, confidence=0.8),
        ]

        # Verify segments have no audio initially
        for seg in segments:
            assert seg.audio == b""

        # Create dummy audio
        sr = 16_000
        num_samples = int(1.0 * sr)
        waveform = torch.zeros(1, num_samples)

        # Attach audio using public method
        aligner.attach_audio_to_segments(
            segments=segments,
            waveform=waveform,
            original_sr=sr,
        )

        # Verify audio was attached
        for seg in segments:
            assert isinstance(seg.audio, (bytes, bytearray))
            assert len(seg.audio) > 0

    def test_normalize_text_strips_diacritics(self) -> None:
        """normalize_text should strip diacritics and normalize to uppercase alphanumeric."""
        # Test with diacritics
        assert SpectrogramGuidedAligner.normalize_text("Núñez") == "NUNEZ"
        assert SpectrogramGuidedAligner.normalize_text("café") == "CAFE"
        assert SpectrogramGuidedAligner.normalize_text("naïve") == "NAIVE"

        # Test with punctuation (should be removed)
        assert SpectrogramGuidedAligner.normalize_text("hello, world!") == "HELLOWORLD"

        # Test with mixed case
        assert SpectrogramGuidedAligner.normalize_text("HeLLo WoRLd") == "HELLOWORLD"

        # Test with spaces (should be removed by the method)
        assert SpectrogramGuidedAligner.normalize_text("hello world") == "HELLOWORLD"

    def test_prepare_transcript_handles_diacritics(self) -> None:
        """__prepare_transcript should strip diacritics for consistent alignment."""
        aligner = self._make_aligner_with_mocks()

        # Mock the vocab to include expected characters
        vocab_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ|"
        aligner.vocab = {c: i for i, c in enumerate(vocab_chars)}
        aligner.ctc_separator = "|"

        # Mock convert_tokens_to_ids to return the character index
        def mock_convert(c):
            return vocab_chars.index(c) if c in vocab_chars else -1

        aligner.processor.tokenizer.convert_tokens_to_ids = mock_convert

        # Test transcript with diacritics
        transcript = "Vasco Núñez"
        (
            full_transcript,
            target_segments,
            clean_text,
            valid_tokens,
        ) = aligner._SpectrogramGuidedAligner__prepare_transcript(transcript)

        # Verify normalization
        assert full_transcript == transcript
        assert target_segments == ["Vasco", "Núñez"]
        # clean_text should have diacritics stripped and spaces replaced with separator
        assert clean_text == "VASCO|NUNEZ"
        # valid_tokens should only include characters in vocab and all be integers
        assert all(isinstance(t, int) for t in valid_tokens)
        assert len(valid_tokens) == len("VASCO|NUNEZ")  # All characters should be valid

    def test_prepare_transcript_raises_when_no_valid_tokens(self) -> None:
        """Transcript with no vocab overlap should raise."""
        aligner = self._make_aligner_with_mocks()
        aligner.vocab = {"A": 1}
        aligner.processor.tokenizer.convert_tokens_to_ids = lambda c: -1
        with pytest.raises(ValueError, match="no valid characters"):
            _ = aligner._SpectrogramGuidedAligner__prepare_transcript("!!!")

    @patch("mllm_shap.connectors.base.audio.forced_align")
    def test_perform_forced_alignment_invokes_forced_align(
        self, mock_forced_align: MagicMock
    ) -> None:
        """Forced align wrapper should pass expected tensors and return alignment path."""
        aligner = self._make_aligner_with_mocks()
        waveform = torch.zeros(1, 1600)
        emissions = torch.randn(1, 5, 4)
        with patch.object(
            SpectrogramGuidedAligner,
            "_SpectrogramGuidedAligner__compute_emissions",
            return_value=emissions,
        ):
            mock_forced_align.return_value = (torch.tensor([[0, 1, 1, 2, 0]]), None)
            path, out_emissions = (
                aligner._SpectrogramGuidedAligner__perform_forced_alignment(
                    waveform, 16_000, [1, 2]
                )
            )
        assert path.tolist() == [0, 1, 1, 2, 0]
        assert out_emissions.shape == (5, 4)

    def test_merge_tokens_skips_blank_and_merges_runs(self) -> None:
        """Merge helper should produce non-blank token spans only."""
        aligner = self._make_aligner_with_mocks()
        spans = aligner._SpectrogramGuidedAligner__merge_tokens(
            torch.tensor([0, 1, 1, 0, 2, 2, 2, 0]), blank_id=0
        )
        assert spans == [(1, 1, 3), (2, 4, 7)]

    def test_set_segment_indices_enforces_min_duration_and_bounds(self) -> None:
        """Index attachment should clamp and enforce minimum duration."""
        aligner = self._make_aligner_with_mocks()
        waveform = torch.zeros(1, 100)
        segments = [
            AudioSegment(token="x", start_time=0.001, end_time=0.0015, confidence=1.0)
        ]
        out = aligner._SpectrogramGuidedAligner__set_segment_indices(
            segments, waveform, 1000
        )
        assert out.shape == (1, 100)
        assert segments[0].start_sample == 1
        # min duration=50 samples at sr=1000, clamped to waveform max 100
        assert segments[0].end_sample == 51

    def test_attach_audio_to_segments_requires_input(self) -> None:
        """Public audio attach helper should validate source inputs."""
        aligner = self._make_aligner_with_mocks()
        with pytest.raises(ValueError, match="Either audio_content or both waveform"):
            aligner.attach_audio_to_segments(segments=[])

    @patch("mllm_shap.connectors.base.audio.TorchAudioHandler.from_bytes")
    def test_attach_audio_to_segments_uses_audio_content(
        self, mock_from_bytes: MagicMock
    ) -> None:
        """Public attach helper should decode bytes when audio_content is provided."""
        aligner = self._make_aligner_with_mocks()
        waveform = torch.zeros(1, 1600)
        mock_from_bytes.return_value = (waveform, 16_000)
        segments = [
            AudioSegment(token="x", start_time=0.0, end_time=0.1, confidence=1.0)
        ]
        aligner.attach_audio_to_segments(
            segments=segments, audio_content=b"abc", audio_format="wav"
        )
        assert len(segments[0].audio) > 0

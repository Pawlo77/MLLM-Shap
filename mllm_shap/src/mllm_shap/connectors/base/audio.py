"""Spectrogram-Guided Forced Aligner using Wav2Vec2."""

import io
import unicodedata
import warnings
import wave
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, cast

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align
from transformers import PreTrainedTokenizerBase, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import logging as hf_logging

from ...utils.audio import TorchAudioHandler
from ...utils.logger import get_logger

hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]

logger: Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*forced_align has been deprecated.*")


UNICODE_CATEGORY_NONSPACING_MARK = "Mn"
ASCII_SPACE = " "


@dataclass
class AudioSegment:  # pylint: disable=too-many-instance-attributes
    """Represents a segment of audio aligned to a token/word."""

    token: str
    """The token/word associated with this segment."""

    start_time: float
    """Start time in seconds."""

    end_time: float
    """End time in seconds."""

    confidence: float
    """Confidence score of the alignment."""

    audio: bytes = field(default=b"")
    """Raw audio bytes for this segment."""

    audio_format: str = field(default="wav")
    """Audio format of the raw audio bytes."""

    sample_rate: int | None = field(default=None, repr=False)
    """Sample rate for `start_sample`/`end_sample` (if set)."""

    start_sample: int | None = field(default=None, repr=False)
    """Start sample index in the source waveform (if set)."""

    end_sample: int | None = field(default=None, repr=False)
    """End sample index in the source waveform (if set)."""

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        """String representation of the AudioSegment."""
        return (
            f"AudioSegment(token='{self.token}', start={self.start_time:.3f}, "
            f"end={self.end_time:.3f}, dur={self.duration:.3f}s)"
        )

    def __add__(self, other: "AudioSegment") -> "AudioSegment":
        """Combine two AudioSegments into one."""
        if self.token != other.token:
            raise ValueError("Cannot combine AudioSegments with different tokens.")
        combined_audio = self.audio + other.audio

        return AudioSegment(
            token=self.token,
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
            confidence=(self.confidence + other.confidence) / 2,
            audio=combined_audio,
            audio_format=self.audio_format,
        )


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class SpectrogramGuidedAligner:
    """
    Spectrogram-Guided Forced Aligner using Wav2Vec2 and Torchaudio.

    It takes raw audio bytes and a transcript (string or list of tokens)
    and produces time-aligned segments with refined boundaries.

    The alignment process consists of several phases:
    1.  **Acoustic Modeling:** A CTC model (Wav2Vec2) maps audio to character probabilities.
    2.  **Forced Alignment:** Dynamic programming finds the optimal alignment path.
    3.  **Boundary Refinement:** Spectrogram features (Energy & Flux) refine boundaries.
    4.  **Aggregation:** Character-level segments are grouped into user-defined tokens.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        device: torch.device,
        model_name: str = "facebook/wav2vec2-large-960h",
        model_revision: str = "main",
        sample_rate: int = 16000,
        ctc_separator: str = "|",
    ):
        """
        Initializes the aligner with the specified CTC model.

        Args:
            device: Torch device to run the model on (CPU or GPU).
            model_name: Hugging Face model name for the Wav2Vec2 CTC model.
            model_revision: Model revision or version to use.
            sample_rate: Expected sample rate for the model (default 16kHz).
            ctc_separator: Separator used in CTC decoding.
        """
        self.device = device
        self.sample_rate = sample_rate
        self.ctc_separator = ctc_separator

        logger.debug("Loading alignment model: %s on %s...", model_name, device)
        try:
            self.processor = cast(Any, Wav2Vec2Processor).from_pretrained(  # nosec B615
                model_name, revision=model_revision
            )
            model = Wav2Vec2ForCTC.from_pretrained(  # nosec B615
                model_name, revision=model_revision
            )
            self.model = cast(Wav2Vec2ForCTC, model.to(cast(Any, device)))
        except OSError as e:
            raise ValueError(
                f"Could not load '{model_name}'. Ensure it is a valid CTC model."
            ) from e

        self.tokenizer = cast(
            PreTrainedTokenizerBase, getattr(self.processor, "tokenizer")
        )
        self.vocab = self.tokenizer.get_vocab()
        self.blank_id = self.tokenizer.pad_token_id or 0

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for alignment by stripping diacritics and keeping only alphanumeric characters.

        This normalization is applied consistently to both the transcript used for forced alignment
        and the target segments used for aggregation to ensure matching.

        Args:
            text: The text to normalize.
        Returns:
            Normalized text (uppercase, no diacritics, only alphanumeric).
        """
        # Decompose Unicode characters (e.g., "ñ" -> "n" + combining tilde)
        # and filter out combining marks (diacritics)
        text_nfd = unicodedata.normalize("NFD", text)
        text_no_diacritics = "".join(
            char
            for char in text_nfd
            if unicodedata.category(char) != UNICODE_CATEGORY_NONSPACING_MARK
        )
        # Keep only alphanumeric characters and convert to uppercase
        return "".join(filter(str.isalnum, text_no_diacritics)).upper()

    def __compute_emissions(
        self, waveform: torch.Tensor, original_sr: int
    ) -> torch.Tensor:
        """
        Computes log-probability emissions from the acoustic model.

        Args:
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
        Returns:
            Emission log-probabilities tensor.
        """
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Model Forward Pass
        inputs = self.processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )

        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
            emissions = torch.log_softmax(logits, dim=-1)

        return emissions

    # pylint: disable=too-many-locals
    def __refine_boundary_smart(
        self, waveform: np.ndarray, sr: int, candidate_time: float
    ) -> float:
        """
        'Smart' Refinement using Energy and Spectral Flux.

        Args:
            waveform: Audio waveform as a numpy array.
            sr: Sampling rate of the waveform.
            candidate_time: Initial candidate boundary time in seconds.
        Returns:
            Refined boundary time in seconds.
        """
        window_samples = int(0.08 * sr)
        center_sample = int(candidate_time * sr)

        start_idx = max(0, center_sample - window_samples)
        end_idx = min(len(waveform), center_sample + window_samples)
        search_region = waveform[start_idx:end_idx]

        # Too short to analyze
        if len(search_region) < 256:  # pylint: disable=magic-value-comparison
            return candidate_time

        # Compute RMS Energy (Loudness)
        rms = librosa.feature.rms(y=search_region, frame_length=256, hop_length=64)[0]

        # Compute Spectral Flux (Change)
        stft = np.abs(librosa.stft(search_region, n_fft=256, hop_length=64))
        flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
        # Pad flux to match rms length
        flux = np.pad(flux, (0, len(rms) - len(flux)), mode="constant")

        # Normalize to [0, 1]
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-9)

        # Weighted cost function: prioritize silence (low RMS) and stability (low Flux)
        cost = 0.8 * rms_norm + 0.2 * flux_norm
        min_idx = np.argmin(cost)

        refined_sample = start_idx + (min_idx * 64)
        return refined_sample / sr

    def __save_wav_mem(self, tensor: torch.Tensor, sample_rate: int) -> bytes:
        """
        Saves a waveform tensor to WAV format in memory.

        Args:
            tensor: Audio waveform tensor.
            sample_rate: Sampling rate for the WAV file.
        Returns:
            Raw WAV bytes.
        """
        src = tensor.cpu()
        if src.dim() == 1:
            src = src.unsqueeze(0)

        n_channels = src.shape[0]
        # Convert float32 to int16 PCM
        src = (src * 32767).clamp(-32768, 32767).to(torch.int16)
        src = src.t().numpy()  # type: ignore[assignment]

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(n_channels)  # pylint: disable=no-member
            wav_file.setsampwidth(2)  # pylint: disable=no-member
            wav_file.setframerate(sample_rate)  # pylint: disable=no-member
            wav_file.writeframes(src.tobytes())  # type: ignore[attr-defined] # pylint: disable=no-member

        return buffer.getvalue()

    def __merge_tokens(
        self, alignment_path: torch.Tensor, blank_id: int
    ) -> list[tuple[int, int, int]]:
        """
        Merges frame-level alignment into (token, start, end) spans.

        Args:
            alignment_path: Tensor of token IDs aligned per frame.
            blank_id: ID of the blank token in CTC.
        Returns:
            list of (token_id, start_frame, end_frame) tuples.
        """
        path = alignment_path.tolist()

        spans = []
        current_token = None
        start_frame = 0
        for i, token in enumerate(path):
            if token != current_token:
                if current_token is not None and current_token != blank_id:
                    spans.append((current_token, start_frame, i))
                current_token = token
                start_frame = i

        # Final span
        if current_token is not None and current_token != blank_id:
            spans.append((current_token, start_frame, len(path)))

        return spans

    def __prepare_transcript(
        self, transcript: str | list[str]
    ) -> tuple[str, list[str], str, list[int]]:
        """
        Prepares the transcript for alignment.

        Args:
            transcript: Either a single string or a list of tokens.
        Returns:
            tuple containing full transcript, target segments, clean text, and valid token IDs.
        """
        if isinstance(transcript, str):
            full_transcript = transcript
            target_segments = transcript.split()
        else:
            target_segments = transcript
            full_transcript = " ".join(transcript)

        # Normalize text: strip diacritics, uppercase, keep alphanumeric + spaces
        # Then replace spaces with CTC separator
        text_upper = full_transcript.upper()
        # Strip diacritics while preserving spaces
        text_nfd = unicodedata.normalize("NFD", text_upper)
        text_no_diacritics = "".join(
            char for char in text_nfd
            if unicodedata.category(char) != UNICODE_CATEGORY_NONSPACING_MARK  # Remove combining marks (diacritics)
        )
        # Keep only alphanumeric and spaces
        text_clean = "".join(
            c for c in text_no_diacritics if c.isalnum() or c == ASCII_SPACE
        )
        # Replace spaces with CTC separator
        clean_text = text_clean.replace(ASCII_SPACE, self.ctc_separator)

        valid_tokens = [
            cast(int, self.tokenizer.convert_tokens_to_ids(c))
            for c in clean_text
            if c in self.vocab
        ]

        if not valid_tokens:
            raise ValueError("Transcript contains no valid characters for this model.")

        return full_transcript, target_segments, clean_text, valid_tokens

    def __perform_forced_alignment(
        self, waveform: torch.Tensor, original_sr: int, valid_tokens: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forced alignment using Torchaudio's forced_align.
        Ensures CPU fallback for compatibility.

        Returns:
            tuple of (alignment_path, emission_log_probs).
        """
        emissions_gpu = self.__compute_emissions(waveform, original_sr).squeeze(0)

        # Move to CPU for forced_align to support MPS (Apple Silicon)
        emissions_cpu = emissions_gpu.unsqueeze(0).cpu()
        targets_cpu = torch.tensor([valid_tokens], dtype=torch.int32).cpu()
        emission_lens_cpu = torch.tensor([emissions_gpu.size(0)]).cpu()
        target_lens_cpu = torch.tensor([len(valid_tokens)]).cpu()

        aligned_tokens, _ = forced_align(
            emissions_cpu,
            targets_cpu,
            emission_lens_cpu,
            target_lens_cpu,
            blank=self.blank_id,
        )
        alignment_path = aligned_tokens[0]
        return alignment_path, emissions_gpu

    # pylint: disable=too-many-locals
    def __refine_token_spans(
        self,
        token_spans: list[tuple[int, int, int]],
        emissions_gpu: torch.Tensor,
        waveform: torch.Tensor,
        original_sr: int,
    ) -> list[dict[str, str | float]]:
        """
        Refines token boundary times using acoustic features.

        Args:
            token_spans: list of (token_id, start_frame, end_frame) tuples.
            emissions_gpu: Emission log-probabilities tensor.
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
        Returns:
            list of refined character segments with start/end times and confidence.
        """
        ratio = waveform.size(1) / emissions_gpu.size(0)
        numpy_wave = waveform.cpu().numpy().squeeze()

        refined_chars: list[dict[str, str | float]] = []
        for sp_token, sp_start, sp_end in token_spans:
            t_start = (sp_start * ratio) / original_sr
            t_end = (sp_end * ratio) / original_sr

            # Calculate confidence from GPU emissions (fast)
            conf = torch.exp(emissions_gpu[sp_start:sp_end, sp_token]).mean().item()

            # Refine boundaries
            r_start = self.__refine_boundary_smart(numpy_wave, original_sr, t_start)
            r_end = self.__refine_boundary_smart(numpy_wave, original_sr, t_end)

            char = cast(str, self.tokenizer.convert_ids_to_tokens(sp_token))

            refined_chars.append(
                {"char": char, "start": r_start, "end": r_end, "confidence": conf}
            )

        return refined_chars

    def __aggregate_chars_to_segments(
        self,
        char_segments: list[dict[str, str | float]],
        target_segments: list[str],
    ) -> list[AudioSegment]:
        """
        Aggregates character-level alignment into the provided target segments.

        Args:
            char_segments: list of character-level segments with timings.
            target_segments: list of target tokens/words to align to.
        Returns:
            list of aggregated AudioSegment objects.
        """
        # Aggregate chars to words/tokens
        final_segments = []
        current_char_idx = 0

        for segment_text in target_segments:
            # Normalize target segment for matching - use the same normalization as __prepare_transcript
            clean_target = self.normalize_text(segment_text)

            if not clean_target:
                continue

            start_time = None
            end_time = None
            confs = []
            found_chars = 0

            # Greedy matching
            while found_chars < len(clean_target) and current_char_idx < len(
                char_segments
            ):
                seg = char_segments[current_char_idx]
                seg_char = cast(str, seg["char"]).replace(self.ctc_separator, "")

                if seg_char == clean_target[found_chars]:
                    if start_time is None:
                        start_time = seg["start"]
                    end_time = seg["end"]
                    confs.append(seg["confidence"])
                    found_chars += 1

                current_char_idx += 1

            if start_time is not None:
                # Fallback for single characters
                if end_time is None:
                    end_time = cast(float, start_time) + 0.1

                avg_conf = sum(confs) / len(confs) if confs else 0.0  # type: ignore

                final_segments.append(
                    AudioSegment(
                        token=segment_text,
                        start_time=cast(float, start_time),
                        end_time=cast(float, end_time),
                        confidence=avg_conf,
                        audio_format="wav",
                    )
                )

        return final_segments

    def __set_segment_indices(
        self,
        final_segments: list[AudioSegment],
        waveform: torch.Tensor,
        original_sr: int,
    ) -> torch.Tensor:
        """
        Attach sample indices to segments based on timing information.

        Args:
            final_segments: list of AudioSegment objects.
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
        Returns:
            CPU waveform tensor in shape [channels, samples].
        """
        cpu_waveform = waveform.cpu()
        if cpu_waveform.dim() == 1:
            cpu_waveform = cpu_waveform.unsqueeze(0)

        for seg in final_segments:
            start_sample = int(seg.start_time * original_sr)
            end_sample = int(seg.end_time * original_sr)

            # Ensure minimum duration (50ms) to avoid degenerate files
            min_duration = int(0.05 * original_sr)
            if end_sample - start_sample < min_duration:
                end_sample = start_sample + min_duration

            # Clamp boundaries
            start_sample = max(0, start_sample)
            end_sample = min(cpu_waveform.size(1), end_sample)

            seg.sample_rate = original_sr
            seg.start_sample = start_sample
            seg.end_sample = end_sample

        return cpu_waveform

    def __attach_audio_to_segments(
        self,
        final_segments: list[AudioSegment],
        waveform: torch.Tensor,
        original_sr: int,
        attach_audio: bool = True,
    ) -> None:
        """
        Attaches segment indices (always) and optionally raw audio bytes.

        Args:
            final_segments: list of AudioSegment objects.
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
        """
        cpu_waveform = self.__set_segment_indices(
            final_segments, waveform, original_sr
        )

        if not attach_audio:
            return

        for seg in final_segments:
            segment_tensor = cpu_waveform[:, seg.start_sample : seg.end_sample]  # noqa: E203
            seg.audio = self.__save_wav_mem(segment_tensor, original_sr)

    def attach_audio_to_segments(
        self,
        segments: list[AudioSegment],
        audio_content: bytes | None = None,
        waveform: torch.Tensor | None = None,
        original_sr: int | None = None,
        audio_format: str = "mp3",
    ) -> None:
        """
        Attach raw audio bytes to existing AudioSegment objects.

        This is a helper method to materialize audio bytes for segments that were
        created without audio attachment (i.e., with attach_audio=False).

        Args:
            segments: list of AudioSegment objects to attach audio to.
            audio_content: Raw audio bytes. Either this or (waveform, original_sr) must be provided.
            waveform: Audio waveform tensor. Either this and original_sr or audio_content must be provided.
            original_sr: Original sampling rate of the waveform.
            audio_format: Format of the audio content (default is "mp3").
        Raises:
            ValueError: If neither audio_content nor (waveform, original_sr) are provided.
        """
        if audio_content is None and (waveform is None or original_sr is None):
            raise ValueError(
                "Either audio_content or both waveform and original_sr must be provided."
            )
        if audio_content is not None:
            waveform, original_sr = TorchAudioHandler.from_bytes(
                audio_content, audio_format=audio_format
            )
        original_sr = cast(int, original_sr)
        waveform = cast(torch.Tensor, waveform)

        # Materialize bytes on demand.
        self.__attach_audio_to_segments(
            segments, waveform, original_sr, attach_audio=True
        )

    def __call__(
        self,
        transcript: str | list[str],
        audio_content: bytes | None = None,
        waveform: torch.Tensor | None = None,
        original_sr: int | None = None,
        audio_format: str = "mp3",
        attach_audio: bool = False,
    ) -> list[AudioSegment]:
        """
        Main pipeline execution.

        Args:
            audio_content: Raw audio bytes.
            transcript: Either a single string (will be split by spaces)
                OR a list of tokens/utterances to align to.
            audio_format: Format of the audio content (default is "mp3").
            attach_audio: Whether to attach raw audio bytes to each segment (default is False).
                If False, segments will have empty audio bytes to save memory.
        Returns:
            list of aligned AudioSegment objects.
        """
        if audio_content is None and (waveform is None or original_sr is None):
            raise ValueError(
                "Either audio_content or both waveform and original_sr must be provided."
            )
        if audio_content is not None:
            waveform, original_sr = TorchAudioHandler.from_bytes(
                audio_content, audio_format=audio_format
            )
        original_sr = cast(int, original_sr)
        waveform = cast(torch.Tensor, waveform)

        _, target_segments, clean_text, valid_tokens = self.__prepare_transcript(
            transcript
        )

        logger.debug("Aligning to transcript: '%s'", clean_text)

        alignment_path, emissions_gpu = self.__perform_forced_alignment(
            waveform, original_sr, valid_tokens
        )

        token_spans = self.__merge_tokens(alignment_path, self.blank_id)

        refined_chars = self.__refine_token_spans(
            token_spans, emissions_gpu, waveform, original_sr
        )

        final_segments = self.__aggregate_chars_to_segments(
            refined_chars, target_segments
        )

        if attach_audio:
            self.__attach_audio_to_segments(
                final_segments, waveform, original_sr
            )
        else:
            self.__set_segment_indices(final_segments, waveform, original_sr)

        return final_segments

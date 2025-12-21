"""Spectrogram-Guided Forced Aligner using Wav2Vec2."""

import io
import warnings
import wave
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Tuple, Union, cast

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import logging as hf_logging

from ...utils.logger import get_logger

hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]

logger: Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*forced_align has been deprecated.*")


@dataclass
class AudioSegment:
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


# pylint: disable=too-few-public-methods
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
            self.processor = Wav2Vec2Processor.from_pretrained(  # nosec B615
                model_name, revision=model_revision
            )  # type: ignore[no-untyped-call]
            self.model = Wav2Vec2ForCTC.from_pretrained(  # nosec B615
                model_name, revision=model_revision
            ).to(device)
        except OSError as e:
            raise ValueError(
                f"Could not load '{model_name}'. Ensure it is a valid CTC model."
            ) from e

        self.vocab = self.processor.tokenizer.get_vocab()  # pylint: disable=no-member
        self.blank_id = self.processor.tokenizer.pad_token_id or 0  # pylint: disable=no-member

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
    ) -> List[Tuple[int, int, int]]:
        """
        Merges frame-level alignment into (token, start, end) spans.

        Args:
            alignment_path: Tensor of token IDs aligned per frame.
            blank_id: ID of the blank token in CTC.
        Returns:
            List of (token_id, start_frame, end_frame) tuples.
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

    def __load_waveform_from_bytes(
        self, audio_bytes: bytes
    ) -> Tuple[torch.Tensor, int]:
        """
        Loads waveform from bytes.

        Args:
            audio_bytes: Raw audio data in bytes.
        Returns:
            Tuple of (waveform tensor, original sample rate).
        """
        waveform, original_sr = torchaudio.load(io.BytesIO(audio_bytes))
        return waveform, original_sr

    def __prepare_transcript(
        self, transcript: Union[str, List[str]]
    ) -> Tuple[str, List[str], str, List[int]]:
        """
        Prepares the transcript for alignment.

        Args:
            transcript: Either a single string or a list of tokens.
        Returns:
            Tuple containing full transcript, target segments, clean text, and valid token IDs.
        """
        if isinstance(transcript, str):
            full_transcript = transcript
            target_segments = transcript.split()
        else:
            target_segments = transcript
            full_transcript = "".join(transcript)

        # Normalize text (UPPER) and handle separators
        clean_text = full_transcript.upper().replace(" ", self.ctc_separator)
        valid_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids(c)  # pylint: disable=no-member
            for c in clean_text
            if c in self.vocab
        ]

        if not valid_tokens:
            raise ValueError("Transcript contains no valid characters for this model.")

        return full_transcript, target_segments, clean_text, valid_tokens

    def __perform_forced_alignment(
        self, waveform: torch.Tensor, original_sr: int, valid_tokens: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forced alignment using Torchaudio's forced_align.
        Ensures CPU fallback for compatibility.

        Returns:
            Tuple of (alignment_path, emission_log_probs).
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
        token_spans: List[Tuple[int, int, int]],
        emissions_gpu: torch.Tensor,
        waveform: torch.Tensor,
        original_sr: int,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Refines token boundary times using acoustic features.

        Args:
            token_spans: List of (token_id, start_frame, end_frame) tuples.
            emissions_gpu: Emission log-probabilities tensor.
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
        Returns:
            List of refined character segments with start/end times and confidence.
        """
        ratio = waveform.size(1) / emissions_gpu.size(0)
        numpy_wave = waveform.cpu().numpy().squeeze()

        refined_chars = []
        for sp_token, sp_start, sp_end in token_spans:
            t_start = (sp_start * ratio) / original_sr
            t_end = (sp_end * ratio) / original_sr

            # Calculate confidence from GPU emissions (fast)
            conf = torch.exp(emissions_gpu[sp_start:sp_end, sp_token]).mean().item()

            # Refine boundaries
            r_start = self.__refine_boundary_smart(numpy_wave, original_sr, t_start)
            r_end = self.__refine_boundary_smart(numpy_wave, original_sr, t_end)

            char = self.processor.tokenizer.convert_ids_to_tokens(sp_token)  # pylint: disable=no-member

            refined_chars.append(
                {"char": char, "start": r_start, "end": r_end, "confidence": conf}
            )

        return refined_chars

    def __aggregate_chars_to_segments(
        self,
        char_segments: List[Dict[str, Union[str, float]]],
        target_segments: List[str],
    ) -> List[AudioSegment]:
        """
        Aggregates character-level alignment into the provided target segments.

        Args:
            char_segments: List of character-level segments with timings.
            target_segments: List of target tokens/words to align to.
        Returns:
            List of aggregated AudioSegment objects.
        """
        # Aggregate chars to words/tokens
        final_segments = []
        current_char_idx = 0

        for segment_text in target_segments:
            # Normalize target segment for matching
            clean_target = "".join(filter(str.isalnum, segment_text)).upper()

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
                    )
                )

        return final_segments

    def __attach_audio_to_segments(
        self,
        final_segments: List[AudioSegment],
        waveform: torch.Tensor,
        original_sr: int,
    ) -> None:
        """
        Attaches raw audio bytes to each segment.

        Args:
            final_segments: List of AudioSegment objects.
            waveform: Audio waveform tensor.
            original_sr: Original sampling rate of the waveform.
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

            segment_tensor = cpu_waveform[:, start_sample:end_sample]
            seg.audio = self.__save_wav_mem(segment_tensor, original_sr)

    def __call__(
        self, audio_bytes: bytes, transcript: Union[str, List[str]]
    ) -> List[AudioSegment]:
        """
        Main pipeline execution.

        Args:
            audio_bytes: Raw audio data.
            transcript: Either a single string (will be split by spaces)
                OR a list of tokens/utterances to align to.
        Returns:
            List of aligned AudioSegment objects.
        """
        waveform, original_sr = self.__load_waveform_from_bytes(audio_bytes)

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

        self.__attach_audio_to_segments(final_segments, waveform, original_sr)

        return final_segments

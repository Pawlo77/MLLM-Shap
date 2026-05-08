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

hf_logging.set_verbosity_error()

logger: Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*forced_align has been deprecated.*")


ASCII_SPACE: str = " "
"""ASCII space character, used for transcript normalization and CTC blank token replacement."""
UNICODE_CATEGORY_NONSPACING_MARK: str = "Mn"
"""Unicode category for non-spacing marks, used to strip diacritics during transcript normalization."""
_SILENCE_THRESHOLD_RATIO: float = 0.5
"""When refining boundaries, the minimum RMS must be less than this ratio of the mean
 RMS in the search window to be accepted as a valid silence point. Prevents false positives in continuous speech."""
_MIN_REFINE_SAMPLES: int = 256
"""Minimum number of audio samples required in the boundary refinement search region to perform analysis.
Prevents unreliable refinements in very short segments."""


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


class SpectrogramGuidedAligner:
    """
    Spectrogram-Guided Forced Aligner using Wav2Vec2 and Torchaudio.

    It takes raw audio bytes and a transcript (string or list of tokens)
    and produces time-aligned segments with refined boundaries.

    The alignment process consists of several phases:
    1.  **Acoustic Modeling:** A CTC model (Wav2Vec2) maps audio to character probabilities.
    2.  **Forced Alignment:** Dynamic programming finds the optimal alignment path.
    3.  **Boundary Refinement:** Spectrogram features (Energy & Flux) refine boundaries,
        using gap midpoints as search anchors and joint negotiation to prevent inversions.
    4.  **Aggregation:** Character-level segments are grouped into user-defined tokens.
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "facebook/wav2vec2-large-960h",
        model_revision: str = "main",
        sample_rate: int = 16000,
        ctc_separator: str = "|",
        boundary_energy_weight: float = 0.8,
        boundary_flux_weight: float = 0.2,
    ):
        """Initialize the aligner with the specified configuration.

        Args:
            device: The torch device to run the model on.
            model_name: The Hugging Face model name for the Wav2Vec2 CTC model.
            model_revision: The revision of the model to load.
            sample_rate: The sample rate to use for audio processing.
            ctc_separator: The character used to replace spaces in the transcript for CTC processing.
            boundary_energy_weight: Weight for energy in boundary refinement (0 to 1).
            boundary_flux_weight: Weight for spectral flux in boundary refinement (0 to 1).
        """
        if boundary_energy_weight < 0 or boundary_flux_weight < 0:
            raise ValueError("Boundary refinement weights must be non-negative.")
        if boundary_energy_weight + boundary_flux_weight <= 0:
            raise ValueError(
                "At least one boundary refinement weight must be greater than zero."
            )

        self.device = device
        self.sample_rate = sample_rate
        self.ctc_separator = ctc_separator
        self.boundary_energy_weight = float(boundary_energy_weight)
        self.boundary_flux_weight = float(boundary_flux_weight)

        logger.debug("Loading alignment model: %s on %s...", model_name, device)
        try:
            self.processor = cast(Any, Wav2Vec2Processor).from_pretrained(
                model_name, revision=model_revision
            )
            model = Wav2Vec2ForCTC.from_pretrained(model_name, revision=model_revision)
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
        """Normalize text by removing diacritics, non-alphanumeric characters, and converting to uppercase."""
        text_nfd = unicodedata.normalize("NFD", text)
        text_no_diacritics = "".join(
            char
            for char in text_nfd
            if unicodedata.category(char) != UNICODE_CATEGORY_NONSPACING_MARK
        )
        return "".join(filter(str.isalnum, text_no_diacritics)).upper()

    def __compute_emissions(
        self, waveform: torch.Tensor, original_sr: int
    ) -> torch.Tensor:
        """Compute the emission probabilities from the audio waveform using the Wav2Vec2 model."""
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        inputs = self.processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )

        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
            emissions = torch.log_softmax(logits, dim=-1)

        return emissions

    def __refine_boundary_smart(
        self,
        waveform: np.ndarray,
        sr: int,
        candidate_time: float,
        left_time: float | None = None,
        right_time: float | None = None,
    ) -> tuple[float, bool]:
        """
        Refine a single boundary timestamp using energy and spectral flux.

        Fix 1: When left_time and right_time are supplied (the raw endpoints of
        the gap between two adjacent character spans), the search window is
        centred on their midpoint so that long silences are searched
        symmetrically rather than being anchored to one character's tail.

        Fix 3: If no frame in the window is clearly quieter than the window
        mean (no genuine silence), the raw candidate_time is returned unchanged
        rather than forcing a cut inside an active phoneme.

        Fix 4: Returns a boolean indicating whether refinement was actually
        applied (False when the search region was too short or no silence was
        found).

        Args:
            waveform: Full audio waveform as a 1-D numpy array.
            sr: Sampling rate.
            candidate_time: Raw CTC boundary estimate in seconds.
            left_time: Start of the blank gap (seconds). Optional.
            right_time: End of the blank gap (seconds). Optional.

        Returns:
            (refined_time_seconds, was_refined)
        """
        # use gap midpoint as centre when gap endpoints are known.
        if left_time is not None and right_time is not None:
            center_time = (left_time + right_time) / 2.0
            half_window = (right_time - left_time) / 2.0 + 0.04  # gap ± 40 ms margin
        else:
            center_time = candidate_time
            half_window = 0.08  # original ±80 ms

        window_samples = int(half_window * sr)
        center_sample = int(center_time * sr)

        start_idx = max(0, center_sample - window_samples)
        end_idx = min(len(waveform), center_sample + window_samples)
        search_region = waveform[start_idx:end_idx]

        # flag when region is too short to analyse.
        if len(search_region) < _MIN_REFINE_SAMPLES:
            return candidate_time, False

        rms = librosa.feature.rms(y=search_region, frame_length=256, hop_length=64)[0]
        stft = np.abs(librosa.stft(search_region, n_fft=256, hop_length=64))
        flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
        flux = np.pad(flux, (0, len(rms) - len(flux)), mode="constant")

        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-9)

        cost = (
            self.boundary_energy_weight * rms_norm
            + self.boundary_flux_weight * flux_norm
        )
        min_idx = int(np.argmin(cost))

        # only accept the refined position if it is genuinely quieter
        # than the window mean.  If no such frame exists (continuous speech,
        # no inter-word pause), fall back to the raw candidate.
        min_rms = rms[min_idx]
        mean_rms = float(np.mean(rms))
        if min_rms > _SILENCE_THRESHOLD_RATIO * mean_rms:
            return candidate_time, False

        refined_sample = start_idx + (min_idx * 64)
        return refined_sample / sr, True

    def __save_wav_mem(self, tensor: torch.Tensor, sample_rate: int) -> bytes:
        """Convert a tensor to WAV format and return as bytes."""
        src = tensor.cpu()
        if src.dim() == 1:
            src = src.unsqueeze(0)

        n_channels = src.shape[0]
        src = (src * 32767).clamp(-32768, 32767).to(torch.int16)
        src = src.t().numpy()

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(src.tobytes())

        return buffer.getvalue()

    def __merge_tokens(
        self, alignment_path: torch.Tensor, blank_id: int
    ) -> list[tuple[int, int, int]]:
        """
        Merge frame-level alignment into (token, start, end) spans.

        Blank-labelled frames are excluded from character spans by construction:
        a span ends at the frame where any token transition occurs, so blank
        regions appear as gaps between adjacent character spans.  These gaps
        are resolved in __refine_token_spans using gap-midpoint anchoring
        (Fix 1 / Fix 2).

        Returns:
            list of (token_id, start_frame, end_frame) tuples.
            Unchanged from the original — blank handling is correct here;
            the original methodological problem was in how the resulting
            gap timestamps were used downstream.
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

        if current_token is not None and current_token != blank_id:
            spans.append((current_token, start_frame, len(path)))

        return spans

    def __prepare_transcript(
        self, transcript: str | list[str]
    ) -> tuple[str, list[str], str, list[int]]:
        """Prepare the transcript for alignment by normalizing and tokenizing."""
        if isinstance(transcript, str):
            full_transcript = transcript
            target_segments = transcript.split()
        else:
            target_segments = transcript
            full_transcript = " ".join(transcript)

        text_upper = full_transcript.upper()
        text_nfd = unicodedata.normalize("NFD", text_upper)
        text_no_diacritics = "".join(
            char
            for char in text_nfd
            if unicodedata.category(char) != UNICODE_CATEGORY_NONSPACING_MARK
        )
        text_clean = "".join(
            c for c in text_no_diacritics if c.isalnum() or c == ASCII_SPACE
        )
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
        """Perform forced alignment between the audio waveform and the transcript."""
        emissions_gpu = self.__compute_emissions(waveform, original_sr).squeeze(0)

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

    def __refine_token_spans(
        self,
        token_spans: list[tuple[int, int, int]],
        emissions_gpu: torch.Tensor,
        waveform: torch.Tensor,
        original_sr: int,
    ) -> list[dict[str, str | float | bool]]:
        """
        Refine token boundary times using acoustic features.

        Fix 1 + Fix 2: Boundaries between adjacent character spans are resolved
        jointly.  For each consecutive pair (span[i], span[i+1]), the raw gap
        is [span[i].end, span[i+1].start] in frame space.  We convert that gap
        to seconds, compute its midpoint, and call __refine_boundary_smart once,
        centred on the midpoint with the full gap as the search window.  The
        resulting single refined time is assigned as the end of span[i] AND the
        start of span[i+1], guaranteeing monotonicity and symmetric coverage of
        blank regions.

        Fix 4: The `boundary_refined` flag from __refine_boundary_smart is
        stored per character so callers can detect heterogeneous boundary quality.
        """
        ratio = waveform.size(1) / emissions_gpu.size(0)
        numpy_wave = waveform.cpu().numpy().squeeze()
        total_samples = waveform.size(1)

        n = len(token_spans)
        if n == 0:
            return []

        # --- Pass 1: convert all raw frame spans to seconds -------------------
        raw_starts: list[float] = []
        raw_ends: list[float] = []
        confs: list[float] = []

        for sp_token, sp_start, sp_end in token_spans:
            raw_starts.append((sp_start * ratio) / original_sr)
            raw_ends.append((sp_end * ratio) / original_sr)
            conf = torch.exp(emissions_gpu[sp_start:sp_end, sp_token]).mean().item()
            confs.append(conf)

        # --- Pass 2: resolve shared inter-character boundaries jointly --------
        # refined_boundaries[i] is the shared boundary between span[i] and span[i+1].
        # len == n-1.
        refined_boundaries: list[tuple[float, bool]] = []
        for i in range(n - 1):
            gap_left = raw_ends[i]  # end of left character span (seconds)
            gap_right = raw_starts[i + 1]  # start of right character span (seconds)

            # Midpoint of the blank gap as the search anchor (Fix 1).
            gap_mid = (gap_left + gap_right) / 2.0

            refined_time, was_refined = self.__refine_boundary_smart(
                numpy_wave,
                original_sr,
                candidate_time=gap_mid,
                left_time=gap_left,
                right_time=gap_right,
            )

            # Safety clamp: must stay within [gap_left, gap_right] to prevent
            # the refined boundary from drifting into neighbouring phonemes.
            refined_time = float(np.clip(refined_time, gap_left, gap_right))
            refined_boundaries.append((refined_time, was_refined))

        # --- Pass 3: refine leading edge of first span and trailing edge of last span ---
        # These have no neighbour on one side, so we use the original ±80 ms window.
        first_start, first_refined = self.__refine_boundary_smart(
            numpy_wave, original_sr, raw_starts[0]
        )
        last_end, last_refined = self.__refine_boundary_smart(
            numpy_wave, original_sr, min(raw_ends[-1], total_samples / original_sr)
        )

        # --- Pass 4: assemble refined character records -----------------------
        refined_chars: list[dict[str, str | float | bool]] = []
        for i, (sp_token, _sp_start, _sp_end) in enumerate(token_spans):
            # Start time: refined leading edge (first span) or shared boundary with left neighbour
            if i == 0:
                start_time = first_start
                start_refined = first_refined
            else:
                start_time, start_refined = refined_boundaries[i - 1]

            # End time: shared boundary with right neighbour, or refined trailing edge (last span)
            if i == n - 1:
                end_time = last_end
                end_refined = last_refined
            else:
                end_time, end_refined = refined_boundaries[i]

            boundary_refined = start_refined and end_refined

            char = cast(str, self.tokenizer.convert_ids_to_tokens(sp_token))
            refined_chars.append(
                {
                    "char": char,
                    "start": start_time,
                    "end": end_time,
                    "confidence": confs[i],
                    "boundary_refined": boundary_refined,
                }
            )

        return refined_chars

    def __aggregate_chars_to_segments(
        self,
        char_segments: list[dict[str, str | float | bool]],
        target_segments: list[str],
    ) -> list[AudioSegment]:
        """Aggregate character-level segments into user-defined token segments."""
        final_segments = []
        current_char_idx = 0

        for segment_text in target_segments:
            clean_target = self.normalize_text(segment_text)

            if not clean_target:
                continue

            start_time = None
            end_time = None
            seg_confs = []
            all_refined = True
            found_chars = 0

            while found_chars < len(clean_target) and current_char_idx < len(
                char_segments
            ):
                seg = char_segments[current_char_idx]
                seg_char = cast(str, seg["char"]).replace(self.ctc_separator, "")

                if seg_char == clean_target[found_chars]:
                    if start_time is None:
                        start_time = seg["start"]
                    end_time = seg["end"]
                    seg_confs.append(seg["confidence"])
                    if not seg.get("boundary_refined", True):
                        all_refined = False
                    found_chars += 1

                current_char_idx += 1

            if start_time is not None:
                if end_time is None:
                    end_time = cast(float, start_time) + 0.1

                avg_conf = sum(seg_confs) / len(seg_confs) if seg_confs else 0.0

                audio_seg = AudioSegment(
                    token=segment_text,
                    start_time=cast(float, start_time),
                    end_time=cast(float, end_time),
                    confidence=avg_conf,
                    audio_format="wav",
                )
                audio_seg.boundary_refined = all_refined
                final_segments.append(audio_seg)

        return final_segments

    def __set_segment_indices(
        self,
        final_segments: list[AudioSegment],
        waveform: torch.Tensor,
        original_sr: int,
    ) -> torch.Tensor:
        """Set the start_sample and end_sample indices for each segment based on the refined start_time and end_time."""
        cpu_waveform = waveform.cpu()
        if cpu_waveform.dim() == 1:
            cpu_waveform = cpu_waveform.unsqueeze(0)

        for seg in final_segments:
            start_sample = int(seg.start_time * original_sr)
            end_sample = int(seg.end_time * original_sr)

            min_duration = int(0.05 * original_sr)
            if end_sample - start_sample < min_duration:
                end_sample = start_sample + min_duration

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
        cpu_waveform = self.__set_segment_indices(final_segments, waveform, original_sr)

        if not attach_audio:
            return

        for seg in final_segments:
            segment_tensor = cpu_waveform[:, seg.start_sample : seg.end_sample]
            seg.audio = self.__save_wav_mem(segment_tensor, original_sr)

    def attach_audio_to_segments(
        self,
        segments: list[AudioSegment],
        audio_content: bytes | None = None,
        waveform: torch.Tensor | None = None,
        original_sr: int | None = None,
        audio_format: str = "mp3",
    ) -> None:
        """Attach audio bytes to existing segments using provided audio input.
        This can be used when segments are generated without audio and need to be enriched later."""
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
        """Align the provided transcript to the audio and return a list of AudioSegments with refined boundaries.

        Args:
            transcript: The transcript to align, either as a single string or a list of token strings.
            audio_content: Raw audio bytes (optional if waveform and original_sr are provided).
            waveform: Pre-loaded audio waveform tensor (optional if audio_content is provided).
            original_sr: Original sample rate of the audio (required if waveform is provided).
            audio_format: Format of the input audio bytes (default: "mp3").
            attach_audio: Whether to attach raw audio bytes to each segment (default: False).
        Returns:            A list of AudioSegment instances with aligned tokens and refined timestamps.
            If attach_audio is True, each segment will also include the corresponding audio bytes in WAV format.
        Raises:
            ValueError: If neither audio_content nor both waveform and original_sr are provided, or if the transcript contains no valid characters for the model.
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
            self.__attach_audio_to_segments(final_segments, waveform, original_sr)
        else:
            self.__set_segment_indices(final_segments, waveform, original_sr)

        return final_segments

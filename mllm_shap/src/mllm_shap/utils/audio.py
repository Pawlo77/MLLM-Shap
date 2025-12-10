"""Utility functions for audio processing and display."""

from typing import TYPE_CHECKING
from io import BytesIO
import numpy as np
import torch
import soundfile as sf
from torch import Tensor
from pydub import AudioSegment

if TYPE_CHECKING:
    from IPython.display import Audio

TARGET_SAMPLE_RATE = 24_000
PCM16_SAMPLE_WIDTH_BYTES = 2
WAVEFORM_2D_DIM = 2
MONO_CHANNELS = 1
AUDIO_FORMAT_WAV = "wav"
AUDIO_FORMAT_MP3 = "mp3"


def display_audio(audio_content: bytes) -> "Audio":
    """
    Display audio content in a Jupyter notebook.

    Args:
        audio_content: The audio content in bytes.
    """
    # Import here to avoid dependency if not used in notebook
    from IPython.display import Audio  # pylint: disable=import-outside-toplevel

    return Audio(data=audio_content, autoplay=True)  # type: ignore


class TorchAudioHandler:
    """Utility class for handling audio content with TorchAudio."""

    @staticmethod
    def from_bytes(audio_content: bytes, audio_format: str = "mp3") -> tuple[Tensor, int]:
        """
        Prepare audio content for processing.

        Args:
            audio_format: The format of the audio content (default is "mp3").
            audio_content: The audio content in bytes.

        Returns:
            A tuple containing the audio tensor and the sample rate.
        """
        try:
            waveform_np, sample_rate = sf.read(BytesIO(audio_content))
            waveform = torch.from_numpy(waveform_np).float()

            if waveform.dim() == MONO_CHANNELS:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == WAVEFORM_2D_DIM:
                waveform = waveform.T

            if waveform.shape[0] > MONO_CHANNELS:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform, int(sample_rate)

        except Exception as e:
            print(f"Error loading with soundfile: {e}, for format: {audio_format}.")
            raise

    @staticmethod
    def to_bytes(
        waveform: torch.Tensor,
        sample_rate: int = TARGET_SAMPLE_RATE,
        audio_format: str = "wav",
        mp3_bitrate: str = "192k",
    ) -> bytes:
        """
        Convert a waveform tensor to audio content in bytes.

        Args:
            waveform: The audio waveform tensor.
            sample_rate: The sample rate of the audio (default is TARGET_SAMPLE_RATE).
            audio_format: The desired audio format ("wav" or "mp3").
            mp3_bitrate: The bitrate for MP3 encoding (default is "192k").

        Returns:
            The audio content in bytes.
        """
        with torch.no_grad():
            wf = waveform.detach().cpu()
            # force mono
            if wf.dim() == WAVEFORM_2D_DIM and wf.size(0) > 1:
                wf = wf.mean(dim=0, keepdim=True)
            if wf.dim() == WAVEFORM_2D_DIM and wf.size(0) == 1:
                wf = wf.squeeze(0)
            elif wf.dim() == 1:
                pass
            else:
                wf = wf.mean(dim=0)

            wf = wf.to(torch.float32)
            # replace NaN/Inf, then clamp
            wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
            wf = wf.contiguous()

        fmt = audio_format.lower()
        if fmt == AUDIO_FORMAT_WAV:
            buf = BytesIO()
            sf.write(buf, wf.numpy(), int(sample_rate), format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.read()

        if fmt == AUDIO_FORMAT_MP3:
            # requires ffmpeg on PATH; pydub delegates to ffmpeg for MP3 encoding
            # (if ffmpeg is missing will get an error from pydub)
            pcm16 = (wf.numpy() * 32767.0).astype(np.int16)
            seg = AudioSegment(
                pcm16.tobytes(),
                frame_rate=int(sample_rate),
                sample_width=PCM16_SAMPLE_WIDTH_BYTES,
                channels=MONO_CHANNELS,
            )
            buf = BytesIO()
            seg.export(buf, format=AUDIO_FORMAT_MP3, bitrate=mp3_bitrate)
            return buf.getvalue()

        raise ValueError(f"Unsupported audio_format: {audio_format!r}")

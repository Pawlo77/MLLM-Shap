"""Utility functions for audio processing and display."""

from typing import TYPE_CHECKING
from io import BytesIO
import soundfile as sf
import torch
from torch import Tensor
from torchaudio import save

if TYPE_CHECKING:
    from IPython.display import Audio


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

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T

        except Exception as e:
            print(f"Error loading with soundfile: {e}, for format: {audio_format}.")
            raise e

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate

    @staticmethod
    def to_bytes(waveform: Tensor, sample_rate: int = 24_000, audio_format: str = "mp3") -> bytes:
        """
        Convert a waveform tensor back to audio bytes.

        Args:
            waveform: The audio waveform tensor.
            sample_rate: The sample rate of the audio. Default is 24,000 Hz.
            audio_format: The desired output format (default is "mp3").

        Returns:
            The audio content in bytes.
        """
        buffer = BytesIO()
        save(buffer, waveform, sample_rate, format=audio_format)
        buffer.seek(0)
        return buffer.read()

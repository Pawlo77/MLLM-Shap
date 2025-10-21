"""Utility functions for audio processing and display."""

from io import BytesIO
from typing import TYPE_CHECKING

from torch import Tensor
from torchaudio import load, save

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
            audio_content: The audio content in bytes.
            audio_format: The format of the audio content (default is "mp3").

        Returns:
            A tuple containing the audio tensor and the sample rate.
        """

        waveform, sample_rate = load(BytesIO(audio_content), format=audio_format)

        # Convert to mono if needed
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

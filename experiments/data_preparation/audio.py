"""Helper functions for audio processing."""

import io

from pydub import AudioSegment


def calculate_audio_duration(audio_bytes_list: list[bytes]) -> list[float]:
    """
    Calculate the duration of an audio file in seconds.

    Args:
        audio_bytes_list: A list of audio files in bytes format.
    Returns:
        The duration of the audio in seconds.
    """
    durations = []
    for audio_bytes in audio_bytes_list:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        durations.append(audio.duration_seconds)
    return durations

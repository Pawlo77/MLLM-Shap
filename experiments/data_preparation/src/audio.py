"""Audio processing and synthesis helpers.

This module contains helpers to convert, normalise and synthesise audio
representations used by the data preparation pipeline. Functions return
bytes and duration metadata suitable for storage in the dataset DataFrame.
"""

import io

import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

from .constants import TTS_CONFIGS, TTSConfig
from .nlp import TTS


def calculate_audio_duration(audio_bytes_list: list[bytes]) -> list[float]:
    """Return duration in seconds for each MP3 blob in ``audio_bytes_list``."""
    durations = []
    for audio_bytes in audio_bytes_list:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        durations.append(audio.duration_seconds)
    return durations


def normalize_original_audio_entry(
    audio_entry: dict,
) -> tuple[list[bytes], list[float]]:
    """Convert HF audio dict into [wav_bytes] and [duration_seconds]."""
    arr = np.asarray(audio_entry["array"], dtype=np.float32)
    sr = int(audio_entry["sampling_rate"])

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    duration = float(len(arr) / sr) if sr > 0 else 0.0
    return [wav_bytes], [duration]


def to_audio_bytes_and_duration(
    audio_entry: dict,
) -> tuple[list[bytes], list[float]]:
    """Convert HF audio dict to (list[bytes], list[float]) — alias for normalize_original_audio_entry."""
    return normalize_original_audio_entry(audio_entry)


async def synthesize_voices(
    df: pd.DataFrame,
    tts: TTS,
    male_config: TTSConfig,
    female_config: TTSConfig,
    column_to_synthesize: str = "sentences",
) -> pd.DataFrame:
    """Synthesize male and female audio columns and compute durations."""
    for name, config in (("male", male_config), ("female", female_config)):
        df = await tts.synthesize_df_from_config(
            df,
            config,
            column_to_synthesize=column_to_synthesize,
            target_column=f"audio__{name}",
        )
        df[f"audio__{name}__duration"] = df[f"audio__{name}"].progress_apply(
            calculate_audio_duration
        )
    return df


async def synthesize_en_voices(
    df: pd.DataFrame,
    tts: TTS,
    column_to_synthesize: str = "sentences",
) -> pd.DataFrame:
    """Synthesize English male/female TTS using ``TTS_CONFIGS['en']``."""
    return await synthesize_voices(
        df,
        tts,
        TTS_CONFIGS["en"]["male"],
        TTS_CONFIGS["en"]["female"],
        column_to_synthesize=column_to_synthesize,
    )

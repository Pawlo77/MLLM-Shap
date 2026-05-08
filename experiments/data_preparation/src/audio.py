"""Audio processing and synthesis helpers."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import soundfile as sf

from ..audio import calculate_audio_duration
from ..constants import TTSConfig
from ..nlp import TTS


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
    """Synthesize male and female audio columns and compute durations.

    Parameters
    ----------
    df                   : DataFrame containing the text to synthesize.
    tts                  : Initialized TTS instance.
    male_config          : TTS configuration for male voice.
    female_config        : TTS configuration for female voice.
    column_to_synthesize : Column with text/sentences to synthesize.

    Returns
    -------
    DataFrame with added audio__male, audio__female, and duration columns.
    """
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

"""Central constants, enums, and modality helpers."""

from enum import StrEnum

DEFAULT_SUBSET: str = "single_sentence"
"""Default dataset subset name."""

DEFAULT_SPLIT: str = "test"
"""Default dataset split name."""

TRUE_JSON: str = "1"
"""String representation of a true boolean value in JSON."""


class InputModality(StrEnum):
    """Input modality options."""

    TEXT = "text"
    AUDIO_ORIGINAL = "audio__original"
    AUDIO_MALE = "audio__male"
    AUDIO_FEMALE = "audio__female"
    INTERLEAVED_TEXT_FIRST_MALE = "interleaved__text_first__male"
    INTERLEAVED_TEXT_FIRST_FEMALE = "interleaved__text_first__female"
    INTERLEAVED_AUDIO_FIRST_MALE = "interleaved__audio_first__male"
    INTERLEAVED_AUDIO_FIRST_FEMALE = "interleaved__audio_first__female"


class OutputModality(StrEnum):
    """Output modality options."""

    TEXT = "text"
    AUDIO = "audio"


class AudioCol(StrEnum):
    """Audio column options in dataset."""

    ORIGINAL = "audio__original"
    MALE = "audio__male"
    FEMALE = "audio__female"


class TokenFilterType(StrEnum):
    """Token filter options."""

    EXCLUDE_PUNCTUATION = "exclude_punctuation"
    NONE = "none"


class DatasetSource(StrEnum):
    """Dataset source/loading strategy."""

    HF_PARQUET = "hf_parquet"
    HF_DATASETS = "hf_datasets"
    LOCAL_PARQUET = "local_parquet"
    LOCAL_CSV = "local_csv"


AUDIO_MODALITIES = frozenset(
    {
        InputModality.AUDIO_ORIGINAL,
        InputModality.AUDIO_MALE,
        InputModality.AUDIO_FEMALE,
    }
)
"""Modalities that require audio input."""

INTERLEAVED_TEXT_FIRST_MODALITIES = frozenset(
    {
        InputModality.INTERLEAVED_TEXT_FIRST_MALE,
        InputModality.INTERLEAVED_TEXT_FIRST_FEMALE,
    }
)
"""Modalities where text comes first, but still require audio input."""

INTERLEAVED_AUDIO_FIRST_MODALITIES = frozenset(
    {
        InputModality.INTERLEAVED_AUDIO_FIRST_MALE,
        InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE,
    }
)
"""Modalities where audio comes first, and also require audio input."""

INTERLEAVED_MODALITIES = (
    INTERLEAVED_TEXT_FIRST_MODALITIES | INTERLEAVED_AUDIO_FIRST_MODALITIES
)
"""All interleaved modalities."""


def needs_audio(modality: InputModality) -> bool:
    """Return True if the input modality requires audio data."""
    return modality in AUDIO_MODALITIES or modality in INTERLEAVED_MODALITIES


def is_text_only_modality(modality: InputModality) -> bool:
    """Return True if the input modality is text-only."""
    return modality == InputModality.TEXT


def audio_column_for(modality: InputModality) -> AudioCol | None:
    """Return the audio column name needed for the given modality, or None for text-only."""
    if modality == InputModality.AUDIO_ORIGINAL:
        return AudioCol.ORIGINAL
    if modality in (
        InputModality.AUDIO_MALE,
        InputModality.INTERLEAVED_TEXT_FIRST_MALE,
        InputModality.INTERLEAVED_AUDIO_FIRST_MALE,
    ):
        return AudioCol.MALE
    if modality in (
        InputModality.AUDIO_FEMALE,
        InputModality.INTERLEAVED_TEXT_FIRST_FEMALE,
        InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE,
    ):
        return AudioCol.FEMALE
    return None

"""Central constants, enums, and modality helpers."""

from enum import StrEnum
from typing import Optional


class ExplainerType(StrEnum):
    """Explainer type strings."""

    EXACT = "exact"
    LIMITED_MC = "limited_mc"
    STANDARD_MC = "standard_mc"
    LIMITED_CC = "limited_cc"
    STANDARD_CC = "standard_cc"
    LIMITED_NEYMAN = "limited_neyman"
    STANDARD_NEYMAN = "standard_neyman"
    HIERARCHICAL = "hierarchical"


MC_LIKE_EXPLAINERS = frozenset({
    ExplainerType.LIMITED_MC,
    ExplainerType.STANDARD_MC,
    ExplainerType.LIMITED_CC,
    ExplainerType.STANDARD_CC,
    ExplainerType.LIMITED_NEYMAN,
    ExplainerType.STANDARD_NEYMAN,
})


class ConnectorType(StrEnum):
    """Available model backends/connectors."""

    LIQUID_AUDIO = "liquid_audio"
    TRANSFORMERS_TEXT = "hf_text"


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


class SimilarityType(StrEnum):
    """Similarity metric options."""

    COSINE = "CosineSimilarity"
    TFIDF_COSINE = "TfIdfCosineSimilarity"
    EUCLIDEAN = "EuclideanSimilarity"


class ModeType(StrEnum):
    """Explanation modes."""

    CONTEXTUAL = "CONTEXTUAL"
    STATIC = "STATIC"


class TextCol(StrEnum):
    """Text column options in dataset."""

    PROMPT = "prompt"
    SENTENCES = "sentences"


class AudioCol(StrEnum):
    """Audio column options in dataset."""

    ORIGINAL = "audio__original"
    MALE = "audio__male"
    FEMALE = "audio__female"


class WandbMode(StrEnum):
    """Weights & Biases operation modes."""

    DISABLED = "disabled"


class DatasetType(StrEnum):
    """Known dataset subset options (non-exhaustive, used for documentation)."""

    SINGLE_SENTENCE__VOICE_BENCH = "single_sentence__voice_bench"
    SINGLE_SENTENCE__LIBRISPEECH_ASR = "single_sentence__librispeech_asr"
    MULTILINGUAL__INFINITY_INSTRUCT = "multi_lingual__infinity_instruct"
    MULTI_SENTENCE__VOICE_BENCH = "multi_sentence__voice_bench"
    MULTI_TURN = "multi_turn"


class TokenFilterType(StrEnum):
    """Token filter options."""

    EXCLUDE_PUNCTUATION = "exclude_punctuation"
    NONE = "none"


class HierarchicalModeType(StrEnum):
    """Hierarchical explainer mode options."""

    MULTI_MODAL_MULTI_USER = "MULTI_MODAL_MULTI_USER"


class DatasetSource(StrEnum):
    """Dataset source/loading strategy."""

    HF_PARQUET = "hf_parquet"
    HF_DATASETS = "hf_datasets"
    LOCAL_PARQUET = "local_parquet"
    LOCAL_CSV = "local_csv"


# ---- Modality helper constants ----

AUDIO_MODALITIES = frozenset({
    InputModality.AUDIO_ORIGINAL,
    InputModality.AUDIO_MALE,
    InputModality.AUDIO_FEMALE,
})

INTERLEAVED_TEXT_FIRST_MODALITIES = frozenset({
    InputModality.INTERLEAVED_TEXT_FIRST_MALE,
    InputModality.INTERLEAVED_TEXT_FIRST_FEMALE,
})

INTERLEAVED_AUDIO_FIRST_MODALITIES = frozenset({
    InputModality.INTERLEAVED_AUDIO_FIRST_MALE,
    InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE,
})

INTERLEAVED_MODALITIES = (
    INTERLEAVED_TEXT_FIRST_MODALITIES | INTERLEAVED_AUDIO_FIRST_MODALITIES
)


# ---- Modality helper predicates ----


def needs_audio(modality: InputModality) -> bool:
    """Return True if the input modality requires audio data."""
    return modality in AUDIO_MODALITIES or modality in INTERLEAVED_MODALITIES


def is_text_only_modality(modality: InputModality) -> bool:
    """Return True if the input modality is text-only."""
    return modality == InputModality.TEXT


def audio_column_for(modality: InputModality) -> Optional[AudioCol]:
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


# ---- Defaults ----

DEFAULT_SUBSET = DatasetType.SINGLE_SENTENCE__VOICE_BENCH.value
DEFAULT_SPLIT = "test"
DEFAULT_SIMILARITY = "CosineSimilarity"
TRUE_JSON = "1"

CHECKPOINT_VERSION = 2

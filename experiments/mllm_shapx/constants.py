"""Central constants and enums to avoid magic strings."""

from __future__ import annotations

from enum import Enum, StrEnum


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


class ConnectorType(Enum):
    """Available model backends/connectors."""

    LIQUID_AUDIO = "liquid_audio"
    TRANSFORMERS_TEXT = "hf_text"


class InputModality(StrEnum):
    """Input modality options."""

    TEXT = "text"
    AUDIO_MALE = "audio__male"
    AUDIO_FEMALE = "audio__female"
    # Interleaved modes: alternate between text and audio in subsequent turns
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

    MALE = "audio__male"
    FEMALE = "audio__female"


class WandbMode(StrEnum):
    """Weights & Biases operation modes."""

    DISABLED = "disabled"


class DatasetType(StrEnum):
    """Dataset subset options."""

    SINGLE_SENTENCE = "single_sentence"
    MULTILINGUAL = "multi_lingual"
    MULTI_SENTENCE = "multi_sentence"


DEFAULT_SUBSET = DatasetType.SINGLE_SENTENCE.value
DEFAULT_SPLIT = "test"
DEFAULT_SIMILARITY = "CosineSimilarity"
TRUE_JSON = "1"

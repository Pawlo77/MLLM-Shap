"""Central constants and enums to avoid magic strings."""
from __future__ import annotations

from enum import StrEnum, Enum


class ExplainerType(StrEnum):
    """Explainer type strings."""
    EXACT = "exact"
    LIMITED_MC = "limited_mc"
    STANDARD_MC = "standard_mc"
    HIERARCHICAL = "hierarchical"


class ConnectorType(Enum):
    """Available model backends/connectors."""
    LIQUID_AUDIO = "liquid_audio"
    TRANSFORMERS_TEXT = "hf_text"


class SimilarityType(StrEnum):
    """Similarity metric options."""
    COSINE = "CosineSimilarity"
    TFIDF_COSINE = "TfIdfCosineSimilarity"


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


DEFAULT_SUBSET = "single_sentence"
DEFAULT_SPLIT = "test"
DEFAULT_SIMILARITY = "CosineSimilarity"
TRUE_JSON = "1"

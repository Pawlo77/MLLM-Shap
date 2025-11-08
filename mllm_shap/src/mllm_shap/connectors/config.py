"""Configuration for Hugging Face interfaces."""

from pydantic import BaseModel


# pylint: disable=too-few-public-methods
class HuggingFaceModelConfig(BaseModel):
    """Holds the necessary information to load a model from the Hugging Face Hub."""

    repo_id: str
    """The repository ID of the model on Hugging Face."""

    revision: str
    """The specific revision or branch of the model to use."""


# pylint: disable=too-few-public-methods
class ModelConfig(BaseModel):
    """Defines settings for controlling text and audio generation behavior."""

    text_temperature: float | None = 0.0
    """Controls the randomness in text generation."""

    text_top_k: int | None = 1
    """Restricts text sampling to the top-k most probable tokens."""

    audio_temperature: float | None = 0.0
    """Controls the randomness in audio generation."""

    audio_top_k: int | None = 1
    """Restricts audio sampling to the top-k most probable tokens."""

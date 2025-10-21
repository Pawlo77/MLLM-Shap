"""Configuration for Hugging Face interfaces."""

from pydantic import BaseModel


class HuggingFaceModelConfig(BaseModel):
    """
    Configuration for Hugging Face models.

    Fields:
        repo_id: The repository ID of the model on Hugging Face.
        revision: The specific revision or version of the model.
    """

    repo_id: str
    revision: str


class ModelConfig(BaseModel):
    """
    Base configuration for models.

    Fields:
        text_temperature: The temperature to use for text generation.
        text_top_k: The top-k sampling parameter for text generation.
        audio_temperature: The temperature to use for audio generation.
        audio_top_k: The top-k sampling parameter for audio generation.
    """

    text_temperature: float | None = 0.0
    text_top_k: int | None = 1
    audio_temperature: float | None = 0.0
    audio_top_k: int | None = 1

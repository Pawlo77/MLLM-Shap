"""Transformers model configuration."""

from ..config import HuggingFaceModelConfig

CONFIG: HuggingFaceModelConfig = HuggingFaceModelConfig(
    repo_id="microsoft/phi-2",
    revision="ef382358ec9e382308935a992d908de099b64c23"
)

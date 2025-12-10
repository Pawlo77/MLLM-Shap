"""Mock model configuration."""

from ..config import HuggingFaceModelConfig

CONFIG: HuggingFaceModelConfig = HuggingFaceModelConfig(
    repo_id="gpt2",
    revision="main"
)

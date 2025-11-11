"""LiquidAudio configuration."""

from ..config import HuggingFaceModelConfig

CONFIG: HuggingFaceModelConfig = HuggingFaceModelConfig(
    repo_id="LiquidAI/LFM2-Audio-1.5B",
    revision="3f9322d8cfdcf3df281227af9ca80d948f5ba878",
)

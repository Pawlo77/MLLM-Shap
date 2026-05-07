"""Mock model configuration."""

from ..config import HuggingFaceModelConfig

CONFIG: HuggingFaceModelConfig = HuggingFaceModelConfig(repo_id="gpt2", revision="main")
"""Mock model configuration module, defining the Hugging Face model configuration
for the GPT-2 model used in the Mock connector. This configuration is used to
initialize the tokenizer"""

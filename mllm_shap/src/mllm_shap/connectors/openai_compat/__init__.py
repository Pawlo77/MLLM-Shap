"""OpenAI-compatible HTTP chat (LM Studio, vLLM, etc.) with local HF embeddings."""

from .model import OpenAICompatCausalText

__all__ = ["OpenAICompatCausalText"]

"""A connector for text generation using a model conforming to
   tranformers interface."""

from .chat import TransformersTextChat
from .model import TransformersCausalText

__all__ = ["TransformersTextChat", "TransformersCausalText"]

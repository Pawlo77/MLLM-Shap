"""A connector for a mock model used for debugging."""

from .chat import MockChat
from .model import Mock

__all__ = ["MockChat", "Mock"]

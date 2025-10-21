"""Base classes for token filtering strategies."""

from abc import ABC

from pydantic import BaseModel


class TokenFilter(ABC, BaseModel):
    """Base class for token filtering strategies."""

    phrased_to_exclude: set[str]

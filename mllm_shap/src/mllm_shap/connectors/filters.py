"""Token filtering strategies for audio-shap connectors."""

from .base.filters import TokenFilter


class KeepAllTokens(TokenFilter):
    """A token filter that does not exclude any tokens."""

    phrases_to_exclude: set[str] = set()
    """No tokens are excluded by this strategy."""


class ExcludePunctuationTokensFilter(TokenFilter):
    """A token filter that removes common punctuation tokens."""

    phrases_to_exclude: set[str] = {".", ",", "!", "?", ";", ":"}
    """Excludes standard inter-punctuation tokens."""

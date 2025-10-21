"""Token filtering strategies for audio-shap connectors."""

from ._base.filters import TokenFilter


class KeepAllTokens(TokenFilter):
    """Strategy to keep all tokens."""

    phrased_to_exclude: set[str] = set()


class ExcludePunctuationTokensFilter(TokenFilter):
    """Strategy to exclude inter-punctuation tokens."""

    phrased_to_exclude: set[str] = {".", ",", "!", "?", ";", ":"}

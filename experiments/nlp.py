"""NLP related utilities."""

from typing import cast

import nltk


def split_into_sentences(text: str) -> list[str]:
    """
    Split the given text into a list of sentences.

    Args:
        text: The input text to split.
    Returns:
        A list of sentences extracted from the text.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    return cast(list[str], nltk.sent_tokenize(text))


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in a given text.

    Args:
        text: The input text to analyze.
    Returns:
        The number of sentences in the text.
    """
    return len(split_into_sentences(text))

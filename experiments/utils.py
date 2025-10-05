"""Utility functions for experiments."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import nltk
from datasets import DatasetDict
from datasets import load_dataset as src_load_dataset

if TYPE_CHECKING:
    from .constants import DatasetConfig


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(hugging_face__config__name: str, config: "DatasetConfig") -> DatasetDict:
    """
    Load a dataset from the Hugging Face Hub using the provided configuration.

    Args:
        hugging_face__config__name: The specific configuration name for the dataset.
        config: The dataset configuration containing the dataset name and cache directory.
    Returns:
        A dictionary containing the dataset splits (e.g., 'train', 'test', 'validation').
    Raises:
        ValueError: If the provided revision is not a valid 40-character hexadecimal string.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", config.revision or ""):
        raise ValueError(f"Unsafe Hugging Face revision: {config.revision!r}. Use a specific commit hash.")

    return cast(
        DatasetDict,
        src_load_dataset(
            config.dataset_name,
            hugging_face__config__name,
            cache_dir=str(config.cache_dir),
            revision=config.revision,  # nosec B615
        ),
    )


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


if __name__ == "__main__":
    print(count_sentences("This is a test. This is only a test!"))

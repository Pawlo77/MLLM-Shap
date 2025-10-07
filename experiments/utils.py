"""Utility functions for experiments."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import nltk
import pandas as pd
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
    if not re.fullmatch(r"[0-9a-f]{40}", config.revision or "", re.IGNORECASE):
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


def save_json(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save a DataFrame to a JSON file.

    Args:
        df: The DataFrame to save.
        path: The file path to save the JSON file.
    """
    df.to_json(
        str(path),
        index=False,
        orient="records",
        indent=4,
        force_ascii=False,
    )


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    """
    Save a DataFrame to a Parquet file.

    Args:
        df: The DataFrame to save. Must not be empty.
        path: The file path to save the Parquet file.
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot save to Parquet.")
    df.to_parquet(str(path), index=False)


def get_final_stats(df: pd.DataFrame, prompt_key: str = "prompt") -> pd.DataFrame:
    """
    Get final statistics for the entire dataframe.

    Args:
        df: The DataFrame to analyze.
        prompt_key: The column name containing the text prompts.
    Returns:
        A DataFrame containing the final statistics.
    """

    unique_entries = df.drop_duplicates(subset=[prompt_key]).shape[0]
    sentences_count = df[prompt_key].apply(count_sentences)
    prompt_lengths = df[prompt_key].apply(len)

    return pd.DataFrame(
        [
            {
                "num_rows": df.shape[0],
                "total_characters": prompt_lengths.sum(),
                "avg_num_characters": prompt_lengths.mean(),
                "total_sentences": sentences_count.sum(),
                "avg_num_sentences": sentences_count.mean(),
                "unique_entries": unique_entries,
                "pct_unique_entries": unique_entries / df.shape[0] * 100,
            }
        ]
    )


if __name__ == "__main__":
    print(count_sentences("This is a test. This is only a test!"))

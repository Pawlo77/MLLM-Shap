"""Utilities for file and dataset I/O operations.

Small helpers to load HF datasets safely and persist DataFrames to JSON
or Parquet with lightweight validation used by the builder notebooks.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from datasets import DatasetDict
from datasets import load_dataset as src_load_dataset

if TYPE_CHECKING:
    from .constants import DatasetConfig


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(
    hugging_face__config__name: str, config: "DatasetConfig"
) -> DatasetDict:
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
        raise ValueError(
            f"Unsafe Hugging Face revision: {config.revision!r}. Use a specific commit hash."
        )

    return cast(
        DatasetDict,
        src_load_dataset(
            config.dataset_name,
            hugging_face__config__name,
            cache_dir=str(config.cache_dir),
            revision=config.revision,  # nosec B615
        ),
    )


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

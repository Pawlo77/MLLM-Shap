"""Dataset loading, row selection, and optional prompt token filtering utilities."""

from __future__ import annotations

import importlib
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import numpy
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .constants import TRUE_JSON, TextCol


def load_single_sentence_df(
    repo_id: str, subset: str, split: str, revision: str
) -> pd.DataFrame:
    """Load the first parquet shard into a DataFrame with optional pin enforcement."""
    _ensure_pinned_revision(revision)
    filename = f"{subset}/{split}/0000.parquet"
    print(
        f"Downloading dataset from HF Hub repo '{repo_id}', file '{filename}', revision '{revision}'"
    )
    parquet_local_path = hf_hub_download(  # nosec B615
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(parquet_local_path)


def load_dataset_from_main(
    repo_id: str,
    subset: str | None,
    split: str,
    revision: str = "main",
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """
    Load dataset from HuggingFace using the datasets library (non-parquet).

    This loads from the main branch (or specified revision) using the dataset's
    native loading scripts, rather than the auto-converted parquet files.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g., "Pawlo77/mllm-shap").
        subset: Dataset subset/config name (can be None for datasets without subsets).
        split: Dataset split (e.g., "train", "test").
        revision: Git revision to load from (default: "main").
        trust_remote_code: Whether to trust remote code in dataset scripts.

    Returns:
        DataFrame containing the loaded dataset.
    """
    print(
        f"Loading dataset '{repo_id}', subset '{subset}', split '{split}', "
        f"revision '{revision}' using datasets library"
    )
    _ensure_pinned_revision(revision)
    dataset = load_dataset(  # nosec B615
        repo_id,
        name=subset,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    return dataset.to_pandas()


def load_df(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    repo_id: str,
    subset: str,
    split: str,
    revision: str,
    use_parquet: bool = True,
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """
    Load dataset as DataFrame, choosing between parquet and main branch methods.

    Args:
        repo_id: HuggingFace Hub repository ID.
        subset: Dataset subset/config name.
        split: Dataset split.
        revision: Git revision to load from.
        use_parquet: If True, load from parquet files (refs/convert/parquet branch).
                     If False, use datasets library to load from main/specified branch.
        trust_remote_code: Whether to trust remote code (only used when use_parquet=False).

    Returns:
        DataFrame containing the loaded dataset.
    """
    if use_parquet:
        return load_single_sentence_df(repo_id, subset, split, revision)
    return load_dataset_from_main(repo_id, subset, split, revision, trust_remote_code)


def choose_prompt_text_column(df: pd.DataFrame) -> str:
    """Pick the correct text column ("prompt" preferred, fallback to "sentences")."""
    if TextCol.SENTENCES.value in df.columns:
        return TextCol.SENTENCES.value
    if TextCol.PROMPT.value in df.columns:
        return TextCol.PROMPT.value
    raise KeyError("Neither 'prompt' nor 'sentences' column found in dataframe.")


def extract_texts_from_row(value: Any) -> List[str]:
    """
    Extract texts from a row value as a list, handling both single strings and lists.

    For multi-sentence datasets, returns all sentences as separate list items.

    Args:
        value: The row value (either a string or list of strings).

    Returns:
        List of text strings (never empty - returns [""] for None/empty input).
    """
    if value is None:
        return [""]
    if isinstance(value, (list, numpy.ndarray)):
        value = value.tolist() if isinstance(value, numpy.ndarray) else value
        # Filter out None/empty strings
        sentences = [str(s) for s in value if s is not None and str(s).strip()]
        return sentences if sentences else [""]
    text = str(value)
    return [text] if text.strip() else [""]


def extract_text_from_row(value: Any, separator: str = " ") -> str:
    """
    Extract text from a row value, handling both single strings and lists of sentences.

    For multi-sentence datasets, concatenates all sentences with the given separator.

    Args:
        value: The row value (either a string or list of strings).
        separator: String to use when joining multiple sentences (default: single space).

    Returns:
        The extracted/concatenated text string.
    """
    texts = extract_texts_from_row(value)
    return separator.join(texts)


def iter_rows_for_selection(
    df: pd.DataFrame,
    start_index: int,
    max_samples: int | None,
    shuffle_seed: int | None,
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """
    Yield (row_index, row_dict) with optional deterministic shuffling and slicing.
    """
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    start = max(0, start_index)
    end = len(df) if max_samples is None else min(len(df), start + max_samples)

    for i in range(start, end):
        yield i, df.iloc[i].to_dict()


def get_hf_text_tokenizer() -> Any:
    """
    Lazily construct the HF tokenizer used by the Transformers text-only connector,
    honoring its pinned repo_id/revision.
    """
    cfg_mod = importlib.import_module("mllm_shap.connectors.transformers_text.config")
    transformers = importlib.import_module("transformers")
    repo_id = getattr(cfg_mod, "CONFIG").repo_id
    revision = getattr(cfg_mod, "CONFIG").revision
    return getattr(transformers, "AutoTokenizer").from_pretrained(
        repo_id, revision=revision
    )  # nosec B615


def filter_df_by_max_prompt_tokens(
    df: pd.DataFrame, text_col: str, tokenizer: Any, max_tokens: int
) -> tuple[pd.DataFrame, int]:
    """
    Return (filtered_df, total_matching_count) where rows are limited to prompts with <= max_tokens.
    """
    lengths = []
    for _, row in df.iterrows():
        text = extract_text_from_row(row[text_col])
        ids = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(ids))
    mask = pd.Series(lengths) <= int(max_tokens + 2)
    filtered = df[mask.values].reset_index(drop=True)
    return filtered, int(mask.sum())


def filter_df_by_min_prompt_tokens(
    df: pd.DataFrame, text_col: str, tokenizer: Any, min_tokens: int
) -> tuple[pd.DataFrame, int]:
    """
    Return (filtered_df, total_matching_count) where rows are limited to prompts with >= min_tokens.
    """
    lengths = []
    for _, row in df.iterrows():
        text = extract_text_from_row(row[text_col])
        ids = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(ids))
    mask = pd.Series(lengths) >= int(min_tokens - 2)
    filtered = df[mask.values].reset_index(drop=True)
    return filtered, int(mask.sum())


def _ensure_pinned_revision(revision: str) -> None:
    """
    Enforce pinned HF revision unless explicitly overridden.
    Allows override with env ALLOW_UNPINNED_HF_DOWNLOAD=1.
    """
    if os.environ.get("ALLOW_UNPINNED_HF_DOWNLOAD") == TRUE_JSON:
        return
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "HuggingFace 'revision' must be a 40-hex commit SHA. "
            "Set ALLOW_UNPINNED_HF_DOWNLOAD=1 to override."
        )

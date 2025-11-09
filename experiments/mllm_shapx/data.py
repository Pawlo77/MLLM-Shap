"""Dataset loading and row selection utilities."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download
from .constants import TextCol, TRUE_JSON


def load_single_sentence_df(
    repo_id: str, subset: str, split: str, revision: str
) -> pd.DataFrame:
    """Load the first parquet shard into a DataFrame with optional pin enforcement."""
    _ensure_pinned_revision(revision)
    filename = f"{subset}/{split}/0000.parquet"
    parquet_local_path = hf_hub_download(  # nosec B615
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(parquet_local_path)


def choose_prompt_text_column(df: pd.DataFrame) -> str:
    """Pick the correct text column (“prompt” preferred, fallback to “sentences”)."""
    if TextCol.PROMPT.value in df.columns:
        return TextCol.PROMPT.value
    if TextCol.SENTENCES.value in df.columns:
        return TextCol.SENTENCES.value
    raise KeyError("Neither 'prompt' nor 'sentences' column found in dataframe.")


def iter_rows_for_selection(
    df: pd.DataFrame, start_index: int, max_samples: int | None, shuffle_seed: int | None
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

"""Dataset loading, row selection, and optional prompt token filtering utilities."""
from __future__ import annotations

import importlib
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


def get_hf_text_tokenizer() -> Any:
    """
    Lazily construct the HF tokenizer used by the Transformers text-only connector,
    honoring its pinned repo_id/revision.
    """
    cfg_mod = importlib.import_module("mllm_shap.connectors.transformers_text.config")
    transformers = importlib.import_module("transformers")
    repo_id = getattr(cfg_mod, "CONFIG").repo_id
    revision = getattr(cfg_mod, "CONFIG").revision
    return getattr(transformers, "AutoTokenizer").from_pretrained(repo_id, revision=revision)  # nosec B615


def filter_df_by_max_prompt_tokens(
    df: pd.DataFrame, text_col: str, tokenizer: Any, max_tokens: int
) -> tuple[pd.DataFrame, int]:
    """
    Return (filtered_df, total_matching_count) where rows are limited to prompts with <= max_tokens.
    """
    def _first_text(row: Any) -> str:
        v = row[text_col]
        return v[0] if isinstance(v, list) and v else (str(v) if v is not None else "")

    lengths = []
    for _, row in df.iterrows():
        text = _first_text(row)
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
    mask = pd.Series(lengths) <= int(max_tokens)
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

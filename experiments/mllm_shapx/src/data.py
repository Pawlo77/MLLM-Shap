"""Dataset loading, row selection, filtering, and prompt extraction utilities."""

import importlib
import logging
import operator
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import DatasetConfig, FilterPredicate
from .constants import (
    TRUE_JSON,
    DatasetSource,
    TextCol,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------
# DATASET LOADING
# ---------------------------


def load_df(cfg: DatasetConfig) -> pd.DataFrame:
    """
    Load dataset as DataFrame using the configured source strategy.

    Supports: hf_parquet, hf_datasets, local_parquet, local_csv.
    """
    if cfg.source == DatasetSource.HF_PARQUET:
        return _load_hf_parquet(cfg.repo_id, cfg.subset, cfg.split, cfg.revision)
    if cfg.source == DatasetSource.HF_DATASETS:
        return _load_hf_datasets(
            cfg.repo_id, cfg.subset, cfg.split, cfg.revision, cfg.trust_remote_code
        )
    if cfg.source == DatasetSource.LOCAL_PARQUET:
        return _load_local_parquet(cfg.path)
    if cfg.source == DatasetSource.LOCAL_CSV:
        return _load_local_csv(cfg.path)
    raise ValueError(f"Unsupported dataset source: {cfg.source}")


def _load_hf_parquet(
    repo_id: str, subset: str, split: str, revision: str
) -> pd.DataFrame:
    """Load the first parquet shard from HF Hub."""
    _ensure_pinned_revision(revision)
    filename = f"{subset}/{split}/0000.parquet"
    LOGGER.info(
        "Downloading dataset from HF Hub repo '%s', file '%s', revision '%s'",
        repo_id,
        filename,
        revision,
    )
    parquet_local_path = hf_hub_download(  # nosec B615
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_parquet(parquet_local_path)


def _load_hf_datasets(
    repo_id: str,
    subset: Optional[str],
    split: str,
    revision: str = "main",
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """Load dataset from HF using datasets library."""
    _ensure_pinned_revision(revision)
    LOGGER.info(
        "Loading dataset '%s', subset '%s', split '%s', revision '%s' via datasets lib",
        repo_id,
        subset,
        split,
        revision,
    )
    dataset = load_dataset(  # nosec B615
        repo_id,
        name=subset,
        split=split,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    return dataset.to_pandas()


def _load_local_parquet(path: Optional[str]) -> pd.DataFrame:
    """Load a local parquet file."""
    if not path:
        raise ValueError("dataset.path is required for local_parquet source.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Local parquet file not found: {resolved}")
    LOGGER.info("Loading local parquet: %s", resolved)
    return pd.read_parquet(resolved)


def _load_local_csv(path: Optional[str]) -> pd.DataFrame:
    """Load a local CSV file."""
    if not path:
        raise ValueError("dataset.path is required for local_csv source.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Local CSV file not found: {resolved}")
    LOGGER.info("Loading local CSV: %s", resolved)
    return pd.read_csv(resolved)


# Legacy aliases for backward compatibility
def load_single_sentence_df(
    repo_id: str, subset: str, split: str, revision: str
) -> pd.DataFrame:
    """Legacy: load HF parquet shard."""
    return _load_hf_parquet(repo_id, subset, split, revision)


def load_dataset_from_main(
    repo_id: str,
    subset: Optional[str],
    split: str,
    revision: str = "main",
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """Legacy: load via datasets library."""
    return _load_hf_datasets(repo_id, subset, split, revision, trust_remote_code)


# ---------------------------
# ROW FILTERING
# ---------------------------

_OP_MAP: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def apply_filters(df: pd.DataFrame, filters: List[FilterPredicate]) -> pd.DataFrame:
    """Apply a list of filter predicates to a DataFrame."""
    for filt in filters:
        col = filt.column
        if col not in df.columns:
            LOGGER.warning("Filter column '%s' not found in DataFrame; skipping.", col)
            continue

        if filt.op == "in":
            df = df[df[col].isin(filt.value)]
        elif filt.op == "not_in":
            df = df[~df[col].isin(filt.value)]
        elif filt.op == "between":
            lo, hi = filt.value[0], filt.value[1]
            df = df[(df[col] >= lo) & (df[col] <= hi)]
        elif filt.op in _OP_MAP:
            op_fn = _OP_MAP[filt.op]
            df = df[df[col].apply(lambda x, v=filt.value, fn=op_fn: fn(x, v))]
        else:
            LOGGER.warning("Unknown filter op '%s'; skipping.", filt.op)

    return df.reset_index(drop=True)


# ---------------------------
# TEXT EXTRACTION
# ---------------------------


def choose_prompt_text_column(df: pd.DataFrame, override: Optional[str] = None) -> str:
    """Pick the correct text column. Uses override if provided, else auto-detect."""
    if override:
        if override in df.columns:
            return override
        raise KeyError(f"Configured text column '{override}' not found in dataframe.")
    if TextCol.SENTENCES.value in df.columns:
        return TextCol.SENTENCES.value
    if TextCol.PROMPT.value in df.columns:
        return TextCol.PROMPT.value
    raise KeyError("Neither 'prompt' nor 'sentences' column found in dataframe.")


def extract_texts_from_row(value: Any) -> List[str]:
    """
    Extract texts from a row value as a list, handling single strings and lists.

    Returns:
        List of text strings (never empty - returns [""] for None/empty input).
    """
    if value is None:
        return [""]
    if isinstance(value, (list, numpy.ndarray)):
        value = value.tolist() if isinstance(value, numpy.ndarray) else value
        sentences = [str(s) for s in value if s is not None and str(s).strip()]
        return sentences if sentences else [""]
    text = str(value)
    return [text] if text.strip() else [""]


def extract_text_from_row(value: Any, separator: str = " ") -> str:
    """Extract text from a row value, concatenating lists with separator."""
    texts = extract_texts_from_row(value)
    return separator.join(texts)


# ---------------------------
# ROW ITERATION / SELECTION
# ---------------------------


def iter_rows_for_selection(
    df: pd.DataFrame,
    start_index: int,
    max_samples: Optional[int],
    shuffle_seed: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield (row_index, row_dict) with optional deterministic shuffling and slicing."""
    if shuffle_seed is not None:
        df = df.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    start = max(0, start_index)
    end = len(df) if max_samples is None else min(len(df), start + max_samples)

    for i in range(start, end):
        yield i, df.iloc[i].to_dict()


def iter_balanced_token_count_rows(
    df: pd.DataFrame,
    token_counts: List[int],
    samples_per_token_count: int,
    start_index: int,
    max_samples: Optional[int],
    shuffle_seed: Optional[int],
    allow_partial_buckets: bool,
    token_count_col: str = "token_count",
) -> List[Tuple[int, Dict[str, Any]]]:
    """Select a deterministic balanced slice by token_count buckets."""
    if token_count_col not in df.columns:
        raise KeyError(
            f"Balanced token-count selection requires a '{token_count_col}' column."
        )

    selected: List[Tuple[int, Dict[str, Any]]] = []
    rng = shuffle_seed if shuffle_seed is not None else 0
    for token_count in token_counts:
        bucket = df[df[token_count_col].astype(int) == int(token_count)]
        if shuffle_seed is not None:
            bucket = bucket.sample(frac=1.0, random_state=rng + int(token_count))
        if len(bucket) < samples_per_token_count:
            if not allow_partial_buckets:
                raise ValueError(
                    f"Only {len(bucket)} rows available for token_count={token_count}; "
                    f"need {samples_per_token_count}."
                )
            LOGGER.warning(
                "Only %d rows available for token_count=%s; using partial bucket.",
                len(bucket),
                token_count,
            )
        for row_idx, row in bucket.head(samples_per_token_count).iterrows():
            selected.append((int(row_idx), row.to_dict()))

    start = max(0, int(start_index))
    end = (
        len(selected)
        if max_samples is None
        else min(len(selected), start + max_samples)
    )
    return selected[start:end]


# ---------------------------
# UTILITIES
# ---------------------------


def get_hf_text_tokenizer() -> Any:
    """Lazily construct the HF tokenizer used by the Transformers text-only connector."""
    cfg_mod = importlib.import_module("mllm_shap.connectors.transformers_text.config")
    transformers = importlib.import_module("transformers")
    repo_id = getattr(cfg_mod, "CONFIG").repo_id
    revision = getattr(cfg_mod, "CONFIG").revision
    return getattr(transformers, "AutoTokenizer").from_pretrained(
        repo_id, revision=revision
    )  # nosec B615


def filter_df_by_max_prompt_tokens(
    df: pd.DataFrame, text_col: str, tokenizer: Any, max_tokens: int
) -> Tuple[pd.DataFrame, int]:
    """Return (filtered_df, total_matching_count) where rows have <= max_tokens."""
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
) -> Tuple[pd.DataFrame, int]:
    """Return (filtered_df, total_matching_count) where rows have >= min_tokens."""
    lengths = []
    for _, row in df.iterrows():
        text = extract_text_from_row(row[text_col])
        ids = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(ids))
    mask = pd.Series(lengths) >= int(min_tokens - 2)
    filtered = df[mask.values].reset_index(drop=True)
    return filtered, int(mask.sum())


def _ensure_pinned_revision(revision: str) -> None:
    """Enforce pinned HF revision unless explicitly overridden."""
    if os.environ.get("ALLOW_UNPINNED_HF_DOWNLOAD") == TRUE_JSON:
        return
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "HuggingFace 'revision' must be a 40-hex commit SHA. "
            "Set ALLOW_UNPINNED_HF_DOWNLOAD=1 to override."
        )

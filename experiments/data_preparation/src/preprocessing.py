"""Shared DataFrame preprocessing for English prompt-centric pipelines.

Provides small helpers to add sentence splits, language flags and perform
deduplication used across the English-focused data-preparation notebooks.
"""

import pandas as pd

from .languages import LanguageClassifier
from .nlp import split_into_sentences


def add_sentence_columns(
    df: pd.DataFrame,
    text_col: str = "prompt",
) -> pd.DataFrame:
    """Add ``sentences`` and ``sentences__num`` from *text_col*."""
    out = df.copy()
    out["sentences"] = out[text_col].progress_apply(split_into_sentences)
    out["sentences__num"] = out["sentences"].apply(len)
    return out


def add_english_flag(
    df: pd.DataFrame,
    classifier: LanguageClassifier,
    text_col: str = "prompt",
) -> pd.DataFrame:
    """Add ``is_english`` using *classifier*."""
    out = df.copy()
    out["is_english"] = out[text_col].progress_apply(classifier.is_english)
    return out


def non_english_prompts(
    df: pd.DataFrame,
    text_col: str = "prompt",
) -> pd.Series:
    """Return non-English prompts for inspection."""
    return df.loc[~df["is_english"], text_col]


def dedupe_by_prompt(
    df: pd.DataFrame,
    keep_audio: bool = False,
) -> pd.DataFrame:
    """Collapse duplicate prompts; merge source datasets into a sorted list."""
    agg: dict[str, str | object] = {
        "dataset": lambda x: sorted(set(x)),
        "sentences": "first",
        "sentences__num": "first",
    }
    if keep_audio:
        agg["audio__original"] = "first"
        agg["audio__original__duration"] = "first"

    return (
        df.groupby(["prompt"], as_index=False)
        .agg(agg)
        .rename(columns={"dataset": "datasets"})
    )


def filter_single_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with exactly one sentence."""
    return df[df["sentences__num"] == 1].copy()


def filter_multi_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with more than one sentence."""
    return df[df["sentences__num"] > 1].copy()


def add_datasets_combined(
    df: pd.DataFrame,
    datasets_col: str = "datasets",
) -> pd.DataFrame:
    """Join dataset name lists into a single stratification key."""
    out = df.copy()
    out["datasets__combined"] = out[datasets_col].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    )
    return out

"""Dataset saving utilities for builder notebooks.

Helpers to prepare DataFrames for persistence and write both Parquet
artifacts and human-readable JSON text samples used for quick inspection.
"""

from pathlib import Path

import pandas as pd

from .constants import (
    MULTI_SENTENCE__VOICE_BENCH,
    SINGLE_SENTENCE__VOICE_BENCH,
)
from .io import save_json, save_parquet

from .statistics import get_sample_df


def prepare_for_save(
    df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
    audio_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Drop temporary columns and reorder so audio columns come last."""
    if cols_to_drop is None:
        cols_to_drop = ["sentences__num", "datasets__combined", "interestingness_score"]
    if audio_cols is None:
        audio_cols = [
            "audio__original",
            "audio__original__duration",
            "audio__male",
            "audio__male__duration",
            "audio__female",
            "audio__female__duration",
        ]

    cols_present = [c for c in cols_to_drop if c in df.columns]
    df_out = df.drop(columns=cols_present)

    audio_present = [c for c in audio_cols if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in audio_present]
    df_out = df_out[other_cols + audio_present]
    return df_out


def save_dataset_and_sample(
    df: pd.DataFrame,
    data_dir: Path,
    name: str,
    group_col: str = "datasets",
    text_col: str = "sentences",
) -> pd.DataFrame:
    """Save a parquet file and a JSON text sample.

    Returns the sample DataFrame for display.
    """
    save_parquet(df, data_dir / f"{name}.parquet")
    print(f"Saved {len(df)} rows to {name}.parquet")

    sample_df = get_sample_df(df, group_col=group_col, text_col=text_col)
    save_json(sample_df, data_dir / f"{name}__text__sample.json")
    return sample_df


def save_single_sentence(
    df: pd.DataFrame,
    data_dir: Path,
    name: str = SINGLE_SENTENCE__VOICE_BENCH,
) -> pd.DataFrame:
    """Prepare NLP-filtered single-sentence data and save (prompt → sentence list)."""
    to_save = prepare_for_save(df)
    to_save["sentences"] = to_save["prompt"].progress_apply(lambda x: [x])
    to_save.drop(columns=["prompt"], inplace=True)
    return save_dataset_and_sample(to_save, data_dir, name)


def save_multi_sentence(
    df: pd.DataFrame,
    data_dir: Path,
    name: str = MULTI_SENTENCE__VOICE_BENCH,
) -> pd.DataFrame:
    """Save multi-sentence split without the raw prompt column."""
    to_save = df.drop(columns=["prompt"])
    return save_dataset_and_sample(to_save, data_dir, name)

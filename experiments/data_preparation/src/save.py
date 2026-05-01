"""Dataset saving utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..io import save_json, save_parquet

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

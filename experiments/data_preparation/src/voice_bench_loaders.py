"""Load and normalize VoiceBench source data.

Helpers to construct a unified DataFrame from VoiceBench dataset configs
and normalise audio entries for downstream processing.
"""

from collections.abc import Callable
from typing import cast

import pandas as pd
from datasets import DatasetDict
from .audio import normalize_original_audio_entry
from .constants import DatasetConfig


def load_voicebench_dataframe(
    load_dataset: Callable[[str], DatasetDict],
    config: DatasetConfig,
    skip_dataset_names: set[str] | frozenset[str],
) -> pd.DataFrame:
    """Load all VoiceBench configs, normalize audio, return a single DataFrame."""
    dt: list[tuple[str, str, str, dict]] = []
    for df_name, config_name in config.configs.items():
        if df_name in skip_dataset_names:
            continue

        print(f"Processing {df_name}...")
        splits = load_dataset(config_name)
        for split_name, split_ds in splits.items():
            print(f"\tSplit: {split_name} ({split_ds.num_rows} rows)")
            dt.extend([
                cast(
                    tuple[str, str, str, dict],
                    [entry["prompt"], df_name, split_name, entry["audio"]],
                )
                for entry in split_ds
            ])

    df = pd.DataFrame(dt, columns=["prompt", "dataset", "split", "audio__original"])
    normalized = df["audio__original"].progress_apply(normalize_original_audio_entry)
    df["audio__original"] = normalized.apply(lambda x: x[0])
    df["audio__original__duration"] = normalized.apply(lambda x: x[1])
    return df

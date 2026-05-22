"""LibriSpeech ASR text-first loading and deferred audio attachment.

Load text-only pools from clean LibriSpeech parquet splits and provide a
second-pass utility to attach audio bytes for a sampled subset of prompts.
"""

import gc
from typing import cast

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import list_repo_files

from .audio import to_audio_bytes_and_duration
from .constants import DatasetConfig

DATASET_LABEL: str = "LibriSpeech-ASR-clean"
"""Label to use in the 'dataset' column for rows originating from LibriSpeech ASR clean splits."""
TARGET_SPLITS: tuple[str, ...] = ("train.100", "train.360", "dev", "test")
"""Target splits to load from the LibriSpeech ASR clean dataset. The loader will attempt to resolve these to available split directories in the dataset repository, using defined aliases for common splits like 'dev' and 'test'. Only resolved splits with available parquet files will be loaded, and the loader will print information about available splits and any missing targets during the loading process."""
LOADER_VERSION: str = "clean_multi_split_text_first_v5"
"""Version identifier for this loading strategy, used for logging and debugging. The version string should be updated when making significant changes to the loading logic, such as how splits are resolved or how audio is attached, to help track which version of the loader was used for different data preparation runs and to facilitate debugging if issues arise with specific versions of the loading code."""

_SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "dev": ("dev", "dev.clean"),
    "test": ("test", "test.clean"),
}
"""Defined aliases for common split names in the LibriSpeech ASR dataset. This mapping allows the loader to resolve target split names like 'dev' and 'test' to the actual directory names used in the dataset repository, which may include suffixes like '.clean'. When the loader attempts to resolve splits, it will check for the presence of these aliases in the available splits and select the first match. This provides flexibility in handling different dataset layouts while still targeting the intended splits for loading."""


def _resolve_clean_splits(repo_files: list[str]) -> list[str]:
    """Resolve target clean splits to available split directories in the dataset repository."""
    available = sorted({
        f.split("/")[1]
        for f in repo_files
        if f.startswith("clean/") and f.endswith(".parquet") and len(f.split("/")) >= 3
    })
    resolved: list[str] = []
    for split_name in TARGET_SPLITS:
        candidates = _SPLIT_ALIASES.get(split_name, (split_name,))
        match = next((c for c in candidates if c in available), None)
        if match is None:
            print(f"Skipping missing split: {split_name} (candidates: {candidates})")
            continue
        resolved.append(match)
    if not resolved:
        raise ValueError("No target clean splits resolved. Check dataset layout.")
    return resolved


def _list_repo_parquet_files(config: DatasetConfig) -> list[str]:
    """List all parquet files in the dataset repository for the given configuration. This is used to discover available splits and shards for loading. The function checks for files that match the expected pattern of being located in a 'clean/{split}/' directory and having a '.parquet' extension, which is the expected layout for the LibriSpeech ASR clean splits. The resulting list of files is used by the loader to determine which splits are available and to construct data file paths for loading the datasets."""
    return list_repo_files(
        repo_id=config.dataset_name,
        repo_type="dataset",
        revision=config.revision,
    )


def load_librispeech_text_pool(config: DatasetConfig) -> pd.DataFrame:
    """Load transcript text from clean splits without audio columns."""
    print(f"Loader: {LOADER_VERSION}")
    repo_files = _list_repo_parquet_files(config)
    available_split_dirs = sorted({
        f.split("/")[1]
        for f in repo_files
        if f.startswith("clean/") and f.endswith(".parquet") and len(f.split("/")) >= 3
    })
    print("Available clean split dirs:", available_split_dirs)
    resolved_splits = _resolve_clean_splits(repo_files)
    print("Resolved splits:", resolved_splits)

    dt: list[tuple[str, str, str]] = []
    for split_name in resolved_splits:
        prefix = f"clean/{split_name}/"
        split_files = sorted(
            f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")
        )
        if not split_files:
            print(f"Skipping empty split prefix: {prefix}")
            continue

        print(f"Loading text-only pool from {prefix} ({len(split_files)} shards)")
        data_files = [
            f"hf://datasets/{config.dataset_name}@{config.revision}/{f}"
            for f in split_files
        ]
        split_ds = hf_load_dataset(
            "parquet",
            data_files={"train": data_files},
            split="train",
            cache_dir=str(config.cache_dir),
        )
        if "audio" in split_ds.column_names:
            split_ds = split_ds.remove_columns(["audio"])

        dt.extend([
            cast(tuple[str, str, str], [entry["text"], DATASET_LABEL, split_name])
            for entry in split_ds
        ])
        del split_ds
        gc.collect()

    df = pd.DataFrame(dt, columns=["prompt", "dataset", "split"])
    print(f"Text pool rows loaded: {len(df)}")
    return df


def attach_librispeech_audio(
    df: pd.DataFrame,
    config: DatasetConfig,
    prompt_col: str = "prompt",
) -> pd.DataFrame:
    """Attach ``audio__original`` for rows in *df* via a second pass over parquet shards."""
    selected_prompts = set(df[prompt_col].tolist())
    audio_lookup: dict[str, tuple[list[bytes], list[float]]] = {}

    repo_files = _list_repo_parquet_files(config)
    resolved_splits = _resolve_clean_splits(repo_files)

    for split_name in resolved_splits:
        if len(audio_lookup) == len(selected_prompts):
            break

        prefix = f"clean/{split_name}/"
        split_files = sorted(
            f for f in repo_files if f.startswith(prefix) and f.endswith(".parquet")
        )
        if not split_files:
            continue

        print(f"Attaching audio from {prefix} ({len(split_files)} shards)")
        data_files = [
            f"hf://datasets/{config.dataset_name}@{config.revision}/{f}"
            for f in split_files
        ]
        split_ds = hf_load_dataset(
            "parquet",
            data_files={"train": data_files},
            split="train",
            cache_dir=str(config.cache_dir),
        )

        for entry in split_ds:
            prompt = entry["text"]
            if prompt not in selected_prompts or prompt in audio_lookup:
                continue
            audio_lookup[prompt] = to_audio_bytes_and_duration(entry["audio"])
            if len(audio_lookup) == len(selected_prompts):
                break

        del split_ds
        gc.collect()

    missing = selected_prompts - set(audio_lookup.keys())
    if missing:
        raise ValueError(f"Missing audio for {len(missing)} sampled prompts")

    out = df.copy()
    out["audio__original"] = out[prompt_col].map(lambda p: audio_lookup[p][0])
    out["audio__original__duration"] = out[prompt_col].map(lambda p: audio_lookup[p][1])
    print(
        f"Attached audio for {len(audio_lookup)} / {len(selected_prompts)} sampled prompts"
    )
    return out

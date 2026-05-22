"""Hub dataset sanity-check helpers for the overview notebook.

Convenience functions to pick configs, load dataset splits, inspect rows,
and play audio from published parquets.
"""

import secrets
from pprint import pprint
from typing import Any

from datasets import DatasetDict, load_dataset
from IPython.display import Audio, display

from .constants import (
    HUB_DEFAULT_SPLIT,
    HUB_REPO_ID,
    MULTI_LINGUAL__INFINITY_INSTRUCT,
    MULTI_SENTENCE__VOICE_BENCH,
    SINGLE_SENTENCE__LIBRISPEECH_ASR,
    SINGLE_SENTENCE__VOICE_BENCH,
)

HUB_OVERVIEW_CONFIGS: tuple[str, ...] = (
    SINGLE_SENTENCE__VOICE_BENCH,
    MULTI_SENTENCE__VOICE_BENCH,
    SINGLE_SENTENCE__LIBRISPEECH_ASR,
    MULTI_LINGUAL__INFINITY_INSTRUCT,
)

PREFER_FEMALE_AUDIO_CONFIGS: frozenset[str] = frozenset({
    MULTI_LINGUAL__INFINITY_INSTRUCT
})


def summarize_hub_configs(
    revision: str,
    dataset_name: str = HUB_REPO_ID,
    configs: tuple[str, ...] = HUB_OVERVIEW_CONFIGS,
    split: str = HUB_DEFAULT_SPLIT,
) -> list[tuple[str, int]]:
    """Load each config and return ``(config_name, test_row_count)``."""
    counts: list[tuple[str, int]] = []
    for config in configs:
        ds = load_hub_dataset(dataset_name, config, revision)
        n = len(ds[split])
        counts.append((config, n))
        print(f"{config}: {n} rows ({split})")
    return counts


def pick_random_config(configs: tuple[str, ...]) -> str:
    """Choose a Hub config name and print it."""
    config = secrets.choice(configs)
    print(f"Using config: {config}")
    return config


def load_hub_dataset(
    dataset_name: str,
    config: str,
    revision: str,
) -> DatasetDict:
    """Load all splits for a Hub config at a pinned revision."""
    return load_dataset(
        dataset_name,
        config,
        revision=revision,  # nosec B615
    )


def inspect_random_test_row(ds: DatasetDict) -> dict[str, Any]:
    """Print keys for a random test-split row."""
    sample_entry: dict[str, Any] = secrets.choice(ds["test"])
    pprint(sample_entry.keys())
    return sample_entry


def play_first_audio_clip(
    entry: dict[str, Any],
    prefer_female: bool = False,
) -> None:
    """Play the first clip from ``audio__male`` or ``audio__female``."""
    col = (
        "audio__female" if prefer_female and "audio__female" in entry else "audio__male"
    )
    display(Audio(data=entry[col][0], autoplay=True))


def show_text_content(entry: dict[str, Any]) -> None:
    """Print sentence lists or legacy prompt text."""
    if "sentences" in entry:
        pprint(entry["sentences"])
    elif "prompt" in entry:
        pprint(entry["prompt"])


def show_prompt(entry: dict[str, Any]) -> None:
    """Backward-compatible alias for :func:`show_text_content`."""
    show_text_content(entry)

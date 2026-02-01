"""Single sentence experiments analysis utilities."""

import os

import pandas as pd

from .common import (
    EXPERIMENTS_DIR as BASE_EXPERIMENTS_DIR,
    load_experiments_results as base_load_experiments_results,
    FIGURES_DIR as BASE_FIGURES_DIR,
    STATS_DIR as BASE_STATS_DIR,
)

# sns.set_style("whitegrid")


EXPERIMENTS_DIR: str = os.path.join(BASE_EXPERIMENTS_DIR, "single_sentence_2026_01_03")
"""Single sentence experiments outputs directory."""

FIGURES_DIR = os.path.join(BASE_FIGURES_DIR, "single_sentence")
os.makedirs(FIGURES_DIR, exist_ok=True)

STATS_DIR = os.path.join(BASE_STATS_DIR, "single_sentence")
os.makedirs(STATS_DIR, exist_ok=True)

CASES: dict[str, str] = {
    "T2T": "text_text_limited_neyman_lin3_0",
    "T2S": "text_audio_limited_neyman_lin3_0",
    "SM2T": "audio_male_text_limited_neyman_lin3_0",
    "SM2S": "audio_male_audio_limited_neyman_lin3_0",
    "SF2T": "audio_female_text_limited_neyman_lin3_0",
    "SF2S": "audio_female_audio_limited_neyman_lin3_0",
}
"""Mapping of experiment case codes to directory names."""


def load_experiments_results(case: str) -> pd.DataFrame:
    """Load experiment results for a given case."""
    return base_load_experiments_results(
        case, cases=CASES, experiments_dir=EXPERIMENTS_DIR
    )

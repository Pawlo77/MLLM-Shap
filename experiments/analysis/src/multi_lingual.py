"""Multi lingual experiments analysis utilities."""

import os

import pandas as pd
import seaborn as sns

from .common import (
    EXPERIMENTS_DIR as BASE_EXPERIMENTS_DIR,
    load_experiments_results as base_load_experiments_results,
    FIGURES_DIR as BASE_FIGURES_DIR,
    STATS_DIR as BASE_STATS_DIR,
)

sns.set_style("whitegrid")


EXPERIMENTS_DIR: str = os.path.join(BASE_EXPERIMENTS_DIR, "multi_lingual_2026_01_03")
"""Multi lingual experiments outputs directory."""

FIGURES_DIR = os.path.join(BASE_FIGURES_DIR, "multi_lingual")
os.makedirs(FIGURES_DIR, exist_ok=True)

STATS_DIR = os.path.join(BASE_STATS_DIR, "multi_lingual")
os.makedirs(STATS_DIR, exist_ok=True)

CASES: dict[str, str] = {
    "T2T": "text_text_limited_neyman_lin3_0",
    "SM2T": "audio_male_text_limited_neyman_lin3_0",
    "SF2T": "audio_female_text_limited_neyman_lin3_0",
}
"""Mapping of experiment case codes to directory names."""


def load_experiments_results(case: str) -> pd.DataFrame:
    """Load experiment results for a given case."""
    return base_load_experiments_results(
        case, cases=CASES, experiments_dir=EXPERIMENTS_DIR, is_multi_lingual=True
    )

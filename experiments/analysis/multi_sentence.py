"""Multi-sentence experiments analysis utilities."""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Literal

import pandas as pd

from analysis_common import LoadedCase, build_units_dataframe, ensure_language_columns, load_case_results

LANGUAGE_COL: str = "language"
ORIGINAL_LANGUAGE_COL: str = "original_language"


DEFAULT_RUN: str = "multi_sentence_2026_01_03"

CASES: dict[str, str] = {
    "T2T": "text_text_limited_neyman_lin3_0",
    "A2T": "audio_text_limited_neyman_lin3_0",
    "I_TF_M": "interleaved__text_first__male_text_limited_neyman_lin3_0",
    "I_AF_M": "interleaved__audio_first__male_text_limited_neyman_lin3_0",
}


def load_experiments_results(
    case: Literal["T2T", "A2T", "I_TF_M", "I_AF_M"],
    run: str = DEFAULT_RUN,
) -> pd.DataFrame:
    """Load multi-sentence experiment results for a given case."""

    loaded: LoadedCase = load_case_results(run_name=run, case_dir=CASES[case])
    df = loaded.df.copy()
    df["case"] = case
    df["run"] = run

    # Keep as nullable columns if missing.
    df = ensure_language_columns(df)
    # (constants retained for readability and to satisfy lint rules)
    _ = (LANGUAGE_COL, ORIGINAL_LANGUAGE_COL)
    return df


def build_units_df(
    case: Literal["T2T", "A2T", "I_TF_M", "I_AF_M"],
    run: str = DEFAULT_RUN,
) -> pd.DataFrame:
    """Build a long-form explainable-units DataFrame."""

    results_df = load_experiments_results(case=case, run=run)
    units_df = build_units_dataframe(results_df)
    units_df["case"] = case
    units_df["run"] = run
    return units_df

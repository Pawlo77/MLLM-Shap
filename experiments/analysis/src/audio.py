"""Audio experiments analysis utilities."""

from typing import Callable

import pandas as pd

CASES: list[str] = ["SM2T", "SM2S", "SF2T", "SF2S"]
"""Supported audio experiment cases."""


def get_audio_df(
    load_experiments_results_callable: Callable[[str], pd.DataFrame],
    cases: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess audio experiment results for a given case.
    Includes SM2T, SM2S, SF2T, SF2S cases.
    Performs basic validation checks.
    """
    if cases is None:
        cases = CASES

    audio_df = pd.DataFrame()
    for case in cases:
        if case not in CASES:
            raise ValueError(f"Unsupported case: {case}")

        case_df = load_experiments_results_callable(case)
        case_df["mode"] = case
        audio_df = pd.concat([audio_df, case_df], ignore_index=True)

    # Preprocess audio experiment results
    audio_df["count_explainable_tokens"] = audio_df["audio_segments"].apply(
        lambda x: len(x["2"]) if "2" in x else None
    )
    audio_df["inputs"] = audio_df["audio_segments"].apply(
        lambda x: " ".join(y["token"] for y in x["2"]) if "2" in x else None
    )
    audio_df["tokens"] = audio_df["audio_segments"].apply(
        lambda x: [y["token"] for y in x["2"]] if "2" in x else None
    )
    audio_df["sv"] = audio_df["audio_segments"].apply(
        lambda x: [y["shap_value"] for y in x["2"]] if "2" in x else None
    )
    audio_df["raw_model_response"] = audio_df["conversation"].apply(
        lambda x: "".join(x[2][0]["content"])
    )

    if (audio_df["count_explainable_tokens"] != audio_df["sv"].apply(len)).any().item():
        raise RuntimeError("Audio token count and shap values length mismatch.")
    for c in ["inputs", "tokens", "sv"]:
        if audio_df[c].isna().any().item():
            raise RuntimeError(f"Column {c} contains NaN values.")

    # Select relevant columns
    audio_df = audio_df[
        [
            "row_index",
            "tokens",
            "sv",
            "inputs",
            "count_explainable_tokens",
            "mode",
            "neyman_steps",
            "n_calls",
            "runtime_sec",
            "raw_model_response",
            "language",
        ]
    ]

    return audio_df

"""Text experiments analysis utilities."""

from typing import Callable
import pandas as pd
import numpy as np
from .common import check_roles

CASES: list[str] = ["T2S", "T2T"]
"""Supported text experiment cases."""


def _filter_valid_sv(row: pd.Series) -> pd.Series:
    """Filter valid shap values and tokens."""
    sv = np.asarray(row["sv"])
    valid_mask = ~pd.isna(sv)
    tokens = np.asarray(row["raw_prompt_text_tokens"])[valid_mask]
    return pd.Series({
        "sv": sv[valid_mask],
        "tokens": tokens,
        "inputs": "".join(tokens),
        "row_index": row["row_index"],
        "count_explainable_tokens": len(tokens),
        "neyman_steps": row["neyman_steps"],
        "n_calls": row["n_calls"],
        "runtime_sec": row["runtime_sec"],
        "raw_model_response": row["raw_model_response"],
        "mode": row["mode"],
        "language": row.get("language", None),
    })


def get_text_df(
    load_experiments_results_callable: Callable[[str], pd.DataFrame],
    cases: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and preprocess text experiment results for a given case.
    Includes T2S, T2T cases.
    Performs basic validation checks.
    """
    if cases is None:
        cases = CASES

    text_df = pd.DataFrame()
    for case in cases:
        if case not in CASES:
            raise ValueError(f"Unsupported case: {case}")

        case_df = load_experiments_results_callable(case)
        case_df["mode"] = case
        text_df = pd.concat([text_df, case_df], ignore_index=True)

    # Preprocess text experiment results
    text_df["count_explainable_tokens"] = text_df["attr_summary"].apply(
        lambda x: x["count_text_tokens"]
    )
    text_df["prompt_text_roles"] = text_df["conversation"].apply(
        lambda x: x[1][0]["roles"]
    )
    text_df["prompt_text"] = text_df["prompt_texts"].apply(lambda x: x[0])
    text_df["sv"] = text_df["conversation"].apply(lambda x: x[1][0]["shap_values"])

    text_df["raw_model_response"] = text_df["conversation"].apply(
        lambda x: "".join(x[2][0]["content"])
    )
    text_df["raw_prompt_text"] = text_df["conversation"].apply(
        lambda x: "".join(x[1][0]["content"])
    )
    text_df["raw_prompt_text_tokens"] = text_df["conversation"].apply(
        lambda x: x[1][0]["content"]
    )
    if not (
        text_df[["prompt_text", "raw_prompt_text"]]
        .apply(lambda row: row["prompt_text"] in row["raw_prompt_text"], axis=1)
        .all()
        .item()
        and text_df[["prompt_text_roles", "sv"]].apply(check_roles, axis=1).all()
    ).item():
        raise RuntimeError("Prompt text or roles check failed.")

    # Filter valid shap values and corresponding tokens
    text_df = text_df.apply(_filter_valid_sv, axis=1)

    return text_df

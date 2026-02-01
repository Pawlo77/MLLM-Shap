"""Multi-sentence experiments analysis utilities.

This dataset differs from single-sentence in that the user prompt may span
multiple user turns (multiple sentences). We therefore standardize the prompt by
collecting explainable units (tokens / audio segments) across all user turns.

The analysis also compares two experiment runs: with SGPA and without SGPA.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .common import (
    EXPERIMENTS_DIR as BASE_EXPERIMENTS_DIR,
    FIGURES_DIR as BASE_FIGURES_DIR,
    STATS_DIR as BASE_STATS_DIR,
    load_experiments_results as base_load_experiments_results,
)


EXPERIMENTS_DIR_WITH_SGPA: str = os.path.join(
    BASE_EXPERIMENTS_DIR, "multi_sentence_2026_01_03"
)
"""Multi-sentence experiments outputs directory (with SGPA)."""

EXPERIMENTS_DIR_WITHOUT_SGPA: str = os.path.join(
    BASE_EXPERIMENTS_DIR, "multi_sentence_2025_06_12"
)
"""Multi-sentence experiments outputs directory (without SGPA)."""


FIGURES_DIR = os.path.join(BASE_FIGURES_DIR, "multi_sentence")
os.makedirs(FIGURES_DIR, exist_ok=True)

STATS_DIR = os.path.join(BASE_STATS_DIR, "multi_sentence")
os.makedirs(STATS_DIR, exist_ok=True)


CASES: dict[str, str] = {
    "T2T": "text_text_limited_neyman_lin3_0",
    "SM2T": "audio_text_limited_neyman_lin3_0",
    "ITF_SM2T": "interleaved__text_first__male_text_limited_neyman_lin3_0",
    "IAF_SM2T": "interleaved__audio_first__male_text_limited_neyman_lin3_0",
}
"""Mapping of experiment case codes to directory names."""


def load_experiments_results(case: str) -> pd.DataFrame:
    """Load experiment results for a given case, for both SGPA settings."""
    with_sgpa = base_load_experiments_results(
        case, cases=CASES, experiments_dir=EXPERIMENTS_DIR_WITH_SGPA, strict=False
    )
    with_sgpa["sgpa"] = True

    without_sgpa = base_load_experiments_results(
        case, cases=CASES, experiments_dir=EXPERIMENTS_DIR_WITHOUT_SGPA, strict=False
    )
    without_sgpa["sgpa"] = False

    return pd.concat([with_sgpa, without_sgpa], ignore_index=True)


def _safe_join_content(content: Any) -> str:
    if isinstance(content, list):
        # content is typically a list[str] for text.
        # For audio, it can be list[dict] with _binary; we do not join those.
        if all(isinstance(x, str) for x in content):
            return "".join(content)
    return ""


def _collect_tokens_and_sv(user_turns: list[list[dict[str, Any]]]) -> tuple[list[str], list[float]]:
    """Collect explainable units (tokens / audio placeholders) and their Shapley values.

    Multi-sentence prompts may span multiple user turns; each turn can contain
    multiple blocks (text or audio). We concatenate all explainable units in the
    order they appear.
    """

    tokens: list[str] = []
    sv: list[float] = []
    audio_counter = 0

    for turn in user_turns:
        for msg in turn:
            shap_values = msg.get("shap_values")
            if shap_values is None:
                continue

            content = msg.get("content", [])
            content_type = msg.get("content_type")

            for i, s in enumerate(shap_values):
                if s is None:
                    continue

                if content_type == 0:
                    tok = content[i] if i < len(content) else ""
                else:
                    tok = f" <AUDIO_{audio_counter}>"
                    audio_counter += 1

                tokens.append(tok)
                sv.append(float(s))

    return tokens, sv


def _standardize_row(row: pd.Series) -> pd.Series:
    """Convert a raw experiment row into standardized analysis columns."""
    conv = row["conversation"]

    # System prompt is conv[0], assistant reply is always the last turn.
    user_turns = conv[1:-1]

    tokens, sv = _collect_tokens_and_sv(user_turns)

    if len(tokens) != len(sv):
        raise RuntimeError("Token/shap length mismatch after standardization.")

    if len(tokens) == 0:
        raise RuntimeError(
            f"No explainable tokens found for row_index={row.get('row_index', None)}."
        )

    # For this dataset we treat language as English (metadata may be 'unknown').
    language = row.get("language", "en")
    if not isinstance(language, str) or language.lower() in {"unknown", ""}:
        language = "en"

    assistant_msg = conv[-1][0] if conv and conv[-1] else {}

    return pd.Series(
        {
            "row_index": row["row_index"],
            "tokens": tokens,
            "sv": sv,
            "inputs": "".join(tokens),
            "count_explainable_tokens": len(tokens),
            "mode": row["mode"],
            "sgpa": row.get("sgpa", None),
            "neyman_steps": row.get("neyman_steps", None),
            "n_calls": row.get("n_calls", None),
            "runtime_sec": row.get("runtime_sec", None),
            "raw_model_response": _safe_join_content(assistant_msg.get("content")),
            "language": language,
            "prompt_texts": row.get("prompt_texts", None),
            "input_modality": row.get("input_modality", None),
            "output_modality": row.get("output_modality", None),
        }
    )


def get_multi_sentence_df(cases: list[str] | None = None) -> pd.DataFrame:
    """Load + standardize multi-sentence experiment results.

    Returns a DataFrame with the same core columns used by the other analyses:
    - row_index, tokens, sv, inputs, count_explainable_tokens
    - mode, sgpa, neyman_steps, n_calls, runtime_sec, raw_model_response, language
    """
    if cases is None:
        cases = list(CASES.keys())

    df = pd.DataFrame()
    for case in cases:
        if case not in CASES:
            raise ValueError(f"Unsupported case: {case}")

        case_df = load_experiments_results(case)
        case_df["mode"] = case

        standardized = case_df.apply(_standardize_row, axis=1)
        df = pd.concat([df, standardized], ignore_index=True)

    # Consistent ordering in plots.
    df["sgpa"] = pd.Categorical(df["sgpa"], categories=[True, False], ordered=True)

    return df

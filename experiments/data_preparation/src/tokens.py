"""Token counting using LiquidAudio model."""

from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from mllm_shap.connectors import LiquidAudio
from mllm_shap.connectors.enums import Role, SystemRolesSetup
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter


def compute_token_counts(
    input_df: pd.DataFrame,
    model: LiquidAudio,
    text_column: str = "prompt",
) -> list[int]:
    """Compute number of explainability tokens (mask sum) for single-turn text entries.

    Parameters
    ----------
    input_df    : DataFrame with a *text_column* containing raw text.
    model       : Pre-initialized LiquidAudio model for tokenization.
    text_column : Column name containing text to tokenize.
    """
    counts: list[int] = []
    for text in tqdm(input_df[text_column].tolist(), desc="token counting"):
        chat = model.get_new_chat(
            system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
            token_filter=ExcludePunctuationTokensFilter(),
        )
        chat.new_turn(Role.SYSTEM)
        chat.add_text("You are a helpful assistant.")
        chat.end_turn()
        chat.new_turn(Role.USER)
        chat.add_text(text)
        chat.end_turn()
        counts.append(int(chat.shap_values_mask.sum().item()))
    return counts


def compute_multi_turn_token_counts(
    input_df: pd.DataFrame,
    model: LiquidAudio,
    sentences_column: str = "sentences",
) -> list[int]:
    """Compute number of explainability tokens for multi-turn conversations.

    Each sentence in the list is treated as a separate USER turn.

    Parameters
    ----------
    input_df         : DataFrame with a *sentences_column* containing lists of strings.
    model            : Pre-initialized LiquidAudio model for tokenization.
    sentences_column : Column name containing sentence lists.
    """
    counts: list[int] = []
    for sentences in tqdm(
        input_df[sentences_column].tolist(), desc="multi-turn token counting"
    ):
        chat = model.get_new_chat(
            system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT,
            token_filter=ExcludePunctuationTokensFilter(),
        )
        chat.new_turn(Role.SYSTEM)
        chat.add_text("You are a helpful assistant.")
        chat.end_turn()
        for sentence in sentences:
            chat.new_turn(Role.USER)
            chat.add_text(sentence)
            chat.end_turn()
        counts.append(int(chat.shap_values_mask.sum().item()))
    return counts


def add_token_counts_and_filter(
    df: pd.DataFrame,
    model: LiquidAudio,
    max_token_count: int = 10,
    text_column: str = "prompt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute token counts and return (full df with counts, filtered candidates).

    Parameters
    ----------
    df             : Input DataFrame.
    model          : LiquidAudio model for tokenization.
    max_token_count: Maximum token count threshold for candidates.
    text_column    : Column containing text to tokenize.

    Returns
    -------
    (df_with_counts, candidates) where candidates have token_count <= max_token_count.
    """
    df["token_count"] = compute_token_counts(df, model=model, text_column=text_column)
    print(f"Computed token counts for {len(df)} entries.")

    candidates = df[df["token_count"] <= max_token_count].copy().reset_index(drop=True)
    print(f"Candidates with token_count <= {max_token_count}: {len(candidates)}")
    return df, candidates

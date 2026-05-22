"""Token counting utilities using the LiquidAudio model.

Functions to compute single- and multi-turn explainability token counts
using a provided ``LiquidAudio`` model and to filter candidates by token
budget for dataset selection.
"""

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
    """Compute number of explainability tokens (mask sum) for single-turn text entries."""
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
    """Compute token counts and return (full df with counts, filtered candidates)."""
    df["token_count"] = compute_token_counts(df, model=model, text_column=text_column)
    print(f"Computed token counts for {len(df)} entries.")

    candidates = df[df["token_count"] <= max_token_count].copy().reset_index(drop=True)
    print(f"Candidates with token_count <= {max_token_count}: {len(candidates)}")
    return df, candidates

"""Multi-sentence dataset preparation utilities.

Helpers to filter, compute token budgets and sample multi-sentence
candidates for the multi-turn VoiceBench arm used by the builder notebooks.
"""

import pandas as pd

from mllm_shap.connectors import LiquidAudio

from .preprocessing import add_datasets_combined
from .reporting import plot_token_sampling_stages
from .sampling import sample_fraction_by_groups, stratified_sample
from .tokens import compute_multi_turn_token_counts


def prepare_multi_sentence_candidates(
    df: pd.DataFrame,
    model: LiquidAudio,
    max_sentences: int = 8,
    max_token_count: int = 30,
    sentences_column: str = "sentences",
) -> pd.DataFrame:
    """Filter by sentence count and token budget; add token counts."""
    out = df[df["sentences__num"] <= max_sentences].copy()
    out["token_count"] = compute_multi_turn_token_counts(
        out, model=model, sentences_column=sentences_column
    )
    print(f"Computed token counts for {len(out)} entries.")
    out = out[out["token_count"] <= max_token_count].copy()
    print(f"Candidates with token_count <= {max_token_count}: {len(out)}")
    return out


def sample_multi_sentence(
    df: pd.DataFrame,
    n_target: int,
    dataset_frac: float = 0.35,
    random_state: int = 0,
) -> pd.DataFrame:
    """Stratify by dataset×sentence count, then token-balanced sample."""
    out = add_datasets_combined(df)
    out["sentences__num__grp"] = out["sentences__num"]
    out = sample_fraction_by_groups(
        out,
        group_cols=["datasets__combined", "sentences__num__grp"],
        frac=dataset_frac,
        random_state=random_state,
    )
    print(f"Size after dataset×sentence stratification: {len(out)}")

    result = stratified_sample(
        pool=out.reset_index(drop=True),
        n_target=n_target,
        strat_col="token_count",
        random_state=random_state,
    )
    print("Selected subset size:", len(result))
    plot_token_sampling_stages(
        out,
        result,
        before_title="Token counts — after dataset×sentence stratification",
        after_title=f"Token counts — final {n_target}-sample dataset",
    )
    return result

"""Single-sentence sampling pipeline helpers.

Functions to run NLP-quality filtering, compute token budgets and perform
stratified sampling for the single-sentence dataset builder pipelines.
"""

import pandas as pd

from mllm_shap.connectors import LiquidAudio

from .filters import nlp_quality_filter
from .sampling import sample_fraction_by_group, stratified_sample
from .statistics import plot_token_count_comparison
from .tokens import add_token_counts_and_filter


def run_nlp_quality_filter(
    df: pd.DataFrame,
    device: str,
) -> tuple[pd.DataFrame, object]:
    """Apply embedding-based quality filtering."""
    return nlp_quality_filter(df, device=device)


def run_token_count_filter(
    df: pd.DataFrame,
    model: LiquidAudio,
    max_token_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute token counts and return (full df, candidates within budget)."""
    return add_token_counts_and_filter(df, model, max_token_count=max_token_count)


def prepare_candidates(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``datasets__combined`` for stratified sampling."""
    out = candidates.copy()
    out["datasets__combined"] = out["datasets"].apply(lambda x: " ".join(x))
    return out


def sample_single_sentence_100(
    candidates: pd.DataFrame,
    n_target: int,
    dataset_frac: float = 0.5,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Halve per combined dataset source, then stratified sample to *n_target*.

    Returns ``(final_sample, after_dataset_stratification)`` for reporting plots.
    """
    stratified = sample_fraction_by_group(
        candidates,
        group_col="datasets__combined",
        frac=dataset_frac,
        random_state=random_state,
    )
    print(f"Size after dataset stratification: {len(stratified)}")
    final = stratified_sample(
        pool=stratified,
        n_target=n_target,
        strat_col="token_count",
        random_state=random_state,
    )
    return final, stratified


def sample_single_sentence_nk(
    candidates: pd.DataFrame,
    n_target: int,
    random_state: int = 0,
    plot_comparison: bool = True,
    plot_label: int | str | None = None,
) -> pd.DataFrame:
    """Stratified sample directly from the candidate pool."""
    result = stratified_sample(
        pool=candidates,
        n_target=n_target,
        strat_col="token_count",
        random_state=random_state,
    )
    print(f"Final dataset size: {len(result)}")
    if plot_comparison and plot_label is not None:
        plot_token_count_comparison(candidates, result, plot_label)
    return result

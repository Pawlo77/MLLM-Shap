"""Notebook reporting helpers: stats prints and sampling plots.

Utilities for printing dataset summaries and plotting sampling-stage
diagnostics used during interactive data preparation and exploratory work.
"""

from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .statistics import compute_budgets, get_df_stats, get_df_stats__by_source


def print_datasets_list_length_counts(
    df: pd.DataFrame,
    datasets_col: str = "datasets",
) -> None:
    """Print how many source datasets each row lists."""
    pprint(df[datasets_col].apply(len).value_counts().sort_index())


def value_counts_at_least(series: pd.Series, minimum: int) -> pd.DataFrame:
    """Return value counts with count >= *minimum*."""
    counts = series.value_counts()
    return counts[counts >= minimum].reset_index()


def report_dataset_stats(
    df: pd.DataFrame,
    text_col: str = "prompt",
    group_col: str = "datasets",
    sentences_num_col: str = "sentences__num",
    include_audio: bool = True,
    show_budgets: bool = True,
    token_col: str = "token_count",
) -> pd.DataFrame:
    """Print aggregate stats and budgets; return by-source table for display."""
    by_source = get_df_stats__by_source(
        df,
        group_col=group_col,
        text_col=text_col,
        sentences_num_col=sentences_num_col,
    )
    pprint(get_df_stats(df, text_col=text_col, include_audio=include_audio))
    if show_budgets and token_col in df.columns:
        pprint(compute_budgets(df[token_col]))
    return by_source


def plot_interestingness_distribution(
    df: pd.DataFrame,
    score_col: str = "interestingness_score",
) -> None:
    """Histogram of interestingness scores after NLP filtering."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df[score_col], bins=40, kde=True)
    plt.title("Interestingness score distribution")
    plt.xlabel(score_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_token_sampling_stages(
    before_sample: pd.DataFrame,
    after_sample: pd.DataFrame,
    before_title: str,
    after_title: str,
    token_col: str = "token_count",
) -> None:
    """Side-by-side token-count histograms for an intermediate and final sample."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(before_sample[token_col], bins=30, kde=False, ax=axes[0])
    axes[0].set_title(before_title)
    axes[0].set_xlabel(token_col)

    sns.histplot(
        after_sample[token_col], bins=30, color="orange", kde=False, ax=axes[1]
    )
    axes[1].set_title(after_title)
    axes[1].set_xlabel(token_col)

    plt.tight_layout()
    plt.show()

    print("Token count summary (final):")
    print(after_sample[token_col].describe())

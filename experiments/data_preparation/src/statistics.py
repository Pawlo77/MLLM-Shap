"""DataFrame statistics helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_sample_df(
    df_to_sample: pd.DataFrame,
    group_col: str = "datasets",
    text_col: str = "sentences",
) -> pd.DataFrame:
    """For each group, get the sample with the longest text (in sentences).

    Parameters
    ----------
    df_to_sample : DataFrame with *group_col* and *text_col* columns.
    group_col    : Column to explode/group on.
    text_col     : Column whose list-length determines "longest".

    Returns
    -------
    DataFrame with one row per group containing the longest entry.
    """
    select_cols = [group_col, text_col]
    return (
        df_to_sample.explode(group_col)
        .groupby(group_col, group_keys=False)
        .apply(lambda g: g.loc[g[text_col].apply(len).idxmax()], include_groups=False)
        .sort_index()
        .reset_index()[select_cols]
    )


def get_df_stats(
    df_to_analyze: pd.DataFrame,
    text_col: str = "prompt",
    sentences_col: str = "sentences",
    include_audio: bool = True,
) -> dict[str, float]:
    """Compute summary statistics for a dataset DataFrame.

    Parameters
    ----------
    df_to_analyze : DataFrame with at least *text_col* and *sentences_col*.
    text_col      : Column with raw text strings.
    sentences_col : Column with lists of sentences.
    include_audio : Whether to include audio duration statistics.
    """
    characters_num = df_to_analyze[text_col].apply(len)
    sentences_num = df_to_analyze[sentences_col].apply(len)
    unique_entries = df_to_analyze[text_col].nunique()

    stats: dict[str, float] = {
        "rows__num": len(df_to_analyze),
        "characters__num": round(characters_num.sum().item(), 2),
        "avg_characters__num": round(characters_num.mean().item(), 2),
        "total_sentences__num": round(sentences_num.sum().item(), 2),
        "avg_sentences__num": round(sentences_num.mean().item(), 2),
        "min_sentences__num": sentences_num.min().item(),
        "unique_entries__num": unique_entries,
        "unique_entries__pct": round(unique_entries / len(df_to_analyze) * 100, 2),
    }

    if include_audio:
        for name in ("female", "male", "original"):
            col = f"audio__{name}__duration"
            if col not in df_to_analyze.columns:
                continue
            durations = df_to_analyze[col].apply(
                lambda x: (
                    sum(sum(e) for e in x)
                    if isinstance(x, list) and x and isinstance(x[0], list)
                    else sum(x)
                    if isinstance(x, list)
                    else x
                )
            )
            stats[f"audio__{name}__duration__average"] = round(
                durations.mean().item(), 2
            )
            stats[f"audio__{name}__duration__total"] = round(durations.sum().item(), 2)

    return stats


def get_df_stats__by_source(
    df_to_analyze: pd.DataFrame,
    group_col: str = "datasets",
    text_col: str = "prompt",
    sentences_num_col: str = "sentences__num",
) -> pd.DataFrame:
    """Statistics grouped by source dataset.

    Parameters
    ----------
    df_to_analyze    : DataFrame with *group_col *text_col and *sentences_num_col*.
    group_col        : Column to explode and group on.
    text_col         : Column with raw text strings.
    sentences_num_col: Column with sentence counts.
    """
    return (
        df_to_analyze.explode(group_col)
        .groupby(group_col)
        .agg(
            num_rows=(text_col, "count"),
            total_characters=(text_col, lambda x: x.str.len().sum()),
            avg_num_characters=(text_col, lambda x: x.str.len().mean()),
            avg_num_sentences=(sentences_num_col, "mean"),
            total_sentences=(sentences_num_col, "sum"),
        )
        .reset_index()
        .sort_values(group_col)
    )


def compute_budgets(token_counts: pd.Series) -> dict[str, float]:
    """Compute Shapley-value budget estimates for different strategies."""
    return {
        "n^2": sum(min(2**t, t**2) for t in token_counts),
        "2n^2": sum(min(2**t, 2 * t**2) for t in token_counts),
        "3n^2": sum(min(2**t, 3 * t**2) for t in token_counts),
        "4n^2": sum(min(2**t, 4 * t**2) for t in token_counts),
        "frac 0.1": sum(0.1 * 2**t for t in token_counts),
        "frac 0.2": sum(0.2 * 2**t for t in token_counts),
    }


def plot_token_count_comparison(
    pool_df: pd.DataFrame,
    final_df: pd.DataFrame,
    n_label: int | str,
    token_col: str = "token_count",
) -> None:
    """Side-by-side histogram of token counts: candidate pool vs final sample."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(pool_df[token_col], bins=30, kde=False, ax=axes[0])
    axes[0].set_title("Token counts — full candidate pool")
    axes[0].set_xlabel(token_col)

    sns.histplot(final_df[token_col], bins=30, color="orange", kde=False, ax=axes[1])
    axes[1].set_title(f"Token counts — final {n_label}-sample dataset")
    axes[1].set_xlabel(token_col)

    plt.tight_layout()
    plt.show()

    print("Token count summary (final):")
    print(final_df[token_col].describe())

"""Plotting functions for experiments analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# pylint: disable=too-many-positional-arguments,too-many-arguments
def plot_token_count_distribution(
    plot_df: pd.DataFrame,
    column: str = "count_text_tokens",
    plot_ax: plt.Axes | None = None,
    hue: str | None = None,
    suffix: str = "",
    legend: bool = True,
) -> None:
    """Plot the distribution of text token counts."""
    if plot_ax is None:
        _, plot_ax = plt.subplots(1, 1, figsize=(8, 5))

    sns.histplot(
        data=plot_df,
        x=column,
        bins=plot_df[column].nunique(),
        palette="Set2",
        stat="percent",
        ax=plot_ax,
        hue=hue,
        multiple="dodge",
        shrink=0.8,
        legend=legend,
    )
    plot_ax.set_xlabel(f"Count of {column.replace('_', ' ').title()}", fontsize=12)
    plot_ax.set_ylabel("Percentage", fontsize=12)
    plot_ax.set_title(
        f"Distribution of {column.replace('_', ' ').title()}{suffix} (%)",
        fontsize=14,
        fontweight="bold",
    )
    plot_ax.grid(axis="y", linestyle="--", alpha=0.7)

    sns.despine(ax=plot_ax)
    plt.tight_layout()


def plot_dist(
    df: pd.DataFrame, hue_col: str, modes: str, hspace: float = -0.4, sv_col: str = "sv"
) -> sns.displot:
    """Create a distribution plot of shapley values for specified modes."""

    g = sns.displot(
        data=df[df["mode"].isin(modes)],
        x=sv_col,
        hue=hue_col,
        row="mode",
        row_order=modes,
        kind="kde",
        palette="Set2",
        fill=True,
        alpha=0.2,
        aspect=4,
        height=1.5,
        common_norm=False,
        facet_kws={"margin_titles": True},
    )

    g.set_titles(row_template="{row_name}", size=12, fontweight="bold")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)
    g.figure.subplots_adjust(hspace=hspace)
    g.set_axis_labels("Shapley Value", "")


def plot_attribution_trend(
    df: pd.DataFrame, n_bins: int = 100, ax: plt.Axes | None = None, legend: bool = True
) -> None:
    """Plot attribution trend by sentence position."""
    df = df.copy()
    df["pos_bin"] = pd.cut(df["normalized_position"], bins=n_bins, labels=False)
    # Convert bin index back to 0-1 scale for plotting
    df["bin_center"] = (df["pos_bin"] + 0.5) / n_bins

    created_new_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
        created_new_fig = True

    sns.lineplot(
        data=df,
        x="bin_center",
        y="sv",
        hue="mode",
        palette="Set2",
        errorbar=None,
        linewidth=2.5,
        alpha=0.85,
        ax=ax,
        legend=legend,
    )

    ax.set_title(
        "Attribution Trend by Sentence Position", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_xlabel("Relative Sentence Position", fontsize=12)

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["Start\n(0%)", "Middle\n(50%)", "End\n(100%)"])
    ax.set_xlim(0, 1)

    sns.despine(ax=ax, trim=True)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, title="Mode")

    if created_new_fig:
        plt.tight_layout()

"""Plotting functions for experiments analysis."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# pylint: disable=too-many-positional-arguments,too-many-arguments
def plot_token_count_distribution(
    plot_df: pd.DataFrame,
    column: str = "count_text_tokens",
    plot_ax: plt.Axes | None = None,
    hue: str | None = None,
    legend: bool = True,
) -> None:
    """Plot the distribution of text token counts."""
    if plot_ax is None:
        _, plot_ax = plt.subplots(1, 1, figsize=(8, 5))

    g = sns.histplot(
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
        "Distribution of Explainable Tokens (%) by Mode",
        fontsize=16,
        fontweight="bold",
    )
    g.grid(axis="y", linestyle="--", alpha=0.7)
    g.grid(axis="x", linestyle="--", alpha=0.0)
    plot_ax.set_ylim(bottom=0)
    g.tick_params(left=False, top=False)

    plot_ax.spines["bottom"].set_color("black")
    plot_ax.spines["left"].set_color("black")

    plot_ax.tick_params(axis="x", colors="black")
    plot_ax.tick_params(axis="y", colors="black")

    sns.despine(ax=plot_ax)
    plt.tight_layout()


def plot_dist(
    df: pd.DataFrame, hue_col: str, modes: str, y_up: Any = None, sv_col: str = "sv"
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

    for ax in g.axes.flat:
        g.set(ylim=(0, y_up))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    g.set_titles(row_template="{row_name}", size=12, fontweight="bold")

    g.tight_layout()
    g.set(ylabel="")
    g.set_axis_labels("Shapley Value", "")


def plot_attribution_trend(
    df: pd.DataFrame,
    n_bins: int = 100,
    ax: plt.Axes | None = None,
    legend: bool = True,
    suffix: str = "",
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
        f"Attribution Trend Over Sentence Position by Mode{suffix}",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_xlabel("Relative Sentence Position", fontsize=12)

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["Start\n(0%)", "Middle\n(50%)", "End\n(100%)"])
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    sns.despine(ax=ax, trim=True)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if legend:
        ax.legend(
            bbox_to_anchor=(0.92, 1), loc="upper left", borderaxespad=0, title="Mode"
        )

    if created_new_fig:
        plt.tight_layout()


def plot_importance_cumsum_and_derivative(
    df_: pd.DataFrame, errorbar: Any = ("ci", 95), alpha: float = 0.3, **kw
) -> None:
    """Plot importance cumulative sum and derivative."""
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Top Plot
    sns.lineplot(
        data=df_,
        x="time (tokens)",
        y="sv_cumsum",
        hue="mode",
        alpha=alpha,
        legend="brief",
        ax=ax1,
        palette="Set2",
        errorbar=errorbar,
        **kw,
    )
    ax1.set_ylabel("Cumulative Sum")
    ax1.set_ylim(bottom=0)

    # Bottom Plot
    sns.lineplot(
        data=df_,
        x="time (tokens)",
        y="sv_derivative",
        hue="mode",
        alpha=alpha,
        legend=False,
        ax=ax2,
        palette="Set2",
        errorbar=errorbar,
        **kw,
    )
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Derivative")

    leg = ax1.legend(
        bbox_to_anchor=(0.89, 0.9),
        loc="upper left",
        frameon=True,
        facecolor="white",
        title="Mode",
    )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine()
    plt.tight_layout()


def plot_derivative(
    df_: pd.DataFrame,
    errorbar: Any = ("ci", 95),
    alpha: float = 0.3,
    hue: str = "language",
    **kw,
) -> None:
    """Plot derivative."""
    g = sns.relplot(
        data=df_,
        x="time (tokens)",
        y="sv_derivative",
        hue=hue,
        row="mode",
        kind="line",
        alpha=alpha,
        palette="Set2",
        errorbar=errorbar,
        height=4,
        aspect=2,
        **kw,
    )
    for ax in g.axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    g.set(ylim=(0, None))
    g.legend.set_bbox_to_anchor((0.98, 0.91))
    g.legend.set_frame_on(True)
    g.set_axis_labels("Time (tokens)", "Derivative")
    sns.despine()
    plt.tight_layout()

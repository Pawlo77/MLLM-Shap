"""Paper-style visualization for all-rank HP-1 deletion faithfulness results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STYLES = {
    "paper_rc": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
    "annotation": dict(
        facecolor="white",
        edgecolor="#cccccc",
        alpha=0.9,
        pad=4.0,
        boxstyle="round,pad=0.3",
    ),
}

VOICE_LABELS = {
    "audio__male": "Male TTS",
    "audio__female": "Female TTS",
}


def _load_summary(result_dir: Path) -> dict:
    path = result_dir / "combined_rankwise_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value_f):
        return "n/a"
    return f"{value_f:.{digits}f}"


def plot_rankwise_faithfulness(result_dir: Path, output_base: Path) -> plt.Figure:
    """Create rank-wise deletion faithfulness figure."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="whitegrid", rc=STYLES["paper_rc"])

    results_path = result_dir / "combined_rankwise_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing rank-wise results CSV: {results_path}")

    df = pd.read_csv(results_path)
    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    df["concentration"] = np.where(
        df["top1_share"]
        >= df[["audio_column", "sample_id", "top1_share"]]
        .drop_duplicates()["top1_share"]
        .median(),
        "Peaked SV distribution",
        "Flat SV distribution",
    )

    rank_counts = df.groupby("segment_rank_abs_sv").size()
    keep_ranks = rank_counts[rank_counts >= 5].index.tolist()
    rank_df = df[df["segment_rank_abs_sv"].isin(keep_ranks)].copy()
    rank_df["rank_label"] = rank_df["segment_rank_abs_sv"].astype(int).astype(str)

    summary = _load_summary(result_dir)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.0),
        gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.82, wspace=0.34)

    ax0, ax1, ax2 = axes
    sns.pointplot(
        data=rank_df,
        x="rank_label",
        y="deletion_drop",
        estimator="mean",
        errorbar=("ci", 95),
        color="#0055A4",
        markers="o",
        linewidth=2.2,
        ax=ax0,
    )
    ax0.set_title("A. Deletion Drop by SV Rank", loc="left", fontweight="bold")
    ax0.set_xlabel("Segment rank by |SV|")
    ax0.set_ylabel("Response-similarity drop")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.grid(axis="x", alpha=0.0)
    ax0.text(
        0.98,
        0.97,
        (
            f"deletions={summary.get('completed_deletions', len(df))}\n"
            f"samples={summary.get('completed_samples', 0)}\n"
            "Spearman |SV|-drop="
            f"{_format_float(summary.get('spearman_abs_sv_vs_drop'))}\n"
            "within-sample rho="
            f"{_format_float(summary.get('mean_within_sample_spearman_abs_sv_vs_drop'))}"
        ),
        transform=ax0.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=STYLES["annotation"],
    )

    sns.pointplot(
        data=rank_df,
        x="rank_label",
        y="deletion_drop",
        hue="concentration",
        estimator="mean",
        errorbar=("ci", 95),
        palette={
            "Peaked SV distribution": "#D55E00",
            "Flat SV distribution": "#777777",
        },
        markers="o",
        linewidth=2.0,
        ax=ax1,
    )
    ax1.set_title("B. Rank Curve by SV Concentration", loc="left", fontweight="bold")
    ax1.set_xlabel("Segment rank by |SV|")
    ax1.set_ylabel("Response-similarity drop")
    ax1.legend(title="", frameon=True, edgecolor="#dddddd", loc="best")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.grid(axis="x", alpha=0.0)

    sample_df = df[["audio_column", "sample_id", "top1_share", "top1_top2_gap"]]
    sample_df = sample_df.drop_duplicates()
    sample_df["voice"] = (
        sample_df["audio_column"].map(VOICE_LABELS).fillna(sample_df["audio_column"])
    )
    sns.scatterplot(
        data=sample_df,
        x="top1_share",
        y="top1_top2_gap",
        hue="voice",
        palette={"Male TTS": "#0055A4", "Female TTS": "#CC79A7"},
        alpha=0.75,
        s=32,
        ax=ax2,
    )
    ax2.set_title("C. What Rank Means Per Sample", loc="left", fontweight="bold")
    ax2.set_xlabel("Top-1 |SV| share")
    ax2.set_ylabel("Top1 - Top2 |SV| gap")
    ax2.legend(title="", frameon=True, edgecolor="#dddddd", loc="best")
    ax2.grid(axis="both", linestyle="--", alpha=0.35)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Rank-Wise Deletion Diagnostic for SGPA-SV Faithfulness",
        fontweight="bold",
        y=0.98,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("experiments/interspeech/outputs/faithfulness_deletion/rankwise"),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("paper/interspeech/figures/faithfulness_rankwise"),
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    plot_rankwise_faithfulness(args.result_dir, args.output_base)
    print(f"Saved {args.output_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()

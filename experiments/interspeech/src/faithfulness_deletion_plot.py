"""Paper-style visualization for HP-1 deletion faithfulness results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
        "legend.fontsize": 11,
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

PALETTE = {
    "Male TTS": "#0055A4",
    "Female TTS": "#CC79A7",
    "Top-SV deletion": "#D55E00",
    "Random deletion": "#777777",
}


def _load_summary(result_dir: Path, name: str) -> dict:
    path = result_dir / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _format_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def plot_faithfulness_deletion(result_dir: Path, output_base: Path) -> plt.Figure:
    """Create the faithfulness deletion figure."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="whitegrid", rc=STYLES["paper_rc"])

    results_path = result_dir / "combined_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing combined results CSV: {results_path}")

    df = pd.read_csv(results_path)
    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    df["pair_id"] = df["voice"] + "_" + df["sample_id"].astype(str)

    long_df = df.melt(
        id_vars=["pair_id", "voice", "audio_column", "sample_id"],
        value_vars=["top_drop", "random_drop"],
        var_name="condition",
        value_name="similarity_drop",
    )
    long_df["condition"] = long_df["condition"].map(
        {
            "top_drop": "Top-SV deletion",
            "random_drop": "Random deletion",
        }
    )

    combined = _load_summary(result_dir, "combined_summary.json")
    male = _load_summary(result_dir, "audio__male_combined_summary.json")
    female = _load_summary(result_dir, "audio__female_combined_summary.json")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.84, wspace=0.32)

    ax0, ax1 = axes
    x_pos = {"Random deletion": 0, "Top-SV deletion": 1}

    for _, row in df.iterrows():
        color = PALETTE.get(row["voice"], "#777777")
        ax0.plot(
            [x_pos["Random deletion"], x_pos["Top-SV deletion"]],
            [row["random_drop"], row["top_drop"]],
            color=color,
            alpha=0.16,
            linewidth=0.8,
            zorder=1,
        )

    sns.pointplot(
        data=long_df,
        x="condition",
        y="similarity_drop",
        hue="voice",
        order=["Random deletion", "Top-SV deletion"],
        estimator="mean",
        errorbar=("ci", 95),
        dodge=0.18,
        markers="o",
        linestyles="-",
        linewidth=2.5,
        palette={k: PALETTE[k] for k in ["Male TTS", "Female TTS"]},
        ax=ax0,
        zorder=3,
    )
    ax0.set_title("A. Deletion Faithfulness", loc="left", fontweight="bold")
    ax0.set_xlabel("")
    ax0.set_ylabel("Response-similarity drop")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.grid(axis="x", alpha=0.0)
    ax0.legend(title="", frameon=True, edgecolor="#dddddd", loc="lower right")

    annotation = (
        f"Combined n={combined.get('completed_samples', len(df))}\n"
        f"Mean paired diff={combined.get('mean_drop_difference', df['drop_difference'].mean()):.3f}\n"
        f"paired $t$={combined.get('paired_t_stat', 0.0):.2f}, "
        f"$p$={_format_p(combined.get('paired_p_value'))}\n"
        f"Cohen's $d_z$={combined.get('cohen_dz', 0.0):.2f}"
    )
    ax0.text(
        0.03,
        0.97,
        annotation,
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=STYLES["annotation"],
    )

    sns.violinplot(
        data=df,
        x="voice",
        y="drop_difference",
        hue="voice",
        order=["Male TTS", "Female TTS"],
        hue_order=["Male TTS", "Female TTS"],
        palette={k: PALETTE[k] for k in ["Male TTS", "Female TTS"]},
        inner=None,
        cut=0,
        linewidth=1.2,
        saturation=0.85,
        legend=False,
        ax=ax1,
    )
    sns.boxplot(
        data=df,
        x="voice",
        y="drop_difference",
        order=["Male TTS", "Female TTS"],
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.1},
        whiskerprops={"color": "#333333", "linewidth": 1.1},
        medianprops={"color": "#333333", "linewidth": 1.5},
        showfliers=False,
        ax=ax1,
    )
    sns.stripplot(
        data=df,
        x="voice",
        y="drop_difference",
        order=["Male TTS", "Female TTS"],
        color="#333333",
        alpha=0.35,
        jitter=0.18,
        size=2.5,
        ax=ax1,
        zorder=3,
    )
    ax1.axhline(0, color="#333333", linestyle=":", linewidth=1.2)
    ax1.set_title("B. Paired Drop Difference", loc="left", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Top-SV drop - random drop")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.grid(axis="x", alpha=0.0)

    voice_note = (
        f"Male: n={male.get('completed_samples', 0)}, "
        f"diff={male.get('mean_drop_difference', 0.0):.3f}, "
        f"p={_format_p(male.get('paired_p_value'))}\n"
        f"Female: n={female.get('completed_samples', 0)}, "
        f"diff={female.get('mean_drop_difference', 0.0):.3f}, "
        f"p={_format_p(female.get('paired_p_value'))}"
    )
    ax1.text(
        0.03,
        0.97,
        voice_note,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=STYLES["annotation"],
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="black")

    fig.suptitle(
        "Top-SV Audio Segment Deletions Cause Larger Response Changes",
        fontweight="bold",
        y=0.98,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "experiments/interspeech/outputs/faithfulness_deletion/remap_fix_50"
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("paper/interspeech/figures/faithfulness_deletion"),
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    plot_faithfulness_deletion(args.result_dir, args.output_base)
    print(f"Saved {args.output_base.with_suffix('.png')}")
    print(f"Saved {args.output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

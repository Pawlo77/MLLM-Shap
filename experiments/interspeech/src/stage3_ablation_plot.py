"""Paper-style visualization for the SGPA Stage 3 ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .sgpa_plot import STYLES


VOICE_LABELS = {
    "audio__male": "Male TTS",
    "audio__female": "Female TTS",
}


def load_ablation_samples(input_dir: Path) -> pd.DataFrame:
    """Load per-sample ablation CSVs and add plotting labels."""
    paths = sorted(input_dir.glob("audio__*_n100_samples.csv"))
    if not paths:
        raise FileNotFoundError(f"No ablation sample CSVs found in {input_dir}")

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    long_df = df.melt(
        id_vars=["sample_id", "voice", "audio_column"],
        value_vars=["raw_mean_flux", "refined_mean_flux"],
        var_name="condition",
        value_name="mean_boundary_flux",
    )
    long_df["condition"] = long_df["condition"].map(
        {
            "raw_mean_flux": "Raw CTC",
            "refined_mean_flux": "SGPA Refined",
        }
    )
    return df, long_df


def plot_stage3_ablation(
    input_dir: Path,
    output_base: Path,
    summary_path: Path | None = None,
) -> plt.Figure:
    """Create and save the Stage 3 ablation figure."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="whitegrid", rc=STYLES["paper_rc"])

    samples_df, long_df = load_ablation_samples(input_dir)
    summary = {}
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.2),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.86, wspace=0.32)

    palette = {
        "Raw CTC": "#D55E00",
        "SGPA Refined": "#009E73",
        "Male TTS": "#0055A4",
        "Female TTS": "#CC79A7",
    }

    ax0, ax1 = axes
    order = ["Raw CTC", "SGPA Refined"]
    x_pos = {"Raw CTC": 0, "SGPA Refined": 1}

    for _, row in samples_df.iterrows():
        color = palette.get(row["voice"], "#777777")
        ax0.plot(
            [x_pos["Raw CTC"], x_pos["SGPA Refined"]],
            [row["raw_mean_flux"], row["refined_mean_flux"]],
            color=color,
            alpha=0.12,
            linewidth=0.8,
            zorder=1,
        )

    sns.pointplot(
        data=long_df,
        x="condition",
        y="mean_boundary_flux",
        hue="voice",
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        dodge=0.18,
        markers="o",
        linestyles="-",
        linewidth=2.5,
        palette={k: palette[k] for k in ["Male TTS", "Female TTS"]},
        ax=ax0,
        zorder=3,
    )
    ax0.set_title("A. Paired Boundary-Flux Reduction", loc="left", fontweight="bold")
    ax0.set_xlabel("")
    ax0.set_ylabel("Mean spectral flux at boundaries")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.grid(axis="x", alpha=0.0)
    ax0.legend(title="", frameon=True, edgecolor="#dddddd", loc="upper right")

    reduction_text = (
        f"Mean reduction: {summary.get('mean_percent_reduction', samples_df['percent_reduction'].mean()):.1f}%\n"
        f"paired $t$={summary.get('paired_t_stat', 0.0):.2f}, "
        f"$p$={summary.get('paired_p_value', 0.0):.1e}\n"
        f"Cohen's $d_z$={summary.get('cohen_dz', 0.0):.2f}"
    )
    ax0.text(
        0.03,
        0.97,
        reduction_text,
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=STYLES["annotation"],
    )

    sns.violinplot(
        data=samples_df,
        x="voice",
        y="percent_reduction",
        hue="voice",
        order=["Male TTS", "Female TTS"],
        hue_order=["Male TTS", "Female TTS"],
        palette={k: palette[k] for k in ["Male TTS", "Female TTS"]},
        inner=None,
        cut=0,
        linewidth=1.2,
        saturation=0.85,
        legend=False,
        ax=ax1,
    )
    sns.boxplot(
        data=samples_df,
        x="voice",
        y="percent_reduction",
        order=["Male TTS", "Female TTS"],
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.1},
        whiskerprops={"color": "#333333", "linewidth": 1.1},
        medianprops={"color": "#333333", "linewidth": 1.5},
        showfliers=False,
        ax=ax1,
    )
    ax1.axhline(0, color="#333333", linestyle=":", linewidth=1.2)
    ax1.set_title("B. Per-Sample Reduction", loc="left", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Flux reduction (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.grid(axis="x", alpha=0.0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="black")

    fig.suptitle(
        "Stage 3 Spectral Refinement Moves Cuts to Lower-Flux Regions",
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
        "--input-dir",
        type=Path,
        default=Path("experiments/interspeech/outputs/stage3_ablation"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "experiments/interspeech/outputs/stage3_ablation/combined_n200_summary.json"
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("paper/interspeech/figures/stage3_ablation"),
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    plot_stage3_ablation(
        input_dir=args.input_dir,
        output_base=args.output_base,
        summary_path=args.summary,
    )
    print(f"Saved {args.output_base.with_suffix('.png')}")
    print(f"Saved {args.output_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

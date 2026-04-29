"""Paper-style visualization for the SGPA Stage 3 ablation."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .sgpa_plot import STYLES


VOICE_LABELS = {
    "audio__male": "Male TTS",
    "audio__female": "Female TTS",
    "audio__original": "Original",
}


def load_ablation_samples(
    input_dir: Path,
    dataset_config: str | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Load per-sample ablation CSVs and add plotting labels."""
    if max_samples is None:
        paths = sorted(input_dir.glob("*_samples.csv"))
    else:
        paths = sorted(input_dir.glob(f"*_n{max_samples}_samples.csv"))
    if not paths:
        raise FileNotFoundError(f"No ablation sample CSVs found in {input_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "dataset_config" not in frame.columns:
            match = re.match(
                r"(?P<dataset_config>.+)__audio__.+_n\d+_samples\.csv$",
                path.name,
            )
            if match:
                frame["dataset_config"] = match.group("dataset_config")
        if "dataset_config" not in frame.columns:
            frame["dataset_config"] = "legacy"
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if dataset_config is not None:
        if "dataset_config" not in df.columns:
            raise ValueError(
                "Samples do not contain dataset_config column. "
                "Please regenerate ablation CSVs with updated stage3_ablation.py."
            )
        df = df[df["dataset_config"] == dataset_config].copy()
        if df.empty:
            raise FileNotFoundError(
                f"No ablation sample CSVs found for dataset_config={dataset_config!r}"
            )

    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    df["series_label"] = df.apply(
        lambda r: f"{r['voice']} | {r['dataset_config']}", axis=1
    )
    long_df = df.melt(
        id_vars=[
            "sample_id",
            "voice",
            "audio_column",
            "dataset_config",
            "series_label",
        ],
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
    dataset_config: str | None = None,
    max_samples: int | None = None,
    summary_path: Path | None = None,
) -> plt.Figure:
    """Create and save the Stage 3 ablation figure."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="white", rc=STYLES["paper_rc"])

    samples_df, long_df = load_ablation_samples(
        input_dir=input_dir,
        dataset_config=dataset_config,
        max_samples=max_samples,
    )
    summary = {}
    if summary_path is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 4.3),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.86, wspace=0.22)

    series_order = sorted(samples_df["series_label"].unique().tolist())
    set2 = sns.color_palette("Set2", n_colors=len(series_order))
    series_palette = dict(zip(series_order, set2))

    ax0, ax1 = axes
    order = ["Raw CTC", "SGPA Refined"]
    x_pos = {"Raw CTC": 0, "SGPA Refined": 1}

    # Reduce overplotting: show a deterministic subset of paired trajectories.
    connector_parts = []
    for _, group in samples_df.groupby("series_label"):
        connector_parts.append(group.sample(n=min(len(group), 250), random_state=7))
    connector_df = pd.concat(connector_parts, ignore_index=True)
    for _, row in connector_df.iterrows():
        line_color = series_palette.get(row["series_label"], "#9A9A9A")
        ax0.plot(
            [x_pos["Raw CTC"], x_pos["SGPA Refined"]],
            [row["raw_mean_flux"], row["refined_mean_flux"]],
            color=line_color,
            alpha=0.05,
            linewidth=0.75,
            zorder=1,
        )

    sns.pointplot(
        data=long_df,
        x="condition",
        y="mean_boundary_flux",
        hue="series_label",
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        dodge=0.0,
        markers="o",
        linestyles="-",
        linewidth=2.8,
        hue_order=series_order,
        palette=series_palette,
        ax=ax0,
        zorder=3,
    )
    ax0.set_title("Paired Boundary-Flux Reduction", loc="center", fontweight="bold")
    ax0.set_xlabel("")
    ax0.set_ylabel("Mean spectral flux at boundaries")
    ax0.set_xlim(-0.12, 1.12)
    ax0.grid(axis="y", linestyle="--", alpha=0.28)
    ax0.grid(axis="x", alpha=0.0)
    if ax0.get_legend() is not None:
        ax0.get_legend().remove()

    sns.violinplot(
        data=samples_df,
        x="series_label",
        y="percent_reduction",
        hue="series_label",
        order=series_order,
        hue_order=series_order,
        palette=series_palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=0.8,
        legend=False,
        ax=ax1,
    )
    sns.boxplot(
        data=samples_df,
        x="series_label",
        y="percent_reduction",
        order=series_order,
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 1.1},
        whiskerprops={"color": "#333333", "linewidth": 1.1},
        medianprops={"color": "#333333", "linewidth": 1.5},
        showfliers=False,
        ax=ax1,
    )
    ax1.axhline(0, color="#333333", linestyle=":", linewidth=1.1)
    ax1.set_title("Per-Sample Reduction", loc="center", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Flux reduction (%)")
    ax1.tick_params(axis="x", labelrotation=20)
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.grid(axis="x", alpha=0.0)
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    # Improve readability when extreme tails compress the central distributions.
    y_lo = float(samples_df["percent_reduction"].quantile(0.01))
    y_hi = float(samples_df["percent_reduction"].quantile(0.99))
    pad = 0.06 * max(y_hi - y_lo, 1.0)
    ax1.set_ylim(y_lo - pad, y_hi + pad)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", colors="black")

    fig.suptitle(
        "Stage 3 Spectral Refinement Ablation"
        + (f" ({dataset_config})" if dataset_config is not None else ""),
        fontweight="bold",
        y=0.98,
    )

    # Save plot metadata beside source ablation CSV files.
    meta = {
        "mean_reduction_percent": float(
            summary.get(
                "mean_percent_reduction", samples_df["percent_reduction"].mean()
            )
        ),
        "paired_t_stat": float(summary.get("paired_t_stat", 0.0)),
        "paired_p_value": float(summary.get("paired_p_value", 0.0)),
        "cohen_dz": float(summary.get("cohen_dz", 0.0)),
        "percent_reduction_quantiles": {
            "q01": y_lo,
            "q99": y_hi,
        },
        "percent_reduction_ylim": {
            "min": float(y_lo - pad),
            "max": float(y_hi + pad),
        },
        "n_samples": int(len(samples_df)),
        "dataset_configs": sorted(samples_df["dataset_config"].unique().tolist()),
    }
    meta_name = (
        "image_meta.json"
        if dataset_config is None
        else f"image_meta__{dataset_config}.json"
    )
    (input_dir / meta_name).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
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
            "experiments/interspeech/outputs/stage3_ablation/combined_n1000_summary.json"
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("paper/interspeech/figures/stage3_ablation"),
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Dataset config to plot (e.g. single_sentence_1k). "
        "If omitted, one image is generated per dataset_config found in CSVs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional filter for n<max_samples> sample files (e.g. 1000). "
        "If omitted, all available sample files are included.",
    )
    return parser


def main() -> None:
    """Run CLI."""
    args = build_argparser().parse_args()
    plot_stage3_ablation(
        input_dir=args.input_dir,
        output_base=args.output_base,
        dataset_config=args.dataset_config,
        max_samples=args.max_samples,
        summary_path=args.summary,
    )
    print(f"Saved {args.output_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()

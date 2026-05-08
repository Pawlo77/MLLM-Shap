"""Visualization for the SGPA Stage 3 ablation."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..sgpa_plot import STYLES

VOICE_LABELS: dict[str, str] = {
    "audio__male": "Male TTS",
    "audio__female": "Female TTS",
    "audio__original": "Original",
}


def load_ablation_samples(
    input_dir: Path,
    dataset_config: str | None = None,
    max_samples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-sample ablation CSVs and add plotting labels."""
    # Try consolidated format first
    consolidated = input_dir / "samples.csv"
    if consolidated.exists():
        df = pd.read_csv(consolidated)
    else:
        # Legacy: glob per-column files
        pattern = f"*_n{max_samples}_samples.csv" if max_samples else "*_samples.csv"
        paths = sorted(input_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No ablation sample CSVs found in {input_dir}")

        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            if "dataset_config" not in frame.columns:
                match = re.match(
                    r"(?P<dataset_config>.+)__audio__.+_n\d+_samples\.csv$", path.name
                )
                frame["dataset_config"] = (
                    match.group("dataset_config") if match else "legacy"
                )
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    if dataset_config is not None:
        df = df[df["dataset_config"] == dataset_config].copy()
        if df.empty:
            raise FileNotFoundError(f"No data for dataset_config={dataset_config!r}")

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
        {"raw_mean_flux": "Raw CTC", "refined_mean_flux": "SGPA Refined"}
    )
    return df, long_df


def plot_stage3_ablation(
    input_dir: Path,
    output_base: Path,
    dataset_config: str | None = None,
    max_samples: int | None = None,
    summary_path: Path | None = None,
    save_meta: bool = False,
) -> plt.Figure:
    """Create and save the Stage 3 ablation figure."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="white", rc=STYLES["paper_rc"])

    samples_df, long_df = load_ablation_samples(input_dir, dataset_config, max_samples)
    summary: dict = {}
    if summary_path is not None and summary_path.exists():
        raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Support both consolidated format (nested under "combined") and legacy flat format
        summary = (
            raw_summary.get("combined", raw_summary)
            if isinstance(raw_summary, dict)
            else raw_summary
        )

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.86, wspace=0.22)

    series_order = sorted(samples_df["series_label"].unique().tolist())
    palette = dict(
        zip(series_order, sns.color_palette("Set2", n_colors=len(series_order)))
    )
    x_pos = {"Raw CTC": 0, "SGPA Refined": 1}

    # Paired trajectories (subsampled)
    connector_parts = [
        grp.sample(n=min(len(grp), 250), random_state=7)
        for _, grp in samples_df.groupby("series_label")
    ]
    for _, row in pd.concat(connector_parts, ignore_index=True).iterrows():
        ax0.plot(
            [x_pos["Raw CTC"], x_pos["SGPA Refined"]],
            [row["raw_mean_flux"], row["refined_mean_flux"]],
            color=palette.get(row["series_label"], "#9A9A9A"),
            alpha=0.05,
            linewidth=0.75,
            zorder=1,
        )

    sns.pointplot(
        data=long_df,
        x="condition",
        y="mean_boundary_flux",
        hue="series_label",
        order=["Raw CTC", "SGPA Refined"],
        estimator="mean",
        errorbar=("ci", 95),
        dodge=0.0,
        markers="o",
        linestyles="-",
        linewidth=2.8,
        hue_order=series_order,
        palette=palette,
        ax=ax0,
        zorder=3,
    )
    ax0.set_title("Paired Boundary-Flux Reduction", loc="center", fontweight="bold")
    ax0.set(xlabel="", ylabel="Mean spectral flux at boundaries", xlim=(-0.12, 1.12))
    ax0.grid(axis="y", linestyle="--", alpha=0.28)
    if ax0.get_legend():
        ax0.get_legend().remove()

    # Violin + box
    sns.violinplot(
        data=samples_df,
        x="series_label",
        y="percent_reduction",
        hue="series_label",
        order=series_order,
        hue_order=series_order,
        palette=palette,
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
        boxprops={"facecolor": "white", "edgecolor": "#333", "linewidth": 1.1},
        whiskerprops={"color": "#333", "linewidth": 1.1},
        medianprops={"color": "#333", "linewidth": 1.5},
        showfliers=False,
        ax=ax1,
    )
    ax1.axhline(0, color="#333", linestyle=":", linewidth=1.1)
    ax1.set_title("Per-Sample Reduction", loc="center", fontweight="bold")
    ax1.set(xlabel="", ylabel="Flux reduction (%)")
    ax1.tick_params(axis="x", labelrotation=20)
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    if ax1.get_legend():
        ax1.get_legend().remove()

    y_lo = float(samples_df["percent_reduction"].quantile(0.01))
    y_hi = float(samples_df["percent_reduction"].quantile(0.99))
    pad = 0.06 * max(y_hi - y_lo, 1.0)
    ax1.set_ylim(y_lo - pad, y_hi + pad)

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Stage 3 Spectral Refinement Ablation"
        + (f" ({dataset_config})" if dataset_config else ""),
        fontweight="bold",
        y=0.98,
    )

    if save_meta:
        meta = {
            "mean_reduction_percent": float(
                summary.get(
                    "mean_percent_reduction", samples_df["percent_reduction"].mean()
                )
            ),
            "paired_t_stat": float(summary.get("paired_t_stat", 0.0)),
            "paired_p_value": float(summary.get("paired_p_value", 0.0)),
            "cohen_dz": float(summary.get("cohen_dz", 0.0)),
            "n_samples": int(len(samples_df)),
            "dataset_configs": sorted(samples_df["dataset_config"].unique().tolist()),
        }
        meta_name = (
            "image_meta.json"
            if dataset_config is None
            else f"image_meta__{dataset_config}.json"
        )
        (input_dir / meta_name).write_text(json.dumps(meta, indent=2))

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return fig

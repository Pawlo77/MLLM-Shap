"""Plotting for HP-1 deletion faithfulness (top-vs-random and rank-wise)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

from experiments.interspeech.src.sgpa_plot import STYLES

VOICE_LABELS: dict[str, str] = {
    "audio__male": "Male TTS",
    "audio__female": "Female TTS",
}

DELETION_METRIC_COLS: dict[str, tuple[str, str, str]] = {
    "embedding": ("top_drop", "random_drop", "drop_difference"),
    "tfidf": ("tfidf_top_drop", "tfidf_random_drop", "tfidf_drop_difference"),
}

RANKWISE_METRIC_COLS: dict[str, str] = {
    "embedding": "deletion_drop",
    "tfidf": "tfidf_deletion_drop",
}

METRIC_TITLES: dict[str, str] = {
    "embedding": "Embedding Cosine",
    "tfidf": "TF-IDF Cosine",
}

VOICE_ORDER: list[str] = ["Male TTS", "Female TTS"]


def _regression_line(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return OLS regression line (xh, yh) and stats (r, p) for x vs y."""
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xh = np.linspace(x[mask].min(), x[mask].max(), 120)
    return xh, slope * xh + intercept, float(r), float(p)


def _voice_palette() -> dict[str, tuple]:
    """Return a color palette mapping each voice to a distinct color."""
    return dict(zip(VOICE_ORDER, sns.color_palette("Set2", n_colors=len(VOICE_ORDER))))


def _apply_style() -> None:
    """Apply consistent styling for all plots."""
    plt.rcParams.update(STYLES["paper_rc"])
    sns.set_theme(style="white", rc=STYLES["paper_rc"])


def _clean_spines(axes) -> None:
    """Remove top and right spines from all axes."""
    for ax in axes.flat if hasattr(axes, "flat") else axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ─── Top-SV vs Random deletion ──────────────────────────────────────────────


def plot_deletion(
    df: pd.DataFrame,
    result_dir: Path,
    output_base: Path,
    metric: str = "embedding",
    save_meta: bool = False,
) -> plt.Figure:
    """Four-panel figure comparing top-SV deletion vs random deletion."""
    if metric not in DELETION_METRIC_COLS:
        raise ValueError(
            f"metric must be one of {list(DELETION_METRIC_COLS)}, got {metric!r}"
        )
    top_col, random_col, diff_col = DELETION_METRIC_COLS[metric]

    _apply_style()

    if metric == "tfidf" and top_col not in df.columns:
        if "top_drop" in df.columns:
            top_col, random_col, diff_col = "top_drop", "random_drop", "drop_difference"
        else:
            raise KeyError(
                f"Column {top_col!r} not found; legacy fallback unavailable."
            )

    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    palette = _voice_palette()

    long_df = df.melt(
        id_vars=["voice", "audio_column", "sample_id"],
        value_vars=[top_col, random_col],
        var_name="condition",
        value_name="similarity_drop",
    )
    long_df["condition"] = long_df["condition"].map({
        top_col: "Top-SV deletion",
        random_col: "Random deletion",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    fig.subplots_adjust(
        left=0.09, right=0.97, bottom=0.10, top=0.88, hspace=0.45, wspace=0.35
    )
    (ax0, ax1), (ax2, ax3) = axes

    # Paired slope
    for _, row in df.iterrows():
        ax0.plot(
            [0, 1],
            [row[random_col], row[top_col]],
            color=palette[row["voice"]],
            alpha=0.15,
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
        dodge=0.0,
        markers="o",
        linestyles="-",
        linewidth=2.5,
        hue_order=VOICE_ORDER,
        palette=palette,
        ax=ax0,
        zorder=3,
    )
    ax0.set(title="", xlabel="", ylabel="Response-similarity drop", xlim=(-0.25, 1.25))
    ax0.set_title("Paired Deletion Comparison", loc="left", fontweight="bold")
    ax0.grid(axis="y", linestyle="--", alpha=0.28)
    if leg := ax0.get_legend():
        leg.remove()

    # Drop-difference violin + box
    sns.violinplot(
        data=df,
        x="voice",
        y=diff_col,
        hue="voice",
        order=VOICE_ORDER,
        hue_order=VOICE_ORDER,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.0,
        saturation=0.8,
        legend=False,
        ax=ax1,
    )
    sns.boxplot(
        data=df,
        x="voice",
        y=diff_col,
        order=VOICE_ORDER,
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#333", "linewidth": 1.1},
        whiskerprops={"color": "#333", "linewidth": 1.1},
        medianprops={"color": "#333", "linewidth": 1.5},
        showfliers=False,
        ax=ax1,
    )
    sns.stripplot(
        data=df,
        x="voice",
        y=diff_col,
        order=VOICE_ORDER,
        color="#444",
        alpha=0.30,
        jitter=0.18,
        size=2.5,
        ax=ax1,
        zorder=3,
    )
    ax1.axhline(0, color="#444", linestyle=":", linewidth=1.2)
    ax1.set_title("Drop Difference by Voice", loc="left", fontweight="bold")
    ax1.set(xlabel="", ylabel="Top-SV drop − Random drop")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)

    # KDE
    for voice, grp in df.groupby("voice"):
        color = palette[voice]
        grp[top_col].plot.kde(ax=ax2, color=color, linewidth=2.0, linestyle="-")
        grp[random_col].plot.kde(
            ax=ax2, color=color, linewidth=1.5, linestyle="--", alpha=0.7
        )
    ax2.set_title("Drop Distribution: Top-SV vs Random", loc="left", fontweight="bold")
    ax2.set(xlabel="Response-similarity drop", ylabel="Density", xlim=(-0.1, 1.4))
    ax2.legend(
        handles=[
            Line2D(
                [0], [0], color="gray", linewidth=2.0, linestyle="-", label="Top-SV"
            ),
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                label="Random",
            ),
        ],
        fontsize=9,
        frameon=False,
        ncol=2,
        loc="upper left",
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.28)

    # |SV| vs drop difference scatter
    for voice, grp in df.groupby("voice"):
        ax3.scatter(
            grp["top_abs_sv"],
            grp[diff_col],
            color=palette[voice],
            alpha=0.55,
            s=22,
            label=voice,
            zorder=3,
        )
    xh, yh, _, _ = _regression_line(
        df["top_abs_sv"].to_numpy(float), df[diff_col].to_numpy(float)
    )
    ax3.plot(xh, yh, color="#555", linewidth=1.5, linestyle="--", zorder=4)
    ax3.axhline(0, color="#aaa", linestyle=":", linewidth=1.0)
    ax3.set_title("Top |SV| vs Paired Drop Difference", loc="left", fontweight="bold")
    ax3.set(xlabel="Top-segment absolute SV", ylabel="Top-SV drop − Random drop")
    ax3.grid(axis="both", linestyle="--", alpha=0.28)

    _clean_spines(axes)
    fig.suptitle(
        f"HP-1 Attribution Faithfulness: Deletion Test ({METRIC_TITLES[metric]})",
        fontweight="bold",
        y=0.95,
    )

    out_path = output_base.parent / f"{output_base.stem}_{metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if save_meta:
        meta = _build_deletion_meta(df, top_col, random_col, diff_col)
        (result_dir / f"image_meta_{metric}.json").write_text(
            json.dumps(meta, indent=2)
        )

    return fig


def _build_deletion_meta(
    df: pd.DataFrame, top_col: str, random_col: str, diff_col: str
) -> dict:
    """Compute summary statistics for the deletion comparison to include as metadata."""
    top = df[top_col].to_numpy(float)
    rand = df[random_col].to_numpy(float)
    diff = df[diff_col].to_numpy(float)
    r_mod, p_mod = stats.linregress(df["top_abs_sv"].to_numpy(float), diff)[2:4]
    has_pairs = len(df) >= 2
    t_stat, p_val = stats.ttest_rel(top, rand) if has_pairs else (None, None)
    cohen_dz = (
        float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)) if has_pairs else None
    )

    by_voice: dict[str, dict] = {}
    for voice, grp in df.groupby("voice"):
        g_diff = grp[diff_col].to_numpy(float)
        g_top, g_rand = grp[top_col].to_numpy(float), grp[random_col].to_numpy(float)
        g_t, g_p = stats.ttest_rel(g_top, g_rand) if len(grp) >= 2 else (None, None)
        by_voice[str(voice)] = {
            "n": int(len(grp)),
            "mean_drop_difference": float(np.mean(g_diff)),
            "paired_t_stat": float(g_t) if g_t is not None else None,
            "paired_p_value": float(g_p) if g_p is not None else None,
            "cohen_dz": float(np.mean(g_diff) / (np.std(g_diff, ddof=1) + 1e-9))
            if len(grp) >= 2
            else None,
        }
    return {
        "n_samples": int(len(df)),
        "mean_top_drop": float(np.mean(top)),
        "mean_random_drop": float(np.mean(rand)),
        "mean_drop_difference": float(np.mean(diff)),
        "median_drop_difference": float(np.median(diff)),
        "paired_t_stat": float(t_stat) if t_stat is not None else None,
        "paired_p_value": float(p_val) if p_val is not None else None,
        "cohen_dz": cohen_dz,
        "top_greater_than_random_rate": float(np.mean(diff > 0)),
        "by_voice": by_voice,
        "effect_moderation_r": float(r_mod),
        "effect_moderation_p": float(p_mod),
    }


# ─── Rank-wise deletion ─────────────────────────────────────────────────────


def _within_sample_spearmans(df: pd.DataFrame, drop_col: str) -> list[float]:
    """Compute within-sample Spearman correlations between |SV| and deletion drop."""
    corrs: list[float] = []
    for _, grp in df.groupby(["audio_column", "sample_id"]):
        if grp["segment_abs_sv"].nunique() < 2 or grp[drop_col].nunique() < 2:
            continue
        r = stats.spearmanr(grp["segment_abs_sv"], grp[drop_col]).statistic
        if np.isfinite(r):
            corrs.append(float(r))
    return corrs


def plot_rankwise(
    df: pd.DataFrame,
    result_dir: Path,
    output_base: Path,
    metric: str = "embedding",
    save_meta: bool = False,
) -> plt.Figure:
    """Four-panel figure showing deletion drop vs absolute-SV rank."""
    if metric not in RANKWISE_METRIC_COLS:
        raise ValueError(
            f"metric must be one of {list(RANKWISE_METRIC_COLS)}, got {metric!r}"
        )
    drop_col = RANKWISE_METRIC_COLS[metric]

    _apply_style()

    if metric == "tfidf" and drop_col not in df.columns:
        if "deletion_drop" in df.columns:
            drop_col = "deletion_drop"
        else:
            raise KeyError(
                f"Column {drop_col!r} not found; legacy fallback unavailable."
            )

    df["voice"] = df["audio_column"].map(VOICE_LABELS).fillna(df["audio_column"])
    palette = _voice_palette()

    rank_counts = df.groupby("segment_rank_abs_sv").size()
    keep_ranks = sorted(rank_counts[rank_counts >= 5].index.tolist())
    rank_df = df[df["segment_rank_abs_sv"].isin(keep_ranks)].copy()

    within_spearmans = _within_sample_spearmans(df, drop_col)
    summary = _load_rankwise_summary(result_dir)
    xh, yh, ols_r, ols_p = _regression_line(
        df["segment_abs_sv"].to_numpy(float), df[drop_col].to_numpy(float)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8))
    fig.subplots_adjust(
        left=0.09, right=0.97, bottom=0.10, top=0.88, hspace=0.45, wspace=0.36
    )
    (ax0, ax1), (ax2, ax3) = axes

    # Deletion curve by rank
    for voice, grp in rank_df.groupby("voice"):
        grouped = grp.groupby("segment_rank_abs_sv")[drop_col]
        means = grouped.mean()
        ci95 = grouped.sem() * stats.t.ppf(0.975, df=grouped.count() - 1)
        ax0.errorbar(
            means.index.astype(int).tolist(),
            means.values,
            yerr=ci95.values,
            color=palette[voice],
            marker="o",
            linewidth=2.0,
            markersize=5,
            capsize=3,
            label=voice,
            zorder=3,
        )
    ax0.set_title("Deletion Drop by |SV| Rank", loc="left", fontweight="bold")
    ax0.set(
        xlabel="Segment rank by |SV|  (1 = highest)", ylabel="Response-similarity drop"
    )
    ax0.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax0.legend(title="", frameon=False)
    ax0.grid(axis="y", linestyle="--", alpha=0.28)

    # |SV| vs drop scatter
    for voice, grp in df.groupby("voice"):
        ax1.scatter(
            grp["segment_abs_sv"],
            grp[drop_col],
            color=palette[voice],
            alpha=0.30,
            s=12,
            label=voice,
            zorder=2,
            rasterized=True,
        )
    ax1.plot(xh, yh, color="#555", linewidth=1.6, linestyle="--", zorder=4, label="OLS")
    ax1.set_title("|SV| vs Deletion Drop", loc="left", fontweight="bold")
    ax1.set(xlabel="Segment absolute SV", ylabel="Response-similarity drop")
    ax1.grid(axis="both", linestyle="--", alpha=0.28)

    # Saturation diagnostic
    ax2_twin = ax2.twinx()
    for voice, grp in df.groupby("voice"):
        vals = grp[drop_col].dropna().to_numpy()
        ax2.hist(
            vals,
            bins=30,
            range=(0.0, 1.05),
            color=palette[voice],
            alpha=0.35,
            density=True,
            label=voice,
        )
        vals_sorted = np.sort(vals)
        ax2_twin.plot(
            vals_sorted,
            np.arange(1, len(vals_sorted) + 1) / len(vals_sorted),
            color=palette[voice],
            linewidth=1.8,
        )
    ax2.axvline(0.8, color="#555", linewidth=1.2, linestyle="--")
    ax2.axvline(0.9, color="#555", linewidth=1.2, linestyle=":")
    ax2.set_title("Saturation Diagnostic", loc="left", fontweight="bold")
    ax2.set(xlabel="Response-similarity drop", ylabel="Density")
    ax2_twin.set(ylabel="CDF", ylim=(0, 1.05))
    ax2.grid(axis="y", linestyle="--", alpha=0.28)
    ax2.spines["top"].set_visible(False)
    ax2_twin.spines["top"].set_visible(False)

    # Within-sample Spearman
    if within_spearmans:
        ws = pd.Series(within_spearmans, name="rho")
        set2 = sns.color_palette("Set2", n_colors=3)
        sns.violinplot(
            y=ws,
            color=set2[2],
            inner=None,
            cut=0,
            linewidth=1.0,
            saturation=0.8,
            ax=ax3,
        )
        sns.stripplot(
            y=ws, color="#333", alpha=0.35, jitter=0.12, size=3, ax=ax3, zorder=3
        )
        ax3.axhline(0, color=set2[0], linewidth=1.4, linestyle="--")
        ax3.axhline(float(ws.median()), color=set2[1], linewidth=1.4, linestyle="-")
        ax3.set_title(
            "Within-Sample Spearman(|SV|, Drop)", loc="left", fontweight="bold"
        )
        ax3.set(ylabel="Spearman ρ per sample", xlabel="")
        ax3.grid(axis="y", linestyle="--", alpha=0.28)

    for ax in [ax0, ax1, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"HP-1 Attribution Faithfulness: Rank-Wise Deletion ({METRIC_TITLES[metric]})",
        fontweight="bold",
        y=0.95,
    )

    out_path = output_base.parent / f"{output_base.stem}_{metric}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if save_meta:
        meta = _build_rankwise_meta(
            df, drop_col, within_spearmans, summary, ols_r, ols_p
        )
        (result_dir / f"image_meta_{metric}.json").write_text(
            json.dumps(meta, indent=2)
        )

    return fig


def _load_rankwise_summary(result_dir: Path) -> dict:
    """Load pre-computed summary statistics for rank-wise deletion from the result directory, if available."""
    for name in ("combined_rankwise_summary.json", "combined_summary.json"):
        p = result_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _build_rankwise_meta(
    df: pd.DataFrame,
    drop_col: str,
    within_spearmans: list[float],
    summary: dict,
    ols_r: float,
    ols_p: float,
) -> dict:
    """Compute summary statistics for the rank-wise deletion comparison to include as metadata."""
    samples = df[["audio_column", "sample_id"]].drop_duplicates()
    drop = df[drop_col]
    return {
        "n_deletions": int(len(df)),
        "n_samples": int(len(samples)),
        "mean_deletion_drop": float(drop.mean()),
        "saturation_frac_ge_0_8": float((drop >= 0.8).mean()),
        "saturation_frac_ge_0_9": float((drop >= 0.9).mean()),
        "spearman_abs_sv_vs_drop_global": summary.get("spearman_abs_sv_vs_drop"),
        "spearman_negative_rank_vs_drop": summary.get("spearman_negative_rank_vs_drop"),
        "mean_within_sample_spearman": float(np.mean(within_spearmans))
        if within_spearmans
        else None,
        "median_within_sample_spearman": float(np.median(within_spearmans))
        if within_spearmans
        else None,
        "iqr_within_sample_spearman": [
            float(np.percentile(within_spearmans, 25)),
            float(np.percentile(within_spearmans, 75)),
        ]
        if within_spearmans
        else None,
        "mean_top1_share": summary.get("mean_top1_share"),
        "mean_abs_sv_entropy_norm": summary.get("mean_abs_sv_entropy_norm"),
        "mean_top1_top2_gap": summary.get("mean_top1_top2_gap"),
        "ols_r": ols_r,
        "ols_p": ols_p,
        "per_rank": summary.get("per_rank"),
    }

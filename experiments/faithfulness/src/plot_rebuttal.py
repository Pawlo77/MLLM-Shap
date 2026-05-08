"""Plot rebuttal faithfulness results across SGPA/raw and input conditions."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


CONDITION_LABELS: dict[str, str] = {
    "stage1_1k_sgpa_male": "1k SGPA / Male TTS",
    "stage1_1k_sgpa_female": "1k SGPA / Female TTS",
    "stage1_1k_sgpa_original": "1k SGPA / Original",
    "stage2_500_sgpa_original": "500 SGPA / Original",
    "stage2_500_raw_original": "500 Raw / Original",
}

CONDITION_ORDER: list[str] = [
    "1k SGPA / Male TTS",
    "1k SGPA / Female TTS",
    "1k SGPA / Original",
    "500 SGPA / Original",
    "500 Raw / Original",
]


def _read_mode_results(root: Path, mode: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    mode_root = root / mode
    tag = "rankwise_results" if mode == "rankwise" else "results"
    combined_name = (
        "combined_rankwise_results.csv"
        if mode == "rankwise"
        else "combined_results.csv"
    )

    for run_dir in sorted(p for p in mode_root.glob("*") if p.is_dir()):
        result_path = run_dir / combined_name
        if result_path.exists():
            df = pd.read_csv(result_path)
        else:
            parts = sorted(run_dir.glob(f"*_part*-of*_{tag}.csv"))
            if not parts:
                continue
            df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

        label = CONDITION_LABELS.get(run_dir.name, run_dir.name)
        df = df.copy()
        df["run_label"] = run_dir.name
        df["condition"] = label
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No {mode} result CSVs found under {mode_root}")
    return pd.concat(frames, ignore_index=True)


def _ordered_conditions(df: pd.DataFrame) -> list[str]:
    present = set(df["condition"].dropna().astype(str))
    ordered = [c for c in CONDITION_ORDER if c in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _within_sample_spearmans(df: pd.DataFrame, drop_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, sample_id), grp in df.groupby(["condition", "sample_id"]):
        if grp["segment_abs_sv"].nunique(dropna=True) < 2:
            continue
        if grp[drop_col].nunique(dropna=True) < 2:
            continue
        rho = stats.spearmanr(grp["segment_abs_sv"], grp[drop_col]).statistic
        if np.isfinite(rho):
            rows.append(
                {
                    "condition": condition,
                    "sample_id": int(sample_id),
                    "spearman_abs_sv_vs_drop": float(rho),
                }
            )
    return pd.DataFrame(rows)


def _per_sample_top_minus_rest(df: pd.DataFrame, drop_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, sample_id), grp in df.groupby(["condition", "sample_id"]):
        top = grp.loc[grp["segment_rank_abs_sv"] == 1, drop_col]
        rest = grp.loc[grp["segment_rank_abs_sv"] > 1, drop_col]
        if top.empty or rest.empty:
            continue
        rows.append(
            {
                "condition": condition,
                "sample_id": int(sample_id),
                "top_drop": float(top.iloc[0]),
                "rest_mean_drop": float(rest.mean()),
                "top_minus_rest": float(top.iloc[0] - rest.mean()),
                "n_segments": int(grp["n_segments"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _rank_informativeness(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, grp in df.groupby("condition"):
        samples = []
        for sample_id, sample_grp in grp.groupby("sample_id"):
            unique_abs = int(sample_grp["segment_abs_sv"].nunique(dropna=True))
            top_gap = sample_grp["top1_top2_gap"].dropna()
            samples.append(
                {
                    "sample_id": int(sample_id),
                    "unique_abs_sv": unique_abs,
                    "is_tied": unique_abs <= 1,
                    "top1_top2_gap": float(top_gap.iloc[0])
                    if not top_gap.empty
                    else 0.0,
                    "top1_share": float(sample_grp["top1_share"].iloc[0]),
                    "entropy": float(sample_grp["abs_sv_entropy_norm"].iloc[0]),
                }
            )
        sample_df = pd.DataFrame(samples)
        rows.append(
            {
                "condition": condition,
                "samples": int(len(sample_df)),
                "tied_abs_sv_rate": float(sample_df["is_tied"].mean()),
                "mean_unique_abs_sv": float(sample_df["unique_abs_sv"].mean()),
                "mean_top1_top2_gap": float(sample_df["top1_top2_gap"].mean()),
                "mean_top1_share": float(sample_df["top1_share"].mean()),
                "mean_entropy": float(sample_df["entropy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(values: pd.Series, seed: int = 20260501) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) else float("nan")
        return val, val
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(5000)]
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def plot_sv_sanity(root: Path, output_dir: Path, metric: str = "tfidf") -> Path:
    """Single-panel answer plot: does highest-SV deletion matter more than the rest?"""
    drop_col = "tfidf_deletion_drop" if metric == "tfidf" else "deletion_drop"
    df = _read_mode_results(root, "rankwise")
    if drop_col not in df.columns:
        raise KeyError(f"Missing rankwise metric column: {drop_col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    order = _ordered_conditions(df)
    top_minus_df = _per_sample_top_minus_rest(df, drop_col)
    info_df = _rank_informativeness(df).set_index("condition")

    summary_rows: list[dict[str, Any]] = []
    for condition in order:
        vals = top_minus_df.loc[
            top_minus_df["condition"] == condition, "top_minus_rest"
        ]
        if vals.empty:
            continue
        ci_low, ci_high = _bootstrap_ci(vals)
        info = info_df.loc[condition]
        summary_rows.append(
            {
                "condition": condition,
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "tied_abs_sv_rate": float(info["tied_abs_sv_rate"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    order = summary["condition"].tolist()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.16, top=0.86)

    y_lookup = {condition: i for i, condition in enumerate(order)}
    rng = np.random.default_rng(20260501)
    for condition in order:
        vals = top_minus_df.loc[
            top_minus_df["condition"] == condition, "top_minus_rest"
        ].to_numpy(dtype=float)
        y = y_lookup[condition]
        jitter = rng.normal(loc=0.0, scale=0.055, size=len(vals))
        color = "#aaaaaa" if len(vals) < 10 else "#4c78a8"
        alpha = 0.65 if len(vals) < 10 else 0.45
        ax.scatter(
            vals,
            np.full(len(vals), y) + jitter,
            s=22,
            color=color,
            alpha=alpha,
            edgecolor="none",
            zorder=2,
        )

    for _, row in summary.iterrows():
        y = y_lookup[row["condition"]]
        color = (
            "#2f855a"
            if row["ci_low"] > 0
            else "#c2410c"
            if row["ci_high"] < 0
            else "#222222"
        )
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            color=color,
            linewidth=4,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            row["mean"],
            y,
            marker="D",
            s=70,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.85,
            y,
            f"n={int(row['n'])}, tied |SV|={row['tied_abs_sv_rate']:.0%}",
            va="center",
            ha="right",
            fontsize=9,
            color="#444444",
        )

    ax.axvspan(-0.75, 0, color="#fee2e2", alpha=0.35, zorder=0)
    ax.axvspan(0, 0.95, color="#dcfce7", alpha=0.35, zorder=0)
    ax.axvline(0, color="#111111", linestyle="--", linewidth=1.4, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(-0.75, 0.95)
    ax.set_xlabel(
        "SV sanity check: top-SV deletion drop - average other-segment deletion drop"
    )
    ax.set_title(
        "Does the Shapley Ranking Identify More Important Audio Segments?",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        -0.37,
        -0.78,
        "top-SV segment is not more important",
        ha="center",
        va="center",
        fontsize=9,
        color="#7f1d1d",
    )
    ax.text(
        0.47,
        -0.78,
        "top-SV segment changes response more",
        ha="center",
        va="center",
        fontsize=9,
        color="#14532d",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_dir / f"rebuttal_sv_sanity_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    (output_dir / f"rebuttal_sv_sanity_{metric}_meta.json").write_text(
        json.dumps(
            {
                "metric": metric,
                "interpretation": (
                    "Positive values mean the highest-SV segment deletion changes the "
                    "response more than deleting the average other segment. Values near "
                    "or below zero do not support rank-wise SV faithfulness."
                ),
                "conditions": summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    return out_path


def plot_rankwise(root: Path, output_dir: Path, metric: str = "tfidf") -> Path:
    drop_col = "tfidf_deletion_drop" if metric == "tfidf" else "deletion_drop"
    df = _read_mode_results(root, "rankwise")
    if drop_col not in df.columns:
        raise KeyError(f"Missing rankwise metric column: {drop_col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    order = _ordered_conditions(df)
    palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))
    rank_counts = df.groupby(["condition", "segment_rank_abs_sv"]).size()
    keep = rank_counts[rank_counts >= 5].reset_index()["segment_rank_abs_sv"].unique()
    rank_df = df[df["segment_rank_abs_sv"].isin(keep)].copy()
    top_minus_df = _per_sample_top_minus_rest(df, drop_col)
    info_df = (
        _rank_informativeness(df).set_index("condition").reindex(order).reset_index()
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.10, top=0.90, hspace=0.42, wspace=0.32
    )
    (ax0, ax1), (ax2, ax3) = axes

    sns.boxplot(
        data=top_minus_df,
        x="condition",
        y="top_minus_rest",
        order=order,
        palette=palette,
        showfliers=False,
        ax=ax0,
    )
    sns.stripplot(
        data=top_minus_df,
        x="condition",
        y="top_minus_rest",
        order=order,
        color="#333333",
        alpha=0.35,
        jitter=0.18,
        size=2.8,
        ax=ax0,
    )
    ax0.axhline(0, color="#444444", linestyle="--", linewidth=1.1)
    ax0.set_title(
        "Does Rank 1 Matter More Than the Rest?", loc="left", fontweight="bold"
    )
    ax0.set_xlabel("")
    ax0.set_ylabel("Per-sample: rank-1 drop - mean non-rank-1 drop")
    ax0.tick_params(axis="x", rotation=25)

    point_df = (
        top_minus_df.groupby("condition")
        .agg(n=("sample_id", "size"), mean=("top_minus_rest", "mean"))
        .reindex(order)
    )
    for i, condition in enumerate(order):
        if condition in point_df.index and pd.notna(point_df.loc[condition, "n"]):
            ax0.text(
                i,
                ax0.get_ylim()[1],
                f"n={int(point_df.loc[condition, 'n'])}",
                ha="center",
                va="top",
                fontsize=8,
            )

    for condition in order:
        grp = rank_df[rank_df["condition"] == condition]
        if grp.empty:
            continue
        grouped = grp.groupby("segment_rank_abs_sv")[drop_col]
        means = grouped.mean()
        sem = grouped.sem().fillna(0.0)
        counts = grouped.count()
        ci95 = sem * stats.t.ppf(0.975, df=np.maximum(counts - 1, 1))
        ax1.errorbar(
            means.index.astype(int),
            means.values,
            yerr=ci95.values,
            marker="o",
            linewidth=2,
            markersize=4,
            capsize=2,
            color=palette[condition],
            label=condition,
        )
    ax1.set_title("Mean Deletion Drop by Rank", loc="left", fontweight="bold")
    ax1.set_xlabel("Segment rank by |SV| (1 = highest)")
    ax1.set_ylabel("Response-similarity drop")
    ax1.legend(frameon=False, fontsize=8)

    x = np.arange(len(info_df))
    bars = ax2.bar(
        x,
        info_df["tied_abs_sv_rate"] * 100.0,
        color=[palette[c] for c in info_df["condition"]],
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(info_df["condition"], rotation=25, ha="right")
    ax2.set_ylim(0, 105)
    ax2.set_title("Can the Segments Actually Be Ranked?", loc="left", fontweight="bold")
    ax2.set_ylabel("Samples with all segment |SV| values tied (%)")
    for bar, unique_mean in zip(bars, info_df["mean_unique_abs_sv"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{unique_mean:.1f} uniq.",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    sns.boxplot(
        data=top_minus_df[top_minus_df["condition"].str.contains("Original")],
        x="condition",
        y="top_minus_rest",
        order=[c for c in order if "Original" in c],
        palette=palette,
        showfliers=False,
        ax=ax3,
    )
    sns.stripplot(
        data=top_minus_df[top_minus_df["condition"].str.contains("Original")],
        x="condition",
        y="top_minus_rest",
        order=[c for c in order if "Original" in c],
        color="#333333",
        alpha=0.35,
        jitter=0.18,
        size=2.8,
        ax=ax3,
    )
    ax3.axhline(0, color="#444444", linestyle="--", linewidth=1.1)
    ax3.set_title("Original Audio: SGPA vs Raw", loc="left", fontweight="bold")
    ax3.set_xlabel("")
    ax3.set_ylabel("Rank-1 advantage")
    ax3.tick_params(axis="x", rotation=25)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Rebuttal Faithfulness: Rank-Wise Deletion ({metric.upper()})",
        fontweight="bold",
    )
    out_path = output_dir / f"rebuttal_rankwise_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "metric": metric,
        "rows": int(len(df)),
        "top_minus_rest": top_minus_df.groupby("condition")["top_minus_rest"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .to_dict(orient="records"),
        "rank_informativeness": info_df.to_dict(orient="records"),
    }
    (output_dir / f"rebuttal_rankwise_{metric}_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    return out_path


def plot_aggregate(root: Path, output_dir: Path, metric: str = "tfidf") -> Path:
    diff_col = "tfidf_drop_difference" if metric == "tfidf" else "drop_difference"
    top_col = "tfidf_top_drop" if metric == "tfidf" else "top_drop"
    rand_col = "tfidf_mean_random_drop" if metric == "tfidf" else "mean_random_drop"
    df = _read_mode_results(root, "aggregate")
    for col in (diff_col, top_col, rand_col):
        if col not in df.columns:
            raise KeyError(f"Missing aggregate metric column: {col}")

    output_dir.mkdir(parents=True, exist_ok=True)
    order = _ordered_conditions(df)
    palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))
    long_df = df.melt(
        id_vars=["condition", "sample_id"],
        value_vars=[top_col, rand_col],
        var_name="condition_type",
        value_name="drop",
    )
    long_df["condition_type"] = long_df["condition_type"].map(
        {top_col: "Top-SV deletion", rand_col: "Uniform random"}
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.84, wspace=0.28)

    sns.pointplot(
        data=long_df,
        x="condition",
        y="drop",
        hue="condition_type",
        order=order,
        errorbar=("ci", 95),
        dodge=0.25,
        ax=axes[0],
    )
    axes[0].set_title("Top-SV vs Random Deletion", loc="left", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Response-similarity drop")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, fontsize=8)

    sns.boxplot(
        data=df,
        x="condition",
        y=diff_col,
        order=order,
        palette=palette,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="condition",
        y=diff_col,
        order=order,
        color="#333333",
        alpha=0.35,
        jitter=0.18,
        size=3,
        ax=axes[1],
    )
    axes[1].axhline(0, color="#444444", linestyle="--", linewidth=1.1)
    axes[1].set_title("Paired Drop Difference", loc="left", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Top-SV drop - random drop")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Rebuttal Faithfulness: Aggregate Deletion ({metric.upper()})",
        fontweight="bold",
    )
    out_path = output_dir / f"rebuttal_aggregate_{metric}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = (
        df.groupby("condition")[diff_col]
        .agg(["count", "mean", "median", "std"])
        .reindex(order)
        .reset_index()
        .to_dict(orient="records")
    )
    (output_dir / f"rebuttal_aggregate_{metric}_meta.json").write_text(
        json.dumps({"metric": metric, "drop_difference": meta}, indent=2)
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("sanity", "rankwise", "aggregate", "both"), default="sanity"
    )
    parser.add_argument("--metric", choices=("tfidf", "embedding"), default="tfidf")
    args = parser.parse_args()

    outputs: list[Path] = []
    if args.mode == "sanity":
        outputs.append(plot_sv_sanity(args.root, args.output_dir, metric=args.metric))
    if args.mode in ("rankwise", "both"):
        outputs.append(plot_rankwise(args.root, args.output_dir, metric=args.metric))
    if args.mode in ("aggregate", "both"):
        outputs.append(plot_aggregate(args.root, args.output_dir, metric=args.metric))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

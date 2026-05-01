"""Summarization and partition-combining functions."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .helpers import EPS, estimate_required_paired_n


def _wilcoxon_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    """One-sided Wilcoxon signed-rank test (greater).

    Returns the test statistic and one-sided p-value when applicable; otherwise
    returns (None, None) for insufficient data or degenerate inputs.
    """
    if values.size < 3:
        return None, None
    if np.allclose(values, 0.0):
        return None, None
    try:
        stat, p_value = stats.wilcoxon(
            values, alternative="greater", zero_method="wilcox"
        )
    except ValueError:
        return None, None
    return float(stat), float(p_value)


def _paired_t_greater(values: np.ndarray) -> tuple[float | None, float | None]:
    """One-sided paired t-test proxy.

    Computes a one-sample t-test against zero and converts the two-sided
    p-value into a one-sided p-value in the direction of the observed mean.
    Returns (stat, p) or (None, None) if not applicable.
    """
    if values.size < 2:
        return None, None
    if np.allclose(values, 0.0):
        return None, None
    stat, p_value_two_sided = stats.ttest_1samp(values, popmean=0.0)
    if not np.isfinite(stat) or not np.isfinite(p_value_two_sided):
        return None, None
    # Convert two-sided p-value into one-sided in the expected direction.
    if stat >= 0:
        p_value = p_value_two_sided / 2.0
    else:
        p_value = 1.0 - (p_value_two_sided / 2.0)
    return float(stat), float(p_value)


def _bh_fdr(p_values: list[float | None]) -> list[float | None]:
    """Benjamini–Hochberg false discovery rate correction.

    Accepts a list of p-values (or None) and returns a list of adjusted q-values
    (or None for entries that were not valid p-values).
    """
    valid = [
        (idx, p) for idx, p in enumerate(p_values) if p is not None and np.isfinite(p)
    ]
    if not valid:
        return [None] * len(p_values)

    order = np.argsort([p for _, p in valid])
    ranked = [(valid[i][0], valid[i][1]) for i in order]
    m = len(ranked)
    adjusted = [0.0] * m
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        raw = ranked[i][1] * m / rank
        prev = min(prev, raw)
        adjusted[i] = min(prev, 1.0)

    out: list[float | None] = [None] * len(p_values)
    for i, (orig_idx, _) in enumerate(ranked):
        out[orig_idx] = float(adjusted[i])
    return out


def _summarize_delta(values: np.ndarray) -> dict[str, float | int | None]:
    """Produce summary statistics for a paired-delta array.

    Returns a dictionary with n, mean, std, median, positive_rate, effect size
    (Cohen's dz), and test statistics (Wilcoxon and paired t where available).
    """
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "positive_rate": None,
            "cohen_dz": None,
            "wilcoxon_stat": None,
            "wilcoxon_p_value": None,
            "paired_t_stat": None,
            "paired_t_p_value": None,
        }

    std = float(np.std(values, ddof=1)) if n >= 2 else None
    wilcoxon_stat, wilcoxon_p = _wilcoxon_greater(values)
    t_stat, t_p = _paired_t_greater(values)
    return {
        "n": n,
        "mean": float(np.mean(values)),
        "std": std,
        "median": float(np.median(values)),
        "positive_rate": float(np.mean(values > 0.0)),
        "cohen_dz": float(np.mean(values) / ((std if std is not None else 0.0) + EPS))
        if n >= 2
        else None,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "paired_t_stat": t_stat,
        "paired_t_p_value": t_p,
    }


def summarize(
    results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    target_effect_size_dz: float = 0.5,
) -> dict[str, Any]:
    """Summarize a set of per-sample `results` into aggregated statistics.

    Produces top-vs-random comparisons, multiple test statistics (Wilcoxon,
    paired t proxy), BH-FDR corrected q-values, basic descriptive stats, and
    a simple power-planning estimate using `estimate_required_paired_n`.
    """
    if results_df.empty:
        return {"completed_samples": 0, "failed_samples": int(len(failures_df))}

    primary_diff = results_df["drop_difference"].to_numpy(dtype=float)
    legacy_top = results_df["top_drop"].to_numpy(dtype=float)
    legacy_rand = results_df["mean_random_drop"].to_numpy(dtype=float)

    tests = {
        "pos_uniform": "drop_difference",
        "pos_stratified": "pos_stratified_drop_difference",
        "neg_uniform": "neg_drop_improvement",
        "neg_stratified": "neg_stratified_drop_improvement",
        "comprehensiveness_uniform": "comprehensiveness_drop_difference",
        "comprehensiveness_stratified": "comprehensiveness_stratified_drop_difference",
        "sufficiency_uniform": "sufficiency_advantage",
        "sufficiency_stratified": "sufficiency_stratified_advantage",
        "monotonicity_embedding": "monotonicity_score",
        "tfidf_pos_uniform": "tfidf_drop_difference",
        "tfidf_pos_stratified": "tfidf_pos_stratified_drop_difference",
        "tfidf_neg_uniform": "tfidf_neg_drop_improvement",
        "tfidf_neg_stratified": "tfidf_neg_stratified_drop_improvement",
        "tfidf_comprehensiveness_uniform": "tfidf_comprehensiveness_drop_difference",
        "tfidf_comprehensiveness_stratified": "tfidf_comprehensiveness_stratified_drop_difference",
        "tfidf_sufficiency_uniform": "tfidf_sufficiency_advantage",
        "tfidf_sufficiency_stratified": "tfidf_sufficiency_stratified_advantage",
        "monotonicity_tfidf": "tfidf_monotonicity_score",
    }

    test_stats: dict[str, dict[str, float | int | None]] = {}
    pvals: list[float | None] = []
    pval_keys: list[str] = []
    for test_name, col in tests.items():
        values = results_df[col].dropna().to_numpy(dtype=float)
        stats_row = _summarize_delta(values)
        test_stats[test_name] = stats_row
        pvals.append(
            stats_row["wilcoxon_p_value"] if isinstance(stats_row, dict) else None
        )
        pval_keys.append(test_name)

    qvals = _bh_fdr(pvals)
    for key, qval in zip(pval_keys, qvals):
        test_stats[key]["wilcoxon_p_value_bh_fdr"] = qval

    required_n = estimate_required_paired_n(target_effect_size_dz=target_effect_size_dz)

    return {
        "completed_samples": int(len(results_df)),
        "failed_samples": int(len(failures_df)),
        "mean_top_drop": float(np.mean(legacy_top)),
        "std_top_drop": float(np.std(legacy_top, ddof=1))
        if len(legacy_top) >= 2
        else None,
        "mean_random_drop": float(np.mean(legacy_rand)),
        "std_random_drop": float(np.std(legacy_rand, ddof=1))
        if len(legacy_rand) >= 2
        else None,
        "mean_drop_difference": float(np.mean(primary_diff)),
        "median_drop_difference": float(np.median(primary_diff)),
        "top_greater_than_random_rate": float(np.mean(primary_diff > 0)),
        "tests": test_stats,
        "power_planning": {
            "target_effect_size_dz": float(target_effect_size_dz),
            "alpha": 0.05,
            "power": 0.8,
            "estimated_required_n": required_n,
        },
        "mean_runtime_sec": float(results_df["runtime_sec"].mean()),
    }


def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Compute Spearman correlation safely, returning None for invalid inputs."""
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return None
    corr = stats.spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float)).statistic
    return float(corr) if math.isfinite(float(corr)) else None


def summarize_rankwise(
    results_df: pd.DataFrame, failures_df: pd.DataFrame
) -> dict[str, Any]:
    """Summarize rank-wise deletion results across samples.

    Aggregates deletion drops by absolute-SV rank, computes within-sample and
    global Spearman correlations between absolute-SV and deletion drop, and
    returns per-rank descriptive statistics and diagnostics.
    """
    if results_df.empty:
        return {
            "completed_deletions": 0,
            "completed_samples": 0,
            "failed_samples": int(len(failures_df)),
        }

    per_rank = (
        results_df.groupby("segment_rank_abs_sv", as_index=True)
        .agg(
            n=("deletion_drop", "size"),
            mean_drop=("deletion_drop", "mean"),
            median_drop=("deletion_drop", "median"),
            mean_abs_sv_share=("segment_abs_sv_share", "mean"),
            mean_abs_sv=("segment_abs_sv", "mean"),
        )
        .sort_index()
    )
    per_rank_summary = {
        str(int(rank)): {k: float(v) for k, v in row.items()}
        for rank, row in per_rank.iterrows()
    }

    top_rows = results_df[results_df["segment_rank_abs_sv"] == 1]
    non_top_rows = results_df[results_df["segment_rank_abs_sv"] > 1]
    sample_count = int(
        results_df[["audio_column", "sample_id"]].drop_duplicates().shape[0]
    )

    per_sample_corrs: list[float] = []
    for _, group in results_df.groupby(["audio_column", "sample_id"]):
        corr = _safe_spearman(group["segment_abs_sv"], group["deletion_drop"])
        if corr is not None:
            per_sample_corrs.append(corr)

    global_corr = _safe_spearman(
        results_df["segment_abs_sv"], results_df["deletion_drop"]
    )
    global_rank_corr = _safe_spearman(
        -results_df["segment_rank_abs_sv"], results_df["deletion_drop"]
    )
    top_minus = (
        float(top_rows["deletion_drop"].mean() - non_top_rows["deletion_drop"].mean())
        if not top_rows.empty and not non_top_rows.empty
        else None
    )

    return {
        "completed_deletions": int(len(results_df)),
        "completed_samples": sample_count,
        "failed_samples": int(len(failures_df)),
        "mean_deletion_drop": float(results_df["deletion_drop"].mean()),
        "mean_top_rank_drop": float(top_rows["deletion_drop"].mean())
        if not top_rows.empty
        else None,
        "mean_non_top_rank_drop": float(non_top_rows["deletion_drop"].mean())
        if not non_top_rows.empty
        else None,
        "mean_top_minus_non_top_drop": top_minus,
        "spearman_abs_sv_vs_drop": global_corr,
        "spearman_negative_rank_vs_drop": global_rank_corr,
        "mean_within_sample_spearman_abs_sv_vs_drop": float(np.mean(per_sample_corrs))
        if per_sample_corrs
        else None,
        "median_within_sample_spearman_abs_sv_vs_drop": float(
            np.median(per_sample_corrs)
        )
        if per_sample_corrs
        else None,
        "within_sample_spearman_n": len(per_sample_corrs),
        "mean_top1_share": float(
            results_df[["audio_column", "sample_id", "top1_share"]]
            .drop_duplicates()["top1_share"]
            .mean()
        ),
        "mean_top1_top2_gap": float(
            results_df[["audio_column", "sample_id", "top1_top2_gap"]]
            .drop_duplicates()["top1_top2_gap"]
            .dropna()
            .mean()
        ),
        "mean_abs_sv_entropy_norm": float(
            results_df[["audio_column", "sample_id", "abs_sv_entropy_norm"]]
            .drop_duplicates()["abs_sv_entropy_norm"]
            .dropna()
            .mean()
        ),
        "per_rank": per_rank_summary,
        "mean_runtime_sec_per_deletion": float(results_df["runtime_sec"].mean()),
    }


def combine_partition_outputs(
    output_dir: Path, all_rank_deletions: bool = False
) -> dict[str, Any]:
    """Combine per-partition CSV outputs into consolidated CSV/JSON summaries.

    Scans `output_dir` for per-partition CSV files, concatenates them,
    deduplicates, writes combined CSVs, and writes per-voice and combined
    summary JSON files using `summarize` / `summarize_rankwise`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    combined_frames: list[pd.DataFrame] = []
    combined_failure_frames: list[pd.DataFrame] = []
    tag = "rankwise_results" if all_rank_deletions else "results"
    summary_tag = "rankwise_summary" if all_rank_deletions else "summary"

    for audio_column in ("audio__male", "audio__female"):
        paths = sorted(output_dir.glob(f"{audio_column}_part*-of*_{tag}.csv"))
        if not paths:
            continue
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        dedup_cols = (
            ["sample_id", "segment_idx"] if all_rank_deletions else ["sample_id"]
        )
        sort_cols = (
            ["sample_id", "segment_rank_abs_sv", "segment_idx"]
            if all_rank_deletions
            else ["sample_id"]
        )
        df = df.drop_duplicates(subset=dedup_cols, keep="last").sort_values(sort_cols)

        result_path = output_dir / f"{audio_column}_combined_{tag}.csv"
        df.to_csv(result_path, index=False)

        failures_paths = sorted(
            output_dir.glob(f"{audio_column}_part*-of*_failures.csv")
        )
        failures_df = (
            pd.concat([pd.read_csv(p) for p in failures_paths], ignore_index=True)
            if failures_paths
            else pd.DataFrame()
        )
        if not failures_df.empty:
            failures_df = failures_df.drop_duplicates(
                subset=["sample_id", "audio_column", "error_type", "error_message"],
                keep="last",
            )
            combined_failure_frames.append(failures_df)

        summary = (
            summarize_rankwise(df, failures_df)
            if all_rank_deletions
            else summarize(df, failures_df)
        )
        summary.update({"audio_column": audio_column, "results_csv": str(result_path)})
        (output_dir / f"{audio_column}_combined_{summary_tag}.json").write_text(
            json.dumps(summary, indent=2)
        )
        summaries[audio_column] = summary
        combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        sort_cols = (
            ["audio_column", "sample_id", "segment_rank_abs_sv", "segment_idx"]
            if all_rank_deletions
            else ["audio_column", "sample_id"]
        )
        combined_df = combined_df.sort_values(sort_cols)
        combined_name = (
            "combined_rankwise_results.csv"
            if all_rank_deletions
            else "combined_results.csv"
        )
        combined_path = output_dir / combined_name
        combined_df.to_csv(combined_path, index=False)

        combined_failures = (
            pd.concat(combined_failure_frames, ignore_index=True)
            if combined_failure_frames
            else pd.DataFrame()
        )
        combined_summary = (
            summarize_rankwise(combined_df, combined_failures)
            if all_rank_deletions
            else summarize(combined_df, combined_failures)
        )
        combined_summary["results_csv"] = str(combined_path)
        summary_name = (
            "combined_rankwise_summary.json"
            if all_rank_deletions
            else "combined_summary.json"
        )
        (output_dir / summary_name).write_text(json.dumps(combined_summary, indent=2))
        summaries["combined"] = combined_summary

    return summaries
